"""Run experiment forwards through nnsight, either locally or remotely on NDIF.

Natural generation uses one request; each path batch uses three forwards. Large
per-sample tensors stay inside the trace; only distances, top-three candidates, and
small endpoint summaries are returned.
"""
from __future__ import annotations

import gc
import os
import threading
from collections import OrderedDict
from dataclasses import dataclass
from functools import reduce

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import nnsight
import torch
import transformers
from nnsight import LanguageModel, ndif
from nnsight.intervention.backends.remote import RemoteBackend

from plateau.core import math as core_math
from plateau.core.math import arc_segments, effect_metrics, interpolate_tokens, top_candidates
from plateau.core.models import ARCHITECTURES


def _hidden(output):
    # Transformers 5 blocks return a tensor; older versions (possibly on NDIF) a tuple.
    return output[0] if isinstance(output, tuple) else output


@dataclass
class LoadedModel:
    model_id: str
    model: LanguageModel
    blocks: object
    head: object

    @property
    def tokenizer(self):
        return self.model.tokenizer

    @property
    def config(self):
        return self.model.config

    @property
    def context_limit(self):
        return self.config.max_position_embeddings


class NnsightBackend:
    """One instance per job; loaded models are shared across instances.

    In remote mode models are meta-device shells (config + tokenizer), so several
    are cached and used by concurrent jobs. Locally only one model is kept.
    `api_key` is the NDIF key for this job's requests (falls back to NDIF_API_KEY).
    """
    _models: OrderedDict = OrderedDict()
    _lock = threading.Lock()
    REMOTE_CACHE = 8

    def __init__(self, remote: bool, device: str | None = None, api_key: str | None = None):
        self.remote = remote
        self.api_key = api_key
        if remote:
            # Helpers called inside traces must be shipped with the request.
            ndif.register(core_math)
            self.device = "cpu"  # inputs are built locally and sent to NDIF
        else:
            self.device = device or ("mps" if torch.backends.mps.is_available()
                                     else "cuda" if torch.cuda.is_available() else "cpu")

    def clear_cache(self):
        if not self.remote:
            gc.collect()
            if self.device == "mps":
                torch.mps.empty_cache()
            elif self.device.startswith("cuda"):
                torch.cuda.empty_cache()

    def describe(self):
        return {"remote": self.remote, "device": "NDIF" if self.remote else self.device,
                "nnsight_version": nnsight.__version__, "transformers_version": transformers.__version__}

    def load(self, model_id, repo):
        key = (self.remote, self.device, model_id, repo)
        with self._lock:
            if key in self._models:
                self._models.move_to_end(key)
                return self._models[key]
            if not self.remote:
                self._models.clear()
                gc.collect()
                if self.device == "mps":
                    torch.mps.empty_cache()
                elif self.device.startswith("cuda"):
                    torch.cuda.empty_cache()
            loaded = self._load(model_id, repo)
            self._models[key] = loaded
            while len(self._models) > (self.REMOTE_CACHE if self.remote else 1):
                self._models.popitem(last=False)
            return loaded

    def _load(self, model_id, repo):
        if self.remote:
            # Remote models stay on the meta device; only config and tokenizer are downloaded.
            model = LanguageModel(repo)
        else:
            dtype_key = "torch_dtype" if int(transformers.__version__.split(".")[0]) < 5 else "dtype"
            model = LanguageModel(repo, device_map=self.device, dispatch=True,
                                  **{dtype_key: torch.float32}, attn_implementation="eager")
        arch = ARCHITECTURES.get(model.config.model_type)
        if arch is None:
            raise ValueError("This model architecture is not supported.")
        attr = lambda path: reduce(getattr, path.split("."), model)
        try:
            head = attr(arch["head"])
        except AttributeError:
            # Transformers 5 renamed GPT-NeoX's embed_out to lm_head. Keep the
            # legacy path for older local/NDIF runtimes and inspect this model.
            if model.config.model_type != "gpt_neox":
                raise
            head = attr("lm_head")
        return LoadedModel(model_id, model, attr(arch["blocks"]), head)

    def _execution(self, lm: LoadedModel):
        """Trace kwargs: a per-job NDIF backend carrying this job's key, or local execution."""
        if not self.remote:
            return {}
        return {"backend": RemoteBackend(lm.model.to_model_key(), api_key=self.api_key or "")}

    def _input(self, rows):
        ids = torch.tensor(rows, device=self.device)
        return ids, torch.ones_like(ids)

    @torch.no_grad()
    def natural(self, lm: LoadedModel, ids, patch_layer, patch_start, max_new_tokens):
        """Greedy generation plus the natural forward's states and logits.

        The first generation step is the natural forward over the full prompt.
        EOS is suppressed (min_new_tokens) so every step runs; the raw argmax of
        each step is recorded and the caller truncates at the first EOS. Stopping
        early would discard everything saved in the trace.
        """
        inp, mask = self._input([ids])
        pad = lm.tokenizer.pad_token_id if lm.tokenizer.pad_token_id is not None else lm.tokenizer.eos_token_id
        # NDIF cannot deserialize the LoadedModel dataclass; only reference its modules inside traces.
        blocks, head = lm.blocks, lm.head
        with lm.model.generate(inp, attention_mask=mask, max_new_tokens=max_new_tokens,
                               min_new_tokens=max_new_tokens, do_sample=False, pad_token_id=pad,
                               **self._execution(lm)) as tracer:
            out = dict().save()
            tokens = list().save()
            probabilities = list().save()
            first = True
            for _ in tracer.iter[:]:
                if first:
                    last_states = []
                    for layer, block in enumerate(blocks):
                        hidden = _hidden(block.output)
                        if layer == patch_layer:
                            out["source"] = hidden[0, patch_start:].clone()
                        # Large models are split across GPUs; gather per-layer results on the CPU.
                        last_states.append(hidden[0, -1].float().cpu())
                    out["layer_states"] = torch.stack(last_states)
                logits = head.output[0, -1].float()
                if first:
                    out["logits"] = logits.clone()
                    first = False
                probs = logits.softmax(-1)
                token = probs.argmax()
                tokens.append(token)
                probabilities.append(probs[token])
        return {"source": out["source"].cpu(), "layer_states": out["layer_states"].cpu(),
                "logits": out["logits"].cpu(), "dtype": str(out["source"].dtype).replace("torch.", ""),
                "tokens": [int(t) for t in tokens], "probabilities": [float(p) for p in probabilities]}

    @torch.no_grad()
    def path(self, lm: LoadedModel, context_ids, patch_layer, patch_start, sources, ts, method,
             record_layers, include_reference_logits=False, previous=None, cancelled=lambda: False):
        """Patch interpolated states into the fixed context and measure every effect metric.

        Rows 0 and 1 are the reference endpoints (sources A and B patched in the
        same context); the remaining rows are the path samples at `ts`.
        """
        inp, mask = self._input([context_ids] * (len(ts) + 2))
        sources, ts = sources.to(self.device), ts.to(self.device)
        previous = previous or {}
        record_layers = list(record_layers)
        blocks, head = lm.blocks, lm.head
        with lm.model.trace(inp, attention_mask=mask, use_cache=False, **self._execution(lm)):
            out = dict().save()
            patch = torch.cat([sources, interpolate_tokens(sources[0], sources[1], ts, method)])
            endpoint_l2 = []
            for layer, block in enumerate(blocks):
                hidden = _hidden(block.output)
                if layer == patch_layer:
                    hidden[:, patch_start:, :] = patch.to(hidden.dtype)
                last = hidden[:, -1].float()
                # Per-layer endpoint L2, moved to the CPU because layers may sit on different GPUs.
                endpoint_l2.append(torch.linalg.vector_norm(last[0] - last[1]).cpu())
                if layer in record_layers:
                    out[str(layer)] = effect_metrics(last[2:], last[0], last[1])
                    out["arc_" + str(layer)] = arc_segments(last[2:], previous.get(str(layer)))
            logits = head.output[:, -1].float()
            out["logits"] = effect_metrics(logits[2:], logits[0], logits[1])
            out["arc_logits"] = arc_segments(logits[2:], previous.get("logits"))
            out["endpoint_l2"] = torch.stack(endpoint_l2)
            out["tokens"] = logits.argmax(-1)
            out["top_ids"], out["top_probs"] = top_candidates(logits)
            out["first_error"] = (logits[2] - logits[0]).abs().max()
            out["last_error"] = (logits[-1] - logits[1]).abs().max()
            if include_reference_logits:
                out["reference_logits"] = logits[:2].clone()
        keys = [str(layer) for layer in record_layers] + ["logits"]
        result = {
            "step_lengths": {key: out["arc_" + key][0].tolist() for key in keys},
            "last_vectors": {key: out["arc_" + key][1] for key in keys},
            "values": {key: {metric: values.cpu().tolist() for metric, values in out[key][0].items()} for key in keys},
            "gaps": {key: float(out[key][1]) for key in keys},
            "endpoint_l2": out["endpoint_l2"].cpu().tolist(),
            "reference_tokens": [int(t) for t in out["tokens"][:2]],
            "tokens": [int(t) for t in out["tokens"][2:]],
            "first_error": float(out["first_error"]), "last_error": float(out["last_error"]),
        }
        if include_reference_logits:
            result["reference_logits"] = out["reference_logits"].cpu()
        candidate_ids = [out["top_ids"][2:].cpu().tolist()]
        candidate_probs = [out["top_probs"][2:].cpu().tolist()]
        next_ids = out["top_ids"][:, :1].to(self.device)
        # Two further raw-model forwards follow the greedy choices. Patch the
        # original context span each time; never overwrite a generated token or
        # feed these later states into the trajectory measurements above.
        for _ in range(2):
            if cancelled():
                raise InterruptedError("Stopped.")
            inp = torch.cat((inp, next_ids), dim=1)
            mask = torch.ones_like(inp)
            context_length = len(context_ids)
            with lm.model.trace(inp, attention_mask=mask, use_cache=False, **self._execution(lm)):
                continuation = dict().save()
                patch = torch.cat([sources, interpolate_tokens(sources[0], sources[1], ts, method)])
                hidden = _hidden(blocks[patch_layer].output)
                hidden[:, patch_start:context_length, :] = patch.to(hidden.dtype)
                continuation["ids"], continuation["probs"] = top_candidates(head.output[:, -1].float())
            candidate_ids.append(continuation["ids"][2:].cpu().tolist())
            candidate_probs.append(continuation["probs"][2:].cpu().tolist())
            next_ids = continuation["ids"][:, :1].to(self.device)
        result["candidate_ids"] = candidate_ids
        result["candidate_probs"] = candidate_probs
        return result
