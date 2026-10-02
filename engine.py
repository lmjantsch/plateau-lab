"""Small, inspectable GPT-2 / Pythia / Qwen activation-plateau experiment engine."""
from __future__ import annotations

import gc
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import snapshot_download

from models import CACHE, DOWNLOAD_FILES, MODELS, cached_snapshot
from hardware import Hardware, is_out_of_memory
from trajectory import METRIC_DEFINITIONS, TrajectoryReadout, relative_distance, summarize
from plateau.core.results import EffectReadout, add_effect
from plateau.core.math import top_candidates
from plateau.core.predictions import sample_prediction

ROOT = Path(__file__).resolve().parent
WORD = re.compile(r"\b[^\W_]+(?:['’\-][^\W_]+)*\b", re.UNICODE)


def next_word_spans(prefix, continuation):
    # A suffix extending the prompt's final word is not a new word.
    return [m for m in WORD.finditer(prefix + continuation) if m.start() >= len(prefix)]


def interpolate(a, b, t, method="slerp"):
    """SLERP directions, linearly interpolated norms (Matthew Shinkle, footnote 1)."""
    t = torch.as_tensor(t, device=a.device, dtype=a.dtype).reshape(-1, 1)
    if method == "linear":
        return (1 - t) * a + t * b
    na, nb = a.norm(), b.norm()
    if min(na.item(), nb.item()) < 1e-10:
        return (1 - t) * a + t * b
    ua, ub = a / na, b / nb
    cos = torch.dot(ua, ub).clamp(-1, 1)
    if cos.item() < -0.9995:
        raise ValueError("The endpoint directions are nearly opposite, so the spherical path is ambiguous. Choose Linear.")
    if cos.item() > 0.9995:
        direction = (1 - t) * ua + t * ub
        direction = direction / direction.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    else:
        theta = torch.acos(cos)
        direction = (torch.sin((1 - t) * theta) * ua + torch.sin(t * theta) * ub) / torch.sin(theta)
    result = direction * ((1 - t) * na + t * nb)
    # Preserve exact endpoints and make endpoint checks meaningful.
    result = torch.where(t == 0, a, result)
    return torch.where(t == 1, b, result)


def interpolate_tokens(a, b, ts, method="slerp"):
    """Interpolate corresponding token vectors independently, with a shared t."""
    if a.shape != b.shape or a.ndim != 2 or len(a) == 0:
        raise ValueError("Interpolation requires matching, nonempty token spans.")
    if len(a) == 1:
        return interpolate(a[0], b[0], ts, method).unsqueeze(1)
    t = torch.as_tensor(ts, device=a.device, dtype=a.dtype).reshape(-1, 1, 1)
    linear = (1 - t) * a + t * b
    if method == "linear":
        return linear
    na, nb = a.norm(dim=-1, keepdim=True), b.norm(dim=-1, keepdim=True)
    nonzero = (na >= 1e-10) & (nb >= 1e-10)
    ua, ub = a / na.clamp_min(1e-10), b / nb.clamp_min(1e-10)
    cos = (ua * ub).sum(dim=-1, keepdim=True).clamp(-1, 1)
    if ((cos < -0.9995) & nonzero).any().item():
        raise ValueError("A pair of token-state directions is nearly opposite. Choose Linear interpolation.")
    direction_linear = (1 - t) * ua + t * ub
    direction_linear = direction_linear / direction_linear.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    theta = torch.acos(cos)
    direction_spherical = (torch.sin((1 - t) * theta) * ua + torch.sin(t * theta) * ub) / torch.sin(theta).clamp_min(1e-12)
    direction = torch.where(cos > 0.9995, direction_linear, direction_spherical)
    result = torch.where(nonzero, direction * ((1 - t) * na + t * nb), linear)
    return torch.where(t == 0, a, torch.where(t == 1, b, result))


class Engine:
    def __init__(self):
        self.model = self.tokenizer = self.model_id = None
        self.hardware = Hardware()

    @property
    def device(self):
        return self.hardware.device

    def load(self, model_id, progress, cancelled=lambda: False):
        if model_id not in MODELS:
            raise ValueError("Choose a model from the list.")
        if self.model_id == model_id:
            return
        label = MODELS[model_id]['label']
        progress(f"Preparing {label}…", None)
        self.model = self.tokenizer = self.model_id = None
        gc.collect()
        self.hardware.clear_cache()
        self.hardware = Hardware()  # Recheck free GPU memory after releasing the previous model.
        snapshot = cached_snapshot(model_id)
        if snapshot is None:
            progress(f"Downloading {label} to this computer. Completed files are kept for future visits; interrupted downloads can resume.", None)
            snapshot = Path(snapshot_download(MODELS[model_id]["repo"], cache_dir=str(CACHE),
                                             allow_patterns=DOWNLOAD_FILES, max_workers=2))
        if cancelled():
            raise InterruptedError("Stopped. Downloaded model files are kept for future use.")
        progress(f"Loading saved {label} from disk into memory. No download needed.", None)
        tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            snapshot, local_files_only=True, torch_dtype=torch.float32,
            attn_implementation="eager", use_safetensors=True)
        model.config._commit_hash = snapshot.name
        model.eval()
        if self.device.startswith("cuda") and self.hardware.requested == "auto":
            free, _ = torch.cuda.mem_get_info(torch.device(self.device))
            weight_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
            if weight_bytes > free * 0.9:
                self.hardware.use_cpu("The GPU has insufficient free memory for this model's float32 weights.")
        fallback = False
        try:
            model.to(self.device)
        except RuntimeError as error:
            if not is_out_of_memory(error) or self.device == "cpu" or self.hardware.requested != "auto":
                raise
            fallback = True
        if fallback:
            model.cpu()  # Also handles a partially completed device transfer.
            gc.collect()
            self.hardware.use_cpu("The model exceeded the GPU's available memory.")
        progress(f"{label} ready on {self.hardware.name} ({self.hardware.backend}) · float32.", 0.06)
        self.model, self.tokenizer, self.model_id = model, tokenizer, model_id

    def architecture(self):
        if self.model.config.model_type == "gpt2":
            return self.model.transformer.h, self.model.transformer.drop
        if self.model.config.model_type == "gpt_neox":
            return self.model.gpt_neox.layers, self.model.gpt_neox.emb_dropout
        if self.model.config.model_type in ("qwen2", "qwen3"):
            return self.model.model.layers, self.model.model.embed_tokens
        raise ValueError("This model architecture is not supported.")

    @property
    def context_limit(self):
        return self.model.config.max_position_embeddings

    def validate(self, request):
        a, b = request.get("sequence_a", ""), request.get("sequence_b", "")
        if not isinstance(a, str) or not isinstance(b, str) or not a.strip() or not b.strip():
            raise ValueError("Enter both sequences first.")
        ids_a = self.tokenizer.encode(a, add_special_tokens=False)
        ids_b = self.tokenizer.encode(b, add_special_tokens=False)
        if max(len(ids_a), len(ids_b)) > min(256, self.context_limit - 48):
            raise ValueError("This tool supports up to 256 tokens per sequence. Shorten the input.")
        if ids_a == ids_b:
            raise ValueError("Both inputs contain identical tokens. Enter two different sequences.")
        return a, b, ids_a, ids_b

    def predict(self, ids, use_cache=True):
        """Greedy decode with one word of look-ahead to avoid truncating subwords."""
        generated, probabilities = [], []
        text, matches, complete = "", [], False
        prefix = self.tokenizer.decode(ids, clean_up_tokenization_spaces=False)
        past = None
        for _ in range(min(48, self.context_limit - len(ids))):
            step_ids = [generated[-1]] if use_cache and past is not None else ids + generated
            inp = torch.tensor([step_ids], device=self.device)
            output = self.model(inp, use_cache=use_cache, past_key_values=past)
            logits = output.logits[0, -1].float()
            past = output.past_key_values if use_cache else None
            next_id = int(logits.argmax())
            if next_id == self.tokenizer.eos_token_id:
                complete = True
                break
            generated.append(next_id)
            probabilities.append(float(logits.softmax(-1)[next_id]))
            text = self.tokenizer.decode(generated, clean_up_tokenization_spaces=False)
            matches = next_word_spans(prefix, text)
            if len(matches) >= 4:
                complete = True
                break
        end = matches[2].end() - len(prefix) if len(matches) >= 3 else len(text)
        return {
            "continuation": text[:end],
            "words": [m.group() for m in matches[:3]],
            "word_count": min(3, len(matches)),
            "complete": complete,
            "tokens": [{"id": token, "text": self.tokenizer.decode([token]), "probability": p}
                       for token, p in zip(generated, probabilities)],
            "note": "Token details include look-ahead tokens used to confirm word boundaries. Only the first three new words are displayed.",
        }

    def _measure_path(self, source_states, natural_logits, context_input, patch_module,
                      blocks, record_layers, ts, method, patch_start, batch_size, progress, cancelled):
        steps = len(ts)
        captures = {}
        sample_keys = {str(layer) for layer in record_layers}
        capture_all = True
        capture_enabled = True

        def capture(name):
            def hook(module, args, output):
                if not capture_enabled or (not capture_all and name not in sample_keys):
                    return
                hidden = output[0] if isinstance(output, tuple) else output
                captures[name] = hidden[:, -1, :].detach().clone()
            return hook

        predictions_path = []
        current_patch = None

        def patch(module, args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            hidden = hidden.clone()
            # During continuation, patch only the original prompt positions.
            # Recompute the full prefix without a cache; generated tokens remain natural.
            hidden[:, patch_start:context_input.shape[1], :] = current_patch
            return (hidden,) + output[1:] if isinstance(output, tuple) else hidden

        handles = []
        endpoint_errors = {}
        try:
            handles.append(patch_module.register_forward_hook(patch))
            for layer, block in enumerate(blocks):
                handles.append(block.register_forward_hook(capture(str(layer))))
            # Define references using the SAME fixed context and intervention as
            # the path. A transferred B state need not reproduce natural B.
            if batch_size == 1:
                reference_logits = []
                reference_hidden = {str(layer): [] for layer in range(len(blocks))}
                for state in source_states:
                    current_patch = state.unsqueeze(0)
                    reference_logits.append(self.model(context_input, use_cache=False).logits[0, -1, :].float().clone())
                    for key in reference_hidden:
                        reference_hidden[key].append(captures[key][0].float().clone())
                endpoints_logits = torch.stack(reference_logits)
                endpoints = {key: torch.stack(values) for key, values in reference_hidden.items()}
            else:
                current_patch = torch.stack(source_states + [source_states[-1]] * (batch_size - 2))
                endpoints_logits = self.model(context_input.expand(batch_size, -1), use_cache=False).logits[:2, -1, :].float().clone()
                endpoints = {str(layer): captures[str(layer)][:2].float().clone() for layer in range(len(blocks))}
            # All-layer L2 needs only the two reference endpoints, not every t.
            endpoint_states = torch.stack([endpoints[str(layer)] for layer in range(len(blocks))])
            endpoint_l2 = torch.linalg.vector_norm(endpoint_states[:, 0] - endpoint_states[:, 1], dim=-1).cpu().tolist()
            del endpoint_states
            endpoints = {key: value for key, value in endpoints.items() if key in sample_keys}
            captures.clear()
            capture_all = False
            endpoint_tokens = [int(scores.argmax()) for scores in endpoints_logits]
            natural_gaps = {
                side: float((endpoints_logits[i] - natural_logits[i]).abs().max())
                for i, side in enumerate(("A", "B"))
            }
            readouts = {key: EffectReadout(value.float()) for key, value in endpoints.items()}
            readouts["logits"] = EffectReadout(endpoints_logits.float())
            for offset in range(0, steps, batch_size):
                if cancelled():
                    raise InterruptedError("Stopped. This incomplete experiment will not be saved.")
                # Form only the current batch: full suffix paths can be large.
                current_patch = interpolate_tokens(*source_states, ts[offset:offset + batch_size], method)
                count = len(current_patch)
                if count < batch_size:
                    current_patch = torch.cat((current_patch, current_patch[-1:].expand(batch_size - count, -1, -1)))
                batch = context_input.expand(batch_size, -1)
                capture_enabled = True
                all_logits = self.model(batch, use_cache=False).logits[:, -1, :].float().clone()
                logits = all_logits[:count]
                for layer in record_layers:
                    key = str(layer)
                    readouts[key].append(captures[key][:count].float())
                readouts["logits"].append(logits)
                if offset == 0:
                    endpoint_errors["A"] = float((logits[0] - endpoints_logits[0]).abs().max())
                if offset + batch_size >= steps:
                    endpoint_errors["B"] = float((logits[-1] - endpoints_logits[1]).abs().max())
                # The first distribution is the very same readout used above.
                # Later generation forwards must never contribute to c/d or arc lengths.
                capture_enabled = False
                captures.clear()
                candidate_ids, candidate_probs = [], []
                for step in range(3):
                    if cancelled():
                        raise InterruptedError("Stopped. This incomplete experiment will not be saved.")
                    if step:
                        batch = torch.cat((batch, top_ids[:, :1]), dim=1)
                        all_logits = self.model(batch, use_cache=False).logits[:, -1, :].float().clone()
                    top_ids, top_probs = top_candidates(all_logits)
                    candidate_ids.append(top_ids[:count].cpu().tolist())
                    candidate_probs.append(top_probs[:count].cpu().tolist())
                for index, alpha in enumerate(ts[offset:offset + batch_size].cpu().tolist()):
                    predictions_path.append(sample_prediction(self.tokenizer, alpha,
                        [step[index] for step in candidate_ids], [step[index] for step in candidate_probs]))
                progress(f"Measuring c(t), d(t) and top-3 token continuations: {min(offset + batch_size, steps)} / {steps} samples", 0.2 + 0.78 * min(offset + batch_size, steps) / steps)
            if cancelled():
                raise InterruptedError("Stopped. This incomplete experiment will not be saved.")
        finally:
            for handle in handles:
                handle.remove()

        return {key: readout.finish() for key, readout in readouts.items()}, predictions_path, endpoint_errors, natural_gaps, endpoint_tokens, endpoint_l2

    @torch.inference_mode()
    def run(self, request, progress=lambda *_: None, cancelled=lambda: False, *, all_layers=False):
        start = time.monotonic()
        model_id = request.get("model", "gpt2-large")
        self.load(model_id, progress, cancelled)
        if cancelled():
            raise InterruptedError("Stopped.")
        a, b, ids_a, ids_b = self.validate(request)
        patch_layer = int(request.get("patch_layer", 0))
        steps = int(request.get("steps", 41))
        method = request.get("interpolation", "slerp")
        context = request.get("context", "a")
        patch_position = request.get("patch_position", "last_token")
        blocks, embedding = self.architecture()
        n_layers = len(blocks)
        if not -1 <= patch_layer < n_layers:
            raise ValueError(f"Choose a layer from 0 to {n_layers - 1}; -1 selects the embedding.")
        if not 5 <= steps <= 201 or method not in ("linear", "slerp"):
            raise ValueError("Use 5–201 samples and select SLERP or Linear interpolation.")
        if context not in ("a", "b"):
            raise ValueError("Choose A or B as the fixed context.")
        if patch_position not in ("last_token", "different_suffix"):
            raise ValueError("Choose First difference → end or Final token only.")
        if patch_position == "different_suffix" and len(ids_a) != len(ids_b):
            raise ValueError(f"Suffix interpolation requires equal token counts (A: {len(ids_a)}, B: {len(ids_b)}). Edit the inputs or choose Final token only for unequal lengths.")
        first_difference = next((i for i, (x, y) in enumerate(zip(ids_a, ids_b)) if x != y), min(len(ids_a), len(ids_b)))
        patch_starts = [first_difference, first_difference] if patch_position == "different_suffix" else [len(ids_a)-1, len(ids_b)-1]
        patch_count = len(ids_a) - patch_starts[0]
        patch_start = patch_starts[0 if context == "a" else 1]

        progress("Generating the next three words for A and B…", 0.08)
        predictions = [self.predict(ids) for ids in (ids_a, ids_b)]
        # Separate forwards preserve natural states. Suffix mode pairs equal
        # positions; final-token mode pairs the last positions of either length.
        inputs = [torch.tensor([ids], device=self.device) for ids in (ids_a, ids_b)]
        context_input = inputs[0 if context == "a" else 1]
        shared_prefix = len(ids_a) == len(ids_b) and ids_a[:-1] == ids_b[:-1]
        # Embedding output, or block output BEFORE final normalization.
        # GPT-2 includes absolute positions here; NeoX/Qwen apply rotary positions in attention.
        patch_module = embedding if patch_layer == -1 else blocks[patch_layer]
        first = min(n_layers - 1, patch_layer + 1)
        record_layers = sorted(set([first, (first + n_layers - 1) // 2, n_layers - 1]))
        representative = list(record_layers)
        if all_layers:
            record_layers = list(range(max(0, patch_layer), n_layers))
        captures = {}
        handles = []

        source_start = 0

        def capture(name):
            def hook(module, args, output):
                hidden = output[0] if isinstance(output, tuple) else output
                captures[name] = (hidden[:, source_start:, :] if name == "patch" else hidden[:, -1, :]).detach().clone()
            return hook

        try:
            handles.append(patch_module.register_forward_hook(capture("patch")))
            for layer, block in enumerate(blocks):
                handles.append(block.register_forward_hook(capture(str(layer))))
            natural_logits, source_states, natural_states = [], [], []
            for inp, source_start in zip(inputs, patch_starts):
                natural_logits.append(self.model(inp, use_cache=False).logits[0, -1, :].clone())
                source_states.append(captures["patch"][0].clone())
                natural_states.append(torch.stack([captures[str(layer)][0] for layer in range(n_layers)]).float())
        finally:
            for handle in handles:
                handle.remove()

        natural_l2 = torch.linalg.vector_norm(natural_states[0] - natural_states[1], dim=-1).cpu().tolist()
        source_l2 = float(torch.linalg.vector_norm(source_states[0].float() - source_states[1].float()))
        source_token_l2 = torch.linalg.vector_norm(source_states[0].float() - source_states[1].float(), dim=-1).cpu().tolist()
        del natural_states
        captures.clear()
        ts = torch.linspace(0, 1, steps, device=self.device)
        batch_size = self.hardware.batch_size(self.model.config, context_input.shape[1], steps)
        batch_retries = 0
        while True:
            try:
                rows, predictions_path, endpoint_errors, natural_gaps, endpoint_tokens, endpoint_l2 = self._measure_path(
                    source_states, natural_logits, context_input, patch_module, blocks,
                    record_layers, ts, method, patch_start, batch_size, progress, cancelled)
                break
            except RuntimeError as error:
                if not is_out_of_memory(error) or batch_size == 1:
                    raise
            # Exit the except block before freeing the allocator cache so the traceback
            # no longer retains failed forward-pass tensors.
            batch_size = max(1, batch_size // 2)
            batch_retries += 1
            self.hardware.clear_cache()
            progress(f"Reducing the batch to {batch_size} to fit available memory. Restarting curve measurement…", 0.2)

        t_values = ts.cpu().tolist()
        curves = [{"key": key, "title": "Logits" if key == "logits" else f"Layer {key} · resid_post",
                   "t": t_values, **values} for key, values in rows.items()
                  if key == "logits" or int(key) in representative]
        d_summary = summarize(t_values, rows["logits"]["d"], "d")
        c_summary = summarize(t_values, rows["logits"]["c"], "c")
        result = {
            "schema_version": 4, "created_at": datetime.now(timezone.utc).isoformat(),
            "primary_metric": "c", "metric_definitions": METRIC_DEFINITIONS,
            "model": model_id, "model_label": MODELS[model_id]["label"],
            "model_architecture": self.model.config.model_type,
            "model_revision": getattr(self.model.config, "_commit_hash", None),
            "device": self.device, "hardware": self.hardware.describe(),
            "dtype": "float32", "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "sequence_a": a, "sequence_b": b,
            "input_tokens": [[{"id": i, "text": self.tokenizer.decode([i])} for i in ids] for ids in (ids_a, ids_b)],
            "predictions": predictions,
            "l2_distances": {
                "metric": "euclidean", "position": "last_token", "representation": "resid_post",
                "source_representation": "embedding" if patch_layer == -1 else "resid_post",
                "patch_layer": patch_layer, "source_l2": source_l2,
                "source_position": patch_position, "source_token_count": patch_count,
                "source_last_token_l2": source_token_l2[-1],
                "source_l2_definition": "L2 over the concatenation of all patched token vectors (Frobenius norm of their difference)",
                "source_token_l2": [{"position_a": patch_starts[0]+i, "position_b": patch_starts[1]+i, "l2": value}
                                    for i, value in enumerate(source_token_l2)],
                "natural_definition": "||h_A(layer)-h_B(layer)||2, using each original sequence's final token",
                "patched_definition": "||h_C(layer,t=0)-h_C(layer,t=1)||2, with context C fixed; measured after the patch at its layer",
                "layers": [{"layer": layer, "natural_l2": natural_l2[layer], "patched_l2": endpoint_l2[layer]}
                           for layer in range(n_layers)],
            },
            "settings": {"patch_layer": patch_layer, "steps": steps, "interpolation": method,
                         "batch_size": batch_size,
                         "prediction_cache": True, "batch_retries": batch_retries,
                         "record_layers": record_layers, "representative_layers": representative,
                         "patch_position": patch_position, "generation": "greedy_3_words",
                         "sample_generation": "top3_greedy_3_tokens",
                         "patch_start_a": patch_starts[0], "patch_start_b": patch_starts[1],
                         "patch_start_context": patch_start, "patch_count": patch_count,
                         "interpolation_unit": "per_token_shared_t", "measurement_position": "last_token",
                         "context": context, "endpoint_reference": "patched_in_fixed_context"},
            "definition": "d(t) = ||x_C(t)-x_C(0)||₂ / (||x_C(t)-x_C(0)||₂ + ||x_C(t)-x_C(1)||₂), with context C held fixed",
            "experiment": {
                "shared_tokenized_prefix": shared_prefix,
                "first_different_token": first_difference,
                "natural_endpoints_reproduced": patch_position == "different_suffix" or shared_prefix,
                "source_lengths": [len(ids_a), len(ids_b)],
                "fixed_context": context.upper(),
                "endpoint_meaning": ("The entire suffix from the first differing token is patched; the identical causal prefix is preserved, so both endpoints reproduce natural A/B outputs."
                                     if patch_position == "different_suffix" else
                                     "A and B final-token source states patched into the same fixed context. These endpoints need not equal the natural A/B outputs."),
                "patched_endpoint_next_tokens": [self.tokenizer.decode([token]) for token in endpoint_tokens],
                "natural_endpoint_next_tokens": [self.tokenizer.decode([int(scores.argmax())]) for scores in natural_logits],
            },
            "source": "https://www.lesswrong.com/posts/WMfSbt7AAcJdHzysB/activation-plateaus-where-and-how-they-emerge",
            "curves": curves, "path_predictions": predictions_path,
            # The original flat summary fields remain d statistics for compatibility.
            "metrics": {**d_summary, "c": c_summary, "d": d_summary,
                        "endpoint_max_abs_logit_error": endpoint_errors,
                        "patched_vs_natural_max_abs_logit_gap": natural_gaps},
            "elapsed_seconds": round(time.monotonic() - start, 2),
        }
        result["backend"] = {"remote": False, "device": self.device, "implementation": "local_torch"}
        return add_effect(result, rows) if all_layers else result
