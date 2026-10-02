"""Experiment orchestration and result record, independent of how forwards run."""
from __future__ import annotations

import time
from datetime import datetime, timezone

import torch

from plateau.core.math import DEFAULT_METRIC, METRICS, interpolate_tokens, transition_width
from plateau.core.models import MODELS
from plateau.core.results import add_effect, arc_from_segments
from plateau.core.predictions import sample_prediction
from plateau.core.trajectory import summarize
import math

SCHEMA_VERSION = 7
MAX_TOKENS = 256
NEXT_TOKENS = 3  # greedy continuation shown per sequence
SOURCE = "https://www.lesswrong.com/posts/WMfSbt7AAcJdHzysB/activation-plateaus-where-and-how-they-emerge"


def encode(tokenizer, text):
    """Token ids as the model was trained to see them.

    The tokenizer's own special tokens are kept: Llama and Gemma prepend BOS, which
    they need (the first position acts as an attention sink); GPT-2, Pythia, Qwen,
    and OLMo add nothing.
    """
    return tokenizer.encode(text, add_special_tokens=True)


def validate(request, tokenizer, context_limit):
    a, b = request.get("sequence_a", ""), request.get("sequence_b", "")
    if not isinstance(a, str) or not isinstance(b, str) or not a.strip() or not b.strip():
        raise ValueError("Enter both sequences first.")
    ids_a, ids_b = encode(tokenizer, a), encode(tokenizer, b)
    if max(len(ids_a), len(ids_b)) > min(MAX_TOKENS, context_limit - NEXT_TOKENS):
        raise ValueError(f"This tool supports up to {MAX_TOKENS} tokens per sequence. Shorten the input.")
    if ids_a == ids_b:
        raise ValueError("Both inputs contain identical tokens. Enter two different sequences.")
    return a, b, ids_a, ids_b


def patch_starts(ids_a, ids_b, patch_position):
    """First differing token index and the first interpolated position in A and B."""
    first_difference = next((i for i, (x, y) in enumerate(zip(ids_a, ids_b)) if x != y), min(len(ids_a), len(ids_b)))
    if patch_position == "different_suffix":
        return first_difference, [first_difference, first_difference]
    return first_difference, [len(ids_a) - 1, len(ids_b) - 1]


def token_pieces(tokenizer, ids):
    return [{"id": i, "text": tokenizer.decode([i])} for i in ids]


def tokenize_preview(tokenizer, request):
    """Input tokens and interpolated positions for the live preview; no model forward."""
    a, b = request.get("sequence_a", ""), request.get("sequence_b", "")
    ids = [encode(tokenizer, text) if text else [] for text in (a, b)]
    patch_position = request.get("patch_position", "last_token")
    starts = None
    if all(ids) and not (patch_position == "different_suffix" and len(ids[0]) != len(ids[1])):
        starts = patch_starts(*ids, patch_position)[1]
    return {"tokens": [token_pieces(tokenizer, side) for side in ids], "patch_starts": starts,
            "max_tokens": MAX_TOKENS}


def next_tokens(tokenizer, tokens, probabilities):
    """The greedy continuation, up to NEXT_TOKENS tokens; ends after an end-of-text token.

    Generation suppresses EOS so every step runs; tokens recorded after an EOS argmax
    were forced and are dropped.
    """
    kept = []
    for token, probability in zip(tokens, probabilities):
        kept.append({"id": token, "text": tokenizer.decode([token]), "probability": probability})
        if token == tokenizer.eos_token_id or len(kept) == NEXT_TOKENS:
            break
    return {"continuation": tokenizer.decode([t["id"] for t in kept], clean_up_tokenization_spaces=False),
            "tokens": kept, "ended": bool(kept) and kept[-1]["id"] == tokenizer.eos_token_id}


def run_experiment(backend, request, batch_size, progress=lambda *_: None, cancelled=lambda: False,
                   models=MODELS):
    """`models` is the catalog the request is checked against (id -> info with "repo" and "label")."""
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("Batch size must be a positive integer.")
    start = time.monotonic()
    model_id = request.get("model") or next(iter(models))
    if model_id not in models:
        raise ValueError("Choose a model from the list.")
    label = models[model_id]["label"]
    progress(f"Preparing {label}…", None)
    lm = backend.load(model_id, models[model_id]["repo"])
    tokenizer = lm.tokenizer
    a, b, ids_a, ids_b = validate(request, tokenizer, lm.context_limit)
    patch_layer = int(request.get("patch_layer", 0))
    steps = int(request.get("steps", 41))
    method = request.get("interpolation", "slerp")
    context = request.get("context", "a")
    patch_position = request.get("patch_position", "last_token")
    n_layers = len(lm.blocks)
    if not 0 <= patch_layer < n_layers:
        raise ValueError(f"Choose a layer from 0 to {n_layers - 1}.")
    if not 5 <= steps <= 201 or method not in ("linear", "slerp"):
        raise ValueError("Use 5–201 samples and select SLERP or Linear interpolation.")
    if context not in ("a", "b"):
        raise ValueError("Choose A or B as the fixed context.")
    if patch_position not in ("last_token", "different_suffix"):
        raise ValueError("Choose First difference → end or Final token only.")
    if patch_position == "different_suffix" and len(ids_a) != len(ids_b):
        raise ValueError(f"Suffix interpolation requires equal token counts (A: {len(ids_a)}, B: {len(ids_b)}). Edit the inputs or choose Final token only for unequal lengths.")
    first_difference, starts = patch_starts(ids_a, ids_b, patch_position)
    patch_count = len(ids_a) - starts[0]
    patch_start = starts[0 if context == "a" else 1]
    context_ids = ids_a if context == "a" else ids_b
    shared_prefix = len(ids_a) == len(ids_b) and ids_a[:-1] == ids_b[:-1]
    # Every block from the patch layer on is measured; earlier blocks have identical endpoints.
    record_layers = list(range(patch_layer, n_layers))
    first = min(n_layers - 1, patch_layer + 1)
    representative = sorted(set([first, (first + n_layers - 1) // 2, n_layers - 1]))

    # Natural forwards run separately, so both sequences keep their own states.
    natural = []
    for side, ids, source_start in (("A", ids_a, starts[0]), ("B", ids_b, starts[1])):
        if cancelled():
            raise InterruptedError("Stopped.")
        progress(f"Generating the next three tokens for {side}…", 0.08 if side == "A" else 0.14)
        natural.append(backend.natural(lm, ids, patch_layer, source_start,
                                       min(NEXT_TOKENS, lm.context_limit - len(ids))))
    predictions = [next_tokens(tokenizer, n["tokens"], n["probabilities"]) for n in natural]
    sources = torch.stack([natural[0]["source"], natural[1]["source"]])
    natural_logits = [n["logits"] for n in natural]
    natural_l2 = torch.linalg.vector_norm(natural[0]["layer_states"] - natural[1]["layer_states"], dim=-1).tolist()
    difference = sources[0].float() - sources[1].float()
    source_l2 = float(torch.linalg.vector_norm(difference))
    source_token_l2 = torch.linalg.vector_norm(difference, dim=-1).tolist()
    ts = torch.linspace(0, 1, steps)
    # Reject ambiguous SLERP paths before sending the path request.
    interpolate_tokens(sources[0].float(), sources[1].float(), ts[steps // 2:steps // 2 + 1], method)

    keys = [str(layer) for layer in record_layers] + ["logits"]
    retries = 0
    while True:
        # Every attempt starts fresh, including the previous vectors and scalar lengths.
        rows = {key: {metric: [] for metric in METRICS} for key in keys}
        lengths = {key: [] for key in keys}
        previous = None
        predictions_path, endpoint_errors, gaps = [], {}, {}
        try:
            for offset in range(0, steps, batch_size):
                if cancelled():
                    raise InterruptedError("Stopped.")
                chunk = backend.path(lm, context_ids, patch_layer, patch_start, sources, ts[offset:offset + batch_size],
                                     method, record_layers, include_reference_logits=offset == 0,
                                     previous=previous, cancelled=cancelled)
                previous = chunk["last_vectors"]
                if offset == 0:
                    gaps = chunk["gaps"]
                    reference_logits = chunk["reference_logits"]
                    endpoint_l2 = chunk["endpoint_l2"]
                    endpoint_tokens = chunk["reference_tokens"]
                    endpoint_errors["A"] = chunk["first_error"]
                if offset + batch_size >= steps:
                    endpoint_errors["B"] = chunk["last_error"]
                for key in keys:
                    lengths[key].extend(chunk["step_lengths"][key])
                    for metric in METRICS:
                        rows[key][metric].extend(chunk["values"][key][metric])
                for index, t in enumerate(ts[offset:offset + batch_size].tolist()):
                    predictions_path.append(sample_prediction(tokenizer, t,
                        [step[index] for step in chunk["candidate_ids"]],
                        [step[index] for step in chunk["candidate_probs"]]))
                done = min(offset + batch_size, steps)
                progress(f"Measuring c(t), endpoint metrics and top-3 token continuations: {done} / {steps} samples", 0.2 + 0.78 * done / steps)
            if cancelled():
                raise InterruptedError("Stopped.")
            break
        except Exception as error:
            if cancelled():
                raise InterruptedError("Stopped.") from None
            if not (isinstance(error, torch.OutOfMemoryError) or "out of memory" in str(error).lower()) or batch_size == 1:
                raise
        previous = chunk = None
        batch_size = max(1, batch_size // 2)
        retries += 1
        if hasattr(backend, "clear_cache"):
            backend.clear_cache()
        progress(f"Reducing the batch to {batch_size}; restarting path measurement…", 0.2)

    natural_gaps = {side: float((reference_logits[i] - natural_logits[i]).abs().max()) for i, side in enumerate(("A", "B"))}
    t_values = ts.tolist()
    readouts = {}
    for key in keys:
        gap = gaps[key]
        status = "nonfinite" if not math.isfinite(gap) else "coincident_endpoints" if gap < 1e-8 else "ok"
        values = {}
        for metric, samples in rows[key].items():
            valid = status == "ok" and all(math.isfinite(value) for value in samples)
            values[metric] = samples if valid else [None] * steps
        if status == "ok" and any(value is None for value in values[DEFAULT_METRIC]):
            status = "nonfinite"
        readouts[key] = {**arc_from_segments(lengths[key]), **values, "d": values[DEFAULT_METRIC],
                         "endpoint_l2": gap if math.isfinite(gap) else None, "d_status": status,
                         "d_undefined_reason": None if status == "ok" else (
                             "Coincident endpoints (L2 < 1e-8); d(t) undefined" if status == "coincident_endpoints"
                             else "Nonfinite endpoint distances; d(t) undefined")}
    curves = [{"key": key, "title": "Logits" if key == "logits" else f"Layer {key} · resid_post",
               "t": t_values, **readouts[key]} for key in [str(layer) for layer in representative] + ["logits"]]
    d_summary = summarize(t_values, readouts["logits"]["d"], "d")
    result = {
        "schema_version": SCHEMA_VERSION, "created_at": datetime.now(timezone.utc).isoformat(),
        "model": model_id, "model_label": label,
        "model_architecture": lm.config.model_type,
        "model_revision": getattr(lm.config, "_commit_hash", None),
        "backend": backend.describe(), "dtype": natural[0]["dtype"],
        "sequence_a": a, "sequence_b": b,
        "input_tokens": [token_pieces(tokenizer, ids) for ids in (ids_a, ids_b)],
        "predictions": predictions,
        "l2_distances": {
            "metric": "euclidean", "position": "last_token", "representation": "resid_post",
            "source_representation": "resid_post",
            "patch_layer": patch_layer, "source_l2": source_l2,
            "source_position": patch_position, "source_token_count": patch_count,
            "source_last_token_l2": source_token_l2[-1],
            "source_l2_definition": "L2 over the concatenation of all patched token vectors (Frobenius norm of their difference)",
            "source_token_l2": [{"position_a": starts[0] + i, "position_b": starts[1] + i, "l2": value}
                                for i, value in enumerate(source_token_l2)],
            "natural_definition": "||h_A(layer)-h_B(layer)||2, using each original sequence's final token",
            "patched_definition": "||h_C(layer,t=0)-h_C(layer,t=1)||2, with context C fixed; measured after the patch at its layer",
            "layers": [{"layer": layer, "natural_l2": natural_l2[layer], "patched_l2": endpoint_l2[layer]}
                       for layer in range(n_layers)],
        },
        "settings": {"patch_layer": patch_layer, "steps": steps, "interpolation": method,
                     "batch_size": batch_size, "batch_retries": retries, "record_layers": record_layers, "representative_layers": representative,
                     "patch_position": patch_position, "generation": "greedy_3_tokens",
                     "sample_generation": "top3_greedy_3_tokens",
                     "patch_start_a": starts[0], "patch_start_b": starts[1],
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
            "patched_endpoint_next_tokens": [tokenizer.decode([token]) for token in endpoint_tokens],
            "natural_endpoint_next_tokens": [tokenizer.decode([int(scores.argmax())]) for scores in natural_logits],
        },
        "source": SOURCE,
        "curves": curves, "path_predictions": predictions_path,
        "metrics": {**d_summary,
                    "endpoint_max_abs_logit_error": endpoint_errors,
                    "patched_vs_natural_max_abs_logit_gap": natural_gaps},
        "elapsed_seconds": round(time.monotonic() - start, 2),
    }

    return add_effect(result, readouts)
