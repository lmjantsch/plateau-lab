# Saved results, cache and exports

[← Back to the visual guide](../README.md)

- `data/runs/`: Automatically saved JSON for each successful experiment.
- `data/examples/`: Saved examples with categories and notes. Saving the same run again updates its annotation without creating a duplicate.
- **JSONL:** Sequences, token IDs, continuations, complete curves, model revision, software versions, settings, and timestamps.
- **CSV:** A long-format table with one row per curve sample, suitable for external plotting. Existing columns (including `d`) retain their meanings and order. Appended columns include `schema_version`, `c`, `step_length`, `cumulative_length`, `total_length`, both metric statuses and undefined reasons, and JSON-encoded definition metadata. Missing or undefined numeric values are blank; they are never fabricated zeroes.
- **L2 measurements:** Records include `l2_distances` with the combined source distance, final-token source distance, individual source token distances, and both final-token distances at every layer; these are preserved in History, saved Examples, and JSONL exports. **Download L2 CSV** exports one row per layer, including the combined/final-token source distances, token scope, starts, and count. Individual source token distances are retained in JSONL. The library's CSV export includes both c(t) and d(t) samples with intervention metadata. Older records without raw distances show a re-run notice; normalized d(t) alone cannot recover the original L2 scale, so no values are invented or old records changed. Captures reuse the natural and reference forwards; all-layer states are not retained for every interpolation sample.
- `.model-cache/`: Persistent downloaded models. GPT-2 Large uses approximately 3 GB; GPT-2 XL uses 6.43 GB. Pythia downloads range from 0.17 GB (70M) to 5.68 GB (2.8B). Qwen downloads range from about 0.99 GB (Qwen2.5 0.5B) to 6.17 GB (Qwen2.5 3B); Qwen3 0.6B is about 1.19 GB. Sharded models are marked Saved only when every weight shard is present. Download size can be smaller than the float32 memory usage.
- `.venv/`: This tool's Python environment.

Schema 4 stores `c`, `step_lengths` (initial sample 0), `cumulative_length`, `total_length`, `c_status`, `d_status`, and undefined reasons alongside each curve’s unchanged `t`/`d` fields. Top-level `metric_definitions` records the formulas, precision, tolerances and representations; `primary_metric` is `c`. JSONL, History, and Examples preserve these fields. New library previews show c; legacy previews are labeled **Legacy · d(t) only**. Records from before schema 4 still show their original d curves. Their missing vector trajectories cannot be recovered from d: the c row explicitly asks for a rerun. Loading or exporting an old result never rewrites it or runs a new experiment automatically.

The integrated local server saves successful runs as JSON files and the Explorer also keeps browser history/imports in IndexedDB. Disk history and annotations from saved examples are shown in the Explorer. Clearing History removes selected run files and browser copies; separately saved examples remain. The classic workbench stores only its input/settings draft in browser storage. On a hosted NDIF server, results are stored only in the viewer's browser; the local library, hardware and file-export endpoints are disabled. Translating the interface does not translate user-entered prompts or model-generated text.

Selection is kept separately for Examples and History while the page stays open. Reloading clears the selection. Opening older records also shows their saved input tokens without re-running inference. The interface exports only selected records; the original `GET /api/export?format=jsonl&scope=examples` endpoint still supports full-collection downloads for scripts. `POST /api/export` accepts `{ "format": "jsonl", "scope": "examples", "ids": ["<record-id>"] }` for a subset; an empty or invalid selection returns an error instead of exporting the whole collection.

**Models stay saved between browser visits, server restarts, and computer restarts.** The menu labels a complete local checkpoint **Saved**. It is loaded directly from its saved snapshot with network access disabled for that load. A **Download** label means files are missing; the first run fetches them into `.model-cache/`. Keep that folder to retain the models. Switching models releases the previous model from memory, but keeps its files on disk. An interrupted download may resume on the next attempt. Downloading and loading show an indeterminate activity indicator; measured experiment progress starts during inference.

To download a model ahead of time without running an experiment, use `.venv/bin/python prepare_model.py pythia-160m` (or another model ID from the menu). This also reuses complete saved checkpoints without making a network request.

## Schema 7 and hosted records

Schema 7 retains `curves`, `metrics.c`, `metrics.d`, the original flat d statistics, raw arc lengths, statuses and definition metadata. `effect` adds a shared t grid and all measured layer/logit rows, with `c`, Shinkle/Heimersheim d, Janiak relative L2 and projected relative L2. Each row has `defined_metrics`, so coincident endpoints can leave d undefined while c remains valid. `total_length`, `step_lengths` and `cumulative_length` are available for every measured row.

The Explorer imports schemas 1–7 from JSON (one record or an array) or JSONL. Older results keep their original schema number and metadata. An in-memory display adapter exposes only the measurements actually saved; it never reconstructs c or additional layers from d. Schema-5/6 runs created by the hosted fork therefore remain d-only until rerun. The classic server's file exports preserve old records byte-for-byte in meaning and do not modify their files.

The Explorer's browser CSV contains all available effect rows and additional endpoint metrics. The local `/api/export` CSV preserves its original column order and representative curve layout. Use JSON/JSONL when you need the entire original record and all metadata.

## Per-sample token matrix

New runs add `settings.sample_generation: "top3_greedy_3_tokens"` and `token_matrix` to each `path_predictions` entry. These are optional additions to schemas 4 (legacy server) and 7 (integrated local/NDIF server); existing `t`, `token_id`, and `token` fields retain their meaning.

`token_matrix.steps` is indexed by generation position, then probability rank. It holds up to three columns, each with the top three candidates as `{id, text, probability, is_eos}`. Probabilities come from the complete vocabulary's softmax, without temperature or top-k renormalization. Later columns condition on the preceding rank-one choices. `stop_reason` is `null` after three positions, `"eos"` when a greedy end-of-text token ends generation, or `"nonfinite"` when a distribution is unavailable. The EOS column is retained; all following columns are omitted.

History, Examples and JSON/JSONL exports retain the matrix. Plotting CSV exports keep their existing columns and omit this nested prediction data. Imports without a matrix show a rerun notice; opening them never generates or invents predictions.
