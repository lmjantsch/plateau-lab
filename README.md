# MARS V · Plateau Lab

**Follow a model’s hidden-state path between two prompts.**

A local and hosted workbench for exploring activation plateaus with **GPT-2, Pythia and Qwen**. Compare continuations, interpolate hidden states, and collect measured examples—with both cumulative path progress **c(t)** and relative endpoint distance **d(t)**.

![Plateau Lab workflow: compare two prompts, interpolate hidden states at a chosen layer, and measure residuals and logits.](docs/assets/workflow.svg)

[Quick start](#quick-start) · [Hosted / NDIF setup](docs/HOSTING.md) · [Your first experiment](#your-first-experiment) · [Read the curves](#read-the-curves) · [Experimental method](docs/METHOD.md) · [Data & exports](docs/DATA.md)

## Quick start

Use Git and **Python 3.12** (tested; 3.10–3.12 recommended), then:

```sh
git clone https://github.com/billy000400/plateau-lab.git
cd plateau-lab
./start.sh
```

Open **[localhost:8765](http://127.0.0.1:8765)** and keep the terminal running. The launcher creates `.venv` and installs dependencies on its first launch. On macOS, you can also double-click **Start Plateau Lab.command**.

> **Apple Silicon:** use native **ARM64 Python**. An Intel/Rosetta Python environment cannot install the pinned macOS PyTorch 2.8 package. Create a fresh environment on each machine; do not copy `.venv` between computers.

The browser is the interface; inference runs on your computer. **No API key is required.** Models download on first use and stay cached for later visits. The integrated Explorer defaults to GPT-2 Small; choose **Pythia 70M** for a smaller first download.

| Where you run | Setup |
| --- | --- |
| Apple Silicon Mac or CPU Linux | `./start.sh` |
| Linux with NVIDIA or AMD GPUs | [GPU installation and device selection](LINUX.md) |
| Remote Linux server | `./start-remote.sh your-user@your-server` · [instructions](LINUX.md) |
| Windows | [Windows setup below](#windows-and-manual-setup) |

The launcher now serves the integrated **FastAPI Explorer**. It keeps the tested local PyTorch engine and adds token previews, all-layer metrics, c/d overview plots, and browser imports. **Local collections & classic view** opens the original workbench at `/classic/`, including saved examples, categories, notes and the original exports. Existing `data/` files are read in place; no migration rewrites them.

For Hugging Face Spaces and per-user NDIF inference, use a separate environment with `requirements-remote.txt`. See [hosting, keys and deployment](docs/HOSTING.md). Local launches continue to use `requirements.txt` and do not require nnsight or an NDIF key.

## Your first experiment

1. **Compare two prompts.** Start with `The house was big` and `The house was in`. Local continuation panels show three words; NDIF panels show three tokens. The result records which generation convention was used.
2. **Choose where to intervene.** The **Interpolation start** slider selects a block output, **After layer N**. Layers start at 0; the default is after layer 0.
3. **Choose which tokens to patch.** **First difference → end** interpolates every token from the first mismatch onward and requires equal token counts. **Final token only** also supports unequal lengths and different prefixes.
4. **Run and inspect.** Choose Linear or SLERP, then click **Run experiment** or press **Cmd/Ctrl + Enter**. Read c(t) first and d(t) directly below. Slide **Inspect sample** to see a **3 × 3 token matrix**: the three most likely candidates for each of the next three token positions at that t, with their probabilities.
5. **Keep an example.** Add a category and notes, then **Save example**. Every successful run is already in **History**; Examples holds your annotated selection.

You can change the model, prompts, layer, token scope and sample count to explore another path. Highlighted token chips show exactly which positions are interpolated. Starter pairs are exploration prompts, not guarantees of a plateau.

The token matrix is available in both the Explorer and classic view. Column 2 follows column 1’s top-ranked token; column 3 follows the top-ranked tokens from columns 1 and 2. The highlighted first row is the greedy continuation from the **patched sample**. Probabilities use the full vocabulary, so the three displayed candidates need not total 100%. Generation stops at a greedy end-of-text token; tokens may be word fragments. Predictions are computed during the run and saved with it, so sliding and reopening a result require no inference. Older results show a rerun notice. Computing the extra two columns adds two forwards per sample batch.

## Read the curves

The image below is **real Pythia 70M output**, measured with 41 SLERP samples after layer 0. Each column follows a different readout of the same intervention. It is a measured example, not a mockup of the application.

![Measured Pythia 70M curves: cumulative path progress c(t) above relative endpoint distance d(t), with matching columns for residual layers 1, 3, 5 and logits.](docs/assets/measured-curves.svg)

[Exact plotted values and model revision](docs/assets/measured-example.json) · [Rebuild the graphics](docs/render_graphics.py)

| Readout | What it tells you | How to interpret it |
| --- | --- | --- |
| **c(t) · Cumulative path progress** | How much of the sampled output trajectory’s total Euclidean path length has accumulated by t? | Flat sections mean little measured movement; steep sections concentrate movement. Detours and backtracking count. |
| **d(t) · Relative endpoint distance** | How does the current vector’s distance to the first endpoint compare with its distances to both endpoints? | An endpoint-relative view of the same vectors. Unlike c, it need not increase monotonically. |
| **Total path L2** | How much absolute movement did this readout accumulate? | Shown below each c chart so normalization does not hide the size of the change. |
| **A ↔ B · L2 distances** | How far apart are original A/B or patched endpoint representations at each layer? | Raw endpoint separation, distinct from the length traveled along the sampled path. |

For recorded vectors `z₀, …, zₙ` in increasing t order:

```text
step length      ℓᵢ = ‖zᵢ − zᵢ₋₁‖₂
path progress    c(tₖ) = (ℓ₁ + … + ℓₖ) / (ℓ₁ + … + ℓₙ)
endpoint metric  d(t)  = ‖z(t) − z(0)‖₂ / (‖z(t) − z(0)‖₂ + ‖z(t) − z(1)‖₂)
```

**A small example:** a one-dimensional trajectory `0 → 3 → 1 → 4` travels `3 + 2 + 3 = 8` units. At its second sample, **c = 3/8**, while **d = 3/4**. The backtrack contributes to c even though it brings the vector closer to the first endpoint.

The c denominator is fixed for the entire run and readout. This is neither a probability nor the fraction of straight-line distance to B. The dashed **Uniform path progress** reference, `c(t) = t`, means constant accumulation per unit t; the trajectory itself need not be straight. More samples may reveal movement that a coarser sampling missed.

<details>
<summary><strong>Undefined metrics, precision and older results</strong></summary>

- A stationary trajectory has total path length zero. It displays **“No measured movement; c(t) undefined”** rather than an invented curve.
- A path can move and return to its starting point: **valid c, undefined d**. One undefined readout does not abort the other measurements.
- Arc lengths use CPU float64 vector differences and norms, with compensated accumulation. The c zero-length tolerance is exactly zero; tiny measured movement is retained. d keeps its existing absolute endpoint tolerance of `1e-8`.
- Old records still display their original d curves, labeled **Legacy · d(t) only** in previews. They lack the vectors needed to recover c. A manual rerun is required; opening an old result never reruns or rewrites it.

See [the experimental definitions](docs/METHOD.md) and [schema/export reference](docs/DATA.md).

</details>

## What happens inside the model?

**Capture A/B states → interpolate selected positions → patch one block output → continue the model → measure the final token.**

All selected positions share the same interpolation coefficient t. Linear interpolation and SLERP are applied independently to each token-state pair. Local measurements use float32 model inference; hosted NDIF precision is reported by the backend; residual readouts are taken **before final normalization**. The integrated Explorer measures every block from the patch onward plus logits. The c/d overview and classic workbench retain representative downstream readouts. The legacy `python app.py` server still measures representative layers only.

| Token scope | Context and endpoint meaning |
| --- | --- |
| **First difference → end** | The identical causal prefix stays fixed and the entire differing suffix is patched. Endpoints reproduce natural A/B outputs up to numerical precision. |
| **Final token only** | Choose A or B as the fixed context. A transferred endpoint may differ from the corresponding prompt’s natural output. |

The continuation panels always show the original, unpatched prompts. [Read the complete method, interpolation rules and endpoint checks →](docs/METHOD.md)

## Models and compute

| Family | Available sizes |
| --- | --- |
| GPT-2 | **Small (Explorer default)** · Medium · Large · XL |
| Pythia | 70M · 160M · 410M · 1B · 1.4B · 2.8B |
| Qwen2.5 Base | 0.5B · 1.5B · 3B |
| Qwen3 Base | 0.6B · 1.7B |

These are **base completion models**, used without chat templates. One model is loaded at a time. The app automatically selects a usable CUDA/ROCm GPU, Apple MPS, or CPU and reports the device in use. Larger models require substantial memory: Pythia 2.8B needs about 11 GB for float32 weights, and Qwen2.5 3B about 12 GB, plus working space.

**Saved** in the model menu means the complete checkpoint is already on disk. Switching models releases the previous model from memory but keeps its cached files. [Cache behavior and sizes →](docs/DATA.md)

## Collect, revisit and export

The Explorer's **History** reads local disk runs and browser imports. It accepts schemas 1–7 as JSON or JSONL, keeps annotations, and lets you select records for CSV/JSONL export. c(t) is displayed only when measured; d-only records ask for a rerun. A completed result also has a direct JSON download.

The following collection controls remain available in **Local collections & classic view**:

| Action | In the app |
| --- | --- |
| Keep a measured run | Every completed experiment appears in **History**. |
| Add interpretation | Set a category and notes, then **Save example**. |
| Reopen a result | Double-click a card, or focus it and press **Enter**. |
| Select several results | Single-click cards or use **Space**; **Select visible** selects the filtered set. |
| Export the selection | Choose **JSONL** for full records or **CSV** for a row per curve sample. |
| Export raw endpoint distances | Choose **Download L2 CSV** on a result. |

Exports preserve **both metrics**, cumulative and total lengths, model/settings metadata, and undefined statuses. Selected records hidden by a search still belong to the selection. Reloading clears the selection, not saved results.

Your files stay in the repository’s local `data/` and `.model-cache/` folders, which are excluded from Git. Schema 7 retains the c/d data introduced in schema 4 and adds the all-layer `effect` table with validity per metric. [Storage layout, schema and API details →](docs/DATA.md)

## Windows and manual setup

Create a fresh environment on the destination machine. On Windows:

```powershell
py -3.12 -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python -m server.launch --open
```

On macOS or CPU Linux, the equivalent manual setup is:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m server.launch --open
```

For GPU Linux, first follow [LINUX.md](LINUX.md) to select the PyTorch build. Optional overrides include `PLATEAU_DEVICE=cpu`, `mps`, `cuda` or `cuda:N`, and `PLATEAU_BATCH_SIZE=1` through `64`; both default to automatic selection. Inspect the device with `.venv/bin/python hardware.py`.

The server binds to **127.0.0.1**. The browser URL opens this computer’s app; use JSONL/CSV exports to share results.

## Validation

The arc-length implementation has independent known-answer tests, real-model vector comparisons for **Linear and SLERP**, OOM/cancellation checks, and temporary HTTP persistence/export tests. Pythia 70M was checked on **CPU and Apple MPS**. Ordinary d values, predictions and endpoint-L2 measurements matched the pre-change engine in the tested cases.

```sh
.venv/bin/python check_trajectory.py
.venv/bin/python check_exports.py
PLATEAU_DEVICE=cpu .venv/bin/python check_arc.py --model pythia-70m
.venv/bin/python check_app.py --model pythia-70m --fastapi
.venv/bin/python check_integration.py  # also requires httpx for the test client
node check_web.js
node check_ui.js
```

[Arc validation →](ARC_VALIDATION.md) · [Integration checks and limitations →](docs/INTEGRATION_VALIDATION.md). Browser interaction and visual checks remain unverified because the automation tool could not verify its admin-enforced policy. Node component checks do not substitute for browser testing.

For isolated manual checks, use `./start.sh --port 8767 --data-dir /absolute/path/to/temporary-test-data` to keep test records separate from your collections.

## References

The intervention is informed by [Matthew Shinkle & StefanHex, *Activation Plateaus: Where and How They Emerge*](https://www.lesswrong.com/posts/WMfSbt7AAcJdHzysB/activation-plateaus-where-and-how-they-emerge), especially footnotes 1–2. Model interfaces and implementation caveats are linked in [the method reference](docs/METHOD.md). Plateau Lab uses Hugging Face forward hooks and does not reproduce TransformerLens weight transformations value for value.
