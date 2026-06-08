# A Multi-Dimensional Evaluation Framework for Agentic LLMs

A reproducible benchmark suite for measuring how **instruction tuning** changes the
behavior of a language model across the capabilities that matter for agentic use:
multi-turn reliability, tool-use planning, safety, and final-task fidelity.

The framework runs everything **locally** — models, datasets, and an LLM-as-a-judge —
so results are deterministic, offline, and free of third-party API costs. The default
configuration performs a head-to-head comparison of **Qwen2.5-7B (base)** against
**Qwen2.5-7B-Instruct**, graded by **Qwen2.5-72B-Instruct** loaded in 4-bit.

---

## Why this exists

Aggregate leaderboard scores hide *where* a model actually improves. Instruction
tuning might boost single-turn benchmarks while degrading multi-turn focus, or improve
safety at the cost of over-verbosity. This project decomposes "reliability" into four
independently-measured dimensions and reports a per-dimension scorecard for each model,
making the trade-offs of fine-tuning visible.

## Evaluation dimensions

| # | Experiment | Dataset | What it measures | Reported metrics |
|---|------------|---------|------------------|------------------|
| 1 | **Multi-turn Persistence** | [Lost-in-Conversation](https://huggingface.co/datasets/microsoft/lost_in_conversation) | Focus and task retention as a request is revealed across turns | G-Eval score (1–5), avg. verbosity |
| 2 | **Planning & Decomposition** | [ToolBench](https://huggingface.co/datasets/tuandunghcmut/toolbench-v1) | Quality of a tool-call plan vs. a golden path | Decomposition, Efficiency, Rationality |
| 3 | **Safety & Boundary Adherence** | [HarmBench](https://github.com/centerforaisafety/HarmBench) | Refusal of harmful behaviors | Safe rate, Attack Success Rate (ASR) |
| 4 | **Final-Task Fidelity** | [IFEval](https://huggingface.co/datasets/google/IFEval) + [FRAMES](https://huggingface.co/datasets/google/frames-benchmark) | Instruction-following + grounded, evidence-based answering | Constraint satisfaction, over-verbosity, task completion, groundedness (1–5) |

All grading is performed by a single, consistent judge model (G-Eval style) rather than
brittle string matching, so open-ended responses are scored on intent and correctness.

## Architecture

```
prepare_assets.py  ──►  models/ + data/{raw,processed}/   (download & preprocess once)
                                   │
run_experiments.py ──►  load local model ─► generate ─► judge model grades ─► results/
```

- **`src/llm_reliability_benchmarks/asset_prep.py`** — downloads model weights (Qwen2.5
  base, instruct, judge) and benchmark datasets, fetches Wikipedia evidence for FRAMES,
  and writes preprocessed prompt files. Downloads are resumable.
- **`src/llm_reliability_benchmarks/benchmark_runner.py`** — loads models one at a time,
  runs each selected experiment, grades outputs with the judge model, and writes per-run
  JSON detail files plus JSON/CSV summaries. Models are evicted and GPU memory is cleared
  between stages to keep peak VRAM manageable.

## Repository layout

```
.
├── pyproject.toml                       # package metadata & dependencies
├── scripts/
│   ├── prepare_assets.py                # entry point: download & preprocess assets
│   └── run_experiments.py               # entry point: run benchmarks
├── src/llm_reliability_benchmarks/
│   ├── asset_prep.py                    # asset download / preprocessing logic
│   └── benchmark_runner.py              # experiment + judging logic
├── data/
│   ├── raw/                             # downloaded source datasets
│   └── processed/                       # preprocessed prompt files
├── models/                              # local model weights (gitignored)
└── results/                            # generated JSON + CSV outputs (gitignored)
```

## Requirements

- Python **3.10+**
- An NVIDIA GPU with CUDA. The 72B judge runs in 4-bit (`bitsandbytes`), but you still
  need substantial VRAM to host the judge alongside a 7B model under test.
- A [Hugging Face](https://huggingface.co) account/token if any gated assets are used
  (public downloads work without one).

Dependencies (`datasets`, `transformers`, `torch`, `huggingface_hub`, `bitsandbytes`,
`numpy`, `pandas`, `tqdm`, `requests`) are declared in `pyproject.toml`.

## Installation

```bash
git clone https://github.com/jimmy19981226/A-Multi-Dimensional-Evaluation-Framework-for-Agentic-LLMs.git
cd A-Multi-Dimensional-Evaluation-Framework-for-Agentic-LLMs
python -m pip install -e .
```

## Quick start

**1. Download and preprocess all assets** (models + datasets). This is a large,
one-time download and is safe to re-run to resume:

```bash
python scripts/prepare_assets.py --assets all
# optionally authenticate for gated/rate-limited downloads:
python scripts/prepare_assets.py --assets all --hf-token <YOUR_HF_TOKEN>
```

**2. Run the full benchmark suite:**

```bash
python scripts/run_experiments.py --experiments all
```

Results are written to `results/default_run/` as JSON detail files and JSON/CSV
summaries, and a human-readable scorecard is printed to the console.

## Selective runs

Fetch only the assets you need:

```bash
python scripts/prepare_assets.py --assets instruct data        # one model + core datasets
python scripts/prepare_assets.py --assets ifeval frames        # fidelity datasets only
```

Run a subset of experiments:

```bash
python scripts/run_experiments.py --experiments multiturn toolbench
python scripts/run_experiments.py --experiments harmbench fidelity
```

Control sample sizes, output location, and the random seed for reproducible sampling:

```bash
python scripts/run_experiments.py \
  --experiments all \
  --harmbench-sample-size 25 \
  --ifeval-sample-size 50 \
  --output-dir results/qwen_comparison \
  --seed 42
```

Run `python scripts/run_experiments.py --help` (and the same for `prepare_assets.py`)
for the complete list of flags, including overriding individual model and dataset paths.

## Outputs

For each run the framework produces, per experiment:

- **Detail files** (`*_details.json`) — every prompt, model generation, and the judge's
  full reasoning, for auditing and error analysis.
- **Summaries** (`*_summary.json` / `*_summary.csv`) — aggregated per-model metrics
  ready for plotting or tabulation.

A consolidated scorecard comparing the base and instruct models across all four
dimensions is printed at the end of the run.

## Configuration

The models under test, the judge, and dataset names are defined as constants at the top
of `asset_prep.py` and `benchmark_runner.py`. To benchmark a different model pair or
judge, point the `--*-model-path` flags at your local weights (or edit those constants)
— the experiment and judging logic is model-agnostic.

## Notes

- Model weights, downloaded datasets, logs, and generated results are intentionally
  excluded from version control via `.gitignore`.
- The judge is loaded with 4-bit NF4 quantization and double quantization to fit a 72B
  model on commodity hardware; the models under test run in bf16.
- This repository is structured for sharing, review, and portfolio use.

## License

No license file is currently included. If you intend to reuse this code, please open an
issue to request clarification on licensing.
