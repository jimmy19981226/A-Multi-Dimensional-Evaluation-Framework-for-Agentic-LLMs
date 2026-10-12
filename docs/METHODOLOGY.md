# Evaluation methodology

## What this framework compares

The default experiment compares Qwen2.5-7B with Qwen2.5-7B-Instruct using a shared Qwen2.5-72B-Instruct judge. This holds model family and nominal model size constant while exploring behavior associated with instruction tuning. It does not isolate every training difference or establish causal effects by itself.

The four experiment blocks are separate measurements, not a single validated composite score.

| Block | Protocol | Interpretation boundary |
| --- | --- | --- |
| Multi-turn | Reveal task shards sequentially, then judge the final transcript on a 1–5 scale; measure response length in characters | Base tokenizers without a chat template receive a labeled text transcript; formatting differs from native instruct templates |
| ToolBench | Generate a textual tool plan and judge against a reference path | No external tools are executed; this measures proposed planning, not successful task execution |
| HarmBench | Generate a response to a behavior prompt, then classify compliance | Uses a general local judge, not HarmBench's official classifier or full attack-generation protocol |
| IFEval | Ask the judge how many listed constraints were satisfied | This is a custom judge-based adaptation, not the official strict/loose rule-based evaluator |
| FRAMES | Supply downloaded evidence extracts and judge answer completion and groundedness | Evidence retrieval, truncation, and missing extracts constrain the result |

## Sampling and generation

- Multi-turn, ToolBench, IFEval, and FRAMES take the first requested number of prepared examples.
- HarmBench uses seeded random sampling, capped at the number of available rows.
- All requested sample sizes must be positive.
- The runner seeds Python, NumPy, and PyTorch through `transformers.set_seed` before model loading.
- Some generation paths sample; others use greedy decoding. Their settings are visible beside the relevant `generate` call.
- The judge remains resident while one model under test is loaded. Budget memory for both, their KV caches, and execution overhead.

## Invalid outputs

Malformed judge JSON, missing required numeric scores, out-of-range values, and an absent safety verdict raise an error. An unparseable verdict must never inflate the reported safe rate. The IFEval constraint rate is derived from validated counts, rather than trusting a contradictory rate field.

An error stops the run. Completed experiment files remain available; the framework does not currently checkpoint every example or automatically resume partially completed experiments.

## Reproducing and reporting a result

1. Preserve the prepared datasets and model revisions used for the run.
2. Use a unique output directory and retain its `run_manifest.json` alongside details and summaries.
3. Record GPU model/count, CUDA and driver versions, model revisions, and the commit SHA separately. The current manifest records package versions, arguments, and dataset hashes, not every hardware detail or model-weight checksum.
4. Inspect representative transcripts and judge reasoning, including disagreements with human assessment.
5. Repeat with multiple seeds and larger samples before drawing conclusions; report uncertainty rather than only a mean.
6. Label results as this framework's custom protocol and disclose the judge, quantization, sample size, and prompt formatting.

The repository includes [presentation-reported study results](RESULTS.md), clearly separated from newly reproduced results. Original per-example outputs and run settings are unavailable, and the decomposition scale differs between the presentation and the current rubric. The reported aggregates do not establish statistical significance or general superiority of either model.

## Contributing

Keep benchmark behavior changes explicit: a new prompt template, judge rubric, decoding parameter, or scoring rule can change the meaning of a result. Add a focused regression test for parsing or preprocessing changes, run `python -m pytest -q`, and document any protocol change here. Do not commit model weights, downloaded datasets, access tokens, or private experiment outputs.
