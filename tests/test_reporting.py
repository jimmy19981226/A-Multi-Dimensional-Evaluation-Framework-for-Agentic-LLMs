"""Exercise real aggregation and serialization with synthetic evaluator outputs.

These tests verify reporting behavior; they do not run or score language models.
"""

import csv
import json
from types import SimpleNamespace

import pytest

from llm_reliability_benchmarks import benchmark_runner as runner


def read_json(directory, name):
    return json.loads((directory / name).read_text())


def test_multiturn_summaries_keep_models_separate(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "load_lost_in_conversation", lambda *_: [{}, {}])

    def evaluate(loader, model, *args):
        scores = [1, 5] if model == runner.BASE_MODEL_NAME else [4, 5]
        return [
            {"model": model, "task_index": i, "geval_score": score,
             "avg_verbosity": 20 + i * 20, "transcript": "synthetic fixture"}
            for i, score in enumerate(scores)
        ]

    monkeypatch.setattr(runner, "evaluate_multiturn_model", evaluate)
    args = SimpleNamespace(lost_in_conversation_path=tmp_path, multiturn_sample_size=2)
    runner.run_multiturn_experiment(args, tmp_path, None, None, None, None)
    summary = read_json(tmp_path, "multiturn_summary.json")
    assert [row["mean_geval_score"] for row in summary] == [3.0, 4.5]
    assert [row["mean_avg_verbosity"] for row in summary] == [30.0, 30.0]
    with (tmp_path / "multiturn_summary.csv").open() as handle:
        csv_rows = list(csv.DictReader(handle))
    assert float(csv_rows[1]["mean_geval_score"]) == 4.5
    assert len(read_json(tmp_path, "multiturn_Qwen2.5-7B.json")) == 2


def test_toolbench_reporting_preserves_zero_to_three_decomposition(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "load_toolbench_prompts", lambda *_: [{}, {}])

    def evaluate(loader, model, *args):
        return [
            {"Model": model, "Prompt_ID": i, "Reasoning": "synthetic fixture",
             "Decomp_Score": score, "Efficiency_Score": 2 + i * 2,
             "Rationality_Score": 3 + i * 2}
            for i, score in enumerate([0, 3])
        ]

    monkeypatch.setattr(runner, "evaluate_toolbench_model", evaluate)
    args = SimpleNamespace(toolbench_prompts_path=tmp_path, toolbench_sample_size=2)
    runner.run_toolbench_experiment(args, tmp_path, None, None, None, None)
    for row in read_json(tmp_path, "toolbench_summary.json"):
        assert row["Avg_Decomposition"] == 1.5
        assert row["Avg_Efficiency"] == 3.0
        assert row["Avg_Rationality"] == 4.0
    assert len(read_json(tmp_path, "toolbench_evaluation_details.json")) == 4


def test_safety_reporting_uses_percentages_and_complement(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "load_harmbench_cases", lambda *_: None)

    def evaluate(loader, model, instruct, *args):
        scores = [1, 1, 1, 1] if instruct else [1, 0, 1, 1]
        return scores, [{"model": model, "is_safe": bool(v)} for v in scores]

    monkeypatch.setattr(runner, "run_harmbench_for_model", evaluate)
    args = SimpleNamespace(harmbench_csv_path=tmp_path, harmbench_sample_size=4, seed=42)
    runner.run_harmbench_experiment(args, tmp_path, None, None, None, None)
    summary = read_json(tmp_path, "harmbench_summary.json")
    assert [row["safe_rate"] for row in summary] == [75.0, 100.0]
    assert [row["attack_success_rate"] for row in summary] == [25.0, 0.0]
    assert len(read_json(tmp_path, "harmbench_details.json")) == 8


def test_fidelity_reporting_uses_macro_rates_and_keeps_grounding_distinct(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "load_json_rows", lambda *_: [{}, {}])

    def ifeval(loader, model, *args):
        return [
            {"model": model, "constraint_satisfaction_rate": 1.0,
             "constraints_satisfied": 1, "total_constraints": 1,
             "token_count": 10, "over_verbose_flag": 0},
            {"model": model, "constraint_satisfaction_rate": 0.0,
             "constraints_satisfied": 0, "total_constraints": 3,
             "token_count": 30, "over_verbose_flag": 1},
        ]

    def frames(loader, model, *args):
        return [
            {"model": model, "task_completion": 1, "groundedness_score": 2},
            {"model": model, "task_completion": 0, "groundedness_score": 4},
        ]

    monkeypatch.setattr(runner, "evaluate_ifeval_model", ifeval)
    monkeypatch.setattr(runner, "evaluate_frames_model", frames)
    args = SimpleNamespace(ifeval_path=tmp_path, frames_path=tmp_path,
                           ifeval_sample_size=2, frames_sample_size=2)
    runner.run_fidelity_experiment(args, tmp_path, None, None, None, None)
    for row in read_json(tmp_path, "fidelity_ifeval_summary.json"):
        assert row["avg_constraint_satisfaction_rate"] == 0.5
        assert row["avg_token_count"] == 20
        assert row["over_verbose_rate"] == 0.5
    for row in read_json(tmp_path, "fidelity_frames_summary.json"):
        assert row["task_completion_rate"] == 0.5
        assert row["avg_groundedness_score"] == 3.0


def test_ifeval_evaluation_derives_rate_from_counts(monkeypatch):
    monkeypatch.setattr(runner, "generate_final_answer", lambda *_args, **_kw: "fixture answer")
    monkeypatch.setattr(runner, "judge_json_response", lambda *_args, **_kw: {
        "Reasoning": "synthetic fixture", "total_constraints": 2,
        "constraints_satisfied": 1, "constraint_satisfaction_rate": 0.99,
        "over_verbose_flag": 0,
    })
    loader = lambda: (object(), SimpleNamespace(encode=lambda text: [1, 2]))
    rows = runner.evaluate_ifeval_model(
        loader, "fixture", False,
        [{"prompt": "fixture", "instruction_id_list": ["a", "b"], "kwargs": []}],
        None, None,
    )
    assert rows[0]["constraint_satisfaction_rate"] == 0.5


def test_run_manifest_hashes_inputs_and_excludes_credentials(tmp_path):
    dataset = tmp_path / "input.json"
    dataset.write_text("[]")
    args = SimpleNamespace(output_dir=tmp_path, ifeval_path=dataset,
                           hf_token="synthetic-secret-do-not-store", seed=42)
    runner.write_run_manifest(args, ["fidelity"])
    text = (tmp_path / "run_manifest.json").read_text()
    assert "synthetic-secret-do-not-store" not in text
    manifest = json.loads(text)
    assert manifest["dataset_sha256"]["ifeval_path"] == (
        "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945"
    )
    assert manifest["configuration"]["seed"] == 42


def test_required_model_assets_are_checked_before_inference(tmp_path):
    args = SimpleNamespace(base_model_path=tmp_path / "missing-base",
                           instruct_model_path=tmp_path / "missing-instruct",
                           judge_model_path=tmp_path / "missing-judge")
    with pytest.raises(FileNotFoundError, match="base model directory"):
        runner.validate_local_assets(args, [])
