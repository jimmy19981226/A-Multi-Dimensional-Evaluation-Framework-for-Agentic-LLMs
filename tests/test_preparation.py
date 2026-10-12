import json

from llm_reliability_benchmarks import asset_prep


def test_toolbench_caps_sample_count_and_accepts_decoded_apis(tmp_path, monkeypatch):
    rows = [
        {
            "query": "Find a route",
            "api_list": "[]",
            "relevant_apis": [["maps", "route"]],
        }
    ]
    monkeypatch.setattr(asset_prep, "load_dataset", lambda *a, **kw: rows)
    destination = tmp_path / "prompts.json"
    asset_prep.build_toolbench_prompts(destination, sample_size=100)
    prompts = json.loads(destination.read_text())
    assert len(prompts) == 1
    assert prompts[0]["solution_path"] == "Step 1: Call route from maps"


def test_toolbench_handles_malformed_api_metadata(tmp_path, monkeypatch):
    monkeypatch.setattr(
        asset_prep,
        "load_dataset",
        lambda *a, **kw: [{"query": "Example", "relevant_apis": None}],
    )
    destination = tmp_path / "prompts.json"
    asset_prep.build_toolbench_prompts(destination, 1)
    assert (
        json.loads(destination.read_text())[0]["solution_path"]
        == "No required tools specified."
    )
