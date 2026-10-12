"""Download and prepare benchmark assets for local experiments."""

import argparse
import json
import os
from pathlib import Path
from urllib.parse import unquote, urlparse

import pandas as pd
import requests
from datasets import load_dataset
from huggingface_hub import login, snapshot_download

from .validation import positive_int


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

BASE_MODEL_NAME = "Qwen/Qwen2.5-7B"
INSTRUCT_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
JUDGE_MODEL_NAME = "Qwen/Qwen2.5-72B-Instruct"
IFEVAL_DATASET_NAME = "google/IFEval"
FRAMES_DATASET_NAME = "google/frames-benchmark"

LOST_IN_CONVERSATION_URL = (
    "https://huggingface.co/datasets/microsoft/lost_in_conversation/resolve/main/"
    "lost_in_conversation.json"
)
HARMBENCH_URL = (
    "https://raw.githubusercontent.com/centerforaisafety/HarmBench/main/data/"
    "behavior_datasets/harmbench_behaviors_text_all.csv"
)
WIKIPEDIA_HEADERS = {
    "User-Agent": "LLM-Reliability-Benchmarks/1.0 (benchmark asset preparation)"
}


def print_rule(char: str = "=", width: int = 88) -> None:
    print(char * width)


def print_section(title: str) -> None:
    print()
    print_rule("=")
    print(title)
    print_rule("=")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download models and datasets for local benchmark runs."
    )
    parser.add_argument("--hf-token", default=None, help="Optional Hugging Face token.")
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    parser.add_argument("--raw-data-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--processed-data-dir", type=Path, default=PROCESSED_DATA_DIR)
    parser.add_argument("--toolbench-sample-size", type=positive_int, default=20)
    parser.add_argument(
        "--max-workers",
        type=positive_int,
        default=2,
        help="Number of concurrent file downloads for Hugging Face snapshots.",
    )
    parser.add_argument(
        "--hf-timeout",
        type=positive_int,
        default=600,
        help="Timeout in seconds for Hugging Face download operations.",
    )
    parser.add_argument(
        "--assets",
        nargs="+",
        choices=["base", "instruct", "judge", "data", "ifeval", "frames", "all"],
        default=["all"],
        help="Choose which assets to download.",
    )
    parser.add_argument("--ifeval-sample-size", type=positive_int, default=100)
    parser.add_argument("--frames-sample-size", type=positive_int, default=100)
    return parser.parse_args()


def ensure_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def maybe_login(token: str | None) -> None:
    if token:
        print("Logging into Hugging Face Hub...")
        login(token=token)
    else:
        print("No HF token supplied. Public downloads only.")


def normalize_assets(assets: list[str]) -> list[str]:
    if "all" in assets:
        return ["base", "instruct", "judge", "data", "ifeval", "frames"]
    return assets


def configure_hf_downloads(timeout_seconds: int) -> None:
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = str(timeout_seconds)
    os.environ["HF_HUB_ETAG_TIMEOUT"] = str(timeout_seconds)


def download_model(repo_id: str, local_dir: Path, max_workers: int) -> None:
    print_section(f"Downloading model: {repo_id}")
    print(f"Saving to: {local_dir}")
    print(f"Max workers: {max_workers}")
    print("If this stops partway through, rerun the same command to resume.")
    snapshot_download(repo_id=repo_id, local_dir=local_dir, max_workers=max_workers)


def download_json(url: str, destination: Path) -> None:
    print(f"Downloading JSON -> {destination}")
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    destination.write_text(response.text, encoding="utf-8")


def download_csv(url: str, destination: Path) -> None:
    print(f"Downloading CSV -> {destination}")
    df = pd.read_csv(url)
    df.to_csv(destination, index=False)


def build_toolbench_prompts(destination: Path, sample_size: int) -> None:
    print_section("Preparing ToolBench prompts")
    dataset = load_dataset("tuandunghcmut/toolbench-v1", "benchmark", split="g3_instruction")
    prompts = []
    for index in range(min(sample_size, len(dataset))):
        item = dataset[index]
        raw_relevant_apis = item.get("relevant_apis", "[]")
        try:
            api_list_parsed = (json.loads(raw_relevant_apis)
                               if isinstance(raw_relevant_apis, str) else raw_relevant_apis)
            if not isinstance(api_list_parsed, list):
                raise ValueError("relevant_apis must be a list")
            golden_steps = []
            for step_num, api_pair in enumerate(api_list_parsed, start=1):
                if isinstance(api_pair, (list, tuple)) and len(api_pair) == 2:
                    golden_steps.append(
                        f"Step {step_num}: Call {api_pair[1]} from {api_pair[0]}"
                    )
            real_golden_path = "\n".join(golden_steps)
        except (json.JSONDecodeError, TypeError, ValueError):
            real_golden_path = "No required tools specified."
        prompts.append(
            {
                "query": item.get("query", ""),
                "available_tools": item.get("api_list", "[]"),
                "solution_path": real_golden_path,
            }
        )
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(prompts, handle, indent=2, ensure_ascii=False)
    print(f"Saved {len(prompts)} ToolBench prompts to: {destination}")


def save_dataset_rows(dataset, destination: Path, sample_size: int) -> None:
    rows = [dataset[index] for index in range(min(sample_size, len(dataset)))]
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2, ensure_ascii=False)
    print(f"Saved {len(rows)} rows to: {destination}")


def wikipedia_title_from_url(url: str) -> str | None:
    if not url or not isinstance(url, str):
        return None
    parsed = urlparse(url)
    if "/wiki/" not in parsed.path:
        return None
    return unquote(parsed.path.split("/wiki/", 1)[1]).replace("_", " ")


def fetch_wikipedia_extract(title: str) -> str:
    endpoint = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts",
        "explaintext": 1,
        "titles": title,
    }
    try:
        response = requests.get(
            endpoint,
            params=params,
            headers=WIKIPEDIA_HEADERS,
            timeout=120,
        )
        response.raise_for_status()
        payload = response.json()
        pages = payload.get("query", {}).get("pages", {})
        if not pages:
            return ""
        page = next(iter(pages.values()))
        return page.get("extract", "")
    except requests.RequestException as exc:
        print(f"Warning: could not fetch Wikipedia extract for '{title}': {exc}")
        return ""


def build_frames_with_context(destination: Path, sample_size: int) -> None:
    print_section("Preparing FRAMES with local evidence")
    dataset = load_dataset(FRAMES_DATASET_NAME, split="test")
    rows = []
    total = min(sample_size, len(dataset))
    for index in range(total):
        item = dataset[index]
        links = []
        for key, value in item.items():
            if key.startswith("wikipedia_link_") and value:
                links.append(value)
        evidence_blocks = []
        for link in links:
            title = wikipedia_title_from_url(link)
            if not title:
                continue
            extract = fetch_wikipedia_extract(title)
            evidence_blocks.append(
                {
                    "title": title,
                    "url": link,
                    "extract": extract[:4000],
                }
            )
        rows.append(
            {
                "Prompt": item.get("Prompt", ""),
                "Answer": item.get("Answer", ""),
                "reasoning_types": item.get("reasoning_types", ""),
                "wiki_links": item.get("wiki_links", ""),
                "evidence_blocks": evidence_blocks,
            }
        )
        print(f"Prepared FRAMES example {index + 1}/{total}")
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2, ensure_ascii=False)
    print(f"Saved {len(rows)} FRAMES examples with evidence to: {destination}")


def main() -> None:
    args = parse_args()
    assets = normalize_assets(args.assets)
    ensure_dirs(args.models_dir, args.raw_data_dir, args.processed_data_dir)
    maybe_login(args.hf_token)
    configure_hf_downloads(args.hf_timeout)

    if "base" in assets:
        download_model(BASE_MODEL_NAME, args.models_dir / "Qwen2.5-7B", args.max_workers)
    if "instruct" in assets:
        download_model(
            INSTRUCT_MODEL_NAME,
            args.models_dir / "Qwen2.5-7B-Instruct",
            args.max_workers,
        )
    if "judge" in assets:
        download_model(
            JUDGE_MODEL_NAME,
            args.models_dir / "Qwen2.5-72B-Instruct",
            args.max_workers,
        )
    if "data" in assets:
        print_section("Downloading raw datasets")
        download_json(
            LOST_IN_CONVERSATION_URL, args.raw_data_dir / "lost_in_conversation.json"
        )
        download_csv(
            HARMBENCH_URL, args.raw_data_dir / "harmbench_behaviors_text_all.csv"
        )
        build_toolbench_prompts(
            args.processed_data_dir / "toolbench_i3_test.json",
            args.toolbench_sample_size,
        )
    if "ifeval" in assets:
        print_section("Downloading IFEval dataset")
        ifeval = load_dataset(IFEVAL_DATASET_NAME, split="train")
        save_dataset_rows(
            ifeval,
            args.processed_data_dir / "ifeval_train.json",
            args.ifeval_sample_size,
        )
    if "frames" in assets:
        build_frames_with_context(
            args.processed_data_dir / "frames_benchmark_with_context.json",
            args.frames_sample_size,
        )

    print_section("Asset preparation complete")
    print(f"Models directory: {args.models_dir.resolve()}")
    print(f"Raw data directory: {args.raw_data_dir.resolve()}")
    print(f"Processed data directory: {args.processed_data_dir.resolve()}")


if __name__ == "__main__":
    main()
