"""Plot presentation-reported values without implying a reproduced experiment.

Requires matplotlib (an optional documentation dependency).
Run from any directory: python scripts/plot_reported_results.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs/evidence/presentation-reported-results.json"
OUTPUT = ROOT / "docs/figures/presentation-reported-results.svg"


def main():
    report = json.loads(SOURCE.read_text())
    metrics = {row["id"]: row for row in report["metrics"] if row["include_in_chart"]}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "svg.hashsalt": "presentation-reported-results"})
    figure = plt.figure(figsize=(9, 8), facecolor="#ffffff")
    figure.text(0.05, 0.962, "Reported subset results", fontsize=20, weight="bold", color="#172b4d")
    figure.text(0.05, 0.925, "From the project presentation · independent reproduction pending", fontsize=10.5, color="#526274")
    colors = ["#63758b", "#00877c"]
    figure.legend([Patch(color=c) for c in colors], ["Qwen2.5-7B", "Qwen2.5-7B-Instruct"],
                  loc="upper left", bbox_to_anchor=(0.04, 0.898), ncol=2, frameon=False)
    panels = [
        ([0.28, 0.49, 0.66, 0.31], ["multiturn_geval", "toolbench_efficiency",
          "toolbench_rationality", "frames_groundedness"],
         ["Multi-turn persistence", "ToolBench efficiency", "ToolBench rationality", "FRAMES groundedness"],
         1, 5, "Judge scores · 1–5 scale", 0.07),
        ([0.28, 0.14, 0.66, 0.25], ["harmbench_safe_rate", "ifeval_constraint_rate", "frames_completion"],
         ["HarmBench safe rate", "IFEval constraint rate", "FRAMES task completion"],
         0, 100, "Rates · percent", 1.6),
    ]
    for rect, ids, labels, lower, upper, title, pad in panels:
        ax = figure.add_axes(rect)
        for i, key in enumerate(ids):
            row = metrics[key]
            for offset, field, color in [(-0.18, "base", colors[0]), (0.18, "instruct", colors[1])]:
                value = row[field]
                ax.barh(i + offset, value - lower, left=lower, height=0.29, color=color, zorder=3)
                label = f"{value:.2f}" if row["unit"] == "score" else f"{value:g}%"
                ax.text(value + pad, i + offset, label, va="center", fontsize=10, color="#172b4d")
        ax.set_yticks(range(len(ids)), labels)
        ax.invert_yaxis()
        ax.set_xlim(lower, upper + (0.52 if upper == 5 else 12))
        ax.set_xticks(range(1, 6) if upper == 5 else range(0, 101, 25))
        ax.set_title(title, loc="left", fontsize=12, weight="bold", pad=13)
        ax.grid(axis="x", color="#e8edf2", zorder=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(axis="both", length=0, pad=8, labelcolor="#36475a")
    figure.text(0.05, 0.065, "Sample counts and original run artifacts are unavailable; no uncertainty intervals are inferred.",
                fontsize=9, color="#526274")
    figure.text(0.05, 0.037, "Decomposition omitted: presentation labels 1–5, while the current code uses 0–3.",
                fontsize=9, color="#526274")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, metadata={"Date": None, "Creator": "scripts/plot_reported_results.py"})
    plt.close(figure)
    print(OUTPUT)


if __name__ == "__main__":
    main()
