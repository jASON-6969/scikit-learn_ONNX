"""Create offline figures and an evidence-based summary of a benchmark run."""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.figure import Figure


_VARIANTS = ("sklearn", "onnx_zipmap", "onnx_tensor")
_BATCH_SIZES = (1, 16, 64, 256)
_LABELS = {
    "sklearn": "scikit-learn",
    "onnx_zipmap": "ONNX Runtime (ZipMap on)",
    "onnx_tensor": "ONNX Runtime (ZipMap off)",
}
_COLORS = {
    "sklearn": "#3366A0",
    "onnx_zipmap": "#D67B21",
    "onnx_tensor": "#33816B",
}


@dataclass(frozen=True)
class _SummaryRow:
    variant: str
    batch_size: int
    calls: int
    median_ms: float
    p25_ms: float
    p75_ms: float
    p95_ms: float
    throughput_samples_per_second: float
    speedup_vs_sklearn: float


def _load_summary(path: Path) -> list[_SummaryRow]:
    """Require the complete agreed matrix before generating comparisons."""
    rows: list[_SummaryRow] = []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        for record in csv.DictReader(stream):
            row = _SummaryRow(
                variant=record["variant"],
                batch_size=int(record["batch_size"]),
                calls=int(record["calls"]),
                median_ms=float(record["median_ms"]),
                p25_ms=float(record["p25_ms"]),
                p75_ms=float(record["p75_ms"]),
                p95_ms=float(record["p95_ms"]),
                throughput_samples_per_second=float(
                    record["throughput_samples_per_second"]
                ),
                speedup_vs_sklearn=float(record["speedup_vs_sklearn"]),
            )
            metrics = (
                row.median_ms,
                row.p25_ms,
                row.p75_ms,
                row.p95_ms,
                row.throughput_samples_per_second,
                row.speedup_vs_sklearn,
            )
            if row.calls <= 0 or any(
                not math.isfinite(value) or value <= 0 for value in metrics
            ):
                raise ValueError(f"Invalid benchmark metrics for {row.variant}.")
            if not row.p25_ms <= row.median_ms <= row.p75_ms <= row.p95_ms:
                raise ValueError(f"Unordered timing quantiles for {row.variant}.")
            rows.append(row)
    expected = {(variant, batch) for variant in _VARIANTS for batch in _BATCH_SIZES}
    actual = {(row.variant, row.batch_size) for row in rows}
    if actual != expected or len(rows) != len(expected):
        raise ValueError("summary.csv must contain each of the 12 configurations once.")
    return sorted(rows, key=lambda row: (_VARIANTS.index(row.variant), row.batch_size))


def _read_json_object(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as stream:
        value: object = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object.")
    return cast(dict[str, object], value)


def _save_figure(figure: Figure, path: Path) -> None:
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_metric(
    rows: list[_SummaryRow],
    selector: Callable[[_SummaryRow], float],
    title: str,
    ylabel: str,
    path: Path,
    *,
    with_iqr: bool = False,
) -> None:
    figure, axis = plt.subplots(figsize=(8.5, 5.2))
    positions = list(range(len(_BATCH_SIZES)))
    for variant in _VARIANTS:
        series = [row for row in rows if row.variant == variant]
        axis.plot(
            positions,
            [selector(row) for row in series],
            marker="o",
            linewidth=2,
            color=_COLORS[variant],
            label=_LABELS[variant],
        )
        if with_iqr:
            axis.fill_between(
                positions,
                [row.p25_ms for row in series],
                [row.p75_ms for row in series],
                color=_COLORS[variant],
                alpha=0.14,
            )
    axis.set_xticks(positions, [str(batch) for batch in _BATCH_SIZES])
    axis.set_xlabel("Batch size (samples per call)")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.set_ylim(bottom=0)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False)
    caption = (
        "Lines: median per-call latency. Shaded areas: P25-P75."
        if with_iqr
        else "Throughput is batch size divided by median per-call latency."
    )
    figure.text(0.5, 0.02, caption, ha="center", fontsize=9)
    figure.tight_layout(rect=(0, 0.05, 1, 1))
    _save_figure(figure, path)


def _zipmap_ratios(rows: list[_SummaryRow]) -> list[tuple[int, float]]:
    lookup = {(row.variant, row.batch_size): row for row in rows}
    return [
        (
            batch,
            lookup[("onnx_zipmap", batch)].median_ms
            / lookup[("onnx_tensor", batch)].median_ms,
        )
        for batch in _BATCH_SIZES
    ]


def _plot_zipmap(rows: list[_SummaryRow], path: Path) -> None:
    ratios = _zipmap_ratios(rows)
    figure, axis = plt.subplots(figsize=(8.5, 5.2))
    positions = list(range(len(ratios)))
    values = [ratio for _, ratio in ratios]
    bars = axis.bar(positions, values, color=_COLORS["onnx_tensor"], width=0.6)
    axis.bar_label(bars, labels=[f"{ratio:.3f}x" for ratio in values], padding=4)
    axis.axhline(1, color="#555555", linestyle="--", linewidth=1, label="Equal latency")
    axis.set_xticks(positions, [str(batch) for batch, _ in ratios])
    axis.set_xlabel("Batch size (samples per call)")
    axis.set_ylabel("Median latency ratio (ZipMap on / ZipMap off)")
    axis.set_title("Effect of the ZipMap output representation")
    axis.set_ylim(0, max(1, max(values)) * 1.22)
    axis.grid(axis="y", alpha=0.25)
    axis.set_axisbelow(True)
    axis.legend(frameon=False)
    figure.text(
        0.5,
        0.02,
        "Above 1: ZipMap off is faster. Below 1: ZipMap on is faster.",
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.05, 1, 1))
    _save_figure(figure, path)


def _format_value(value: object) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _format_percent(value: object) -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{value * 100:.4f}%"
    return _format_value(value)


def _correctness_lines(correctness: dict[str, object]) -> list[str]:
    variants = correctness.get("variants")
    if not isinstance(variants, dict):
        raise ValueError("correctness.json must contain a variants object.")
    lines = [
        "## Prediction correctness",
        "",
        f"Overall check passed: **{_format_value(correctness.get('passed'))}**. "
        f"Original held-out cases: **{_format_value(correctness.get('test_count'))}**. "
        f"Absolute probability tolerance: "
        f"**{_format_value(correctness.get('probability_tolerance'))}**.",
        "",
        "Baseline scikit-learn accuracy: "
        f"**{_format_percent(correctness.get('baseline_accuracy'))}**.",
        "",
        "| Backend | Label agreement | Accuracy | Maximum absolute probability error | "
        "Finite probabilities | Normalized probability rows |",
        "| --- | ---: | ---: | ---: | --- | --- |",
    ]
    for variant in _VARIANTS:
        result = variants.get(variant)
        if result is None:
            continue
        if not isinstance(result, dict):
            raise ValueError(f"Correctness metrics for {variant} must be an object.")
        lines.append(
            f"| {_LABELS[variant]} | {_format_percent(result.get('label_agreement'))} | "
            f"{_format_percent(result.get('accuracy'))} | "
            f"{_format_value(result.get('max_probability_abs_error'))} | "
            f"{_format_value(result.get('probabilities_finite'))} | "
            f"{_format_value(result.get('probability_rows_normalized'))} |"
        )
    if correctness.get("passed") is not True:
        lines.extend(
            [
                "",
                "**Correctness checks did not pass. Performance measurements alone "
                "do not establish equivalent model behavior; resolve the reported "
                "prediction differences before recommending a backend.**",
            ]
        )
    return lines


def _results_markdown(
    rows: list[_SummaryRow], correctness: dict[str, object], has_setup: bool,
    n_estimators: int,
) -> str:
    lines = [
        "# CPU inference benchmark results",
        "",
        f"This run compares the same trained {n_estimators}-tree Random Forest on the Breast "
        "Cancer Wisconsin dataset using scikit-learn and two ONNX Runtime output "
        "representations. All measurements come from this run on one CPU machine. "
        "Hardware and software versions are recorded in "
        "[environment.json](environment.json); the data split is recorded in "
        "[dataset.json](dataset.json), model parameters in "
        "[model_parameters.json](model_parameters.json), effective runtime and thread "
        "settings in [runtime_settings.json](runtime_settings.json), and experiment "
        "and timing settings in [config.json](config.json).",
        "",
        "## All measured configurations",
        "",
        "| Backend | Batch size | Timed calls | Median (ms) | P25-P75 (ms) | "
        "P95 (ms) | Throughput (samples/s) | Speedup vs scikit-learn |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {_LABELS[row.variant]} | {row.batch_size} | {row.calls} | "
            f"{row.median_ms:.6g} | {row.p25_ms:.6g}-{row.p75_ms:.6g} | "
            f"{row.p95_ms:.6g} | {row.throughput_samples_per_second:.6g} | "
            f"{row.speedup_vs_sklearn:.4f}x |"
        )
    lines.extend(
        [
            "",
            "The median is the middle observed per-call timing. P25-P75 shows the "
            "central 50% of timings (the interquartile range); P95 is the 95th "
            "percentile. These describe timing variation and are not confidence "
            "intervals for a population of machines. Throughput is batch size "
            "divided by median call time. Speedup is the scikit-learn median "
            "divided by the compared backend median for the same batch size: "
            "above 1 means faster and below 1 means slower.",
            "",
            "![Median latency and interquartile range](figures/latency.png)",
            "",
            "![Throughput from median latency](figures/throughput.png)",
            "",
            "## ZipMap comparison",
            "",
            "| Batch size | Median latency: ZipMap on / ZipMap off |",
            "| ---: | ---: |",
        ]
    )
    for batch, ratio in _zipmap_ratios(rows):
        lines.append(f"| {batch} | {ratio:.4f}x |")
    lines.extend(
        [
            "",
            "A ratio above 1 favors disabling ZipMap; below 1 favors enabling it. "
            "This is a measured change between converted graphs and their output "
            "representations, not an isolated measurement of one operator's cost.",
            "",
            "![ZipMap median latency ratios](figures/zipmap_speedup.png)",
            "",
        ]
    )
    lines.extend(_correctness_lines(correctness))
    lines.extend(
        [
            "",
            "## Measurement scope and limitations",
            "",
            "- Accuracy and agreement use the original held-out cases. Repeated "
            "inputs used to form synthetic performance batches are not additional "
            "independent accuracy samples.",
            "- Timings cover warmed, in-process inference calls and the output "
            "objects materialized by those APIs. Model loading, session creation, "
            "input preparation, output object destruction, and dictionary-to-matrix "
            "conversion for correctness checks are outside the timed region.",
            "- scikit-learn is measured through predict_proba; ONNX Runtime fetches "
            "the probability output only. ZipMap on returns dictionaries and ZipMap "
            "off returns a tensor. This compares probability-producing APIs rather "
            "than isolated model computation; scikit-learn's Python tree dispatch "
            "is included. Consumers needing a different final format may incur "
            "additional conversion costs.",
            "- OS power settings and thread placement are not controlled by this "
            "benchmark. Record the machine's power mode, concurrent workload, and "
            "other relevant run conditions manually when interpreting or repeating "
            "these measurements.",
            "- One machine, one dataset, and one Random Forest configuration "
            "cannot establish performance for other models, hardware, GPU execution, "
            "or production workloads. Repeated fixed inputs and warmed sessions "
            "also differ from cold requests and changing inputs.",
            "- Conclusions should follow the observed ratios and timing spread. "
            "A slower ONNX result is still a valid observation; no acceleration "
            "factor was assumed in advance.",
            "",
            "Raw per-call observations are retained in [timings.csv](timings.csv); "
            "the table above is generated from [summary.csv](summary.csv). "
            "Prediction checks are recorded in [correctness.json](correctness.json).",
        ]
    )
    if has_setup:
        lines.extend(
            [
                "",
                "Setup durations are recorded separately in "
                "[setup_times.json](setup_times.json) and are excluded from the "
                "steady-state inference ratios above.",
            ]
        )
    return "\n".join(lines) + "\n"


def generate_report(output_dir: Path) -> None:
    """Read completed run artifacts and write three PNGs plus results.md."""
    rows = _load_summary(output_dir / "summary.csv")
    correctness = _read_json_object(output_dir / "correctness.json")
    _read_json_object(output_dir / "environment.json")
    setup_path = output_dir / "setup_times.json"
    has_setup = setup_path.is_file()
    if has_setup:
        _read_json_object(setup_path)
    config = _read_json_object(output_dir / "config.json")
    n_estimators = config.get("n_estimators")
    if not isinstance(n_estimators, int) or isinstance(n_estimators, bool) or n_estimators <= 0:
        raise ValueError("config.json must contain a positive integer n_estimators.")
    markdown = _results_markdown(rows, correctness, has_setup, n_estimators)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    _plot_metric(
        rows,
        lambda row: row.median_ms,
        "CPU inference latency by batch size",
        "Median latency (ms per call)",
        figures_dir / "latency.png",
        with_iqr=True,
    )
    _plot_metric(
        rows,
        lambda row: row.throughput_samples_per_second,
        "CPU inference throughput by batch size",
        "Throughput (samples per second)",
        figures_dir / "throughput.png",
    )
    _plot_zipmap(rows, figures_dir / "zipmap_speedup.png")
    (output_dir / "results.md").write_text(markdown, encoding="utf-8")
