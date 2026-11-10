"""Recompute study tables and create offline comparisons from saved evidence."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from statistics import median
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from .reporting import _COLORS, _LABELS, _VARIANTS, _load_summary, _save_figure

TableRow = dict[str, Any]


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _aggregate_runs(rows: list[TableRow], repeats: int) -> list[TableRow]:
    groups: dict[tuple[int, str, int], list[TableRow]] = defaultdict(list)
    for row in rows:
        groups[(row["n_estimators"], row["variant"], row["batch_size"])].append(row)
    result: list[TableRow] = []
    for (trees, variant, batch), group in groups.items():
        if len(group) != repeats or len({r["repeat"] for r in group}) != repeats:
            raise ValueError("Missing or repeated independent study runs")
        item: TableRow = {
            "n_estimators": trees, "variant": variant,
            "batch_size": batch, "repeats": repeats,
        }
        for metric in ("median_ms", "p95_ms", "throughput_samples_per_second", "speedup_vs_sklearn"):
            values = [float(row[metric]) for row in group]
            item[metric] = float(median(values))
            if metric == "speedup_vs_sklearn":
                item["min_speedup"], item["max_speedup"] = min(values), max(values)
            else:
                item["min_" + metric], item["max_" + metric] = min(values), max(values)
        result.append(item)
    return sorted(result, key=lambda r: (r["n_estimators"], r["batch_size"], _VARIANTS.index(r["variant"])))


def _profile_rows(profile: dict[str, Any], identity: TableRow) -> list[TableRow]:
    rows: list[TableRow] = []
    for batch in profile["sklearn"]:
        categories: dict[str, float] = defaultdict(float)
        calls: dict[str, int] = defaultdict(int)
        for function in batch["functions"]:
            categories[function["category"]] += float(function["self_seconds"])
            calls[function["category"]] += int(function["total_calls"])
        for category in sorted(categories):
            rows.append({
                **identity, "profiler": "cProfile", "variant": "sklearn",
                "batch_size": batch["batch_size"], "category": category,
                "profile_calls": batch["profile_calls"], "event_count": calls[category],
                "mean_us_per_call": categories[category] * 1e6 / batch["profile_calls"],
            })
    for batch in profile["onnx"]:
        for operator in batch["operators"]:
            rows.append({
                **identity, "profiler": "ORT", "variant": batch["variant"],
                "batch_size": batch["batch_size"], "category": operator["op_name"],
                "profile_calls": batch["profile_calls"], "event_count": operator["event_count"],
                "mean_us_per_call": operator["mean_us_per_call"],
            })
    if any(not math.isfinite(float(row["mean_us_per_call"])) or row["mean_us_per_call"] < 0 for row in rows):
        raise ValueError("Invalid profiling duration")
    return rows


def collect_study_tables(output_dir: Path) -> dict[str, list[TableRow]]:
    """Read evidence without modifying it; verification can independently reuse this."""
    config = _read_object(output_dir / "study_config.json")
    trees = config["trees"]
    repeats = int(config["repeats"])
    batches = config["experiment"]["batch_sizes"]
    expected_runs = {(repeat, tree) for repeat in range(1, repeats + 1) for tree in trees}
    entries = config["execution_order"]
    if len(entries) != len(expected_runs) or {(e["repeat"], e["n_estimators"]) for e in entries} != expected_runs:
        raise ValueError("Study manifest has missing or duplicate runs")
    runs: list[TableRow] = []
    sizes: list[TableRow] = []
    setups: list[TableRow] = []
    profiles: list[TableRow] = []
    for entry in entries:
        run_dir = output_dir / entry["run"]
        if _read_object(run_dir / "status.json").get("status") != "complete":
            raise ValueError(f"Incomplete child run: {entry['run']}")
        run_config = _read_object(run_dir / "config.json")
        if run_config["n_estimators"] != entry["n_estimators"]:
            raise ValueError("Child tree count differs from the study manifest")
        identity = {key: entry[key] for key in ("run", "repeat", "n_estimators", "order")}
        child_rows = _load_summary(run_dir / "summary.csv")
        expected = {(variant, batch) for variant in _VARIANTS for batch in batches}
        if {(row.variant, row.batch_size) for row in child_rows} != expected:
            raise ValueError("Child timing matrix differs from study settings")
        count = int(config["experiment"]["rounds"]) * int(config["experiment"]["calls_per_round"])
        if any(row.calls != count for row in child_rows):
            raise ValueError("Child timing call counts differ from study settings")
        runs.extend({**identity, **asdict(row)} for row in child_rows)
        metrics = _read_object(run_dir / "model_metrics.json")
        if metrics["n_estimators"] != entry["n_estimators"]:
            raise ValueError("Model metrics tree count differs from the study manifest")
        sizes.append({**identity, **metrics})
        setup = _read_object(run_dir / "setup_times.json")
        for phase in ("training", "sklearn_model_load"):
            setups.append({**identity, "phase": phase, "variant": "sklearn", "seconds": setup[phase]})
        for phase in ("conversion", "ort_session_creation"):
            setups.extend({**identity, "phase": phase, "variant": variant, "seconds": seconds}
                          for variant, seconds in setup[phase].items())
        profile = _read_object(run_dir / "profiling" / "summary.json")
        if profile["calls"] != config["profile_calls"]:
            raise ValueError("Profile call count differs from study settings")
        if {b["batch_size"] for b in profile["sklearn"]} != set(batches):
            raise ValueError("Incomplete sklearn profile matrix")
        if {(b["variant"], b["batch_size"]) for b in profile["onnx"]} != {
            (variant, batch) for variant in _VARIANTS[1:] for batch in batches
        }:
            raise ValueError("Incomplete ORT profile matrix")
        profiles.extend(_profile_rows(profile, identity))
    return {
        "runs.csv": runs,
        "summary.csv": _aggregate_runs(runs, repeats),
        "model_sizes.csv": sizes,
        "setup_times.csv": setups,
        "profiling_summary.csv": profiles,
    }


def _write_table(path: Path, rows: list[TableRow]) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty study table: {path.name}")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot_performance(rows: list[TableRow], figures: Path, *, speedup: bool) -> None:
    batches = sorted({r["batch_size"] for r in rows})
    trees = sorted({r["n_estimators"] for r in rows})
    figure, axes = plt.subplots(1, len(batches), figsize=(4 * len(batches), 4.5), squeeze=False)
    field = "speedup_vs_sklearn" if speedup else "median_ms"
    low_field = "min_speedup" if speedup else "min_median_ms"
    high_field = "max_speedup" if speedup else "max_median_ms"
    for axis, batch in zip(axes[0], batches):
        for variant in (_VARIANTS[1:] if speedup else _VARIANTS):
            series = [r for r in rows if r["batch_size"] == batch and r["variant"] == variant]
            series.sort(key=lambda r: r["n_estimators"])
            positions = list(range(len(trees)))
            axis.plot(positions, [r[field] for r in series], marker="o", label=_LABELS[variant], color=_COLORS[variant])
            axis.fill_between(positions, [r[low_field] for r in series], [r[high_field] for r in series], alpha=0.15, color=_COLORS[variant])
        axis.set_xticks(range(len(trees)), [str(t) for t in trees])
        axis.set_xlabel("Trees")
        axis.set_title(f"Batch {batch}")
        axis.grid(axis="y", alpha=0.25)
        if speedup:
            axis.axhline(1, color="gray", linestyle="--", linewidth=1)
            axis.set_ylabel("Speedup vs scikit-learn")
        else:
            axis.set_yscale("log")
            axis.set_ylabel("Median API latency (ms, log scale)")
    handles, labels = axes[0][0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncols=len(handles), frameon=False)
    figure.suptitle("Lines: median across independent runs; bands: observed run range")
    figure.tight_layout(rect=(0, 0.1, 1, 0.93))
    _save_figure(figure, figures / ("speedup.png" if speedup else "latency.png"))


def _plot_sizes(rows: list[TableRow], figures: Path) -> None:
    trees = sorted({r["n_estimators"] for r in rows})
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    positions = list(range(len(trees)))
    for offset, (variant, field) in enumerate((("sklearn", "sklearn_bytes"), ("onnx_zipmap", "onnx_zipmap_bytes"), ("onnx_tensor", "onnx_tensor_bytes"))):
        values = [float(median(r[field] for r in rows if r["n_estimators"] == tree)) / 1024 for tree in trees]
        axes[0].bar([p + (offset - 1) * 0.24 for p in positions], values, width=0.24, label=_LABELS[variant], color=_COLORS[variant])
    axes[0].set_ylabel("Serialized file size (KiB)")
    axes[0].set_title("Model files (not runtime memory use)")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].plot(positions, [median(r["total_nodes"] for r in rows if r["n_estimators"] == t) for t in trees], marker="o", color=_COLORS["sklearn"])
    axes[1].set_ylabel("Total nodes in fitted forest")
    axes[1].set_title("Actual model structure")
    for axis in axes:
        axis.set_xticks(positions, [str(t) for t in trees])
        axis.set_xlabel("Trees")
        axis.grid(axis="y", alpha=0.25)
        axis.set_axisbelow(True)
    figure.tight_layout()
    _save_figure(figure, figures / "model_size.png")


def _plot_profiles(rows: list[TableRow], figures: Path) -> None:
    batches = sorted({r["batch_size"] for r in rows})
    trees = sorted({r["n_estimators"] for r in rows})
    figure, axes = plt.subplots(2, len(batches), figsize=(4.2 * len(batches), 9), squeeze=False)
    for row_index, profiler in enumerate(("cProfile", "ORT")):
        selected = [r for r in rows if r["profiler"] == profiler]
        categories = sorted({r["category"] for r in selected})
        for axis, batch in zip(axes[row_index], batches):
            labels = [f"{tree}" if profiler == "cProfile" else f"{tree}\n{'map' if variant == 'onnx_zipmap' else 'tensor'}"
                      for tree in trees for variant in (("sklearn",) if profiler == "cProfile" else _VARIANTS[1:])]
            keys = [(tree, variant) for tree in trees for variant in (("sklearn",) if profiler == "cProfile" else _VARIANTS[1:])]
            bottom = [0.0] * len(keys)
            for category_index, category in enumerate(categories):
                values = []
                for tree, variant in keys:
                    samples = [float(r["mean_us_per_call"]) for r in selected if r["n_estimators"] == tree and r["variant"] == variant and r["batch_size"] == batch and r["category"] == category]
                    values.append(float(median(samples)) if samples else 0.0)
                axis.bar(range(len(keys)), values, bottom=bottom, label=category, color=plt.get_cmap("tab20")(category_index % 20))
                bottom = [a + b for a, b in zip(bottom, values)]
            axis.set_xticks(range(len(keys)), labels, fontsize=8)
            axis.set_title(f"{profiler}: batch {batch}")
            axis.set_xlabel("Trees / output format")
            axis.set_ylabel("Profiled self time (us/call)" if profiler == "cProfile" else "Profiled operator time (us/call)")
            axis.grid(axis="y", alpha=0.2)
            axis.set_axisbelow(True)
        handles, legends = axes[row_index][0].get_legend_handles_labels()
        axes[row_index][-1].legend(handles, legends, fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False)
    figure.suptitle("Separate profilers with instrumentation overhead; not benchmark latency or a common cost breakdown")
    figure.tight_layout(rect=(0, 0, 0.88, 0.95))
    _save_figure(figure, figures / "profiling.png")


def _markdown(output_dir: Path, config: dict[str, Any], tables: dict[str, list[TableRow]]) -> str:
    summary = tables["summary.csv"]
    sizes = tables["model_sizes.csv"]
    profiles = tables["profiling_summary.csv"]
    trees = sorted(config["trees"])
    batches = sorted(config["experiment"]["batch_sizes"])
    count = sum(r["calls"] for r in tables["runs.csv"])
    repeats = config["repeats"]
    lines = [
        "# 隨機森林樹數與 ONNX 推理效能研究", "",
        f"本次比較 {'、'.join(map(str, trees))} 棵樹，每種模型獨立執行 {repeats} 次，"
        f"共 {len(config['execution_order'])} 個子實驗、{count:,} 次正式計時。"
        "資料為 Breast Cancer Wisconsin，固定分層 80/20 切分、seed 42、float32、CPU 單執行緒。"
        "每個森林分別比較 scikit-learn 與由同一森林轉換的兩版 ONNX。", "",
        f"每配置先暖機 {config['experiment']['warmup_calls']} 次，再執行 "
        f"{config['experiment']['rounds']} 輪 × {config['experiment']['calls_per_round']} 次。"
        "樹數順序逐次輪換；所有正式計時完成後才執行 profiling 與繪圖。", "",
        "## 實測延遲及加速比", "",
        "表中延遲、P95、吞吐量及加速比均先按各次實驗計算，再取跨次中位數。"
        "加速比的範圍是各次觀察到的最小／最大值，不是信賴區間。"
        "加速比大於 1 表示 ONNX 較快，小於 1 表示較慢。", "",
        "| 樹數 | Batch | 執行方式 | 延遲 ms | 跨次延遲範圍 ms | P95 ms | 吞吐量 samples/s | 加速比 | 跨次加速比範圍 |",
        "| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary:
        lines.append(f"| {row['n_estimators']} | {row['batch_size']} | {_LABELS[row['variant']]} | {row['median_ms']:.6g} | "
                     f"{row['min_median_ms']:.6g}–{row['max_median_ms']:.6g} | {row['p95_ms']:.6g} | "
                     f"{row['throughput_samples_per_second']:.6g} | {row['speedup_vs_sklearn']:.3f}× | {row['min_speedup']:.3f}–{row['max_speedup']:.3f}× |")
    lines.extend(["", "![API 延遲](figures/latency.png)", "", "![加速比](figures/speedup.png)", "",
                  "## 模型大小是否影響加速效果", "",
                  "| 樹數 | 總節點數 | 最大深度 | sklearn 檔案 KiB | ONNX ZipMap KiB | ONNX tensor KiB |",
                  "| ---: | ---: | ---: | ---: | ---: | ---: |"])
    for tree in trees:
        group = [r for r in sizes if r["n_estimators"] == tree]
        fields = [float(median(r[field] for r in group)) for field in ("total_nodes", "max_depth", "sklearn_bytes", "onnx_zipmap_bytes", "onnx_tensor_bytes")]
        lines.append(f"| {tree} | {fields[0]:g} | {fields[1]:g} | {fields[2]/1024:.3f} | {fields[3]/1024:.3f} | {fields[4]/1024:.3f} |")
    lines.extend(["", "![模型大小與節點數](figures/model_size.png)", ""])
    for batch in batches:
        for variant in _VARIANTS[1:]:
            series = sorted([r for r in summary if r["variant"] == variant and r["batch_size"] == batch], key=lambda r: r["n_estimators"])
            ratios = [r["speedup_vs_sklearn"] for r in series]
            trend = "上升" if all(a < b for a, b in zip(ratios, ratios[1:])) else "下降" if all(a > b for a, b in zip(ratios, ratios[1:])) else "沒有單調變化"
            lines.append(f"- Batch {batch}，{_LABELS[variant]}：樹數由小到大時，加速比分別為 "
                         f"{'、'.join(f'{value:.3f}×' for value in ratios)}，跨次中位數呈現{trend}。")
    lines.extend(["", "這是同一資料集、同一種森林在不同樹數下的觀察。樹數增加也改變總節點數及模型結構；"
                  "序列化檔案大小不是 runtime 記憶體占用量，亦不能單獨證明快取或向量化機制。"
                  "不同樹數的模型各自訓練；一致性驗證只要求同一森林在不同 runtime 間一致，不要求三個森林預測完全相同。", "",
                  "## 各步驟耗時與 ONNX 加速原因", "",
                  "正式延遲涵蓋 probability API 呼叫及返回物件建立，排除輸入準備、模型載入、session 建立、驗證、"
                  "ZipMap 字典轉矩陣與返回物件銷毀。一次性準備成本另外列出：", "",
                  "| 樹數 | 步驟 | 執行方式 | 耗時中位數 ms | 跨次範圍 ms |",
                  "| ---: | --- | --- | ---: | ---: |"])
    for tree in trees:
        keys = sorted({(r["phase"], r["variant"]) for r in tables["setup_times.csv"] if r["n_estimators"] == tree})
        for phase, variant in keys:
            values = [r["seconds"] * 1000 for r in tables["setup_times.csv"] if r["n_estimators"] == tree and r["phase"] == phase and r["variant"] == variant]
            lines.append(f"| {tree} | {phase} | {_LABELS[variant]} | {median(values):.6g} | {min(values):.6g}–{max(values):.6g} |")
    lines.extend(["", "上述準備步驟在每個子實驗各量一次，沒有逐步冷啟動重複測試。訓練不含 joblib 儲存；"
                  "ONNX 轉換不含 checker 及序列化；這些資料不能相加當作完整啟動耗時。", "",
                  "![獨立 profiling](figures/profiling.png)", "",
                  f"每個子實驗與 batch 另外剖析 {config['profile_calls']} 次呼叫。sklearn 使用 cProfile，"
                  "以 self time 分類，不重複加總 cumulative time；ORT 使用獨立 profiling session 的實際算子事件，依 model_run 時間窗排除暖機。", "",
                  "| 樹數 | Batch | profiler／執行方式 | 步驟 | 剖析耗時 μs／call（跨次中位數） |",
                  "| ---: | ---: | --- | --- | ---: |"])
    profile_keys = sorted({(r["n_estimators"], r["batch_size"], r["profiler"], r["variant"], r["category"]) for r in profiles})
    for tree, batch, profiler, variant, category in profile_keys:
        values = [r["mean_us_per_call"] for r in profiles if (r["n_estimators"], r["batch_size"], r["profiler"], r["variant"], r["category"]) == (tree, batch, profiler, variant, category)]
        lines.append(f"| {tree} | {batch} | {profiler} / {_LABELS[variant]} | {category} | {median(values):.6g} |")
    lines.extend(["", "scikit-learn 的樹遍歷本身使用編譯的原生程式；predict_proba 仍須在 Python 層驗證輸入、"
                  "透過 joblib 逐樹調度、累加機率及求平均。ONNX 將森林編碼為 TreeEnsembleClassifier，"
                  "由 runtime 的原生 ensemble kernel 執行。上述剖析列出這些路徑的實測時間，"
                  "支持分析 API 組織及調度成本的差別；不能簡化成『Python 對 C++』，亦沒有直接量測快取、SIMD 或編譯器最佳化。", "",
                  "ZipMap 將機率 tensor 轉為類別標籤對應的 map，並在返回時建立 Python 容器。"
                  "同一森林的兩版 ONNX 在正式計時中的差異，是整體圖與輸出表示路徑的差異；"
                  "ORT 中 ZipMap 算子事件只能描述原生算子部分，不能涵蓋全部 Python 字典物件建立成本。"
                  "Cast 等算子是否執行以 trace 為準。", "",
                  "cProfile 與 ORT profiling 都會增加測量開銷，且觀察邊界不同。圖表上下兩列是分開的量測；"
                  "不能把它們視為共同的精確成本分解，不能用正式 API 延遲減去另一組 profile 算子時間來宣稱剩餘值是 wrapper 成本。"
                  "cProfile 亦不能分解原生 tree traversal 內部每個步驟。", "",
                  "## 正確性、證據及限制", "",
                  "| 子實驗 | 樹數 | 原始測試例數 | 驗證通過 | 最大 ONNX 機率誤差 |", "| --- | ---: | ---: | --- | ---: |"])
    for entry in config["execution_order"]:
        correctness = _read_object(output_dir / entry["run"] / "correctness.json")
        maximum = max(correctness["variants"][v]["max_probability_abs_error"] for v in _VARIANTS[1:])
        lines.append(f"| [{entry['run']}]({entry['run']}/correctness.json) | {entry['n_estimators']} | "
                     f"{correctness['test_count']} | {'是' if correctness['passed'] else '否'} | {maximum:.8g} |")
    lines.extend(["", "準確率僅使用原始、未重複的 114 個 held-out cases。Batch 256 的重複輸入是人工效能負載，"
                  "不增加獨立測試例數。所有數字限於這台機器、此資料集、此設定；三次重跑也不能代表其他機器。"
                  "CPU 頻率、電源模式與背景負載沒有由程式控制，未記錄的條件不作推測。", "",
                  "[study_config.json](study_config.json) 記錄執行順序與設定；[runs.csv](runs.csv) 保留各次摘要；"
                  "[summary.csv](summary.csv) 為跨次彙總；[model_sizes.csv](model_sizes.csv)、"
                  "[setup_times.csv](setup_times.csv)、[profiling_summary.csv](profiling_summary.csv) 可追溯至各子實驗。"
                  "原始 timings、模型、環境、profiling trace 與函式資料均保留在各子實驗目錄；"
                  "study-verify 重新載入模型、檢查雜湊並重算彙總。", ""])
    return "\n".join(lines)


def generate_study_report(output_dir: Path) -> None:
    """Regenerate tables, four figures and a Chinese report without remeasurement."""
    tables = collect_study_tables(output_dir)
    config = _read_object(output_dir / "study_config.json")
    markdown = _markdown(output_dir, config, tables)
    for filename, rows in tables.items():
        _write_table(output_dir / filename, rows)
    figures = output_dir / "figures"
    figures.mkdir(exist_ok=True)
    _plot_performance(tables["summary.csv"], figures, speedup=False)
    _plot_performance(tables["summary.csv"], figures, speedup=True)
    _plot_sizes(tables["model_sizes.csv"], figures)
    _plot_profiles(tables["profiling_summary.csv"], figures)
    (output_dir / "results_zh.md").write_text(markdown, encoding="utf-8")
