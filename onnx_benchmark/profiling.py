"""Collect diagnostic profiles separately from uninstrumented latency results."""

import cProfile
import json
import math
import pstats
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Callable

import joblib
import numpy as np
import onnxruntime as ort
from threadpoolctl import threadpool_limits

from .artifacts import write_json
from .config import ExperimentConfig
from .data_model import FloatArray, build_batches
from .runtime import create_session_options


LIMITATIONS = [
    "Profiles are separate instrumented executions; their durations and percentages are not benchmark latency measurements.",
    "cProfile categories group exclusive function self time by source location; cumulative times overlap and must not be summed.",
    "cProfile cannot resolve Cython tree traversal internals; untraced native work may be charged to its Python caller, including tree predict_proba.",
    "ORT kernel events cover graph operators, not all Python binding, feed handling or output materialization costs.",
    "ORT operator means and medians aggregate each operator's events within each measured model_run, after excluding warmup windows; zero-duration events are valid at trace clock resolution.",
    "Do not subtract profiled kernel time from separately measured API latency to claim an exact wrapper cost, or attribute the entire ZipMap/tensor latency difference to the ZipMap kernel.",
]


def _repeat(call: Callable[[], Any], count: int) -> None:
    for _ in range(count):
        result = call()
        del result


def _function_category(filename: str, function: str) -> str:
    """Use exclusive time only; native work may remain in its Python caller."""
    source = filename.replace("\\", "/").lower()
    if source.endswith("/profiling.py"):
        return "profiler_driver"
    if "/sklearn/utils/validation.py" in source or function in {
        "_validate_X_predict", "_validate_for_predict",
    }:
        return "input_validation"
    if "/sklearn/tree/" in source:
        return "tree_prediction"
    if "/sklearn/ensemble/_forest.py" in source:
        return "probability_aggregation" if function == "_accumulate_prediction" else "forest_orchestration"
    if "/joblib/" in source or "/sklearn/utils/parallel.py" in source:
        return "parallel_dispatch"
    if filename == "~":
        return "native_or_builtin"
    return "other"


def _profile_sklearn(
    model: Any, x: FloatArray, calls: int, warmup: int, output: Path,
) -> dict[str, Any]:
    call = lambda: model.predict_proba(x)
    _repeat(call, warmup)
    profiler = cProfile.Profile()
    try:
        profiler.runcall(_repeat, call, calls)
    finally:
        profiler.dump_stats(str(output))
    stats = pstats.Stats(profiler)
    functions: list[dict[str, Any]] = []
    raw_stats: dict[tuple[str, int, str], Any] = getattr(stats, "stats")
    for (filename, line, function), values in raw_stats.items():
        primitive_calls, total_calls, self_seconds, cumulative_seconds, _ = values
        functions.append({
            "filename": filename,
            "line": line,
            "function": function,
            "primitive_calls": primitive_calls,
            "total_calls": total_calls,
            "self_seconds": self_seconds,
            "cumulative_seconds": cumulative_seconds,
            "category": _function_category(filename, function),
        })
    functions.sort(key=lambda row: (-row["self_seconds"], row["filename"], row["line"], row["function"]))
    return {
        "batch_size": len(x),
        "profile_calls": calls,
        "total_self_seconds": sum(row["self_seconds"] for row in functions),
        "functions": functions,
    }


def _event_interval(event: dict[str, Any]) -> tuple[float, float]:
    start_value, duration_value = event.get("ts"), event.get("dur")
    if (
        isinstance(start_value, bool) or not isinstance(start_value, (int, float))
        or isinstance(duration_value, bool) or not isinstance(duration_value, (int, float))
    ):
        raise ValueError(f"Invalid timestamp/duration in ORT event: {event.get('name')}")
    start, duration = float(start_value), float(duration_value)
    if not math.isfinite(start) or not math.isfinite(duration) or duration < 0:
        raise ValueError(f"Invalid timestamp/duration in ORT event: {event.get('name')}")
    end = start + duration
    if not math.isfinite(end):
        raise ValueError(f"Invalid event endpoint in ORT event: {event.get('name')}")
    return start, end


def _summarize_onnx_trace(events: Any, calls: int, warmup: int) -> dict[str, Any]:
    """Assign kernels by time containment, never by the JSON event ordering."""
    if not isinstance(events, list) or any(not isinstance(event, dict) for event in events):
        raise ValueError("ORT trace must be a list of event objects")
    runs = [
        event for event in events
        if event.get("cat") == "Session" and event.get("name") == "model_run" and event.get("ph") == "X"
    ]
    if len(runs) != calls + warmup:
        raise ValueError(f"ORT model_run count {len(runs)} differs from expected {calls + warmup}")
    runs.sort(key=lambda event: _event_interval(event)[0])
    windows = [_event_interval(event) for event in runs]
    kernels = [
        event for event in events
        if event.get("cat") == "Node" and event.get("ph") == "X"
        and str(event.get("name", "")).endswith("_kernel_time")
    ]
    if not kernels:
        raise ValueError("ORT trace contains no matching kernel events")
    per_run: list[dict[str, float]] = [defaultdict(float) for _ in runs]
    event_counts: dict[str, int] = defaultdict(int)
    for event in kernels:
        start, end = _event_interval(event)
        matches = [index for index, (left, right) in enumerate(windows) if left <= start and end <= right]
        if len(matches) != 1:
            raise ValueError(f"ORT kernel event {event.get('name')} belongs to {len(matches)} model_run windows")
        args = event.get("args")
        op_name = args.get("op_name") if isinstance(args, dict) else None
        if not isinstance(op_name, str) or not op_name:
            raise ValueError(f"ORT kernel event lacks op_name: {event.get('name')}")
        index = matches[0]
        per_run[index][op_name] += end - start
        if index >= warmup:
            event_counts[op_name] += 1
    measured = per_run[warmup:]
    for index, row in enumerate(measured, start=warmup):
        if not row:
            raise ValueError(f"No kernel events matched measured model_run {index + 1}")
    operators = []
    for op_name in sorted(event_counts):
        durations = [row.get(op_name, 0.0) for row in measured]
        operators.append({
            "op_name": op_name,
            "event_count": event_counts[op_name],
            "total_us": sum(durations),
            "mean_us_per_call": mean(durations),
            "median_us_per_call": median(durations),
        })
    return {
        "profile_calls": calls,
        "operators": operators,
        "model_run_mean_us": mean(right - left for left, right in windows[warmup:]),
    }


def _profile_onnx(
    model_path: Path, variant: str, x: FloatArray, calls: int, warmup: int,
    output_dir: Path,
) -> dict[str, Any]:
    prefix = output_dir / f"{variant}_batch_{len(x)}"
    trace_path: Path | None = None
    try:
        session = ort.InferenceSession(
            str(model_path), sess_options=create_session_options(prefix),
            providers=["CPUExecutionProvider"],
        )
        try:
            inputs = {session.get_inputs()[0].name: x}
            outputs = [session.get_outputs()[1].name]
            _repeat(lambda: session.run(outputs, inputs)[0], warmup + calls)
        finally:
            trace_path = Path(session.end_profiling())
        result = _summarize_onnx_trace(
            json.loads(trace_path.read_text(encoding="utf-8")), calls, warmup,
        )
    except Exception as error:
        write_json(prefix.with_name(prefix.name + "_error.json"), {
            "error_type": type(error).__name__,
            "error": str(error),
            "trace_file": trace_path.name if trace_path is not None else None,
            "expected_profile_calls": calls,
            "expected_warmup_calls": warmup,
        })
        raise
    return {"variant": variant, "batch_size": len(x), **result}


def run_profiles(run_dir: Path, calls: int = 100, warmup: int = 50) -> None:
    """Profile saved models on their fixed batches, preserving raw evidence."""
    if isinstance(calls, bool) or not isinstance(calls, int) or calls <= 0:
        raise ValueError("Profile calls must be a positive integer")
    if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 0:
        raise ValueError("Profile warmup must be a non-negative integer")
    settings = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    settings["batch_sizes"] = tuple(settings["batch_sizes"])
    config = ExperimentConfig(**settings)
    with np.load(run_dir / "dataset.npz", allow_pickle=False) as saved:
        x_test = np.ascontiguousarray(saved["x"][saved["test_indices"]], dtype=np.float32)
    model = joblib.load(run_dir / "random_forest.joblib")
    if model.n_jobs != 1:
        raise ValueError("Saved forest must use n_jobs=1 for comparable profiles")
    batches, _ = build_batches(x_test, config.batch_sizes, config.seed)
    output_dir = run_dir / "profiling"
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("Profiling directory is not empty; existing evidence will not be overwritten")
    output_dir.mkdir(exist_ok=True)
    sklearn_rows = []
    onnx_rows = []
    with threadpool_limits(limits=1):
        for size, x in batches.items():
            sklearn_rows.append(_profile_sklearn(
                model, x, calls, warmup, output_dir / f"sklearn_batch_{size}.pstats",
            ))
            for variant in ("onnx_zipmap", "onnx_tensor"):
                onnx_rows.append(_profile_onnx(
                    run_dir / f"{variant}.onnx", variant, x, calls, warmup, output_dir,
                ))
    write_json(output_dir / "summary.json", {
        "calls": calls,
        "warmup": warmup,
        "sklearn": sklearn_rows,
        "onnx": onnx_rows,
        "limitations": LIMITATIONS,
    })
