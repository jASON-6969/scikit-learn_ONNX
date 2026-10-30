"""Run the complete study, or regenerate figures from existing measurements."""

import argparse
import csv
import json
import sys
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from .artifacts import sha256_file, source_hashes, write_json
from .benchmark import measure
from .config import ExperimentConfig
from .conversion import convert_models
from .data_model import build_batches, prepare_data, train_model
from .environment import collect_environment
from .reporting import generate_report
from .runtime import create_adapters
from .validation import validate_runtimes


def run_experiment(output_dir: Path, config: ExperimentConfig) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("Output directory is not empty; use a new path to preserve existing evidence")
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "config.json", config.to_dict())
    write_json(output_dir / "status.json", {"status": "running"})
    try:
        with threadpool_limits(limits=1):
            data = prepare_data(config, output_dir)
            model, train_time = train_model(config, output_dir)
            paths, conversion_times = convert_models(model, data.x_test.shape[1], config, output_dir)
            # Separately measure deserialization of the already-saved original model.
            start = perf_counter()
            loaded_model = joblib.load(output_dir / "random_forest.joblib")
            load_time = perf_counter() - start
            adapters, session_times = create_adapters(loaded_model, paths)
            write_json(output_dir / "setup_times.json", {
                "unit": "seconds", "repetitions": 1,
                "training": train_time,
                "sklearn_model_load": load_time,
                "conversion": conversion_times,
                "ort_session_creation": session_times,
                "scope": "single observed setup durations, excluded from steady-state inference",
            })
            write_json(output_dir / "environment.json", collect_environment())
            write_json(output_dir / "runtime_settings.json", {adapter.variant: adapter.settings for adapter in adapters})
            batches, batch_indices = build_batches(data.x_test, config.batch_sizes, config.seed)
            saved_batches: dict[str, Any] = {str(size): values for size, values in batch_indices.items()}
            np.savez_compressed(output_dir / "batch_indices.npz", **saved_batches)
            validate_runtimes(
                adapters, data.x_test, data.y_test, batches, batch_indices,
                config.probability_tolerance, output_dir,
            )
            print(f"Correctness passed on {len(data.y_test)} distinct held-out cases", flush=True)
            write_json(output_dir / "provenance.json", {
                "source_sha256": source_hashes(),
                "artifacts_sha256": {path.name: sha256_file(path) for path in (
                    output_dir / "dataset.npz", output_dir / "random_forest.joblib", *paths.values(),
                    output_dir / "batch_indices.npz",
                )},
                "command": sys.argv,
                "benchmark_scope": "probability API wall time, including native output materialization",
                "excluded": ["load", "session creation", "input preparation", "validation", "ZipMap-to-matrix normalization", "result destruction"],
            })
            measure(adapters, batches, config, output_dir)
        # Plotting happens after measurement to avoid benchmark interference.
        generate_report(output_dir)
        write_json(output_dir / "status.json", {"status": "complete"})
    except Exception as error:
        write_json(output_dir / "status.json", {"status": "failed", "error": str(error)})
        raise


def verify_saved_run(output_dir: Path) -> dict[str, Any]:
    """Independently verify provenance, split isolation and saved timing statistics."""
    provenance = json.loads((output_dir / "provenance.json").read_text(encoding="utf-8"))
    if json.loads((output_dir / "status.json").read_text(encoding="utf-8")).get("status") != "complete":
        raise ValueError("Run did not complete; inspect status.json")
    for filename, expected in provenance["artifacts_sha256"].items():
        if sha256_file(output_dir / filename) != expected:
            raise ValueError(f"Artifact hash mismatch: {filename}")
    if provenance["source_sha256"] != source_hashes():
        raise ValueError("Current source differs from the measured source")
    with np.load(output_dir / "dataset.npz", allow_pickle=False) as data:
        train_indices, test_indices = data["train_indices"], data["test_indices"]
        if np.intersect1d(train_indices, test_indices).size or len(np.unique(test_indices)) != len(test_indices):
            raise ValueError("Saved split has overlap or repeated held-out samples")
        all_indices = np.concatenate((train_indices, test_indices))
        if not np.array_equal(np.sort(all_indices), np.arange(len(data["y"]))):
            raise ValueError("Saved split does not cover each dataset sample exactly once")
        x_test, y_test = data["x"][test_indices], data["y"][test_indices]
    config_data = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))
    config_data["batch_sizes"] = tuple(config_data["batch_sizes"])
    config = ExperimentConfig(**config_data)
    model = joblib.load(output_dir / "random_forest.joblib")
    with threadpool_limits(limits=1):
        adapters, _ = create_adapters(model, {name: output_dir / (name + ".onnx") for name in ("onnx_zipmap", "onnx_tensor")})
        batches, indices = build_batches(x_test, config.batch_sizes, config.seed)
        with np.load(output_dir / "batch_indices.npz", allow_pickle=False) as saved:
            if set(saved.files) != {str(size) for size in config.batch_sizes} or any(
                not np.array_equal(saved[str(size)], indices[size]) for size in config.batch_sizes
            ):
                raise ValueError("Saved performance batch indices differ from the recorded configuration")
        # Write repeat verification to a separate folder, preserving original evidence.
        verification_dir = output_dir / "verification"
        verification_dir.mkdir(exist_ok=True)
        fresh_correctness = validate_runtimes(adapters, x_test, y_test, batches, indices, config.probability_tolerance, verification_dir)
        original_correctness = json.loads((output_dir / "correctness.json").read_text(encoding="utf-8"))
        if fresh_correctness != original_correctness:
            raise ValueError("Saved correctness metrics differ from reloaded predictions")
    with (output_dir / "timings.csv").open(encoding="utf-8", newline="") as stream:
        groups: dict[tuple[str, int], list[int]] = {}
        for row in csv.DictReader(stream):
            groups.setdefault((row["variant"], int(row["batch_size"])), []).append(int(row["duration_ns"]))
    expected_keys = {(variant, size) for variant in ("sklearn", "onnx_zipmap", "onnx_tensor") for size in config.batch_sizes}
    if set(groups) != expected_keys:
        raise ValueError("Missing or unexpected timing groups")
    with (output_dir / "summary.csv").open(encoding="utf-8", newline="") as stream:
        summary = list(csv.DictReader(stream))
    if len(summary) != len(expected_keys) or {(row["variant"], int(row["batch_size"])) for row in summary} != expected_keys:
        raise ValueError("Missing or duplicate summary rows")
    for row in summary:
        key = (row["variant"], int(row["batch_size"]))
        values = groups[key]
        expected_count = config.rounds * config.calls_per_round
        if len(values) != expected_count or int(row["calls"]) != expected_count or min(values) <= 0:
            raise ValueError("Invalid timing count or duration")
        median = float(np.median(values))
        baseline = float(np.median(groups[("sklearn", key[1])]))
        q25, q75, q95 = np.percentile(values, [25, 75, 95])
        expected_values = {
            "median_ms": median / 1e6,
            "p25_ms": float(q25) / 1e6,
            "p75_ms": float(q75) / 1e6,
            "p95_ms": float(q95) / 1e6,
            "throughput_samples_per_second": key[1] * 1e9 / median,
            "speedup_vs_sklearn": baseline / median,
        }
        if any(not np.isclose(float(row[field]), expected, rtol=1e-12, atol=0) for field, expected in expected_values.items()):
            raise ValueError(f"Summary statistics disagree with raw timing: {key}")
    result = {"passed": True, "groups": len(groups), "timing_rows": sum(map(len, groups.values())), "checks": ["source and model hashes", "disjoint held-out split", "reloaded prediction equivalence", "raw timing counts", "summary statistics"]}
    write_json(output_dir / "verification" / "verification.json", result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Same-model CPU inference comparison")
    parser.add_argument("command", choices=("run", "report", "verify"))
    parser.add_argument("--output", type=Path, default=Path("output/experiment"))
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--calls", type=int, default=100)
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            config = ExperimentConfig(warmup_calls=args.warmup, rounds=args.rounds, calls_per_round=args.calls)
            run_experiment(args.output, config)
        elif args.command == "report":
            generate_report(args.output)
        else:
            print(json.dumps(verify_saved_run(args.output), indent=2))
    except (ValueError, OSError) as error:
        parser.exit(1, f"Error: {error}\n")
    return 0

