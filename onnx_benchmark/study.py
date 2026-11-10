"""Run matched tree-count experiments, then profile and verify their evidence."""

import csv
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from .artifacts import sha256_file, source_hashes, write_json
from .config import ExperimentConfig, StudyConfig
from .experiment import run_experiment, verify_saved_run
from .profiling import run_profiles
from .reporting import generate_report
from .study_reporting import collect_study_tables, generate_study_report


def execution_order(config: StudyConfig) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for repeat in range(1, config.repeats + 1):
        offset = (repeat - 1) % len(config.trees)
        ordered = config.trees[offset:] + config.trees[:offset]
        for order, tree_count in enumerate(ordered, 1):
            entries.append({
                "run": f"repeat_{repeat}/trees_{tree_count}",
                "repeat": repeat, "n_estimators": tree_count, "order": order,
            })
    return entries


def _evidence_paths(output_dir: Path) -> list[Path]:
    """Figures and rendered prose are derived; hash all measurement evidence."""
    return sorted(
        path for path in output_dir.rglob("*")
        if path.is_file() and path.suffix in {".json", ".csv", ".npz", ".joblib", ".onnx", ".pstats"}
        and "verification" not in path.relative_to(output_dir).parts
        and path != output_dir / "provenance.json"
        and path != output_dir / "status.json"
    )


def run_study(
    output_dir: Path, config: ExperimentConfig,
    trees: tuple[int, ...] = (10, 100, 500), repeats: int = 3, profile_calls: int = 100,
) -> None:
    study = StudyConfig(trees, repeats, profile_calls, config)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("Output directory is not empty; use a new path to preserve existing evidence")
    output_dir.mkdir(parents=True, exist_ok=True)
    entries = execution_order(study)
    write_json(output_dir / "study_config.json", {**study.to_dict(), "execution_order": entries})
    write_json(output_dir / "status.json", {"status": "running", "stage": "measurement"})
    try:
        # Never run a profiler or render a chart between timed experiments.
        for entry in entries:
            print(f"Study measurement: {entry['run']} ({entry['order']} in repeat)", flush=True)
            run_experiment(
                output_dir / entry["run"], replace(config, n_estimators=entry["n_estimators"]),
                generate_figures=False,
            )
        write_json(output_dir / "status.json", {"status": "running", "stage": "profiling"})
        for entry in entries:
            print(f"Study profiling: {entry['run']}", flush=True)
            run_profiles(output_dir / entry["run"], calls=profile_calls, warmup=config.warmup_calls)
        write_json(output_dir / "status.json", {"status": "running", "stage": "reporting"})
        for entry in entries:
            generate_report(output_dir / entry["run"])
        generate_study_report(output_dir)
        write_json(output_dir / "provenance.json", {
            "source_sha256": source_hashes(),
            "artifacts_sha256": {
                path.relative_to(output_dir).as_posix(): sha256_file(path)
                for path in _evidence_paths(output_dir)
            },
            "command": sys.argv,
            "timing_scope": "unprofiled probability API calls; same boundary as individual runs",
            "profiling_scope": "separate diagnostic sessions after all measurements",
            "power_conditions": "not controlled; power mode and concurrent workload not independently verified",
        })
        write_json(output_dir / "status.json", {"status": "complete"})
        verify_study(output_dir)
    except Exception as error:
        write_json(output_dir / "status.json", {"status": "failed", "error": str(error)})
        raise


def _load_study(output_dir: Path) -> tuple[StudyConfig, list[dict[str, Any]]]:
    raw = json.loads((output_dir / "study_config.json").read_text(encoding="utf-8"))
    experiment = dict(raw["experiment"])
    experiment["batch_sizes"] = tuple(experiment["batch_sizes"])
    config = StudyConfig(tuple(raw["trees"]), raw["repeats"], raw["profile_calls"], ExperimentConfig(**experiment))
    entries = execution_order(config)
    if raw["execution_order"] != entries:
        raise ValueError("Study execution order does not match the recorded configuration")
    return config, entries


def _compare_npz(first: Path, second: Path) -> None:
    with np.load(first, allow_pickle=False) as left, np.load(second, allow_pickle=False) as right:
        if set(left.files) != set(right.files) or any(
            not np.array_equal(left[key], right[key]) for key in left.files
        ):
            raise ValueError(f"Study inputs differ between runs: {second}")


def _verify_tables(output_dir: Path) -> None:
    for filename, expected in collect_study_tables(output_dir).items():
        with (output_dir / filename).open(encoding="utf-8-sig", newline="") as stream:
            actual = list(csv.DictReader(stream))
        if len(actual) != len(expected):
            raise ValueError(f"Study summary statistics disagree with original runs: {filename}")
        for recorded, computed in zip(actual, expected):
            if set(recorded) != set(computed):
                raise ValueError(f"Study table columns disagree: {filename}")
            for key, value in computed.items():
                if isinstance(value, (float, int)):
                    matches = bool(np.isclose(float(recorded[key]), value, rtol=1e-12, atol=0))
                else:
                    matches = recorded[key] == str(value)
                if not matches:
                    raise ValueError(f"Study summary statistics disagree with original runs: {filename}:{key}")


def verify_study(output_dir: Path) -> dict[str, Any]:
    if json.loads((output_dir / "status.json").read_text(encoding="utf-8")).get("status") != "complete":
        raise ValueError("Study did not complete; inspect status.json")
    provenance = json.loads((output_dir / "provenance.json").read_text(encoding="utf-8"))
    if provenance["source_sha256"] != source_hashes():
        raise ValueError("Current source differs from the measured source")
    evidence = {path.relative_to(output_dir).as_posix() for path in _evidence_paths(output_dir)}
    if evidence != set(provenance["artifacts_sha256"]):
        raise ValueError("Study evidence inventory differs from provenance")
    for filename, expected in provenance["artifacts_sha256"].items():
        if sha256_file(output_dir / filename) != expected:
            raise ValueError(f"Artifact hash mismatch: {filename}")
    config, entries = _load_study(output_dir)
    first_dir = output_dir / entries[0]["run"]
    reference_runtime = json.loads((first_dir / "runtime_settings.json").read_text(encoding="utf-8"))
    timing_rows = 0
    for entry in entries:
        run_dir = output_dir / entry["run"]
        raw_config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        raw_config["batch_sizes"] = tuple(raw_config["batch_sizes"])
        if ExperimentConfig(**raw_config) != replace(config.experiment, n_estimators=entry["n_estimators"]):
            raise ValueError(f"Study tree count or configuration mismatch: {entry['run']}")
        result = verify_saved_run(run_dir)
        timing_rows += int(result["timing_rows"])
        _compare_npz(first_dir / "dataset.npz", run_dir / "dataset.npz")
        _compare_npz(first_dir / "batch_indices.npz", run_dir / "batch_indices.npz")
        runtime = json.loads((run_dir / "runtime_settings.json").read_text(encoding="utf-8"))
        if runtime != reference_runtime:
            raise ValueError("Study runtime settings differ between runs")
        profile = json.loads((run_dir / "profiling" / "summary.json").read_text(encoding="utf-8"))
        sizes = set(config.experiment.batch_sizes)
        if profile["calls"] != config.profile_calls or profile["warmup"] != config.experiment.warmup_calls:
            raise ValueError("Profile call counts differ from study configuration")
        if len(profile["sklearn"]) != len(sizes) or {item["batch_size"] for item in profile["sklearn"]} != sizes:
            raise ValueError("Incomplete sklearn profiling matrix")
        expected_ort = {(variant, size) for variant in ("onnx_zipmap", "onnx_tensor") for size in sizes}
        if len(profile["onnx"]) != len(expected_ort) or {
            (item["variant"], item["batch_size"]) for item in profile["onnx"]
        } != expected_ort:
            raise ValueError("Incomplete ONNX profiling matrix")
        if any(item["profile_calls"] != config.profile_calls for item in profile["sklearn"] + profile["onnx"]):
            raise ValueError("Incomplete profile call counts")
    _verify_tables(output_dir)
    result = {
        "passed": True, "runs": len(entries), "timing_rows": timing_rows,
        "groups": len(entries) * 3 * len(config.experiment.batch_sizes),
        "checks": ["source and evidence hashes", "rotating tree order", "matched split and batch inputs",
                   "child prediction equivalence and raw timing statistics", "matching runtime settings",
                   "complete profiling matrix", "recomputed study tables"],
    }
    verification_dir = output_dir / "verification"
    verification_dir.mkdir(exist_ok=True)
    write_json(verification_dir / "verification.json", result)
    return result
