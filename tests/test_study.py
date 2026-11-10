"""Exercise the tree-size study with real inference and preserved evidence."""

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from onnx_benchmark.config import ExperimentConfig
from onnx_benchmark.study import run_study, verify_study


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


class StudyIntegrationTests(unittest.TestCase):
    """Share one small real study instead of repeating its setup for each check."""

    temporary: tempfile.TemporaryDirectory[str]
    output: Path
    config: ExperimentConfig
    trees = (2, 3, 4)

    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.output = Path(cls.temporary.name) / "study"
        cls.config = ExperimentConfig(warmup_calls=0, rounds=1, calls_per_round=1)
        run_study(cls.output, cls.config, trees=cls.trees, repeats=3, profile_calls=2)

    def _child_runs(self) -> list[tuple[int, int, Path]]:
        return [
            (repeat, trees, self.output / f"repeat_{repeat}" / f"trees_{trees}")
            for repeat in range(1, 4)
            for trees in self.trees
        ]

    def _refresh_root_hash(self, path: Path) -> None:
        """Keep the manifest valid to test content checks independently of hashes."""
        manifest_path = self.output / "provenance.json"
        manifest = read_json(manifest_path)
        relative = path.relative_to(self.output).as_posix()
        hashes = manifest["artifacts_sha256"]
        for key in list(hashes):
            if key.replace("\\", "/") == relative:
                hashes[key] = hashlib.sha256(path.read_bytes()).hexdigest()
        write_json(manifest_path, manifest)

    def test_complete_study_preserves_matched_data_and_rotates_tree_order(self) -> None:
        self.assertEqual(read_json(self.output / "status.json")["status"], "complete")
        settings = read_json(self.output / "study_config.json")
        self.assertEqual(settings["trees"], list(self.trees))
        self.assertEqual(settings["repeats"], 3)
        self.assertEqual(settings["profile_calls"], 2)
        expected_order: list[dict[str, Any]] = []
        for repeat in range(1, 4):
            offset = (repeat - 1) % len(self.trees)
            ordered = self.trees[offset:] + self.trees[:offset]
            expected_order.extend({
                "run": f"repeat_{repeat}/trees_{trees}",
                "repeat": repeat,
                "n_estimators": trees,
                "order": order,
            } for order, trees in enumerate(ordered, start=1))
        self.assertEqual(settings["execution_order"], expected_order)

        reference_data: dict[str, Any] | None = None
        reference_batches: dict[str, Any] | None = None
        raw_timing_count = 0
        for repeat, trees, child in self._child_runs():
            with self.subTest(repeat=repeat, trees=trees):
                config = read_json(child / "config.json")
                self.assertEqual(config["n_estimators"], trees)
                self.assertEqual(config["seed"], self.config.seed)
                self.assertEqual(config["batch_sizes"], list(self.config.batch_sizes))
                self.assertEqual(config["warmup_calls"], 0)
                self.assertEqual(config["rounds"] * config["calls_per_round"], 1)
                self.assertIs(read_json(child / "correctness.json")["passed"], True)
                self.assertEqual(read_json(child / "correctness.json")["test_count"], 114)
                with np.load(child / "dataset.npz", allow_pickle=False) as saved:
                    data = {key: saved[key] for key in saved.files}
                with np.load(child / "batch_indices.npz", allow_pickle=False) as saved:
                    batches = {key: saved[key] for key in saved.files}
                if reference_data is None or reference_batches is None:
                    reference_data, reference_batches = data, batches
                self.assertEqual(set(data), set(reference_data))
                self.assertEqual(set(batches), set(reference_batches))
                for key, values in data.items():
                    np.testing.assert_array_equal(values, reference_data[key])
                for key, values in batches.items():
                    np.testing.assert_array_equal(values, reference_batches[key])

                model = joblib.load(child / "random_forest.joblib")
                metrics = read_json(child / "model_metrics.json")
                self.assertEqual(len(model.estimators_), trees)
                self.assertEqual(metrics["n_estimators"], trees)
                self.assertEqual(metrics["total_nodes"], sum(
                    estimator.tree_.node_count for estimator in model.estimators_
                ))
                self.assertEqual(metrics["max_depth"], max(
                    estimator.tree_.max_depth for estimator in model.estimators_
                ))
                for name, filename in (
                    ("sklearn_bytes", "random_forest.joblib"),
                    ("onnx_zipmap_bytes", "onnx_zipmap.onnx"),
                    ("onnx_tensor_bytes", "onnx_tensor.onnx"),
                ):
                    self.assertEqual(metrics[name], (child / filename).stat().st_size)
                timings = read_csv(child / "timings.csv")
                self.assertEqual(len(timings), 12)
                self.assertTrue(all(int(row["duration_ns"]) > 0 for row in timings))
                raw_timing_count += len(timings)
                report = (child / "results.md").read_text(encoding="utf-8")
                self.assertIn(f"{trees}-tree", report)
                self.assertNotIn("100-tree", report)
        self.assertEqual(raw_timing_count, 108)

    def test_aggregate_statistics_match_the_independent_repeat_summaries(self) -> None:
        runs = read_csv(self.output / "runs.csv")
        summary = read_csv(self.output / "summary.csv")
        self.assertEqual(len(runs), 108)
        self.assertEqual(len(summary), 36)
        expected_keys = {
            (trees, variant, batch)
            for trees in self.trees
            for variant in ("sklearn", "onnx_zipmap", "onnx_tensor")
            for batch in self.config.batch_sizes
        }
        self.assertEqual({
            (int(row["n_estimators"]), row["variant"], int(row["batch_size"]))
            for row in summary
        }, expected_keys)
        for row in summary:
            key = (int(row["n_estimators"]), row["variant"], int(row["batch_size"]))
            with self.subTest(key=key):
                matching = [item for item in runs if (
                    int(item["n_estimators"]), item["variant"], int(item["batch_size"])
                ) == key]
                self.assertEqual(len(matching), 3)
                self.assertEqual({int(item["repeat"]) for item in matching}, {1, 2, 3})
                self.assertEqual(int(row["repeats"]), 3)
                medians = [float(item["median_ms"]) for item in matching]
                speedups = [float(item["speedup_vs_sklearn"]) for item in matching]
                for field, expected in (
                    ("median_ms", float(np.median(medians))),
                    ("min_median_ms", min(medians)),
                    ("max_median_ms", max(medians)),
                    ("speedup_vs_sklearn", float(np.median(speedups))),
                    ("min_speedup", min(speedups)),
                    ("max_speedup", max(speedups)),
                ):
                    self.assertTrue(np.isclose(float(row[field]), expected, rtol=1e-12, atol=0), field)

        for filename in ("model_sizes.csv", "setup_times.csv", "profiling_summary.csv"):
            self.assertGreater(len(read_csv(self.output / filename)), 0, filename)
        self.assertGreater((self.output / "results_zh.md").stat().st_size, 0)
        figures = list((self.output / "figures").glob("*.png"))
        self.assertTrue(figures)
        self.assertTrue(all(path.stat().st_size > 0 for path in figures))

    def test_verification_preserves_original_measurements(self) -> None:
        paths = [self.output / "runs.csv", self.output / "summary.csv"]
        paths.extend(
            child / filename
            for _, _, child in self._child_runs()
            for filename in ("timings.csv", "summary.csv", "correctness.json")
        )
        original = {path: path.read_bytes() for path in paths}
        self.assertIs(verify_study(self.output)["passed"], True)
        for path, content in original.items():
            self.assertEqual(path.read_bytes(), content, str(path))

    def test_verification_rejects_corrupted_or_missing_evidence(self) -> None:
        provenance_path = self.output / "provenance.json"
        original_provenance = provenance_path.read_bytes()
        child = self.output / "repeat_1" / "trees_2"

        with self.subTest(corruption="source hash"):
            try:
                provenance = read_json(provenance_path)
                source_hashes = provenance["source_sha256"]
                self.assertTrue(source_hashes)
                source_hashes[next(iter(source_hashes))] = "0" * 64
                write_json(provenance_path, provenance)
                with self.assertRaises((ValueError, OSError)):
                    verify_study(self.output)
            finally:
                provenance_path.write_bytes(original_provenance)

        for filename in ("random_forest.joblib", "onnx_tensor.onnx"):
            with self.subTest(corruption=filename):
                path = child / filename
                original = path.read_bytes()
                try:
                    path.write_bytes(original + b"corrupted evidence")
                    with self.assertRaises((ValueError, OSError)):
                        verify_study(self.output)
                finally:
                    path.write_bytes(original)

        with self.subTest(corruption="rehashed summary statistic"):
            path = self.output / "summary.csv"
            original = path.read_bytes()
            try:
                rows = read_csv(path)
                rows[0]["median_ms"] = str(float(rows[0]["median_ms"]) * 2)
                with path.open("w", newline="", encoding="utf-8") as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
                self._refresh_root_hash(path)
                with self.assertRaises((ValueError, OSError)):
                    verify_study(self.output)
            finally:
                path.write_bytes(original)
                provenance_path.write_bytes(original_provenance)

        with self.subTest(corruption="tree configuration mismatch"):
            path = child / "config.json"
            original = path.read_bytes()
            try:
                config = read_json(path)
                config["n_estimators"] = 99
                write_json(path, config)
                self._refresh_root_hash(path)
                with self.assertRaises((ValueError, OSError)):
                    verify_study(self.output)
            finally:
                path.write_bytes(original)
                provenance_path.write_bytes(original_provenance)

        with self.subTest(corruption="missing child"):
            moved = Path(self.temporary.name) / "temporarily_missing_child"
            child.rename(moved)
            try:
                with self.assertRaises((ValueError, OSError)):
                    verify_study(self.output)
            finally:
                moved.rename(child)
        self.assertIs(verify_study(self.output)["passed"], True)

    def test_nonempty_output_is_rejected_without_changing_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            marker = output / "existing.txt"
            marker.write_bytes(b"evidence must survive")
            with self.assertRaises(ValueError):
                run_study(output, self.config, trees=self.trees, repeats=3, profile_calls=2)
            self.assertEqual(marker.read_bytes(), b"evidence must survive")
            self.assertEqual(list(output.iterdir()), [marker])


if __name__ == "__main__":
    unittest.main()
