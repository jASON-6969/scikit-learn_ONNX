"""Test scientific acceptance criteria using real sklearn and ONNX runtimes."""

import csv
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

import numpy as np
from threadpoolctl import threadpool_limits

from onnx_benchmark.cli import run_experiment, verify_saved_run
from onnx_benchmark.config import ExperimentConfig
from onnx_benchmark.conversion import convert_models
from onnx_benchmark.data_model import FloatArray, build_batches, prepare_data, train_model
from onnx_benchmark.runtime import InferenceAdapter, create_adapters, normalize_probabilities
from onnx_benchmark.validation import compare_predictions, validate_runtimes


def fixed_adapter(
    variant: str, probabilities: Any, labels: Any,
    classes: tuple[int, ...] = (0, 1),
) -> InferenceAdapter:
    """Supply intentionally valid or invalid runtime output to validation."""
    def infer_probabilities(x: FloatArray) -> Any:
        return probabilities

    def infer_labels(x: FloatArray) -> Any:
        return labels

    return InferenceAdapter(
        variant, infer_probabilities, infer_labels,
        np.asarray(classes, dtype=np.int64), {},
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


class DataAndOutputTests(unittest.TestCase):
    def test_split_is_reproducible_disjoint_and_unrepeated(self) -> None:
        config = ExperimentConfig()
        with tempfile.TemporaryDirectory() as temporary:
            first_dir = Path(temporary) / "first"
            second_dir = Path(temporary) / "second"
            first_dir.mkdir()
            second_dir.mkdir()
            first = prepare_data(config, first_dir)
            second = prepare_data(config, second_dir)
            np.testing.assert_array_equal(first.train_indices, second.train_indices)
            np.testing.assert_array_equal(first.test_indices, second.test_indices)
            np.testing.assert_array_equal(first.x_test, second.x_test)
            np.testing.assert_array_equal(first.y_test, second.y_test)
            self.assertEqual(len(first.train_indices), 455)
            self.assertEqual(len(first.test_indices), 114)
            self.assertEqual(np.intersect1d(first.train_indices, first.test_indices).size, 0)
            combined = np.concatenate((first.train_indices, first.test_indices))
            np.testing.assert_array_equal(np.sort(combined), np.arange(569))
            self.assertEqual(len(np.unique(first.test_indices)), len(first.test_indices))
            self.assertEqual(first.x_test.dtype, np.dtype(np.float32))
            with np.load(first_dir / "dataset.npz", allow_pickle=False) as saved:
                np.testing.assert_array_equal(first.x_test, saved["x"][saved["test_indices"]])

    def test_batches_repeat_only_as_fixed_synthetic_load(self) -> None:
        # Unique rows make unintended duplicated source cases visible.
        x = np.arange(114 * 3, dtype=np.float32).reshape(114, 3)
        sizes = (1, 16, 64, 256)
        batches, indices = build_batches(x, sizes, 42)
        repeated_batches, repeated_indices = build_batches(x, sizes, 42)
        self.assertEqual(set(batches), set(sizes))
        self.assertEqual(len(x), 114)
        for size in sizes:
            with self.subTest(batch_size=size):
                self.assertEqual(batches[size].shape, (size, 3))
                self.assertEqual(batches[size].dtype, np.dtype(np.float32))
                self.assertTrue(batches[size].flags.c_contiguous)
                np.testing.assert_array_equal(batches[size], repeated_batches[size])
                np.testing.assert_array_equal(indices[size], repeated_indices[size])
                np.testing.assert_array_equal(batches[size], x[indices[size]])
                if size <= len(x):
                    self.assertEqual(len(np.unique(indices[size])), size)
        np.testing.assert_array_equal(np.sort(indices[256][:114]), np.arange(114))
        np.testing.assert_array_equal(indices[256][114:228], indices[256][:114])
        np.testing.assert_array_equal(indices[1], indices[16][:1])
        np.testing.assert_array_equal(indices[16], indices[64][:16])

    def test_zipmap_uses_fitted_class_order_not_dictionary_order(self) -> None:
        classes = np.asarray([2, 9], dtype=np.int64)
        # Both insertion order and numeric ordering differ from the desired columns.
        raw = [{9: 0.75, 2: 0.25}, {2: 0.8, 9: 0.2}]
        np.testing.assert_array_equal(
            normalize_probabilities(raw, classes),
            np.asarray([[0.25, 0.75], [0.8, 0.2]]),
        )
        np.testing.assert_array_equal(
            normalize_probabilities(raw, classes[::-1]),
            np.asarray([[0.75, 0.25], [0.2, 0.8]]),
        )

    def test_validation_rejects_missing_keys_shapes_and_class_order(self) -> None:
        x = np.zeros((2, 3), dtype=np.float32)
        y = np.asarray([0, 1], dtype=np.int64)
        baseline = fixed_adapter("sklearn", [[0.8, 0.2], [0.1, 0.9]], y)
        candidates = (
            fixed_adapter("missing_key", [{0: 0.8}, {0: 0.1, 1: 0.9}], y),
            fixed_adapter("extra_key", [{0: 0.8, 1: 0.2, 2: 0.0}] * 2, y),
            fixed_adapter("vector", [0.8, 0.2], y),
            fixed_adapter("wrong_columns", [[1.0], [1.0]], y),
            fixed_adapter("missing_row", [[0.8, 0.2]], y[:1]),
            fixed_adapter("class_order", [[0.8, 0.2], [0.1, 0.9]], y, (1, 0)),
        )
        for candidate in candidates:
            with self.subTest(candidate=candidate.variant), self.assertRaises(ValueError):
                compare_predictions(baseline, candidate, x, y, 1e-5)

    def test_invalid_predictions_prevent_benchmark_and_save_failure_evidence(self) -> None:
        x = np.zeros((2, 3), dtype=np.float32)
        y = np.asarray([0, 1], dtype=np.int64)
        baseline = fixed_adapter("sklearn", [[0.8, 0.2], [0.1, 0.9]], y)
        candidates = (
            fixed_adapter("nan", [[np.nan, 0.2], [0.1, 0.9]], y),
            fixed_adapter("probability_error", [[0.7, 0.3], [0.1, 0.9]], y),
            fixed_adapter("label_error", [[0.8, 0.2], [0.1, 0.9]], y[::-1]),
            fixed_adapter("not_normalized", [[0.8, 0.8], [0.1, 0.9]], y),
            fixed_adapter("out_of_bounds", [[1.1, -0.1], [0.1, 0.9]], y),
        )
        for candidate in candidates:
            with self.subTest(candidate=candidate.variant), tempfile.TemporaryDirectory() as temporary:
                output = Path(temporary)
                with self.assertRaisesRegex(ValueError, "Prediction validation failed"):
                    validate_runtimes(
                        [baseline, candidate], x, y, {2: x},
                        {2: np.asarray([0, 1], dtype=np.int64)}, 1e-5, output,
                    )
                evidence = json.loads((output / "correctness.json").read_text(encoding="utf-8"))
                self.assertIs(evidence["passed"], False)
                self.assertIs(evidence["variants"][candidate.variant]["passed"], False)
                self.assertFalse((output / "timings.csv").exists())
                if candidate.variant == "nan":
                    self.assertIsNone(evidence["variants"][candidate.variant]["max_probability_abs_error"])


class RealRuntimeTests(unittest.TestCase):
    def test_same_fitted_forest_converts_and_matches_every_validation_batch(self) -> None:
        config = ExperimentConfig()
        with tempfile.TemporaryDirectory() as temporary, threadpool_limits(limits=1):
            output = Path(temporary)
            data = prepare_data(config, output)
            model, _ = train_model(config, output)
            paths, _ = convert_models(model, data.x_test.shape[1], config, output)
            adapters, _ = create_adapters(model, paths)
            batches, indices = build_batches(data.x_test, config.batch_sizes, config.seed)
            result = validate_runtimes(
                adapters, data.x_test, data.y_test, batches, indices,
                config.probability_tolerance, output,
            )
            self.assertIs(result["passed"], True)
            self.assertEqual(result["test_count"], 114)
            self.assertEqual(set(result["variants"]), {"sklearn", "onnx_zipmap", "onnx_tensor"})
            for variant, heldout in result["variants"].items():
                with self.subTest(variant=variant):
                    self.assertEqual(heldout["label_agreement"], 1.0)
                    self.assertLessEqual(heldout["max_probability_abs_error"], 1e-5)
                    self.assertEqual(heldout["accuracy"], result["baseline_accuracy"])
                    self.assertEqual(set(result["performance_batch_checks"][variant]), {"1", "16", "64", "256"})
                    self.assertTrue(all(check["passed"] for check in result["performance_batch_checks"][variant].values()))
                    self.assertTrue(all("accuracy" not in check for check in result["performance_batch_checks"][variant].values()))
            conversion = json.loads((output / "conversion.json").read_text(encoding="utf-8"))
            for variant, enabled in (("onnx_zipmap", True), ("onnx_tensor", False)):
                nodes = conversion[variant]["nodes"]
                self.assertEqual(sum(node["op_type"] == "ZipMap" for node in nodes), int(enabled))
            # Refit independently with the frozen split and seed; prediction
            # reproducibility does not rely on matching serialized file bytes.
            retrained, _ = train_model(config, output)
            np.testing.assert_array_equal(
                model.predict_proba(data.x_test), retrained.predict_proba(data.x_test),
            )

    def test_complete_smoke_run_verifies_statistics_and_rejects_corruption(self) -> None:
        config = ExperimentConfig(warmup_calls=1, rounds=2, calls_per_round=2)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            run_experiment(output, config)
            status = json.loads((output / "status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["status"], "complete")
            timings = read_csv(output / "timings.csv")
            summary = read_csv(output / "summary.csv")
            self.assertEqual(len(timings), 48)
            self.assertEqual(len(summary), 12)
            self.assertTrue(all(int(row["duration_ns"]) > 0 for row in timings))
            self.assertTrue(all(int(row["calls"]) == 4 for row in summary))
            for size in config.batch_sizes:
                for round_number, order in ((1, ("sklearn", "onnx_zipmap", "onnx_tensor")), (2, ("onnx_zipmap", "onnx_tensor", "sklearn"))):
                    rows = [row for row in timings if int(row["batch_size"]) == size and int(row["round"]) == round_number]
                    actual = tuple(row["variant"] for row in rows if row["call"] == "1")
                    self.assertEqual(actual, order)
                    self.assertEqual([int(row["order_in_round"]) for row in rows], [1, 1, 2, 2, 3, 3])
            for filename in ("latency.png", "throughput.png", "zipmap_speedup.png"):
                self.assertGreater((output / "figures" / filename).stat().st_size, 0)
            self.assertTrue((output / "results.md").is_file())
            original_correctness = (output / "correctness.json").read_bytes()
            verification = verify_saved_run(output)
            self.assertIs(verification["passed"], True)
            self.assertEqual(verification["groups"], 12)
            self.assertEqual(verification["timing_rows"], 48)
            self.assertEqual((output / "correctness.json").read_bytes(), original_correctness)
            self.assertTrue((output / "verification" / "verification.json").is_file())
            original_summary = (output / "summary.csv").read_bytes()
            summary[0]["median_ms"] = str(float(summary[0]["median_ms"]) * 2)
            with (output / "summary.csv").open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
                writer.writeheader()
                writer.writerows(summary)
            with self.assertRaisesRegex(ValueError, "Summary statistics disagree"):
                verify_saved_run(output)
            (output / "summary.csv").write_bytes(original_summary)
            batch_path = output / "batch_indices.npz"
            batch_path.write_bytes(batch_path.read_bytes() + b"corrupted evidence")
            with self.assertRaisesRegex(ValueError, "Artifact hash mismatch: batch_indices.npz"):
                verify_saved_run(output)

    def test_occupied_output_is_rejected_without_modifying_existing_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            marker = output / "existing.txt"
            marker.write_bytes(b"evidence must survive")
            with self.assertRaisesRegex(ValueError, "Output directory is not empty"):
                run_experiment(output, ExperimentConfig())
            self.assertEqual(marker.read_bytes(), b"evidence must survive")
            self.assertEqual(list(output.iterdir()), [marker])


if __name__ == "__main__":
    unittest.main()
