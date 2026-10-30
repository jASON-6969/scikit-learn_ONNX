"""Validate predictions on distinct held-out cases and every timing batch."""

from pathlib import Path
from typing import Any

import numpy as np

from .artifacts import write_json
from .data_model import FloatArray, IntArray
from .runtime import InferenceAdapter, normalize_probabilities


def compare_predictions(
    baseline: InferenceAdapter, candidate: InferenceAdapter,
    x: FloatArray, y: IntArray, tolerance: float, *, include_accuracy: bool = True,
) -> dict[str, Any]:
    baseline_p = normalize_probabilities(baseline.infer_probabilities(x), baseline.classes)
    baseline_labels = np.asarray(baseline.infer_labels(x)).reshape(-1)
    probabilities = normalize_probabilities(candidate.infer_probabilities(x), candidate.classes)
    labels = np.asarray(candidate.infer_labels(x)).reshape(-1)
    if probabilities.shape != baseline_p.shape or len(labels) != len(y):
        raise ValueError("Prediction count does not match the input batch")
    if not np.array_equal(candidate.classes, baseline.classes):
        raise ValueError("Runtime class order differs from the fitted model")
    finite = bool(np.isfinite(probabilities).all())
    bounded = bool(((probabilities >= -tolerance) & (probabilities <= 1 + tolerance)).all())
    normalized = bool(np.allclose(probabilities.sum(axis=1), 1, rtol=0, atol=tolerance))
    error = float(np.max(np.abs(probabilities - baseline_p)))
    agreement = float(np.mean(labels == baseline_labels))
    result: dict[str, Any] = {
        "label_agreement": agreement,
        "max_probability_abs_error": error if finite else None,
        "probabilities_finite": finite,
        "probabilities_bounded": bounded,
        "probability_rows_normalized": normalized,
        "passed": bool(finite and bounded and normalized and agreement == 1 and error <= tolerance),
    }
    if include_accuracy:
        result["accuracy"] = float(np.mean(labels == y))
    return result


def validate_runtimes(
    adapters: list[InferenceAdapter], x_test: FloatArray, y_test: IntArray,
    batches: dict[int, FloatArray], batch_indices: dict[int, IntArray],
    tolerance: float, output_dir: Path,
) -> dict[str, Any]:
    baseline = adapters[0]
    heldout: dict[str, Any] = {}
    batch_checks: dict[str, Any] = {}
    for adapter in adapters:
        heldout[adapter.variant] = compare_predictions(baseline, adapter, x_test, y_test, tolerance)
        batch_checks[adapter.variant] = {
            str(size): compare_predictions(
                baseline, adapter, x, y_test[batch_indices[size]], tolerance, include_accuracy=False,
            )
            for size, x in batches.items()
        }
    result = {
        "passed": all(row["passed"] for row in heldout.values())
        and all(row["passed"] for variants in batch_checks.values() for row in variants.values()),
        "probability_tolerance": tolerance,
        "test_count": len(y_test),
        "baseline_accuracy": heldout[baseline.variant]["accuracy"],
        "accuracy_scope": "original non-repeated held-out test set only",
        "variants": heldout,
        "performance_batch_checks": batch_checks,
    }
    write_json(output_dir / "correctness.json", result)
    if not result["passed"]:
        raise ValueError("Prediction validation failed; inspect correctness.json before benchmarking")
    return result

