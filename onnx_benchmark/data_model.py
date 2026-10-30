"""Freeze a stratified split, fit one forest, and build synthetic timing loads."""

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import joblib
import numpy as np
from numpy.typing import NDArray
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from .artifacts import write_json
from .config import ExperimentConfig

FloatArray = NDArray[np.float32]
IntArray = NDArray[np.int64]


@dataclass
class PreparedData:
    x_test: FloatArray
    y_test: IntArray
    train_indices: IntArray
    test_indices: IntArray


def prepare_data(config: ExperimentConfig, output_dir: Path) -> PreparedData:
    dataset = load_breast_cancer()
    # Both training and inference use the same float32 representation.
    x = np.ascontiguousarray(dataset.data, dtype=np.float32)
    y = np.asarray(dataset.target, dtype=np.int64)
    train_indices, test_indices = train_test_split(
        np.arange(len(y), dtype=np.int64),
        test_size=config.test_size,
        stratify=y,
        random_state=config.seed,
    )
    np.savez_compressed(
        output_dir / "dataset.npz",
        x=x,
        y=y,
        train_indices=train_indices,
        test_indices=test_indices,
    )
    write_json(output_dir / "dataset.json", {
        "name": "Breast Cancer Wisconsin Diagnostic",
        "source": "sklearn.datasets.load_breast_cancer",
        "samples": len(y),
        "features": x.shape[1],
        "train_count": len(train_indices),
        "test_count": len(test_indices),
        "feature_names": dataset.feature_names.tolist(),
        "target_names": dataset.target_names.tolist(),
        "dtype": str(x.dtype),
        "seed": config.seed,
        "test_size": config.test_size,
        "stratified": True,
    })
    return PreparedData(x[test_indices], y[test_indices], train_indices, test_indices)


def train_model(
    config: ExperimentConfig, output_dir: Path,
) -> tuple[Any, float]:
    with np.load(output_dir / "dataset.npz", allow_pickle=False) as saved:
        x, y, train_indices = saved["x"], saved["y"], saved["train_indices"]
    model = RandomForestClassifier(
        n_estimators=config.n_estimators,
        random_state=config.seed,
        n_jobs=1,
    )
    start = perf_counter()
    model.fit(x[train_indices], y[train_indices])
    elapsed = perf_counter() - start
    joblib.dump(model, output_dir / "random_forest.joblib")
    write_json(output_dir / "model_parameters.json", model.get_params())
    return model, elapsed


def build_batches(
    x_test: FloatArray, sizes: tuple[int, ...], seed: int,
) -> tuple[dict[int, FloatArray], dict[int, IntArray]]:
    """Use one fixed shuffled test sequence, repeated only as artificial load."""
    if len(x_test) == 0:
        raise ValueError("Cannot build performance batches from an empty test set")
    order = np.random.default_rng(seed).permutation(len(x_test)).astype(np.int64)
    batches: dict[int, FloatArray] = {}
    batch_indices: dict[int, IntArray] = {}
    for size in sizes:
        if size <= 0:
            raise ValueError("batch sizes must be positive")
        indices = np.resize(order, size)
        batch_indices[size] = indices
        batches[size] = np.ascontiguousarray(x_test[indices], dtype=np.float32)
    return batches, batch_indices

