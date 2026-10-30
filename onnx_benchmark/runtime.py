"""Adapters expose matching probability inference while preserving output costs."""

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

import numpy as np
import onnxruntime as ort
from numpy.typing import NDArray

from .data_model import FloatArray

ProbabilityArray = NDArray[np.float64]


@dataclass
class InferenceAdapter:
    variant: str
    infer_probabilities: Callable[[FloatArray], Any]
    infer_labels: Callable[[FloatArray], Any]
    classes: NDArray[np.int64]
    settings: dict[str, Any]
    prepare_probability_call: Callable[[FloatArray], Callable[[], Any]] | None = None

    def bind_probabilities(self, x: FloatArray) -> Callable[[], Any]:
        if self.prepare_probability_call is not None:
            return self.prepare_probability_call(x)
        return lambda: self.infer_probabilities(x)


def create_adapters(
    model: Any, paths: dict[str, Path],
) -> tuple[list[InferenceAdapter], dict[str, float]]:
    adapters = [InferenceAdapter(
        variant="sklearn",
        infer_probabilities=model.predict_proba,
        infer_labels=model.predict,
        classes=np.asarray(model.classes_, dtype=np.int64),
        settings={"n_jobs": 1, "native_threadpools": 1, "requested_output": "probabilities"},
    )]
    setup_times: dict[str, float] = {}
    for variant, path in paths.items():
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        start = perf_counter()
        session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
        setup_times[variant] = perf_counter() - start
        input_name = session.get_inputs()[0].name
        label_name, probability_name = [output.name for output in session.get_outputs()]

        def probability_call(
            x: FloatArray, current_session: Any = session,
            current_input: str = input_name, output: str = probability_name,
        ) -> Any:
            return current_session.run([output], {current_input: x})[0]

        def label_call(
            x: FloatArray, current_session: Any = session,
            current_input: str = input_name, output: str = label_name,
        ) -> Any:
            return current_session.run([output], {current_input: x})[0]

        def prepare_call(
            x: FloatArray, current_session: Any = session,
            current_input: str = input_name, output: str = probability_name,
        ) -> Callable[[], Any]:
            outputs = [output]
            inputs = {current_input: x}
            return lambda: current_session.run(outputs, inputs)[0]

        adapters.append(InferenceAdapter(
            variant, probability_call, label_call, np.asarray(model.classes_, dtype=np.int64), {
                "providers": session.get_providers(),
                "intra_op_num_threads": 1,
                "inter_op_num_threads": 1,
                "execution_mode": "ORT_SEQUENTIAL",
                "graph_optimization_level": "ORT_ENABLE_ALL",
                "input": input_name,
                "requested_output": probability_name,
                "output_type": session.get_outputs()[1].type,
                "python_output_normalization_timed": False,
                "input_feed_prepared_before_timing": True,
            },
            prepare_call,
        ))
    return adapters, setup_times


def normalize_probabilities(raw: Any, classes: NDArray[np.int64]) -> ProbabilityArray:
    """Reorder ZipMap columns using model.classes_; never call inside the timer."""
    if isinstance(raw, list) and raw and isinstance(raw[0], dict):
        expected = {int(label) for label in classes}
        if any(set(row) != expected for row in raw):
            raise ValueError("ZipMap class keys differ from model.classes_")
        return np.asarray([[row[int(label)] for label in classes] for row in raw], dtype=np.float64)
    matrix = np.asarray(raw, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != len(classes):
        raise ValueError("Probability output must be a samples-by-classes matrix")
    return matrix

