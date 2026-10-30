"""Convert the same fitted forest with and without a ZipMap output node."""

from pathlib import Path
from time import perf_counter
from typing import Any

import onnx
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

from .artifacts import sha256_file, write_json
from .config import ExperimentConfig


def convert_models(
    model: Any, feature_count: int, config: ExperimentConfig, output_dir: Path,
) -> tuple[dict[str, Path], dict[str, float]]:
    paths: dict[str, Path] = {}
    timings: dict[str, float] = {}
    manifests: dict[str, Any] = {}
    for variant, zipmap in (("onnx_zipmap", True), ("onnx_tensor", False)):
        start = perf_counter()
        graph = convert_sklearn(
            model,
            initial_types=[("features", FloatTensorType([None, feature_count]))],
            options={id(model): {"zipmap": zipmap}},
            target_opset={"": config.target_opset, "ai.onnx.ml": 3},
        )
        timings[variant] = perf_counter() - start
        onnx.checker.check_model(graph)
        path = output_dir / (variant + ".onnx")
        path.write_bytes(graph.SerializeToString())
        paths[variant] = path
        nodes = [{"op_type": node.op_type, "domain": node.domain} for node in graph.graph.node]
        if sum(node["op_type"] == "ZipMap" for node in nodes) != int(zipmap):
            raise ValueError("Converted graph does not have the requested ZipMap setting")
        manifests[variant] = {
            "file": path.name,
            "sha256": sha256_file(path),
            "zipmap": zipmap,
            "bytes": path.stat().st_size,
            "nodes": nodes,
            "requested_opsets": {"": config.target_opset, "ai.onnx.ml": 3},
            "opsets": {entry.domain: entry.version for entry in graph.opset_import},
            "opset_imports": [{"domain": entry.domain, "version": entry.version} for entry in graph.opset_import],
            "outputs": [output.name for output in graph.graph.output],
        }
    write_json(output_dir / "conversion.json", manifests)
    return paths, timings

