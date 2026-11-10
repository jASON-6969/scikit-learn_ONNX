"""Check diagnostic profile attribution and real runtime trace compatibility."""

import json
import math
import pstats
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

from threadpoolctl import threadpool_limits

from onnx_benchmark.artifacts import write_json
from onnx_benchmark.config import ExperimentConfig
from onnx_benchmark.conversion import convert_models
from onnx_benchmark.data_model import prepare_data, train_model
from onnx_benchmark.profiling import _summarize_onnx_trace, run_profiles


def run_event(start: float, duration: float) -> dict[str, Any]:
    return {"cat": "Session", "name": "model_run", "ph": "X", "ts": start, "dur": duration}


def kernel_event(start: float, duration: Any, op: str = "TreeEnsembleClassifier") -> dict[str, Any]:
    return {
        "cat": "Node", "name": f"{op}_kernel_time", "ph": "X", "ts": start, "dur": duration,
        "args": {"op_name": op, "provider": "CPUExecutionProvider"},
    }


class TraceAttributionTests(unittest.TestCase):
    def test_warmup_is_excluded_by_sorted_time_windows_and_per_call_operators(self) -> None:
        # Deliberately unordered like real traces; warmup is much slower.
        events = [
            kernel_event(401, 2, "ZipMap"), run_event(400, 20),
            kernel_event(301, 3), kernel_event(305, 5), run_event(300, 20),
            kernel_event(110, 80), run_event(100, 100),
            kernel_event(404, 6),
            {"cat": "Node", "name": "TreeEnsembleClassifier_fence_before", "ph": "X", "ts": 0, "dur": 0},
        ]
        result = _summarize_onnx_trace(events, calls=2, warmup=1)
        operators = {row["op_name"]: row for row in result["operators"]}
        tree = operators["TreeEnsembleClassifier"]
        self.assertEqual(tree["event_count"], 3)
        self.assertEqual(tree["total_us"], 14)
        self.assertEqual(tree["mean_us_per_call"], 7)
        self.assertEqual(tree["median_us_per_call"], 7)
        self.assertEqual(operators["ZipMap"]["mean_us_per_call"], 1)
        self.assertEqual(operators["ZipMap"]["median_us_per_call"], 1)
        self.assertEqual(result["model_run_mean_us"], 20)

    def test_zero_duration_events_are_valid(self) -> None:
        result = _summarize_onnx_trace([kernel_event(10, 0), run_event(10, 0)], calls=1, warmup=0)
        self.assertEqual(result["operators"][0]["total_us"], 0)
        self.assertEqual(result["operators"][0]["event_count"], 1)
        self.assertEqual(result["model_run_mean_us"], 0)

    def test_missing_extra_and_unattributed_events_are_rejected(self) -> None:
        cases = (
            ([run_event(0, 10), kernel_event(1, 2)], 2, "model_run count"),
            ([run_event(0, 10)], 1, "no matching kernel events"),
            ([run_event(0, 10), kernel_event(11, 1)], 1, "belongs to 0"),
            ([run_event(0, 10), kernel_event(9, 2)], 1, "belongs to 0"),
            ([run_event(0, 10), run_event(20, 10), kernel_event(1, 2)], 2, "No kernel events matched"),
        )
        for events, calls, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                _summarize_onnx_trace(events, calls=calls, warmup=0)

    def test_ambiguous_containment_and_boundary_zero_event_are_rejected(self) -> None:
        cases = (
            [run_event(0, 20), run_event(10, 20), kernel_event(11, 1)],
            [run_event(0, 10), run_event(10, 10), kernel_event(10, 0)],
        )
        for events in cases:
            with self.subTest(events=events), self.assertRaisesRegex(ValueError, "belongs to 2"):
                _summarize_onnx_trace(events, calls=2, warmup=0)

    def test_invalid_timestamps_durations_or_missing_op_name_are_rejected(self) -> None:
        for duration in (-1, math.nan, math.inf, "1", True):
            with self.subTest(duration=duration), self.assertRaisesRegex(ValueError, "Invalid timestamp/duration"):
                _summarize_onnx_trace([run_event(0, 10), kernel_event(1, duration)], calls=1, warmup=0)
        event = kernel_event(1, 2)
        event["args"] = {}
        with self.assertRaisesRegex(ValueError, "lacks op_name"):
            _summarize_onnx_trace([run_event(0, 10), event], calls=1, warmup=0)


class ProfileRuntimeTests(unittest.TestCase):
    def test_real_profiles_preserve_all_batches_and_raw_evidence(self) -> None:
        config = ExperimentConfig(n_estimators=3)
        with tempfile.TemporaryDirectory() as temporary, threadpool_limits(limits=1):
            output = Path(temporary)
            data = prepare_data(config, output)
            model, _ = train_model(config, output)
            convert_models(model, data.x_test.shape[1], config, output)
            write_json(output / "config.json", config.to_dict())
            original_files = {path.name: path.read_bytes() for path in output.iterdir() if path.is_file()}
            run_profiles(output, calls=2, warmup=1)
            summary = json.loads((output / "profiling" / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["calls"], 2)
            self.assertEqual(summary["warmup"], 1)
            self.assertEqual(len(summary["sklearn"]), 4)
            self.assertEqual(len(summary["onnx"]), 8)
            self.assertTrue(summary["limitations"])
            for row in summary["sklearn"]:
                self.assertEqual(row["profile_calls"], 2)
                self.assertGreater(row["total_self_seconds"], 0)
                self.assertAlmostEqual(row["total_self_seconds"], sum(fn["self_seconds"] for fn in row["functions"]))
                filename = output / "profiling" / f"sklearn_batch_{row['batch_size']}.pstats"
                stats = pstats.Stats(str(filename))
                self.assertTrue(getattr(stats, "stats"))
                forest_calls = [fn for fn in row["functions"] if fn["function"] == "predict_proba" and fn["category"] == "forest_orchestration"]
                self.assertEqual(len(forest_calls), 1)
                self.assertEqual(forest_calls[0]["total_calls"], 2)
                self.assertTrue(any(fn["category"] == "tree_prediction" for fn in row["functions"]))
            for row in summary["onnx"]:
                self.assertEqual(row["profile_calls"], 2)
                self.assertGreaterEqual(row["model_run_mean_us"], 0)
                operators = {op["op_name"]: op for op in row["operators"]}
                self.assertEqual(operators["TreeEnsembleClassifier"]["event_count"], 2)
                if row["variant"] == "onnx_zipmap":
                    self.assertEqual(operators["ZipMap"]["event_count"], 2)
                else:
                    self.assertNotIn("ZipMap", operators)
            traces = list((output / "profiling").glob("onnx_*.json"))
            self.assertEqual(len(traces), 8)
            for filename in traces:
                events = json.loads(filename.read_text(encoding="utf-8"))
                self.assertEqual(sum(event.get("name") == "model_run" for event in events), 3)
            for name, content in original_files.items():
                self.assertEqual((output / name).read_bytes(), content)
            with self.assertRaisesRegex(ValueError, "Profiling directory is not empty"):
                run_profiles(output, calls=2, warmup=1)

    def test_trace_parse_failure_retains_raw_trace_and_diagnostics_without_summary(self) -> None:
        config = ExperimentConfig(n_estimators=1, batch_sizes=(1,))
        with tempfile.TemporaryDirectory() as temporary, threadpool_limits(limits=1):
            output = Path(temporary)
            data = prepare_data(config, output)
            model, _ = train_model(config, output)
            convert_models(model, data.x_test.shape[1], config, output)
            write_json(output / "config.json", config.to_dict())
            with patch("onnx_benchmark.profiling._summarize_onnx_trace", side_effect=ValueError("Malformed trace")):
                with self.assertRaisesRegex(ValueError, "Malformed trace"):
                    run_profiles(output, calls=1, warmup=0)
            directory = output / "profiling"
            self.assertFalse((directory / "summary.json").exists())
            diagnostics = json.loads((directory / "onnx_zipmap_batch_1_error.json").read_text(encoding="utf-8"))
            self.assertEqual(diagnostics["error"], "Malformed trace")
            self.assertTrue((directory / diagnostics["trace_file"]).is_file())
            self.assertTrue((directory / "sklearn_batch_1.pstats").is_file())

    def test_invalid_call_counts_are_rejected_before_reading_or_writing(self) -> None:
        for calls, warmup in ((0, 0), (1, -1), (True, 0), (1, False)):
            with self.subTest(calls=calls, warmup=warmup), tempfile.TemporaryDirectory() as temporary:
                output = Path(temporary)
                with self.assertRaises(ValueError):
                    run_profiles(output, calls=calls, warmup=warmup)
                self.assertEqual(list(output.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
