"""Warm up, rotate execution order, and persist each measured API call."""

import csv
from pathlib import Path
from time import perf_counter_ns
from typing import Any, Callable

import numpy as np

from .config import ExperimentConfig
from .data_model import FloatArray
from .runtime import InferenceAdapter


def time_calls(call: Callable[[], Any], count: int) -> list[int]:
    """Discard each result after the stop time; include API output construction."""
    timings: list[int] = []
    for _ in range(count):
        start = perf_counter_ns()
        result = call()
        elapsed = perf_counter_ns() - start
        del result
        if elapsed <= 0:
            raise ValueError("Clock produced a non-positive measurement")
        timings.append(elapsed)
    return timings


def measure(
    adapters: list[InferenceAdapter], batches: dict[int, FloatArray],
    config: ExperimentConfig, output_dir: Path,
) -> None:
    timings: dict[tuple[str, int], list[int]] = {}
    with (output_dir / "timings.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "variant", "batch_size", "round", "order_in_round", "call", "duration_ns",
        ])
        writer.writeheader()
        for size, x in batches.items():
            calls = {adapter.variant: adapter.bind_probabilities(x) for adapter in adapters}
            for adapter in adapters:
                for _ in range(config.warmup_calls):
                    calls[adapter.variant]()
                timings[(adapter.variant, size)] = []
            for round_index in range(config.rounds):
                offset = round_index % len(adapters)
                ordered = adapters[offset:] + adapters[:offset]
                for position, adapter in enumerate(ordered):
                    # Closure creation and CSV output are outside measured calls.
                    durations = time_calls(calls[adapter.variant], config.calls_per_round)
                    timings[(adapter.variant, size)].extend(durations)
                    writer.writerows({
                        "variant": adapter.variant,
                        "batch_size": size,
                        "round": round_index + 1,
                        "order_in_round": position + 1,
                        "call": call_index + 1,
                        "duration_ns": duration,
                    } for call_index, duration in enumerate(durations))
                stream.flush()
            print(f"Measured batch {size}: {config.rounds * config.calls_per_round} calls per variant", flush=True)
    write_summary(timings, output_dir)


def write_summary(timings: dict[tuple[str, int], list[int]], output_dir: Path) -> None:
    medians = {key: float(np.median(values)) for key, values in timings.items()}
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "variant", "batch_size", "calls", "median_ms", "p25_ms", "p75_ms", "p95_ms",
            "throughput_samples_per_second", "speedup_vs_sklearn",
        ])
        writer.writeheader()
        for (variant, size), values in timings.items():
            p25, p75, p95 = np.percentile(values, [25, 75, 95])
            median = medians[(variant, size)]
            writer.writerow({
                "variant": variant,
                "batch_size": size,
                "calls": len(values),
                "median_ms": median / 1e6,
                "p25_ms": float(p25) / 1e6,
                "p75_ms": float(p75) / 1e6,
                "p95_ms": float(p95) / 1e6,
                "throughput_samples_per_second": size * 1e9 / median,
                "speedup_vs_sklearn": medians[("sklearn", size)] / median,
            })

