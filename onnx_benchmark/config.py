"""Central experiment settings shared by training, conversion and measurement."""

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 42
    test_size: float = 0.2
    n_estimators: int = 100
    batch_sizes: tuple[int, ...] = (1, 16, 64, 256)
    warmup_calls: int = 50
    rounds: int = 10
    calls_per_round: int = 100
    probability_tolerance: float = 1e-5
    target_opset: int = 17

    def __post_init__(self) -> None:
        if not 0 < self.test_size < 1:
            raise ValueError("test_size must be between 0 and 1")
        if self.n_estimators <= 0 or any(size <= 0 for size in self.batch_sizes):
            raise ValueError("tree and batch counts must be positive")
        if not self.batch_sizes or len(set(self.batch_sizes)) != len(self.batch_sizes):
            raise ValueError("batch_sizes must contain distinct positive sizes")
        if self.warmup_calls < 0 or self.rounds <= 0 or self.calls_per_round <= 0:
            raise ValueError("invalid benchmark repetition counts")
        if self.probability_tolerance <= 0:
            raise ValueError("probability_tolerance must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

