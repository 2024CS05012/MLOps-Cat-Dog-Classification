import time
from dataclasses import dataclass


@dataclass
class AppMetrics:
    request_count: int = 0
    total_latency_seconds: float = 0.0

    def observe(self, latency_seconds: float) -> None:
        self.request_count += 1
        self.total_latency_seconds += latency_seconds

    @property
    def average_latency_seconds(self) -> float:
        if self.request_count == 0:
            return 0.0
        return self.total_latency_seconds / self.request_count


metrics = AppMetrics()


def now() -> float:
    return time.perf_counter()
