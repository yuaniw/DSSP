import json
import statistics
import time
from pathlib import Path

import torch


def cuda_available():
    return torch.cuda.is_available()


def synchronize_if_needed():
    if cuda_available():
        torch.cuda.synchronize()


def reset_peak_memory_stats():
    if cuda_available():
        torch.cuda.reset_peak_memory_stats()


def peak_memory_mb():
    if not cuda_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024 ** 2)


class RunningStats:
    def __init__(self, name):
        self.name = name
        self.values = []

    def add(self, value):
        self.values.append(float(value))

    def summary(self):
        if not self.values:
            return {
                "name": self.name,
                "count": 0,
                "mean": 0.0,
                "median": 0.0,
                "min": 0.0,
                "max": 0.0,
                "p95": 0.0,
            }
        values = sorted(self.values)
        p95_index = min(len(values) - 1, max(0, int(0.95 * len(values)) - 1))
        return {
            "name": self.name,
            "count": len(values),
            "mean": statistics.fmean(values),
            "median": statistics.median(values),
            "min": values[0],
            "max": values[-1],
            "p95": values[p95_index],
        }


class ModelRuntimeTracker:
    def __init__(self):
        self.inference_time_ms = RunningStats("inference_time_ms")
        self.peak_memory_mb = RunningStats("peak_memory_mb")

    def record_inference(self, fn):
        reset_peak_memory_stats()
        synchronize_if_needed()
        start = time.perf_counter()
        result = fn()
        synchronize_if_needed()
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        self.inference_time_ms.add(elapsed_ms)
        self.peak_memory_mb.add(peak_memory_mb())
        return result

    def summary(self):
        return {
            "inference_time_ms": self.inference_time_ms.summary(),
            "peak_memory_mb": self.peak_memory_mb.summary(),
        }


def write_metrics(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
