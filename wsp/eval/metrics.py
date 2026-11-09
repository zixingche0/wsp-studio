"""评估指标辅助。"""

from __future__ import annotations

import math


def perplexity_from_logprobs(logprobs: list[float]) -> float:
    """从逐 token logprob 计算困惑度。"""
    if not logprobs:
        return float("inf")
    avg = sum(logprobs) / len(logprobs)
    return math.exp(-avg)


def normalize_score(value: float, low: float = 0.0, high: float = 1.0) -> float:
    """把分数归一化到 0~1。"""
    if high == low:
        return 0.0
    return max(0.0, min(1.0, (value - low) / (high - low)))


def delta_report(before: dict[str, float], after: dict[str, float]) -> dict:
    """给出每个指标的绝对/相对变化。"""
    out = {}
    for key in sorted(set(before) | set(after)):
        b = before.get(key)
        a = after.get(key)
        if b is None or a is None:
            out[key] = {"before": b, "after": a, "delta": None}
            continue
        out[key] = {
            "before": round(b, 4),
            "after": round(a, 4),
            "delta": round(a - b, 4),
            "relative": round((a - b) / b, 4) if b else None,
        }
    return out
