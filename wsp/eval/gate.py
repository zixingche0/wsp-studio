"""评估门控。

判定规则（全部基于"编辑前后跑同一组任务"）：

* 主指标提升 >= ``min_delta``
* 每个 guard 指标退化 <= ``max_regression``
* 可选：困惑度恶化不超过 ``ppl_worsen_ratio``

没通过就回滚。门控的价值在于"大多数时候它说不"——
社区实测（wsp-kit 7 次运行 2 次通过，OpenWSP 复跑只拿到 12.9%）
说明这是常态，不是失败。
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GateDecision:
    """门控结论。"""

    passed: bool
    primary: str
    primary_before: float
    primary_after: float
    min_delta: float
    reasons: list[str] = field(default_factory=list)
    guards: dict[str, dict] = field(default_factory=dict)
    ppl: dict | None = None

    def as_dict(self) -> dict:
        return {
            "passed": self.passed,
            "primary": self.primary,
            "primary_before": round(self.primary_before, 4),
            "primary_after": round(self.primary_after, 4),
            "delta": round(self.primary_after - self.primary_before, 4),
            "min_delta": self.min_delta,
            "reasons": self.reasons,
            "guards": self.guards,
            "ppl": self.ppl,
        }


def evaluate_gate(
    before: dict[str, float],
    after: dict[str, float],
    gate: dict,
) -> GateDecision:
    """比较编辑前后的指标。

    ``before`` / ``after`` 形如 ``{"math_short": 0.5, "ifeval_mini": 0.8}``。
    """
    primary = gate.get("primary") or next(iter(before), "")
    min_delta = float(gate.get("min_delta", 0.05))
    delta_unit = gate.get("delta_unit", "absolute")
    max_regression = float(gate.get("max_regression", 0.03))
    regression_unit = gate.get("regression_unit", "absolute")
    guards = gate.get("guards") or []

    if primary not in before or primary not in after:
        return GateDecision(
            passed=False,
            primary=primary,
            primary_before=before.get(primary, 0.0),
            primary_after=after.get(primary, 0.0),
            min_delta=min_delta,
            reasons=[f"主指标 {primary!r} 在编辑前后缺失，无法判定"],
        )

    b = float(before[primary])
    a = float(after[primary])

    if delta_unit == "relative":
        delta = (a - b) / b if b else (float("inf") if a > 0 else 0.0)
        threshold = min_delta
    else:
        delta = a - b
        threshold = min_delta

    reasons: list[str] = []
    if delta < threshold:
        reasons.append(
            f"主指标 {primary} 提升不足：{b:.4f} → {a:.4f} "
            f"(Δ={delta:+.4f}，需要 ≥ {threshold:+.4f})"
        )

    guard_report: dict[str, dict] = {}
    for g in guards:
        if g not in before or g not in after:
            guard_report[g] = {"status": "missing"}
            continue
        gb, ga = float(before[g]), float(after[g])
        if regression_unit == "relative":
            reg = (gb - ga) / gb if gb else 0.0
        else:
            reg = gb - ga
        ok = reg <= max_regression
        guard_report[g] = {
            "before": round(gb, 4),
            "after": round(ga, 4),
            "regression": round(reg, 4),
            "limit": max_regression,
            "ok": ok,
        }
        if not ok:
            reasons.append(
                f"guard {g} 退化过大：{gb:.4f} → {ga:.4f} "
                f"(退化 {reg:.4f} > {max_regression:.4f})"
            )

    return GateDecision(
        passed=len(reasons) == 0,
        primary=primary,
        primary_before=b,
        primary_after=a,
        min_delta=min_delta,
        reasons=reasons,
        guards=guard_report,
    )


def evaluate_gate_with_ppl(
    before: dict[str, float],
    after: dict[str, float],
    gate: dict,
    *,
    ppl_before: float | None = None,
    ppl_after: float | None = None,
) -> GateDecision:
    """在基础门控之上，附加困惑度检查。"""
    decision = evaluate_gate(before, after, gate)
    limit = gate.get("ppl_worsen_ratio")
    if limit is None or ppl_before is None or ppl_after is None:
        return decision

    ratio = ppl_after / ppl_before if ppl_before else float("inf")
    ok = ratio <= float(limit)
    decision.ppl = {
        "before": round(ppl_before, 4),
        "after": round(ppl_after, 4),
        "worsen_ratio": round(ratio, 4),
        "limit": float(limit),
        "ok": ok,
    }
    if not ok:
        decision.reasons.append(
            f"困惑度恶化超限：{ppl_before:.3f} → {ppl_after:.3f} "
            f"(×{ratio:.3f} > {limit})"
        )
        decision.passed = False
    return decision
