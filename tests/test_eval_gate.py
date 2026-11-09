"""评估门控测试。"""

from __future__ import annotations

import pytest

from wsp.eval import gate


def _gate(**kw):
    base = {
        "primary": "math_short",
        "min_delta": 0.05,
        "max_regression": 0.03,
        "guards": ["ifeval_mini"],
        "delta_unit": "absolute",
        "regression_unit": "absolute",
    }
    base.update(kw)
    return base


def test_pass_when_primary_improves():
    d = gate.evaluate_gate(
        {"math_short": 0.50, "ifeval_mini": 0.80},
        {"math_short": 0.60, "ifeval_mini": 0.79},
        _gate(),
    )
    assert d.passed is True
    assert d.reasons == []


def test_fail_when_improvement_too_small():
    d = gate.evaluate_gate(
        {"math_short": 0.50, "ifeval_mini": 0.80},
        {"math_short": 0.52, "ifeval_mini": 0.80},
        _gate(),
    )
    assert d.passed is False
    assert "提升不足" in d.reasons[0]


def test_fail_when_primary_drops():
    d = gate.evaluate_gate(
        {"math_short": 0.60, "ifeval_mini": 0.80},
        {"math_short": 0.40, "ifeval_mini": 0.80},
        _gate(),
    )
    assert d.passed is False


def test_fail_when_guard_regresses_too_much():
    d = gate.evaluate_gate(
        {"math_short": 0.50, "ifeval_mini": 0.80},
        {"math_short": 0.70, "ifeval_mini": 0.70},
        _gate(),
    )
    assert d.passed is False
    assert any("ifeval_mini" in r for r in d.reasons)


def test_pass_when_guard_regresses_within_limit():
    d = gate.evaluate_gate(
        {"math_short": 0.50, "ifeval_mini": 0.80},
        {"math_short": 0.70, "ifeval_mini": 0.78},
        _gate(),
    )
    assert d.passed is True


def test_relative_delta_unit():
    d = gate.evaluate_gate(
        {"math_short": 0.10, "ifeval_mini": 0.8},
        {"math_short": 0.11, "ifeval_mini": 0.8},
        _gate(delta_unit="relative", min_delta=0.05),
    )
    assert d.passed is True  # 10% 相对提升


def test_missing_primary_fails():
    d = gate.evaluate_gate({"ifeval_mini": 0.8}, {"ifeval_mini": 0.8}, _gate())
    assert d.passed is False
    assert "缺失" in d.reasons[0]


def test_noop_comparison_fails_to_improve():
    """noop 对照：前后一致，主指标没有提升，必须拒绝。"""
    scores = {"math_short": 0.5, "ifeval_mini": 0.8}
    d = gate.evaluate_gate(scores, dict(scores), _gate())
    assert d.passed is False


def test_ppl_guard_blocks():
    d = gate.evaluate_gate_with_ppl(
        {"math_short": 0.5, "ifeval_mini": 0.8},
        {"math_short": 0.7, "ifeval_mini": 0.8},
        _gate(ppl_worsen_ratio=1.10),
        ppl_before=10.0,
        ppl_after=13.0,
    )
    assert d.passed is False
    assert d.ppl["ok"] is False


def test_ppl_guard_allows_small_change():
    d = gate.evaluate_gate_with_ppl(
        {"math_short": 0.5, "ifeval_mini": 0.8},
        {"math_short": 0.7, "ifeval_mini": 0.8},
        _gate(ppl_worsen_ratio=1.10),
        ppl_before=10.0,
        ppl_after=10.5,
    )
    assert d.passed is True
    assert d.ppl["ok"] is True


def test_gate_decision_serializable():
    d = gate.evaluate_gate(
        {"math_short": 0.5, "ifeval_mini": 0.8},
        {"math_short": 0.6, "ifeval_mini": 0.79},
        _gate(),
    )
    payload = d.as_dict()
    assert payload["passed"] is True
    assert payload["delta"] == pytest.approx(0.1, abs=1e-6)
    assert "guards" in payload


def test_metrics_delta_report():
    from wsp.eval import metrics

    rep = metrics.delta_report(
        {"a": 0.5, "b": 1.0}, {"a": 0.6, "b": 0.9}
    )
    assert rep["a"]["delta"] == pytest.approx(0.1)
    assert rep["b"]["delta"] == pytest.approx(-0.1)
