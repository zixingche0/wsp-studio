"""编辑计划 Schema 测试。"""

from __future__ import annotations

import pytest

from wsp.editor import plan_schema


def _valid_plan():
    return {
        "goal": "修复循环",
        "edits": [
            {
                "type": "direction_ablation",
                "layer": 15,
                "module": "mlp.down_proj",
                "axis": 1,
                "index": {"kind": "range", "start": 0, "stop": 64},
                "method": "project_out",
                "magnitude": 0.8,
                "norm_preserve": True,
                "expected_effect": "减少重复",
                "risk": "可能影响通用生成",
            }
        ],
        "eval_gate": {"primary": "math_short", "min_delta": 0.05},
    }


def test_valid_plan_passes():
    plan = plan_schema.validate_plan(_valid_plan())
    assert plan["goal"] == "修复循环"


def test_noop_plan_passes():
    plan = plan_schema.empty_plan("测试")
    plan_schema.validate_plan(plan)
    assert plan["edits"] == []


def test_missing_goal_rejected():
    plan = _valid_plan()
    del plan["goal"]
    with pytest.raises(plan_schema.PlanValidationError):
        plan_schema.validate_plan(plan)


def test_missing_eval_gate_rejected():
    plan = _valid_plan()
    del plan["eval_gate"]
    with pytest.raises(plan_schema.PlanValidationError):
        plan_schema.validate_plan(plan)


def test_direction_ablation_requires_axis():
    plan = _valid_plan()
    del plan["edits"][0]["axis"]
    with pytest.raises(plan_schema.PlanValidationError) as exc:
        plan_schema.validate_plan(plan)
    assert "axis" in str(exc.value)


def test_direction_ablation_requires_index():
    plan = _valid_plan()
    del plan["edits"][0]["index"]
    with pytest.raises(plan_schema.PlanValidationError):
        plan_schema.validate_plan(plan)


def test_edit_requires_expected_effect_and_risk():
    """说明书要求：每条编辑必须有预期效果与风险说明。"""
    for field in ("expected_effect", "risk"):
        plan = _valid_plan()
        del plan["edits"][0][field]
        with pytest.raises(plan_schema.PlanValidationError):
            plan_schema.validate_plan(plan)


def test_unknown_edit_type_rejected():
    plan = _valid_plan()
    plan["edits"][0]["type"] = "super_bake"
    with pytest.raises(plan_schema.PlanValidationError):
        plan_schema.validate_plan(plan)


def test_magnitude_out_of_range_rejected():
    plan = _valid_plan()
    plan["edits"][0]["magnitude"] = 9.0
    with pytest.raises(plan_schema.PlanValidationError):
        plan_schema.validate_plan(plan)


def test_extra_field_rejected():
    plan = _valid_plan()
    plan["edits"][0]["secret_sauce"] = True
    with pytest.raises(plan_schema.PlanValidationError):
        plan_schema.validate_plan(plan)


def test_module_scale_requires_factor():
    plan = _valid_plan()
    plan["edits"] = [{
        "type": "module_scale",
        "layer": 3,
        "module": "mlp.down_proj",
        "expected_effect": "x",
        "risk": "y",
    }]
    with pytest.raises(plan_schema.PlanValidationError):
        plan_schema.validate_plan(plan)


def test_too_many_edits_rejected():
    plan = _valid_plan()
    plan["edits"] = [_valid_plan()["edits"][0]] * 9
    with pytest.raises(plan_schema.PlanValidationError):
        plan_schema.validate_plan(plan)


def test_plan_summary_text():
    text = plan_schema.plan_summary(_valid_plan())
    assert "L15" in text
    assert "direction_ablation" not in text  # 摘要用人话


def test_empty_plan_summary():
    text = plan_schema.plan_summary(plan_schema.empty_plan("x"))
    assert "noop" in text
