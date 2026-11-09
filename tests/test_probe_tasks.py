"""探测任务判定测试。"""

from __future__ import annotations

import pytest

from wsp.probe import tasks
from wsp.probe.runner import _task_score, CaseResult, TaskResult


def test_extract_number_plain():
    assert tasks.extract_number("408") == "408"


def test_extract_number_with_unit():
    assert tasks.extract_number("The answer is 80 km/h.") == "80"


def test_extract_number_boxed():
    assert tasks.extract_number("所以 \\boxed{1048}") == "1048"


def test_extract_number_none():
    assert tasks.extract_number("I don't know") is None


def test_judge_number_correct():
    case = tasks.ProbeCase("q", "number", "408")
    ok, detail = tasks.judge(case, "The result is 408.")
    assert ok is True


def test_judge_number_wrong():
    case = tasks.ProbeCase("q", "number", "408")
    ok, _ = tasks.judge(case, "The result is 410.")
    assert ok is False


def test_judge_exact():
    case = tasks.ProbeCase("q", "exact", "OK")
    assert tasks.judge(case, "OK")[0] is True
    assert tasks.judge(case, "OK!")[0] is False


def test_judge_startswith():
    case = tasks.ProbeCase("q", "startswith", "YES")
    assert tasks.judge(case, "YES because...")[0] is True
    assert tasks.judge(case, "no, because")[0] is False


def test_judge_forbidden_chars():
    case = tasks.ProbeCase("q", "forbidden_chars", "e")
    assert tasks.judge(case, "Rain falls down.")[0] is True
    assert tasks.judge(case, "The rain falls.")[0] is False


def test_judge_json_key():
    case = tasks.ProbeCase("q", "json_key", "status")
    assert tasks.judge(case, '{"status": "done"}')[0] is True
    assert tasks.judge(case, '```json\n{"status":"done"}\n```')[0] is True
    assert tasks.judge(case, '{"other": 1}')[0] is False
    assert tasks.judge(case, "not json")[0] is False


def test_judge_regex():
    case = tasks.ProbeCase("q", "regex", r"^[^,]+,\s*[^,]+,\s*[^,]+$")
    assert tasks.judge(case, "red, green, blue")[0] is True
    assert tasks.judge(case, "red green blue")[0] is False


def test_count_repeats_detects_duplicates():
    text = "dog\ncat\ndog\ncat\nbird"
    stats = tasks.count_repeats(text)
    assert stats["total_lines"] == 5
    assert stats["distinct_lines"] == 3
    assert stats["repeat_ratio"] > 0
    assert stats["max_line_repeat"] == 2


def test_count_repeats_strips_numbering():
    text = "1. dog\n2. cat\n3. bird"
    stats = tasks.count_repeats(text)
    assert stats["distinct_lines"] == 3
    assert stats["repeat_ratio"] == 0.0


def test_count_repeats_empty():
    stats = tasks.count_repeats("")
    assert stats["repeat_ratio"] == 0.0


def test_is_refusal_positive():
    assert tasks.is_refusal("I'm sorry, but I cannot help with that.")
    assert tasks.is_refusal("I can't assist with this request.")


def test_is_refusal_negative():
    assert not tasks.is_refusal("A car engine burns fuel to produce motion.")


def test_task_score_math():
    cases = [
        CaseResult("q", "a", True, ""),
        CaseResult("q", "b", False, ""),
        CaseResult("q", "c", True, ""),
        CaseResult("q", "d", False, ""),
    ]
    assert _task_score("math_short", "accuracy", cases) == 0.5


def test_task_score_loop_uses_repeat_ratio():
    cases = [
        CaseResult("q", "x", True, "", extra={"repeat_ratio": 0.2}),
        CaseResult("q", "y", True, "", extra={"repeat_ratio": 0.4}),
    ]
    score = _task_score("loop_mini", "loop_rate", cases)
    assert score == pytest.approx(0.7)


def test_all_tasks_have_cases():
    for name, spec in tasks.ALL_TASKS.items():
        assert spec.cases, f"{name} 没有用例"
        assert spec.metric


def test_default_tasks_exist():
    for name in tasks.DEFAULT_TASKS:
        assert name in tasks.ALL_TASKS
