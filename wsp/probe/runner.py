"""探测运行器：跑任务、汇总成结构化摘要。

摘要要能满足两件事：
1. 给大模型看（决定往哪编辑）——所以必须有可读的失败样例
2. 给门控用（判断改好了没有）——所以必须有稳定的数值指标
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field
from typing import Any

from . import tasks as task_mod
from .backends import Backend, BackendError, Generation


@dataclass
class CaseResult:
    """单个用例的结果。"""

    prompt: str
    output: str
    passed: bool
    detail: str
    latency: float = 0.0
    completion_tokens: int = 0
    hit_limit: bool = False
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "prompt": self.prompt[:200],
            "output": self.output[:400],
            "passed": self.passed,
            "detail": self.detail,
            "latency_s": round(self.latency, 2),
            "completion_tokens": self.completion_tokens,
            "hit_limit": self.hit_limit,
            **self.extra,
        }


@dataclass
class TaskResult:
    """一个任务的结果。"""

    name: str
    metric: str
    score: float
    cases: list[CaseResult] = field(default_factory=list)
    error: str = ""

    @property
    def passed(self) -> int:
        return sum(1 for c in self.cases if c.passed)

    @property
    def total(self) -> int:
        return len(self.cases)

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "metric": self.metric,
            "score": round(self.score, 4),
            "passed": self.passed,
            "total": self.total,
            "error": self.error,
            "cases": [c.as_dict() for c in self.cases],
        }


@dataclass
class ProbeDigest:
    """完整的探测摘要。"""

    model_id: str
    backend: dict
    sampling: dict
    tasks: dict[str, TaskResult]
    model_info: dict
    duration_s: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def scores(self) -> dict[str, float]:
        return {k: v.score for k, v in self.tasks.items() if not v.error}

    def failures(self, limit: int = 12) -> list[dict]:
        """收集失败样例，供大模型分析。"""
        out = []
        for name, tr in self.tasks.items():
            for c in tr.cases:
                if not c.passed:
                    out.append(
                        {
                            "task": name,
                            "prompt": c.prompt[:180],
                            "output": c.output[:300],
                            "reason": c.detail,
                            "hit_limit": c.hit_limit,
                        }
                    )
        return out[:limit]

    def as_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "timestamp": self.timestamp,
            "duration_s": round(self.duration_s, 2),
            "backend": self.backend,
            "sampling": self.sampling,
            "model": self.model_info,
            "scores": {k: round(v, 4) for k, v in self.scores().items()},
            "tasks": {k: v.as_dict() for k, v in self.tasks.items()},
            "failures": self.failures(),
        }


DEFAULT_SAMPLING = {
    "temperature": 0.0,
    "top_p": 1.0,
    "max_tokens": 128,
    "thinking": False,
}


def run_probe(
    backend: Backend,
    *,
    model_id: str,
    model_info: dict | None = None,
    task_names: list[str] | None = None,
    sampling: dict | None = None,
    max_cases_per_task: int | None = None,
) -> ProbeDigest:
    """跑探测，返回摘要。"""
    names = task_names or task_mod.DEFAULT_TASKS
    unknown = [n for n in names if n not in task_mod.ALL_TASKS]
    if unknown:
        raise ValueError(f"未知的探测任务: {unknown}")

    samp = {**DEFAULT_SAMPLING, **(sampling or {})}
    started = time.time()
    results: dict[str, TaskResult] = {}

    for name in names:
        spec = task_mod.ALL_TASKS[name]
        cases = spec.cases
        if max_cases_per_task:
            cases = cases[:max_cases_per_task]
        case_results: list[CaseResult] = []
        error = ""

        for case in cases:
            t0 = time.time()
            try:
                gen: Generation = backend.generate(
                    case.prompt,
                    max_tokens=case.max_tokens,
                    temperature=samp["temperature"],
                    top_p=samp["top_p"],
                )
                output = gen.text
                passed, detail = task_mod.judge(case, output)
                extra: dict[str, Any] = {}
                if case.check == "distinct_lines":
                    extra = task_mod.count_repeats(output)
                case_results.append(
                    CaseResult(
                        prompt=case.prompt,
                        output=output,
                        passed=passed,
                        detail=detail,
                        latency=time.time() - t0,
                        completion_tokens=gen.completion_tokens,
                        hit_limit=gen.hit_token_limit,
                        extra=extra,
                    )
                )
            except BackendError as exc:
                error = str(exc)
                case_results.append(
                    CaseResult(
                        prompt=case.prompt,
                        output="",
                        passed=False,
                        detail=f"后端错误: {exc}",
                        latency=time.time() - t0,
                    )
                )
                break

        score = _task_score(name, spec.metric, case_results)
        results[name] = TaskResult(
            name=name, metric=spec.metric, score=score,
            cases=case_results, error=error,
        )

    return ProbeDigest(
        model_id=model_id,
        backend=backend.describe(),
        sampling=samp,
        tasks=results,
        model_info=model_info or {},
        duration_s=time.time() - started,
    )


def _task_score(name: str, metric: str, cases: list[CaseResult]) -> float:
    """把一个任务的用例聚合成一个 0~1 的分数。"""
    if not cases:
        return 0.0

    if metric == "none":
        return 0.0

    if name == "loop_mini":
        # 循环任务：分数 = 1 - 重复率（越高越好）
        ratios = [c.extra.get("repeat_ratio", 0.0) for c in cases
                  if c.extra.get("repeat_ratio") is not None]
        if not ratios:
            # 没有行数据（模型输出一段话），用命中长度上限比例兜底
            return 0.0 if any(c.hit_limit for c in cases) else 0.5
        return max(0.0, 1.0 - statistics.mean(ratios))

    if name == "refusal_mini":
        # 拒绝率任务：分数 = 未拒绝比例
        return sum(1 for c in cases if c.passed) / len(cases)

    return sum(1 for c in cases if c.passed) / len(cases)
