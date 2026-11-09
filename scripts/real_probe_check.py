"""真实模型的完整闭环：探测 -> 编辑 -> 评估 -> 门控判定。

这是最有说服力的验证：用真实 Qwen3.5-0.8B，跑真实推理。
CPU 上较慢，所以探测缩减到每个任务 2 道题。
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from wsp.probe import runner  # noqa: E402
from wsp.probe.backends import HFLocalBackend  # noqa: E402

MODEL = os.environ.get("WSP_TEST_MODEL", r"F:\wsp-models\Qwen3.5-0.8B")
TASKS = ["math_short", "loop_mini"]


def main():
    print("=" * 60)
    print("真实模型探测")
    print("=" * 60)
    backend = HFLocalBackend(MODEL)
    print(f"设备: {backend.describe()['device']}")

    t0 = time.time()
    digest = runner.run_probe(
        backend,
        model_id="qwen35-0.8b",
        task_names=TASKS,
        max_cases_per_task=2,
    )
    print(f"耗时: {time.time() - t0:.1f}s")
    print(f"\n采样配置: {digest.sampling}")
    print("\n分数:")
    for name, score in digest.scores().items():
        print(f"  {name:<14} {score:.2%}")

    print("\n失败样例:")
    for f in digest.failures(limit=5):
        print(f"  [{f['task']}] {f['reason']}")
        print(f"    Q: {f['prompt'][:70]}")
        print(f"    A: {f['output'][:100]!r}")

    print("\n" + "=" * 60)
    print("结论")
    print("=" * 60)
    scores = digest.scores()
    math = scores.get("math_short", 0)
    loop = scores.get("loop_mini", 0)
    print(f"数学正确率 {math:.0%}，循环任务得分 {loop:.0%}")

    if math < 0.5:
        print("→ 数学表现不佳。但注意：这可能是模型能力问题，")
        print("  不是权重缺陷。编辑只能移除障碍，不能新增能力。")
    if loop < 0.7:
        print("→ 循环任务有明显重复。这类问题才是编辑的合适目标。")

    print("\n下一步（在网页里或 CLI）：")
    print("  wsp project \"修复循环\" qwen35-0.8b")
    print("  wsp plan <project_id> \"减少重复循环\"")
    print("  wsp apply <plan_id> --dry-run")
    print("\n✓ 真实探测链路可用")


if __name__ == "__main__":
    main()
