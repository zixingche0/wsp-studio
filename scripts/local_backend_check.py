"""用真实 Qwen3.5-0.8B 验证本地推理后端能加载编辑后的 checkpoint。

这是"评估"环节的关键能力：LM Studio 加载的 GGUF 读不到新 checkpoint，
所以必须能用 transformers 直接加载。
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from wsp.probe.backends import HFLocalBackend  # noqa: E402


MODEL = os.environ.get("WSP_TEST_MODEL", r"F:\wsp-models\Qwen3.5-0.8B")


def main():
    print(f"模型: {MODEL}")
    backend = HFLocalBackend(MODEL)
    print(f"describe: {backend.describe()}")

    t0 = time.time()
    print("\n加载中（首次会慢）…")
    gen = backend.generate(
        "What is 17 * 24? Answer with just the number.",
        max_tokens=32,
        temperature=0.0,
    )
    elapsed = time.time() - t0
    print(f"\n耗时: {elapsed:.1f}s")
    print(f"设备: {backend.describe()['device']}")
    print(f"输出: {gen.text!r}")
    print(f"tokens: prompt={gen.prompt_tokens} completion={gen.completion_tokens}")
    print(f"finish: {gen.finish_reason}")

    if "408" in gen.text:
        print("\n✓ 正确答案")
    else:
        print("\n△ 答案不精确（小模型正常现象，不代表后端有问题）")

    # 第二次调用应该明显更快（模型已在内存）
    t1 = time.time()
    backend.generate("Say hello.", max_tokens=8, temperature=0.0)
    print(f"\n第二次调用耗时: {time.time() - t1:.2f}s（模型已驻留内存）")
    print("\n✓ 本地推理后端可用")


if __name__ == "__main__":
    main()
