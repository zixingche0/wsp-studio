"""一键跑全部测试。

    python scripts/run_all_tests.py

分六层：
  1. 单元测试（pytest，不需要服务）
  2. 系统冒烟（需要服务）
  3. 前端渲染（需要服务 + Edge）
  4. Agent 流程（需要服务）
  5. 并发压力（需要服务）
  6. 安全完整性（需要服务 + 真实模型）

服务没起来时会自动提示怎么启动。
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE = "http://127.0.0.1:8765"


def service_up() -> bool:
    try:
        import httpx

        r = httpx.get(BASE + "/api/health", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def run(name: str, cmd: list[str], cwd: Path = ROOT, timeout: int = 1800) -> tuple[bool, str]:
    print(f"\n{'=' * 70}")
    print(f"{name}")
    print("=" * 70)
    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd, cwd=str(cwd), capture_output=True, text=True,
            timeout=timeout, encoding="utf-8", errors="replace",
        )
    except subprocess.TimeoutExpired:
        print("  超时")
        return False, "timeout"

    out = (proc.stdout or "") + (proc.stderr or "")
    # 只打印关键行，避免刷屏
    for line in out.splitlines():
        low = line.lower()
        if any(k in low for k in ("passed", "failed", "通过", "失败", "✗", "error")):
            print("  " + line.strip())
    elapsed = time.time() - t0
    ok = proc.returncode == 0
    print(f"  -> {'PASS' if ok else 'FAIL'} ({elapsed:.1f}s)")
    return ok, out


def main() -> int:
    results: list[tuple[str, bool]] = []

    print("WSP Studio — 全量测试")
    print(f"项目: {ROOT}")
    print(f"服务: {'在跑' if service_up() else '未启动'}")

    # 1. 单元测试
    ok, _ = run("1/6 单元测试", [
        sys.executable, "-m", "pytest", "tests/", "-q"
    ])
    results.append(("单元测试", ok))

    needs_service = [
        ("2/6 系统冒烟", ["scripts/smoke_full.py"]),
        ("3/6 前端渲染", ["scripts/test_frontend.py"]),
        ("4/6 Agent 流程", ["scripts/test_agent_flow.py"]),
        ("5/6 并发压力", ["scripts/test_concurrency.py"]),
        ("6/6 安全完整性", ["scripts/test_security.py"]),
    ]

    if not service_up():
        print("\n服务未启动，跳过 2~6。")
        print("启动方式：")
        print("  wsp serve")
        print("  或 python -m uvicorn wsp.api.app:app --port 8765")
        for name, _ in needs_service:
            results.append((name.split(" ", 1)[1], None))  # type: ignore
    else:
        for name, script in needs_service:
            ok, _ = run(name, [sys.executable, str(ROOT / script[0])])
            results.append((name.split(" ", 1)[1], ok))

    print("\n" + "=" * 70)
    print("汇总")
    print("=" * 70)
    total_pass = 0
    total_fail = 0
    for name, ok in results:
        if ok is None:
            mark = "SKIP"
        elif ok:
            mark = "PASS"
            total_pass += 1
        else:
            mark = "FAIL"
            total_fail += 1
        print(f"  [{mark}] {name}")

    print(f"\n{total_pass} 层通过，{total_fail} 层失败")

    if total_fail:
        print("\n重新跑失败项可以看完整输出。")
    return 1 if total_fail else 0


if __name__ == "__main__":
    sys.exit(main())
