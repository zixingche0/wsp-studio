"""并发与压力测试。

验证：服务在多请求下不崩溃、不串数据、响应时间可控。
"""

from __future__ import annotations

import concurrent.futures
import json
import statistics
import sys
import time

import httpx

API = "http://127.0.0.1:8765/api"

passed, failed = [], []


def check(name, fn):
    try:
        v = fn()
        passed.append(name)
        print(f"  ✓ {name}" + (f" -> {v}" if v not in (True, None) else ""))
        return v
    except Exception as exc:
        failed.append((name, str(exc)))
        print(f"  ✗ {name}\n      {exc}")
        return None


def timed(fn, n: int = 10) -> dict:
    """跑 n 次，返回统计。"""
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return {
        "min": round(min(times), 4),
        "median": round(statistics.median(times), 4),
        "max": round(max(times), 4),
    }


def main() -> int:
    print("=" * 70)
    print("并发与压力")
    print("=" * 70)

    client = httpx.Client(timeout=60.0)

    print("\n第 1 部分：单请求延迟")
    for path in ("/health", "/models", "/models/sources", "/edits", "/projects"):
        check(f"GET {path}", lambda p=path: timed(
            lambda: client.get(API + p).raise_for_status()))

    print("\n第 2 部分：并发读（20 路）")
    def one(i: int) -> int:
        r = client.get(API + "/models")
        return len(r.json()["models"])

    t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as pool:
        results = list(pool.map(one, range(40)))
    elapsed = time.perf_counter() - t0
    check("40 次并发 GET /models 全部成功",
          lambda: all(r >= 1 for r in results) or (_ for _ in ()).throw(
              AssertionError(f"有失败: {results}")))
    check("并发结果一致", lambda: len(set(results)) == 1 or
          (_ for _ in ()).throw(AssertionError(f"数据不一致: {set(results)}")))
    print(f"      40 请求 / {elapsed:.2f}s = {40/elapsed:.0f} req/s")

    print("\n第 3 部分：并发写（隔离性）")
    def make_project(i: int) -> str:
        r = client.post(API + "/projects", json={
            "name": f"并发项目-{i}", "model_id": "qwen35-0.8b", "goal": "x"
        })
        r.raise_for_status()
        return r.json()["id"]

    ids = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as pool:
        ids = list(pool.map(make_project, range(10)))

    check("10 个并发创建全部成功", lambda: len(ids) == 10 or
          (_ for _ in ()).throw(AssertionError(f"只成功 {len(ids)}")))
    check("id 互不重复", lambda: len(set(ids)) == 10 or
          (_ for _ in ()).throw(AssertionError("有重复 id")))

    lst = client.get(API + "/projects").json()["projects"]
    names = {p["name"] for p in lst if p["name"].startswith("并发项目-")}
    check("10 个项目都能查到", lambda: len(names) == 10 or
          (_ for _ in ()).throw(AssertionError(f"只查到 {len(names)}")))

    print("\n第 4 部分：并发干跑（重计算路径）")
    model = next(m for m in client.get(API + "/models").json()["models"]
                 if m["can_edit"])
    pid = client.post(API + "/projects", json={
        "name": "干跑并发", "model_id": model["id"]
    }).json()["id"]

    plan = {
        "goal": "并发干跑",
        "edits": [{
            "type": "direction_ablation", "layer": 15,
            "module": "mlp.down_proj", "axis": 1,
            "index": {"kind": "range", "start": 0, "stop": 32},
            "method": "project_out", "magnitude": 0.5,
            "direction_source": {"kind": "random"},
            "expected_effect": "x", "risk": "y",
        }],
        "eval_gate": {"primary": "loop_mini", "min_delta": 0.05},
    }
    plan_ids = []
    for _ in range(6):
        r = client.post(API + "/plans", json={"project_id": pid, "plan": plan})
        plan_ids.append(r.json()["id"])

    def do_dry(pid_: str) -> dict:
        r = client.post(API + f"/plans/{pid_}/apply", json={"dry_run": True})
        r.raise_for_status()
        return r.json()

    t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        drys = list(pool.map(do_dry, plan_ids))
    elapsed = time.perf_counter() - t0
    check("6 个并发干跑全部成功",
          lambda: all(d["result"]["ok"] for d in drys) or
          (_ for _ in ()).throw(AssertionError("有失败")))
    counts = {d["result"]["total_changed"] for d in drys}
    check("并发干跑结果一致", lambda: len(counts) == 1 or
          (_ for _ in ()).throw(AssertionError(f"结果不一致: {counts}")))
    print(f"      6 干跑 / {elapsed:.2f}s，每次改动 {list(counts)[0]:,}")

    print("\n第 5 部分：非法请求不拖垮服务")
    def bad_request(i: int) -> int:
        r = client.post(API + "/plans", json={"garbage": i})
        return r.status_code

    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as pool:
        codes = list(pool.map(bad_request, range(40)))
    check("40 个非法请求都返回 4xx",
          lambda: all(400 <= c < 500 for c in codes) or
          (_ for _ in ()).throw(AssertionError(f"异常码: {set(codes)}")))

    check("服务仍然存活", lambda: (
        client.get(API + "/health").status_code == 200 or
        (_ for _ in ()).throw(AssertionError("服务挂了"))))

    # 清理
    for p in client.get(API + "/projects").json()["projects"]:
        client.delete(API + f"/projects/{p['id']}")

    print()
    print("=" * 70)
    print(f"通过 {len(passed)}，失败 {len(failed)}")
    print("=" * 70)
    if failed:
        for n, e in failed:
            print(f"  ✗ {n}: {e}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
