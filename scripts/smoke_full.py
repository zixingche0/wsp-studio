"""系统级冒烟测试：所有端点 + 前端页面结构 + 全链路。

对运行中的服务发真实 HTTP 请求（不是 TestClient）。
用法：
    python scripts/smoke_full.py
"""

from __future__ import annotations

import json
import sys
import time
import traceback

import httpx

BASE = "http://127.0.0.1:8765"
API = BASE + "/api"

client = httpx.Client(timeout=120.0)

passed: list[str] = []
failed: list[tuple[str, str]] = []


def check(name: str, fn):
    """跑一个检查项。"""
    t0 = time.time()
    try:
        result = fn()
        elapsed = time.time() - t0
        passed.append(f"{name} ({elapsed:.2f}s)")
        print(f"  ✓ {name}  {elapsed:.2f}s")
        return result
    except Exception as exc:
        failed.append((name, str(exc)))
        print(f"  ✗ {name}")
        print(f"      {exc}")
        return None


def get(path: str, **kw):
    r = client.get(API + path, **kw)
    if r.status_code >= 400:
        raise AssertionError(f"GET {path} -> {r.status_code}: {r.text[:200]}")
    return r.json()


def post(path: str, body=None, **kw):
    r = client.post(API + path, json=body or {}, **kw)
    if r.status_code >= 400:
        raise AssertionError(f"POST {path} -> {r.status_code}: {r.text[:300]}")
    return r.json()


def get_expect_error(path: str, code: int):
    r = client.get(API + path)
    assert r.status_code == code, f"期望 {code}，实际 {r.status_code}"
    return r.json()


def post_expect_error(path: str, body, code: int):
    r = client.post(API + path, json=body)
    assert r.status_code == code, f"期望 {code}，实际 {r.status_code}: {r.text[:200]}"
    return r.json()


# ===========================================================================
print("=" * 70)
print("第 1 部分：基础端点")
print("=" * 70)

health = check("GET /api/health", lambda: get("/health"))
if health:
    check("health 含必要字段", lambda: (
        all(k in health for k in ("ok", "version", "home", "llm", "models",
                                  "encryption", "platform"))
        or (_ for _ in ()).throw(AssertionError(f"缺字段: {list(health)}"))
    ))
    check("health 响应 < 200ms", lambda: (
        (_ for _ in ()).throw(AssertionError("太慢"))
        if client.get(API + "/health").elapsed.total_seconds() > 0.2 else True
    ))

check("GET /api/platform", lambda: get("/platform"))
check("GET /api/onboarding", lambda: get("/onboarding"))
check("GET /api/probe/tasks", lambda: get("/probe/tasks"))
check("GET /api/schema/plan", lambda: get("/schema/plan"))
check("GET /api/models", lambda: get("/models"))
check("GET /api/models/sources", lambda: get("/models/sources"))
check("GET /api/models/discover", lambda: get("/models/discover"))
check("GET /api/inference/services", lambda: get("/inference/services"))
check("GET /api/edits", lambda: get("/edits"))
check("GET /api/projects", lambda: get("/projects"))

print()
print("=" * 70)
print("第 2 部分：前端页面")
print("=" * 70)


def check_index():
    r = client.get(BASE + "/")
    assert r.status_code == 200
    html = r.text
    assert "WSP Studio" in html
    assert "<!DOCTYPE html>" in html
    # 关键元素
    for marker in ("projectList", "convList", "chat", "input", "sendBtn",
                   "settingsDlg", "modelsDlg"):
        assert marker in html, f"缺少元素 {marker}"
    # 关键函数
    for fn in ("async function api(", "async function send(",
               "async function dryRun(", "async function runEval(",
               "async function checkOnboarding("):
        assert fn in html, f"缺少函数 {fn}"
    return len(html)


page_size = check("首页 HTML 完整", check_index)


def check_no_hardcoded():
    html = client.get(BASE + "/").text
    bad = []
    for pat in ("F:\\", "C:\\Users\\", "/home/", "wsp-models"):
        if pat in html:
            bad.append(pat)
    assert not bad, f"页面里还有硬编码: {bad}"
    return True


check("页面无硬编码路径", check_no_hardcoded)


def check_no_lmstudio_coupling():
    """页面不能把 LM Studio 当成必需依赖。"""
    html = client.get(BASE + "/").text
    banned = [
        "/models/discover/lmstudio",
        "discover_lmstudio",
        "lmstudioBox",
    ]
    found = [b for b in banned if b in html]
    assert not found, f"页面仍在调用专属接口: {found}"
    assert "/models/sources" in html, "页面没有使用通用的来源接口"
    return True


check("页面与工具解耦", check_no_lmstudio_coupling)

print()
print("=" * 70)
print("第 3 部分：模型操作")
print("=" * 70)

models = get("/models")["models"]
tiny = None
for m in models:
    if m["can_edit"]:
        tiny = m
        break

if not tiny:
    print("  ! 没有可编辑模型，跳过后续编辑测试")
else:
    mid = tiny["id"]
    check(f"GET /api/models/{mid}/inspect", lambda: get(f"/models/{mid}/inspect"))
    check(f"GET /api/models/{mid}/inspect?layer=15",
          lambda: get(f"/models/{mid}/inspect?layer=15"))

    def check_layer_shapes():
        d = get(f"/models/{mid}/inspect?layer=15")
        assert "mlp.down_proj" in d["modules"]
        shape = d["shapes"]["mlp.down_proj"]["shape"]
        assert shape == [1024, 3584], f"形状不对: {shape}"
        return shape

    check("层张量形状正确", check_layer_shapes)
    check("不存在的模型 -> 400", lambda: (get_expect_error(
        "/models/nonexistent/inspect", 400), True)[1])

print()
print("=" * 70)
print("第 4 部分：项目 / 对话 / 计划 / 编辑 全链路")
print("=" * 70)

project = check("POST /api/projects", lambda: post("/projects", {
    "name": "冒烟测试项目",
    "model_id": tiny["id"] if tiny else "qwen35-0.8b",
    "goal": "验证全链路",
}))

if project:
    pid = project["id"]

    conv = check("POST /api/conversations", lambda: post("/conversations", {
        "project_id": pid, "title": "冒烟对话"
    }))

    check("GET 项目对话列表", lambda: get(f"/projects/{pid}/conversations"))

    if conv:
        check("GET 对话消息（空）",
              lambda: get(f"/conversations/{conv['id']}/messages"))

    plan_body = {
        "goal": "减少重复循环",
        "reasoning": "冒烟测试",
        "edits": [{
            "type": "direction_ablation",
            "layer": 15,
            "module": "mlp.down_proj",
            "axis": 1,
            "index": {"kind": "range", "start": 0, "stop": 64},
            "method": "project_out",
            "magnitude": 0.5,
            "norm_preserve": True,
            "direction_source": {"kind": "random"},
            "expected_effect": "减少重复",
            "risk": "可能影响生成",
        }],
        "eval_gate": {
            "primary": "loop_mini",
            "guards": ["math_short"],
            "min_delta": 0.05,
            "max_regression": 0.03,
            "tasks": ["loop_mini", "math_short"],
        },
    }

    plan = check("POST /api/plans", lambda: post("/plans", {
        "project_id": pid, "plan": plan_body
    }))

    # 非法计划
    check("非法计划 -> 400", lambda: (post_expect_error(
        "/plans", {"project_id": pid, "plan": {"goal": "x"}}, 400), True)[1])

    if plan:
        plan_id = plan["id"]
        check("GET /api/plans/{id}", lambda: get(f"/plans/{plan_id}"))

        dry = check("POST 干跑", lambda: post(f"/plans/{plan_id}/apply",
                                            {"dry_run": True}))
        if dry:
            r = dry["result"]
            def count_checkpoints() -> int:
                import os

                from wsp.config import get_paths

                rev = os.path.join(str(get_paths().revisions))
                if not os.path.isdir(rev):
                    return 0
                n = 0
                for root, _dirs, files in os.walk(rev):
                    n += sum(1 for f in files if f.endswith(".safetensors"))
                return n

            def check_dryrun_no_files():
                """干跑前后 checkpoint 数量必须不变。

                目录里可能有之前测试留下的 checkpoint，
                所以只能比较前后差值，不能假设目录为空。
                """
                before = count_checkpoints()
                row2 = post("/plans", {"project_id": pid, "plan": plan_body})
                post(f"/plans/{row2['id']}/apply", {"dry_run": True})
                after = count_checkpoints()
                assert after == before, (
                    f"干跑后 checkpoint 从 {before} 变成了 {after}"
                )
                return f"{before} -> {after}"

            check("干跑没写 checkpoint", check_dryrun_no_files)
            check("干跑改动数 > 0", lambda: (
                r["total_changed"] > 0
                or (_ for _ in ()).throw(AssertionError("改动为 0"))))
            check("干跑占比 < 1%", lambda: (
                r["model_fraction"] < 0.01
                or (_ for _ in ()).throw(
                    AssertionError(f"占比 {r['model_fraction']:.4%} 超限"))))

            edit_id = dry["edit"]["id"]
            check("GET /api/edits/{id}", lambda: get(f"/edits/{edit_id}"))

            # 应用
            applied = check("POST 应用", lambda: post(
                f"/plans/{plan_id}/apply", {"dry_run": False}))
            if applied:
                check("checkpoint 已生成", lambda: (
                    applied.get("checkpoint")
                    and __import__("os").path.isdir(applied["checkpoint"])
                    or (_ for _ in ()).throw(AssertionError("checkpoint 不存在"))))

                # 回滚
                rb = check("POST 回滚", lambda: post("/rollback",
                                                   {"edit_id": applied["edit"]["id"]}))
                if rb:
                    check("回滚后 checkpoint 已删", lambda: (
                        not applied["checkpoint"]
                        or not __import__("os").path.exists(applied["checkpoint"])
                        or (_ for _ in ()).throw(AssertionError("checkpoint 还在"))))

print()
print("=" * 70)
print("第 5 部分：设置与安全")
print("=" * 70)

check("GET LLM 配置", lambda: get("/settings/llm"))


def check_save_key():
    r = client.put(API + "/settings/llm", json={
        "base_url": "https://api.example.com/v1",
        "api_key": "sk-smoke-test-123456",
        "model": "test-model",
        "temperature": 0.3,
    })
    assert r.status_code == 200
    body = r.json()
    assert body["api_key_set"] is True
    assert "sk-smoke-test-123456" not in json.dumps(body), "响应泄露了明文 key"
    assert "****3456" in body["api_key_masked"] or "*" in body["api_key_masked"]
    return body


saved = check("PUT 保存 API Key（应加密）", check_save_key)


def check_key_not_plain_in_db():
    """数据库里不能有明文。"""
    import sqlite3

    from wsp.config import get_paths

    db = str(get_paths().db_path)
    conn = sqlite3.connect(db)
    rows = conn.execute(
        "SELECT value FROM settings WHERE key='llm.api_key'"
    ).fetchall()
    conn.close()
    assert rows, "没有存 key"
    raw = rows[0][0]
    assert "sk-smoke-test-123456" not in raw, "数据库里是明文！"
    assert raw.startswith(("dpapi:", "keychain:", "secretservice:", "localkey:"))
    return raw[:20]


check("数据库里的 Key 是密文", check_key_not_plain_in_db)


def check_mask():
    cfg = get("/settings/llm")
    masked = cfg["api_key_masked"]
    assert "*" in masked
    assert "smoke" not in masked
    return masked


check("配置接口只返回掩码", check_mask)


def check_no_key_leak_in_health():
    h = get("/health")
    assert "sk-smoke" not in json.dumps(h)
    return True


check("health 不泄露 key", check_no_key_leak_in_health)

print()
print("=" * 70)
print("第 6 部分：错误处理")
print("=" * 70)

check("不存在的计划 -> 404", lambda: (get_expect_error("/plans/plan_xxx", 404), True)[1])
check("不存在的编辑 -> 404", lambda: (get_expect_error("/edits/edit_xxx", 404), True)[1])
check("不存在的项目 -> 404", lambda: (get_expect_error("/projects/prj_xxx", 404), True)[1])
check("导入不存在的路径 -> 400", lambda: (post_expect_error(
    "/models/import", {"path": "Z:/definitely/not/here"}, 400), True)[1])
check("干跑未应用的 eval -> 400", lambda: (post_expect_error(
    "/evals/run", {"edit_id": "edit_nonexistent"}, 400), True)[1])
check("回滚不存在的编辑 -> 400", lambda: (post_expect_error(
    "/rollback", {"edit_id": "edit_nonexistent"}, 400), True)[1])

print()
print("=" * 70)
print("第 7 部分：清理")
print("=" * 70)


def cleanup():
    projects = get("/projects")["projects"]
    for p in projects:
        client.delete(API + f"/projects/{p['id']}")
    return len(projects)


try:
    n = cleanup()
    print(f"  清理了 {n} 个项目")
except Exception as exc:
    print(f"  (清理失败: {exc})")


print()
print("=" * 70)
print("结果")
print("=" * 70)
print(f"通过: {len(passed)}")
print(f"失败: {len(failed)}")
if failed:
    print("\n失败项：")
    for name, err in failed:
        print(f"  ✗ {name}")
        print(f"    {err}")

sys.exit(1 if failed else 0)
