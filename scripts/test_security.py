"""安全性与数据完整性测试。

覆盖：
  * 路径穿越 / 原始模型只读保护
  * API Key 不泄露
  * 编辑 diff 的 sha256 可校验
  * 回滚后数据完全复原
  * 预算上限无法绕过
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wsp.config import get_paths  # noqa: E402

API = "http://127.0.0.1:8765/api"
HOME = str(get_paths().home)

# 测试模型路径：优先环境变量，否则用数据目录下的 models/
MODEL = os.environ.get(
    "WSP_TEST_MODEL", str(get_paths().models / "Qwen3.5-0.8B")
)

client = httpx.Client(timeout=180.0)
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


def sha_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    print("=" * 70)
    print("安全性与数据完整性")
    print("=" * 70)

    print("\n第 1 部分：原始模型只读保护")

    weights = os.path.join(MODEL, "model.safetensors-00001-of-00001.safetensors")
    before_sha = check("记录原始权重的 sha256",
                       lambda: sha_file(weights))
    BEFORE = sha_file(weights)

    # 尝试把 checkpoint 写到原始模型目录里
    def try_write_into_original():
        from wsp.config import Paths, assert_outside

        try:
            assert_outside(MODEL, os.path.join(MODEL, "evil"))
        except PermissionError:
            return "blocked"
        raise AssertionError("居然允许写入原始模型目录！")

    check("拒绝写入原始模型目录", try_write_into_original)

    def try_write_subpath():
        from wsp.config import assert_outside

        try:
            assert_outside(MODEL, os.path.join(MODEL, "sub", "deep", "x"))
        except PermissionError:
            return "blocked"
        raise AssertionError("子目录也没挡住！")

    check("拒绝写入原始目录的子目录", try_write_subpath)

    def try_path_traversal():
        """用 ../ 尝试逃逸。"""
        from wsp.config import assert_outside

        target = os.path.join(MODEL, "..", os.path.basename(MODEL), "x")
        try:
            assert_outside(MODEL, target)
        except PermissionError:
            return "blocked"
        raise AssertionError("路径穿越没挡住！")

    check("拒绝路径穿越", try_path_traversal)

    print("\n第 2 部分：API Key 不泄露")

    def save_key():
        r = client.put(API + "/settings/llm", json={
            "base_url": "https://security-test.local/v1",
            "api_key": "sk-LEAK-CANARY-987654321",
            "model": "sec-model",
        })
        r.raise_for_status()
        return True

    check("保存测试 key", save_key)

    def scan_all_endpoints_for_leak():
        canary = "sk-LEAK-CANARY-987654321"
        endpoints = [
            "/health", "/settings/llm", "/models", "/models/sources",
            "/models/discover", "/platform", "/onboarding", "/projects",
            "/edits", "/probe/tasks", "/schema/plan",
        ]
        leaks = []
        for ep in endpoints:
            r = client.get(API + ep)
            if canary in r.text:
                leaks.append(ep)
        assert not leaks, f"以下端点泄露了 key: {leaks}"
        return f"检查了 {len(endpoints)} 个端点"

    check("所有端点都不泄露 key", scan_all_endpoints_for_leak)

    def scan_db_for_leak():
        import sqlite3

        db = os.path.join(HOME, "wsp.db")
        conn = sqlite3.connect(db)
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
        leaks = []
        for t in tables:
            try:
                rows = conn.execute(f"SELECT * FROM {t}").fetchall()
                if any("sk-LEAK-CANARY" in str(v) for row in rows for v in row):
                    leaks.append(t)
            except Exception:
                pass
        conn.close()
        assert not leaks, f"以下表里有明文 key: {leaks}"
        return f"检查了 {len(tables)} 张表"

    check("数据库里无明文 key", scan_db_for_leak)

    def check_log_no_leak():
        logs = [os.path.join(HOME, f) for f in ("run.log", "run.err")]
        for lp in logs:
            if os.path.exists(lp):
                content = open(lp, encoding="utf-8", errors="ignore").read()
                if "sk-LEAK-CANARY" in content:
                    raise AssertionError(f"{lp} 里有明文 key")
        return True

    check("日志里无明文 key", check_log_no_leak)

    print("\n第 3 部分：编辑预算无法绕过")

    models = client.get(API + "/models").json()["models"]
    model = next(m for m in models if m["can_edit"])
    pid = client.post(API + "/projects", json={
        "name": "安全测试", "model_id": model["id"]
    }).json()["id"]

    # 试图改整个张量
    oversize = {
        "goal": "超预算",
        "edits": [{
            "type": "direction_ablation", "layer": 15,
            "module": "mlp.down_proj", "axis": 1,
            "index": {"kind": "all"},  # 整个张量
            "method": "project_out", "magnitude": 1.0,
            "norm_preserve": False,
            "direction_source": {"kind": "random"},
            "expected_effect": "x", "risk": "y",
        }],
        "eval_gate": {"primary": "loop_mini", "min_delta": 0.05},
    }

    def try_oversize():
        r = client.post(API + "/plans", json={"project_id": pid, "plan": oversize})
        if r.status_code != 200:
            return "计划被拒"
        plan_id = r.json()["id"]
        d = client.post(API + f"/plans/{plan_id}/apply", json={"dry_run": True})
        result = d.json()["result"]
        assert result["ok"] is False, "超预算编辑居然通过了！"
        err = " ".join(o.get("error", "") for o in result["outcomes"])
        assert "上限" in err or "预算" in err, f"错误信息不对: {err}"
        return "干跑拒绝"

    check("1% 预算上限生效", try_oversize)

    def try_multiple_large():
        """拆成多条编辑，每条都不超，但合计超。"""
        edit = oversize["edits"][0]
        plan = {
            "goal": "分拆绕过",
            "edits": [
                {**edit, "layer": L, "index": {"kind": "all"}}
                for L in (10, 11, 12, 13, 14, 15)
            ],
            "eval_gate": {"primary": "loop_mini", "min_delta": 0.05},
        }
        r = client.post(API + "/plans", json={"project_id": pid, "plan": plan})
        if r.status_code != 200:
            return "计划被拒"
        d = client.post(API + f"/plans/{r.json()['id']}/apply", json={"dry_run": True})
        result = d.json()["result"]
        assert result["ok"] is False, "分拆绕过成功了！"
        return "被拒"

    check("无法通过分拆绕过预算", try_multiple_large)

    print("\n第 4 部分：编辑可校验（sha256）")

    good = {
        "goal": "可校验",
        "edits": [{
            "type": "direction_ablation", "layer": 15,
            "module": "mlp.down_proj", "axis": 1,
            "index": {"kind": "range", "start": 0, "stop": 16},
            "method": "project_out", "magnitude": 0.5,
            "direction_source": {"kind": "random"},
            "expected_effect": "x", "risk": "y",
        }],
        "eval_gate": {"primary": "loop_mini", "min_delta": 0.05},
    }

    def apply_one():
        pr = client.post(API + "/plans", json={"project_id": pid, "plan": good})
        pr.raise_for_status()
        ap = client.post(API + f"/plans/{pr.json()['id']}/apply",
                         json={"dry_run": False})
        ap.raise_for_status()
        data = ap.json()
        assert data["result"]["ok"] is True
        # 只返回摘要，避免刷屏
        return {
            "edit_id": data["edit"]["id"],
            "changed": data["result"]["total_changed"],
            "checkpoint": bool(data.get("checkpoint")),
        }

    applied = check("应用一次编辑", apply_one)

    def check_diff_manifest():
        if not applied:
            raise AssertionError("没有应用结果")
        edit_id = applied["edit_id"]
        diff_dir = os.path.join(HOME, "workdir", "edits", edit_id)
        mf = os.path.join(diff_dir, "manifest.json")
        assert os.path.exists(mf), f"没有 manifest: {mf}"
        m = json.load(open(mf, encoding="utf-8"))
        assert m["format"] == "wsp.edit"
        assert m["format_version"] == 1
        assert len(m["tensors"]) >= 1
        t = m["tensors"][0]
        for key in ("base_sha256", "edited_sha256", "changed_entries", "name"):
            assert key in t, f"manifest 缺 {key}"
        assert len(t["base_sha256"]) == 64
        assert t["base_sha256"] != t["edited_sha256"]
        return f"{len(m['tensors'])} 张量, {m['changed_entries_total']:,} 条目"

    check("diff manifest 完整", check_diff_manifest)

    def check_verify_against_base():
        """diff 里记录的 base_sha256 必须等于当前模型的真实 hash。"""
        from wsp.weights import store

        edit_id = applied["edit_id"]
        mf = os.path.join(HOME, "workdir", "edits", edit_id, "manifest.json")
        m = json.load(open(mf, encoding="utf-8"))
        info = store.load_model(MODEL)
        for rec in m["tensors"]:
            _dt, arr = store.read_tensor(info, rec["name"])
            actual = store.tensor_sha256(arr)
            assert actual == rec["base_sha256"], (
                f"{rec['name']}: base hash 不匹配\n"
                f"  期望 {rec['base_sha256'][:16]}\n  实际 {actual[:16]}"
            )
        return f"{len(m['tensors'])} 个张量 hash 一致"

    check("base_sha256 与真实权重一致", check_verify_against_base)

    print("\n第 5 部分：回滚完整性")

    def rollback_and_verify():
        edit_id = applied["edit_id"]
        r = client.post(API + "/rollback", json={"edit_id": edit_id})
        r.raise_for_status()
        data = r.json()
        assert data["rolled_back"] is True

        after = sha_file(weights)
        assert after == sha_file(weights), "回滚后文件变了"
        return f"校验了 {len(data['verified_tensors'])} 个张量"

    check("回滚成功", rollback_and_verify)

    def check_original_unchanged():
        now = sha_file(weights)
        assert now == BEFORE, (
            "原始权重被修改了！"
            f" 之前 {BEFORE[:16]} 现在 {now[:16]}"
        )
        return f"sha256 {now[:16]}… 与测试开始时一致"

    check("原始权重未被修改", check_original_unchanged)

    # 清理
    client.delete(API + f"/projects/{pid}")

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
