"""端到端验证：真实 Qwen3.5-0.8B，走完 dry-run -> apply -> diff -> rollback。

不依赖 LLM（用固定的官方编辑几何），验证平台自身的正确性。
"""

import json
import os
import sys

import httpx

BASE = "http://127.0.0.1:8765/api"
client = httpx.Client(timeout=600.0)


def show(title, obj):
    print(f"\n=== {title} ===")
    if isinstance(obj, (dict, list)):
        print(json.dumps(obj, ensure_ascii=False, indent=2)[:2500])
    else:
        print(obj)


def main():
    h = client.get(f"{BASE}/health").json()
    show("health", {"ok": h["ok"], "models": h["models"]})

    models = client.get(f"{BASE}/models").json()["models"]
    target = next(m for m in models if m["id"] == "qwen35-0.8b")
    show("model", {k: target[k] for k in
                   ("id", "name", "num_layers", "num_tensors", "capabilities")})

    proj = client.post(f"{BASE}/projects", json={
        "name": "官方编辑复现",
        "model_id": "qwen35-0.8b",
        "goal": "修复数学推理中的重复循环",
    }).json()
    show("project", {"id": proj["id"]})

    plan = {
        "goal": "修复数学推理中的重复循环",
        "reasoning": (
            "按官方 exp8 manifest：down_proj 形状 [1024,3584]，"
            "沿 axis=1 选 64 列，1024 行全覆盖，project_out。"
        ),
        "edits": [
            {
                "type": "direction_ablation",
                "layer": L,
                "module": "mlp.down_proj",
                "axis": 1,
                "index": {"kind": "range", "start": 0, "stop": 64},
                "method": "project_out",
                "magnitude": 0.8,
                "norm_preserve": True,
                "direction_source": {"kind": "svd_top1"},
                "expected_effect": "减少重复循环",
                "risk": "可能影响通用文本生成",
            }
            for L in (15, 19)
        ],
        "eval_gate": {
            "primary": "loop_mini",
            "guards": ["math_short"],
            "min_delta": 0.05,
            "delta_unit": "absolute",
            "max_regression": 0.03,
            "regression_unit": "absolute",
            "tasks": ["loop_mini", "math_short"],
            "ppl_worsen_ratio": 1.10,
        },
    }

    row = client.post(f"{BASE}/plans", json={
        "project_id": proj["id"], "plan": plan
    }).json()
    plan_id = row["id"]
    show("plan created", {"id": plan_id, "summary": row.get("summary")})

    dry = client.post(f"{BASE}/plans/{plan_id}/apply", json={"dry_run": True}).json()
    r = dry["result"]
    print("\n=== DRY RUN ===")
    print(f"ok={r['ok']}  changed={r['total_changed']:,}  "
          f"fraction={r['model_fraction']:.6%}")
    for o in r["outcomes"]:
        s = o.get("stats", {})
        print(f"  {o['tensor_name']}")
        print(f"    changed={s.get('changed_entries')}  "
              f"rows={s.get('changed_rows')}  cols={s.get('changed_cols')}  "
              f"normΔ={s.get('relative_norm_change', 0):+.6f}")

    print("\n[官方参考] changed=130,188  rows=1024/1023  cols=64  "
          "relative_change≈0.106/0.081  fraction=0.016%")

    edit_id = dry["edit"]["id"]
    print(f"dry-run 写入文件？ {os.path.exists(os.path.join(os.environ['WSP_HOME'], 'workdir'))}")

    app = client.post(f"{BASE}/plans/{plan_id}/apply", json={"dry_run": False}).json()
    show("APPLY", {"edit_id": app["edit"]["id"], "checkpoint": app["checkpoint"]})

    edits = client.get(f"{BASE}/edits").json()["edits"]
    show("edits", [{"id": e["id"], "status": e["status"], "dry_run": e["dry_run"]}
                   for e in edits])

    rb = client.post(f"{BASE}/rollback", json={"edit_id": app["edit"]["id"]}).json()
    show("ROLLBACK", rb)

    print("\nDONE")


if __name__ == "__main__":
    main()
