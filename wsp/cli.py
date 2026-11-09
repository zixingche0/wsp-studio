"""WSP Studio 命令行。

    wsp serve             启动 Web 界面
    wsp models            列出模型
    wsp import <path>     导入 safetensors 模型
    wsp probe <model>     跑探测
    wsp plan <project>    调用 LLM 生成计划
    wsp apply <plan> --dry-run
    wsp apply <plan>      真正应用
    wsp eval <edit>
    wsp rollback <edit>
    wsp edits             编辑历史
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from .service import Service, ServiceError


def _print(obj) -> None:
    if isinstance(obj, (dict, list)):
        print(json.dumps(obj, ensure_ascii=False, indent=2))
    else:
        print(obj)


def _service(args) -> Service:
    import os

    if getattr(args, "home", None):
        os.environ["WSP_HOME"] = args.home
    return Service()


# ---------------------------------------------------------------------------

def cmd_serve(args) -> int:
    from .api.app import main as run

    run()
    return 0


def cmd_models(args) -> int:
    svc = _service(args)
    _print(svc.list_models())
    return 0


def cmd_import(args) -> int:
    svc = _service(args)
    entry = svc.import_model(args.path, args.model_id)
    _print(entry)
    return 0


def cmd_inspect(args) -> int:
    svc = _service(args)
    _print(svc.inspect_model(args.model, args.layer))
    return 0


def cmd_probe(args) -> int:
    svc = _service(args)
    digest = svc.run_probe(
        args.model,
        tasks=args.tasks.split(",") if args.tasks else None,
        max_cases_per_task=args.max_cases,
    )
    _print({k: v for k, v in digest.items() if k != "tasks"})
    return 0


def cmd_project(args) -> int:
    svc = _service(args)
    _print(svc.create_project(args.name, args.model, args.goal or ""))
    return 0


def cmd_plan(args) -> int:
    from .llm.prompts import PLANNER_SYSTEM, planner_user_prompt
    from .editor.plan_schema import PLAN_SCHEMA

    svc = _service(args)
    project = svc.get_project(args.project)
    model_info = svc.inspect_model(project["model_id"])
    probe = svc.run_probe(project["model_id"], max_cases_per_task=3)
    client = svc.llm_client()
    prompt = planner_user_prompt(args.goal, probe, model_info, PLAN_SCHEMA)
    data, resp = client.chat_json(PLANNER_SYSTEM, prompt)
    row = svc.create_plan(project["id"], data, goal=args.goal)
    _print({"plan_id": row["id"], "plan": row["plan"], "repaired": resp.repaired})
    return 0


def cmd_apply(args) -> int:
    svc = _service(args)
    if args.dry_run:
        _print(svc.dry_run_plan(args.plan_id))
    else:
        _print(svc.apply_plan(args.plan_id))
    return 0


def cmd_eval(args) -> int:
    svc = _service(args)
    _print(svc.run_eval(args.edit_id))
    return 0


def cmd_rollback(args) -> int:
    svc = _service(args)
    _print(svc.rollback_edit(args.edit_id))
    return 0


def cmd_edits(args) -> int:
    svc = _service(args)
    _print(svc.list_edits(args.project))
    return 0


def cmd_health(args) -> int:
    svc = _service(args)
    _print(svc.health())
    return 0


def cmd_platform(args) -> int:
    from . import platform as plat

    _print(plat.platform_report())
    return 0


def cmd_discover(args) -> int:
    """扫描本机所有模型来源。"""
    svc = _service(args)
    extra = args.dirs.split(os.pathsep) if getattr(args, "dirs", None) else None
    result = svc.discover_models(extra)
    if args.json:
        _print(result)
        return 0

    print(f"发现 {result['count']} 个模型\n")
    if result["models"]:
        print(f"{'来源':<12}{'模型':<48}{'大小':>10}  能力")
        print("-" * 84)
        for m in result["models"]:
            cap = "可编辑" if m["editable"] else "只读"
            print(f"{m['source']:<12}{m['name'][:46]:<48}"
                  f"{m['size_gb']:>8}GB  {cap}")
    print("\n扫描的目录：")
    for r in result["roots"]:
        print(f"  {r['path']}")
    up = [s for s in result["services"] if s["reachable"]]
    print("\n运行中的推理服务：")
    if up:
        for s in up:
            print(f"  {s['name']:<22} {s['base_url']}  (模型: {', '.join(s['models']) or '无'})")
    else:
        print("  （无）")
    return 0


def cmd_storage(args) -> int:
    """查看磁盘占用。"""
    svc = _service(args)
    _print(svc.storage_usage())
    return 0


def cmd_gc(args) -> int:
    """清理不再需要的 checkpoint。"""
    svc = _service(args)
    if args.dry_run:
        plan = svc.plan_storage_gc(args.include_applied, args.keep_latest)
        _print({
            "dry_run": True,
            "can_free_gb": plan["total_gb"],
            "checkpoints": plan["checkpoint_count"],
            "db_rows": len(plan["delete_db_rows"]),
            "keep": plan["keep"],
            "detail": plan["delete_checkpoints"],
        })
        print()
        print(f"预计可回收 {plan['total_gb']} GB。加 --apply 执行。")
        return 0
    result = svc.run_storage_gc(args.include_applied, args.keep_latest)
    _print(result)
    return 0


# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="wsp", description="WSP Studio 命令行")
    p.add_argument("--home", help="数据目录（默认 ../wsp-data）")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("serve", help="启动 Web 界面").set_defaults(func=cmd_serve)
    sub.add_parser("health", help="环境体检").set_defaults(func=cmd_health)
    sub.add_parser("platform", help="平台信息（系统/设备/目录）").set_defaults(
        func=cmd_platform
    )
    sub.add_parser("models", help="列出已注册模型").set_defaults(func=cmd_models)

    s = sub.add_parser("discover", help="扫描本机所有模型来源")
    s.add_argument("--dirs", help="额外目录，用系统路径分隔符分隔")
    s.add_argument("--json", action="store_true", help="输出 JSON")
    s.set_defaults(func=cmd_discover)

    s = sub.add_parser("import", help="导入 safetensors 模型")
    s.add_argument("path")
    s.add_argument("--model-id")
    s.set_defaults(func=cmd_import)

    s = sub.add_parser("inspect", help="查看模型结构")
    s.add_argument("model")
    s.add_argument("--layer", type=int)
    s.set_defaults(func=cmd_inspect)

    s = sub.add_parser("probe", help="运行探测")
    s.add_argument("model")
    s.add_argument("--tasks")
    s.add_argument("--max-cases", type=int)
    s.set_defaults(func=cmd_probe)

    s = sub.add_parser("project", help="创建项目")
    s.add_argument("name")
    s.add_argument("model")
    s.add_argument("--goal")
    s.set_defaults(func=cmd_project)

    s = sub.add_parser("plan", help="用 LLM 生成编辑计划")
    s.add_argument("project")
    s.add_argument("goal")
    s.set_defaults(func=cmd_plan)

    s = sub.add_parser("apply", help="应用计划")
    s.add_argument("plan_id")
    s.add_argument("--dry-run", action="store_true", help="只预览，不写文件")
    s.set_defaults(func=cmd_apply)

    s = sub.add_parser("eval", help="评估门控")
    s.add_argument("edit_id")
    s.set_defaults(func=cmd_eval)

    s = sub.add_parser("rollback", help="回滚编辑")
    s.add_argument("edit_id")
    s.set_defaults(func=cmd_rollback)

    sub.add_parser("storage", help="磁盘占用").set_defaults(func=cmd_storage)

    s = sub.add_parser("gc", help="清理不再需要的 checkpoint")
    s.add_argument("--dry-run", action="store_true", default=True,
                   help="只预览（默认）")
    s.add_argument("--apply", dest="dry_run", action="store_false",
                   help="真正执行删除")
    s.add_argument("--include-applied", action="store_true",
                   help="连已应用的 checkpoint 也删（diff 仍保留）")
    s.add_argument("--keep-latest", type=int, default=0,
                   help="无论如何保留最近 N 个")
    s.set_defaults(func=cmd_gc)

    s = sub.add_parser("edits", help="编辑历史")
    s.add_argument("--project")
    s.set_defaults(func=cmd_edits)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except ServiceError as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
