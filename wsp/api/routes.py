"""FastAPI 路由。"""

from __future__ import annotations

import json
import os
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..agent.loop import Agent, AgentError
from ..editor.plan_schema import PLAN_SCHEMA, plan_summary
from ..models.registry import (
    discover_all_models,
    discover_running_services,
    discover_sources,
)
from ..probe import tasks as probe_tasks
from ..service import Service, ServiceError

router = APIRouter()

_service: Service | None = None


def svc() -> Service:
    global _service
    if _service is None:
        _service = Service()
    return _service


def reset_service() -> None:
    global _service
    _service = None


# ---------------------------------------------------------------------------
# 通用
# ---------------------------------------------------------------------------

@router.get("/health")
def health() -> dict:
    return svc().health()


@router.get("/schema/plan")
def get_plan_schema() -> dict:
    return PLAN_SCHEMA


@router.get("/probe/tasks")
def list_probe_tasks() -> dict:
    return {
        "tasks": [
            {"name": t.name, "description": t.description, "metric": t.metric,
             "num_cases": len(t.cases)}
            for t in probe_tasks.ALL_TASKS.values()
        ],
        "default": probe_tasks.DEFAULT_TASKS,
    }


# ---------------------------------------------------------------------------
# 设置
# ---------------------------------------------------------------------------

class LLMConfigIn(BaseModel):
    base_url: str | None = None
    api_key: str | None = None
    model: str | None = None
    temperature: float | None = None


@router.get("/settings/llm")
def get_llm_settings() -> dict:
    return svc().llm_config_public()


@router.put("/settings/llm")
def put_llm_settings(body: LLMConfigIn) -> dict:
    try:
        return svc().save_llm_config(
            base_url=body.base_url,
            api_key=body.api_key,
            model=body.model,
            temperature=body.temperature,
        )
    except Exception as exc:
        raise HTTPException(400, str(exc))


@router.post("/settings/llm/test")
def test_llm() -> dict:
    client = svc().llm_client()
    return client.test_connection()


# ---------------------------------------------------------------------------
# 模型
# ---------------------------------------------------------------------------

class ImportModelIn(BaseModel):
    path: str
    model_id: str | None = None


class EndpointIn(BaseModel):
    endpoint: str
    served_model: str


@router.get("/models")
def list_models() -> dict:
    return {"models": svc().list_models()}


@router.post("/models/import")
def import_model(body: ImportModelIn) -> dict:
    try:
        return svc().import_model(body.path, body.model_id)
    except ServiceError as exc:
        raise HTTPException(400, str(exc))


@router.post("/models/{model_id}/endpoint")
def set_endpoint(model_id: str, body: EndpointIn) -> dict:
    try:
        return svc().attach_endpoint(model_id, body.endpoint, body.served_model)
    except ServiceError as exc:
        raise HTTPException(400, str(exc))


@router.get("/models/{model_id}/inspect")
def inspect_model(model_id: str, layer: int | None = None) -> dict:
    try:
        return svc().inspect_model(model_id, layer)
    except ServiceError as exc:
        raise HTTPException(400, str(exc))


@router.get("/models/sources")
def model_sources() -> dict:
    """各模型来源的可用状态（内置插件 + 用户插件）。"""
    return {"sources": discover_sources()}


@router.get("/inference/services")
def inference_services() -> dict:
    """本机运行中的推理服务（不绑定任何特定工具）。"""
    return discover_running_services()


@router.get("/models/discover")
def discover_all(extra: str | None = None) -> dict:
    """扫描本机所有模型来源。"""
    dirs = [d for d in (extra or "").split(os.pathsep) if d.strip()]
    return svc().discover_models(dirs or None)


@router.get("/platform")
def platform_info() -> dict:
    """平台信息：操作系统、设备、内存、已知模型目录、工具可用性。"""
    from .. import platform as plat

    return plat.platform_report()


@router.get("/onboarding")
def onboarding_info() -> dict:
    """首次运行引导：当前环境还缺什么、怎么补。"""
    from .. import onboarding as ob
    from .. import platform as plat

    service = svc()
    steps = ob.build_guide(
        has_model=bool(service.state.registry.list()),
        has_llm_config=bool(service.get_llm_config().api_key),
        running_services=[e.as_dict() for e in plat.scan_services()],
    )
    return {
        "steps": [s.as_dict() for s in steps],
        "summary": ob.guide_summary(steps),
        "recommended_models": ob.recommend_models(),
    }


@router.get("/models/download-command")
def model_download_command(model_id: str) -> dict:
    """给出跨平台的模型下载命令。"""
    from .. import onboarding as ob

    return {"model_id": model_id, "commands": ob.download_command(model_id)}


# ---------------------------------------------------------------------------
# 探测
# ---------------------------------------------------------------------------

class ProbeIn(BaseModel):
    model_id: str
    tasks: list[str] | None = None
    max_cases_per_task: int | None = None
    sampling: dict | None = None


@router.post("/probes/run")
def run_probe(body: ProbeIn) -> dict:
    try:
        return svc().run_probe(
            body.model_id, body.tasks, body.max_cases_per_task, body.sampling
        )
    except ServiceError as exc:
        raise HTTPException(400, str(exc))


# ---------------------------------------------------------------------------
# 项目 / 对话
# ---------------------------------------------------------------------------

class ProjectIn(BaseModel):
    name: str
    model_id: str
    goal: str = ""


class ConversationIn(BaseModel):
    project_id: str
    title: str = ""


class MessageIn(BaseModel):
    content: str


@router.get("/projects")
def list_projects() -> dict:
    return {"projects": svc().list_projects()}


@router.post("/projects")
def create_project(body: ProjectIn) -> dict:
    try:
        return svc().create_project(body.name, body.model_id, body.goal)
    except ServiceError as exc:
        raise HTTPException(400, str(exc))


@router.get("/projects/{project_id}")
def get_project(project_id: str) -> dict:
    try:
        return svc().get_project(project_id)
    except ServiceError as exc:
        raise HTTPException(404, str(exc))


@router.get("/projects/{project_id}/conversations")
def list_conversations(project_id: str) -> dict:
    return {"conversations": svc().list_conversations(project_id)}


@router.post("/conversations")
def create_conversation(body: ConversationIn) -> dict:
    try:
        return svc().create_conversation(body.project_id, body.title)
    except ServiceError as exc:
        raise HTTPException(400, str(exc))


@router.get("/conversations/{conversation_id}/messages")
def list_messages(conversation_id: str) -> dict:
    return {"messages": svc().list_messages(conversation_id)}


@router.post("/conversations/{conversation_id}/chat")
def chat(conversation_id: str, body: MessageIn) -> dict:
    """Agent 循环：用户说话，模型调工具。"""
    agent = Agent(svc())
    try:
        run = agent.run(conversation_id, body.content)
    except AgentError as exc:
        raise HTTPException(400, str(exc))
    return run.as_dict()


# ---------------------------------------------------------------------------
# 计划
# ---------------------------------------------------------------------------

class PlanIn(BaseModel):
    project_id: str
    plan: dict
    goal: str = ""
    conversation_id: str | None = None


class GeneratePlanIn(BaseModel):
    project_id: str
    goal: str
    probe: dict | None = None
    tasks: list[str] | None = None
    constraints: dict | None = None


@router.post("/plans")
def create_plan(body: PlanIn) -> dict:
    try:
        row = svc().create_plan(
            body.project_id, body.plan, body.conversation_id, body.goal
        )
        row["summary"] = plan_summary(row["plan"])
        return row
    except ServiceError as exc:
        raise HTTPException(400, str(exc))


@router.post("/plans/generate")
def generate_plan(body: GeneratePlanIn) -> dict:
    """调用大模型 API 生成编辑计划。"""
    from ..llm.prompts import PLANNER_SYSTEM, planner_user_prompt

    service = svc()
    project = service.get_project(body.project_id)
    entry = service.get_model_entry(project["model_id"])
    model_info = service.inspect_model(project["model_id"])

    probe = body.probe
    if probe is None:
        probe = service.run_probe(project["model_id"], tasks=body.tasks)

    client = service.llm_client()
    prompt = planner_user_prompt(
        body.goal, probe, model_info, PLAN_SCHEMA, body.constraints
    )
    try:
        data, resp = client.chat_json(
            PLANNER_SYSTEM, prompt, schema_hint=json.dumps(PLAN_SCHEMA)[:2000]
        )
    except Exception as exc:
        raise HTTPException(400, f"规划器调用失败: {exc}")

    row = service.create_plan(project["id"], data, goal=body.goal)
    row["summary"] = plan_summary(data)
    row["repaired"] = resp.repaired
    row["probe"] = probe
    return row


@router.get("/plans/{plan_id}")
def get_plan(plan_id: str) -> dict:
    try:
        return svc().get_plan(plan_id)
    except ServiceError as exc:
        raise HTTPException(404, str(exc))


class ApplyIn(BaseModel):
    dry_run: bool = True


@router.post("/plans/{plan_id}/apply")
def apply_plan(plan_id: str, body: ApplyIn | None = None) -> dict:
    dry = True if body is None else body.dry_run
    try:
        if dry:
            return svc().dry_run_plan(plan_id)
        return svc().apply_plan(plan_id)
    except ServiceError as exc:
        raise HTTPException(400, str(exc))


# ---------------------------------------------------------------------------
# 编辑 / 评估 / 回滚
# ---------------------------------------------------------------------------

class EvalIn(BaseModel):
    edit_id: str
    tasks: list[str] | None = None


class RollbackIn(BaseModel):
    edit_id: str


@router.get("/edits")
def list_edits(project_id: str | None = None) -> dict:
    return {"edits": svc().list_edits(project_id)}


@router.get("/edits/{edit_id}")
def get_edit(edit_id: str) -> dict:
    try:
        return svc().get_edit(edit_id)
    except ServiceError as exc:
        raise HTTPException(404, str(exc))


@router.post("/evals/run")
def run_eval(body: EvalIn) -> dict:
    try:
        return svc().run_eval(body.edit_id, body.tasks)
    except ServiceError as exc:
        raise HTTPException(400, str(exc))


@router.post("/rollback")
def rollback(body: RollbackIn) -> dict:
    try:
        return svc().rollback_edit(body.edit_id)
    except ServiceError as exc:
        raise HTTPException(400, str(exc))


# ---------------------------------------------------------------------------
# 存储清理
# ---------------------------------------------------------------------------

class GcIn(BaseModel):
    dry_run: bool = True
    include_applied: bool = False
    keep_latest: int = 0


@router.get("/storage/usage")
def storage_usage() -> dict:
    """磁盘使用情况。"""
    return svc().storage_usage()


@router.post("/storage/gc")
def storage_gc(body: GcIn | None = None) -> dict:
    """清理不再需要的 checkpoint。

    ``dry_run=True``（默认）只返回计划，不删任何东西。
    """
    opts = body or GcIn()
    if opts.dry_run:
        return svc().plan_storage_gc(opts.include_applied, opts.keep_latest)
    return svc().run_storage_gc(opts.include_applied, opts.keep_latest)
