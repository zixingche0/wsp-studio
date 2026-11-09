"""服务层：把存储、注册表、执行器、探测、门控串起来。

API 层和 Agent 层都调用这里，保证行为一致。
"""

from __future__ import annotations

import json

import numpy as np
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

from .config import Paths, assert_outside, get_paths
from .editor import executor, ops, plan_schema
from .eval import gate as gate_mod
from .eval import metrics as metrics_mod
from .llm.client import LLMClient, LLMConfig, LLMError
from .models.registry import ModelEntry, Registry
from .probe import runner as probe_runner
from .probe.backends import Backend, HFLocalBackend, OpenAICompatBackend
from .store import db as db_mod
from .store import secrets

SETTINGS_KEY_BASE_URL = "llm.base_url"
SETTINGS_KEY_API_KEY = "llm.api_key"
SETTINGS_KEY_MODEL = "llm.model"
SETTINGS_KEY_TEMPERATURE = "llm.temperature"


class ServiceError(Exception):
    """业务错误（可直接展示给用户）。"""


@dataclass
class AppState:
    """全局状态。"""

    paths: Paths
    db: db_mod.Database
    registry: Registry

    @classmethod
    def create(cls) -> "AppState":
        paths = get_paths()
        return cls(
            paths=paths,
            db=db_mod.Database(str(paths.db_path)),
            registry=Registry(paths),
        )


class Service:
    """面向 API / Agent 的统一门面。"""

    def __init__(self, state: AppState | None = None):
        self.state = state or AppState.create()

    # ------------------------------------------------------------------
    # LLM 配置
    # ------------------------------------------------------------------

    def get_llm_config(self) -> LLMConfig:
        """从数据库读配置；没有则回落到环境变量。"""
        db = self.state.db
        base_url = db.get_setting(SETTINGS_KEY_BASE_URL) or os.environ.get(
            "WSP_LLM_BASE_URL", "https://api.openai.com/v1"
        )
        model = db.get_setting(SETTINGS_KEY_MODEL) or os.environ.get(
            "WSP_LLM_MODEL", "gpt-4o-mini"
        )
        encrypted = db.get_setting(SETTINGS_KEY_API_KEY) or ""
        api_key = ""
        if encrypted:
            try:
                api_key = secrets.decrypt_secret(encrypted, self.state.paths.home)
            except secrets.SecretsError:
                api_key = ""
        if not api_key:
            api_key = os.environ.get("WSP_LLM_API_KEY", "")
        temperature = float(db.get_setting(SETTINGS_KEY_TEMPERATURE) or 0.2)
        return LLMConfig(
            base_url=base_url, api_key=api_key, model=model, temperature=temperature
        )

    def save_llm_config(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
    ) -> dict:
        """保存配置。api_key 传空字符串表示"不修改"。"""
        db = self.state.db
        if base_url is not None:
            db.set_setting(SETTINGS_KEY_BASE_URL, base_url.strip().rstrip("/"))
        if model is not None:
            db.set_setting(SETTINGS_KEY_MODEL, model.strip())
        if temperature is not None:
            db.set_setting(SETTINGS_KEY_TEMPERATURE, str(float(temperature)))
        if api_key:
            encrypted = secrets.encrypt_secret(api_key.strip(), self.state.paths.home)
            db.set_setting(SETTINGS_KEY_API_KEY, encrypted)
        return self.llm_config_public()

    def llm_config_public(self) -> dict:
        """给 UI 的配置视图：key 只返回掩码。"""
        cfg = self.get_llm_config()
        return {
            "base_url": cfg.base_url,
            "model": cfg.model,
            "temperature": cfg.temperature,
            "api_key_masked": secrets.mask_secret(cfg.api_key) if cfg.api_key else "",
            "api_key_set": bool(cfg.api_key),
            "source": "database"
            if self.state.db.get_setting(SETTINGS_KEY_API_KEY)
            else ("env" if os.environ.get("WSP_LLM_API_KEY") else "none"),
            "encryption": secrets.scheme_name(),
        }

    def llm_client(self) -> LLMClient:
        return LLMClient(self.get_llm_config())

    # ------------------------------------------------------------------
    # 模型
    # ------------------------------------------------------------------

    def list_models(self) -> list[dict]:
        return [m.as_dict() for m in self.state.registry.list()]

    def get_model_entry(self, model_id: str) -> ModelEntry:
        entry = self.state.registry.get(model_id)
        if not entry:
            raise ServiceError(f"模型 {model_id!r} 未注册")
        return entry

    def import_model(self, path: str, model_id: str | None = None) -> dict:
        if not os.path.isdir(path):
            raise ServiceError(f"目录不存在: {path}")
        try:
            entry = self.state.registry.import_safetensors(path, model_id)
        except Exception as exc:
            raise ServiceError(f"导入失败: {exc}") from exc
        return entry.as_dict()

    def attach_endpoint(
        self, model_id: str, endpoint: str, served_model: str
    ) -> dict:
        entry = self.state.registry.attach_endpoint(model_id, endpoint, served_model)
        if not entry:
            raise ServiceError(f"模型 {model_id!r} 未注册")
        return entry.as_dict()

    def inspect_model(self, model_id: str, layer: int | None = None) -> dict:
        from .weights import names as name_utils
        from .weights import store as weight_store

        entry = self.get_model_entry(model_id)
        if not entry.can_edit:
            # GGUF 之类没有结构可读
            return {
                "model_id": model_id,
                "kind": entry.kind,
                "editable": False,
                "note": "这个模型是 GGUF，只能用于探测/评估，不能编辑权重。",
            }
        info = weight_store.load_model(entry.path)
        layers = name_utils.list_layers(info.tensor_names)
        result = {
            "model_id": model_id,
            "name": info.name,
            "editable": True,
            "num_layers": info.num_layers,
            "num_tensors": info.num_tensors,
            "total_gb": round(info.total_bytes / 1024**3, 3),
            "architectures": info.config.get("architectures", []),
            "model_type": info.config.get("model_type"),
            "layers": layers,
        }
        if layer is not None:
            mods = name_utils.list_modules(info.tensor_names, layer=layer)
            result["layer"] = layer
            result["modules"] = mods
            shapes = {}
            for mod in mods:
                res = name_utils.resolve(info.tensor_names, layer=layer, module=mod)
                if res.ok and res.tensor_name:
                    dtype_str, arr = weight_store.read_tensor(info, res.tensor_name)
                    shapes[mod] = {
                        "tensor": res.tensor_name,
                        "shape": list(arr.shape),
                        "dtype": dtype_str,
                    }
            result["shapes"] = shapes
        else:
            result["modules"] = name_utils.list_modules(info.tensor_names)
        return result

    def model_backend(self, model_id: str) -> Backend:
        """给探测/评估用的推理后端。"""
        entry = self.get_model_entry(model_id)
        if entry.endpoint and entry.served_model:
            return OpenAICompatBackend(entry.endpoint, entry.served_model)
        if entry.can_edit:
            return HFLocalBackend(entry.path)
        raise ServiceError(
            f"模型 {model_id!r} 既没有服务端点也不是 safetensors，无法推理"
        )

    # ------------------------------------------------------------------
    # 探测
    # ------------------------------------------------------------------

    def run_probe(
        self,
        model_id: str,
        tasks: list[str] | None = None,
        max_cases_per_task: int | None = None,
        sampling: dict | None = None,
    ) -> dict:
        entry = self.get_model_entry(model_id)
        backend = self.model_backend(model_id)
        info = {}
        if entry.can_edit:
            from .weights import store as weight_store
            try:
                info = weight_store.load_model(entry.path).as_dict()
            except Exception:
                info = {}
        digest = probe_runner.run_probe(
            backend,
            model_id=model_id,
            model_info=info,
            task_names=tasks,
            sampling=sampling,
            max_cases_per_task=max_cases_per_task,
        )
        return digest.as_dict()

    # ------------------------------------------------------------------
    # 计划
    # ------------------------------------------------------------------

    def create_plan(
        self,
        project_id: str,
        plan: dict,
        conversation_id: str | None = None,
        goal: str = "",
    ) -> dict:
        project = self.state.db.get_project(project_id)
        if not project:
            raise ServiceError(f"项目 {project_id!r} 不存在")
        try:
            plan_schema.validate_plan(plan)
        except plan_schema.PlanValidationError as exc:
            raise ServiceError(str(exc)) from exc
        return self.state.db.create_plan(
            project_id,
            project["model_id"],
            plan,
            conversation_id=conversation_id,
            goal=goal or plan.get("goal", ""),
        )

    def get_plan(self, plan_id: str) -> dict:
        plan = self.state.db.get_plan(plan_id)
        if not plan:
            raise ServiceError(f"计划 {plan_id!r} 不存在")
        return plan

    def _model_info_for_plan(self, plan_row: dict):
        from .weights import store as weight_store

        entry = self.get_model_entry(plan_row["model_id"])
        if not entry.can_edit:
            raise ServiceError(
                "这个模型不可编辑（GGUF 只能探测）。请先在设置里登记一个 safetensors 模型。"
            )
        return entry, weight_store.load_model(entry.path)

    def dry_run_plan(self, plan_id: str) -> dict:
        row = self.get_plan(plan_id)
        entry, info = self._model_info_for_plan(row)
        result = executor.execute_plan(
            row["plan"], info, self.state.paths,
            dry_run=True, model_id=entry.id,
        )
        edit_row = self.state.db.create_edit(
            project_id=row["project_id"],
            model_id=entry.id,
            plan_id=plan_id,
            dry_run=True,
            base_revision=info.name,
            diff=result.as_dict(),
            status="preview" if result.ok else "preview_failed",
            reason="" if result.ok else result.error,
        )
        return {"edit": edit_row, "result": result.as_dict()}

    def apply_plan(self, plan_id: str) -> dict:
        row = self.get_plan(plan_id)
        entry, info = self._model_info_for_plan(row)
        edit_row = self.state.db.create_edit(
            project_id=row["project_id"],
            model_id=entry.id,
            plan_id=plan_id,
            dry_run=False,
            base_revision=info.name,
            status="applying",
        )
        result = executor.execute_plan(
            row["plan"], info, self.state.paths,
            dry_run=False, model_id=entry.id, edit_id=edit_row["id"],
        )
        if not result.ok:
            self.state.db.set_edit_status(edit_row["id"], "apply_failed", result.error)
            self.state.db.update_project(row["project_id"], status="error")
            raise ServiceError(f"应用失败: {result.error or '未知错误'}")

        # 导出位级 diff，并把 checkpoint 路径写回
        diff_dir = self.state.paths.edits / edit_row["id"]
        diff = executor.export_edit_diff(
            info, _pairs_from_plan(row["plan"], info), diff_dir
        )
        self.state.db.set_edit_status(edit_row["id"], "applied")
        updated = self.state.db.get_edit(edit_row["id"])
        if updated:
            updated["diff"] = {"execution": result.as_dict(), "manifest": diff}
        self.state.db.set_plan_status(plan_id, "applied")
        self.state.db.update_project(row["project_id"], status="evaluating")
        return {"edit": updated, "result": result.as_dict(), "checkpoint": result.edit_dir}

    def list_edits(self, project_id: str | None = None) -> list[dict]:
        return self.state.db.list_edits(project_id)

    def get_edit(self, edit_id: str) -> dict:
        edit = self.state.db.get_edit(edit_id)
        if not edit:
            raise ServiceError(f"编辑 {edit_id!r} 不存在")
        return edit

    # ------------------------------------------------------------------
    # 评估与回滚
    # ------------------------------------------------------------------

    def run_eval(self, edit_id: str, tasks: list[str] | None = None) -> dict:
        """对一次已应用的编辑跑门控。"""
        edit = self.get_edit(edit_id)
        if edit["dry_run"]:
            raise ServiceError("这是一次干跑，没有实际 checkpoint 可评估")
        plan_row = self.state.db.get_plan(edit["plan_id"]) if edit["plan_id"] else None
        gate_cfg = (plan_row or {}).get("plan", {}).get("eval_gate") or {
            "primary": "math_short",
            "min_delta": 0.05,
            "max_regression": 0.03,
        }
        entry = self.get_model_entry(edit["model_id"])

        # 编辑前：原始模型
        base_backend = self.model_backend(entry.id)
        probe_tasks = tasks or gate_cfg.get("tasks") or ["math_short", "ifeval_mini"]
        before = probe_runner.run_probe(
            base_backend, model_id=entry.id, task_names=probe_tasks
        ).scores()

        # 编辑后：如果后端是 HTTP（LM Studio），它加载的是原始 GGUF，
        # 读不到新 checkpoint；这种情况必须用本地后端。
        checkpoint = self._checkpoint_path(edit)
        if checkpoint:
            after_backend: Backend = HFLocalBackend(str(checkpoint))
        else:
            after_backend = base_backend
        after = probe_runner.run_probe(
            after_backend, model_id=entry.id, task_names=probe_tasks
        ).scores()

        decision = gate_mod.evaluate_gate(before, after, gate_cfg)
        passed = decision.passed
        self.state.db.create_eval(edit_id, edit["project_id"], passed, decision.as_dict())

        rolled_back = False
        if not passed:
            self.state.db.set_edit_status(edit_id, "eval_failed", "; ".join(decision.reasons))
            try:
                self.rollback_edit(edit_id, auto=True)
                rolled_back = True
            except ServiceError:
                rolled_back = False
        else:
            self.state.db.set_edit_status(edit_id, "eval_passed")
        self.state.db.update_project(
            edit["project_id"], status="active" if passed else "rolled_back"
        )

        return {
            "edit_id": edit_id,
            "passed": passed,
            "gate": decision.as_dict(),
            "before": before,
            "after": after,
            "delta": metrics_mod.delta_report(before, after),
            "rolled_back": rolled_back,
        }

    def _checkpoint_path(self, edit: dict) -> Path | None:
        diff = edit.get("diff") or {}
        exec_info = diff.get("execution") or {}
        path = exec_info.get("edit_dir")
        if path and Path(path).is_dir():
            return Path(path)
        candidate = self.state.paths.revisions / edit["model_id"] / "revisions" / edit["id"]
        return candidate if candidate.is_dir() else None

    def rollback_edit(self, edit_id: str, auto: bool = False) -> dict:
        """回滚到 base_revision，并校验 hash。"""
        edit = self.get_edit(edit_id)
        checkpoint = self._checkpoint_path(edit)
        if checkpoint and checkpoint.exists():
            shutil.rmtree(checkpoint, ignore_errors=True)
        diff_dir = self.state.paths.edits / edit_id
        verified = []
        if diff_dir.is_dir():
            verified = self._verify_diff_against_base(edit, diff_dir)
        self.state.db.set_edit_status(
            edit_id,
            "rolled_back",
            "自动回滚（门控未通过）" if auto else "手动回滚",
        )
        self.state.db.update_project(edit["project_id"], status="active")
        return {
            "edit_id": edit_id,
            "rolled_back": True,
            "checkpoint_removed": str(checkpoint) if checkpoint else None,
            "verified_tensors": verified,
            "auto": auto,
        }

    def _verify_diff_against_base(self, edit: dict, diff_dir: Path) -> list[str]:
        """校验：当前模型张量是否与 diff 记录的 base_sha256 一致。"""
        from .weights import store as weight_store

        manifest_path = diff_dir / "manifest.json"
        if not manifest_path.exists():
            return []
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return []
        entry = self.state.registry.get(edit["model_id"])
        if not entry or not entry.can_edit:
            return []
        info = weight_store.load_model(entry.path)
        verified = []
        for rec in manifest.get("tensors", []):
            name = rec.get("name")
            if name not in info.tensor_names:
                continue
            _dt, arr = weight_store.read_tensor(info, name)
            if weight_store.tensor_sha256(arr) == rec.get("base_sha256"):
                verified.append(name)
        return verified

    # ------------------------------------------------------------------
    # 项目 / 对话
    # ------------------------------------------------------------------

    def create_project(self, name: str, model_id: str, goal: str = "") -> dict:
        self.get_model_entry(model_id)
        return self.state.db.create_project(name, model_id, goal)

    def list_projects(self) -> list[dict]:
        return self.state.db.list_projects()

    def get_project(self, project_id: str) -> dict:
        project = self.state.db.get_project(project_id)
        if not project:
            raise ServiceError(f"项目 {project_id!r} 不存在")
        return project

    def create_conversation(self, project_id: str, title: str = "") -> dict:
        self.get_project(project_id)
        return self.state.db.create_conversation(project_id, title)

    def list_conversations(self, project_id: str) -> list[dict]:
        return self.state.db.list_conversations(project_id)

    def list_messages(self, conversation_id: str) -> list[dict]:
        return self.state.db.list_messages(conversation_id)

    # ------------------------------------------------------------------
    # 诊断
    # ------------------------------------------------------------------

    def health(self, scan: bool = False) -> dict:
        """环境体检。

        ``scan=False``（默认）不探测推理服务端口——那要几秒钟，
        会让前端打开页面时卡住。要探测请显式传 True。
        """
        from . import platform as plat

        cfg = self.llm_config_public()
        if scan:
            from .models.registry import discover_running_services

            lm = discover_running_services()
        else:
            lm = {"running": None, "services": [], "note": "未扫描"}

        # 模型目录扫描也跳过：health 只报告配置状态
        return {
            "ok": True,
            "version": __import__("wsp").__version__,
            "home": str(self.state.paths.home),
            "llm": cfg,
            "inference_services": lm,
            "models": len(self.state.registry.list()),
            "encryption": secrets.scheme_name(),
            "encryption_detail": secrets.scheme_description(),
            "platform": plat.platform_report(scan_services=False),
        }

    # ------------------------------------------------------------------
    # 存储清理
    # ------------------------------------------------------------------

    def storage_usage(self) -> dict:
        """磁盘使用情况。"""
        from . import storage

        return storage.usage_report(self.state.paths, self.state.db)

    def plan_storage_gc(
        self, include_applied: bool = False, keep_latest: int = 0
    ) -> dict:
        """算出能清理什么（不执行）。"""
        from . import storage

        plan = storage.plan_gc(
            self.state.db, self.state.paths,
            include_applied=include_applied, keep_latest=keep_latest,
        )
        return plan.as_dict()

    def run_storage_gc(
        self, include_applied: bool = False, keep_latest: int = 0
    ) -> dict:
        """执行清理。"""
        from . import storage

        plan = storage.plan_gc(
            self.state.db, self.state.paths,
            include_applied=include_applied, keep_latest=keep_latest,
        )
        result = storage.apply_gc(self.state.db, self.state.paths, plan)
        result["planned"] = plan.as_dict()
        return result

    def discover_models(self, extra_dirs: list[str] | None = None) -> dict:
        """扫描本机所有模型来源。"""
        from . import platform as plat
        from .models import sources

        found = sources.discover_all(extra_dirs)
        # 只报告实际启用的来源会扫描的目录
        roots: list[dict] = []
        for src in sources.all_sources():
            try:
                for p in src.roots():
                    roots.append({"source": src.label, "path": str(p)})
            except Exception:
                continue
        return {
            "models": [m.as_dict() for m in found],
            "count": len(found),
            "sources": sources.source_report(),
            "services": [ep.as_dict() for ep in plat.scan_services()],
            "roots": roots,
        }


def _pairs_from_plan(plan: dict, info) -> dict:
    """把计划里被编辑的张量重新算一遍，供导出 diff 用。

    这是个简化：真正执行时的内存结果没有持久化，所以这里重放一次。
    重放是确定性的（同一 plan、同一 base），结果与 apply 时一致。
    """
    from .weights import names as name_utils
    from .weights import store as weight_store

    modified: dict = {}
    for edit in plan.get("edits") or []:
        if edit.get("type") == "noop":
            continue
        res = name_utils.resolve(
            info.tensor_names,
            layer=edit.get("layer"),
            module=edit.get("module"),
            name=edit.get("tensor_name"),
        )
        if not res.ok or not res.tensor_name:
            continue
        tname = res.tensor_name
        if tname in modified:
            dtype_str, arr = modified[tname]
        else:
            dtype_str, arr = weight_store.read_tensor(info, tname)
        if edit["type"] == "rank1_subtract":
            src = edit.get("direction_source") or {}
            new_arr, _stats = ops.apply_rank1_subtract(
                arr, dtype_str,
                axis=edit["axis"],
                index_spec=edit["index"],
                u=np.asarray(src["u"], dtype=np.float64),
                v=np.asarray(src["v"], dtype=np.float64),
                alpha=edit["magnitude"],
            )
        elif edit["type"] == "direction_ablation":
            from .editor import executor as exec_mod

            direction = exec_mod._direction_from_source(
                edit.get("direction_source"),
                rows=arr.shape[0],
                layer=edit.get("layer"),
                module=edit.get("module"),
            )
            if direction is None:
                direction = exec_mod._svd_direction(
                    arr, dtype_str, edit.get("axis", 1), edit.get("index")
                )
            new_arr, _stats = ops.apply_direction_ablation(
                arr, dtype_str,
                axis=edit["axis"],
                index_spec=edit["index"],
                direction=direction,
                magnitude=edit["magnitude"],
                method=edit.get("method") or "project_out",
                norm_preserve=bool(edit.get("norm_preserve", True)),
            )
        else:
            new_arr, _stats = ops.apply_module_scale(
                arr, dtype_str, factor=edit["factor"]
            )
        modified[tname] = (dtype_str, new_arr)
    return modified
