"""SQLite 存储层。

五层实体，对应"定向管理"的要求：

    Project  ── 一个模型 + 一个目标 + 一条主线
      └─ Conversation  ── 用户与大模型的对话
          └─ Plan     ── 大模型给出的编辑计划
              └─ Edit ── 一次实际应用（可 dry-run）
                  └─ Eval ── 门控结果

每条下游记录都带上游外键，**不允许孤儿记录**。
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
import uuid
from contextlib import contextmanager
from typing import Any, Iterable, Iterator

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS projects (
    id           TEXT PRIMARY KEY,
    name         TEXT NOT NULL,
    model_id     TEXT NOT NULL,
    goal         TEXT NOT NULL DEFAULT '',
    status       TEXT NOT NULL DEFAULT 'active',
    created_at   REAL NOT NULL,
    updated_at   REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS conversations (
    id           TEXT PRIMARY KEY,
    project_id   TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    title        TEXT NOT NULL DEFAULT '',
    created_at   REAL NOT NULL,
    updated_at   REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id              TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role            TEXT NOT NULL,
    content         TEXT NOT NULL DEFAULT '',
    tool_calls      TEXT,
    tool_call_id    TEXT,
    created_at      REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id, created_at);

CREATE TABLE IF NOT EXISTS plans (
    id              TEXT PRIMARY KEY,
    conversation_id TEXT REFERENCES conversations(id) ON DELETE SET NULL,
    project_id      TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    model_id        TEXT NOT NULL,
    goal            TEXT NOT NULL DEFAULT '',
    plan_json       TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'proposed',
    created_at      REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_plans_project ON plans(project_id, created_at);

CREATE TABLE IF NOT EXISTS edits (
    id             TEXT PRIMARY KEY,
    plan_id        TEXT REFERENCES plans(id) ON DELETE SET NULL,
    project_id     TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    model_id       TEXT NOT NULL,
    dry_run        INTEGER NOT NULL DEFAULT 1,
    status         TEXT NOT NULL DEFAULT 'preview',
    base_revision  TEXT NOT NULL DEFAULT '',
    edit_dir       TEXT NOT NULL DEFAULT '',
    diff_json      TEXT NOT NULL DEFAULT '{}',
    reason         TEXT NOT NULL DEFAULT '',
    created_at     REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_edits_project ON edits(project_id, created_at);

CREATE TABLE IF NOT EXISTS evals (
    id          TEXT PRIMARY KEY,
    edit_id     TEXT NOT NULL REFERENCES edits(id) ON DELETE CASCADE,
    project_id  TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    passed      INTEGER NOT NULL DEFAULT 0,
    result_json TEXT NOT NULL DEFAULT '{}',
    created_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_evals_edit ON evals(edit_id);

CREATE TABLE IF NOT EXISTS settings (
    key        TEXT PRIMARY KEY,
    value      TEXT NOT NULL,
    updated_at REAL NOT NULL
);
"""


def new_id(prefix: str) -> str:
    """生成带前缀的短 id。"""
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def now() -> float:
    return time.time()


class Database:
    """轻量 SQLite 封装。每次操作开连接，适合桌面端低频写入。"""

    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def cursor(self) -> Iterator[sqlite3.Connection]:
        conn = self._connect()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ---------------- projects ----------------

    def create_project(self, name: str, model_id: str, goal: str = "") -> dict:
        pid = new_id("prj")
        ts = now()
        with self.cursor() as conn:
            conn.execute(
                "INSERT INTO projects(id,name,model_id,goal,status,created_at,updated_at)"
                " VALUES(?,?,?,?,'active',?,?)",
                (pid, name, model_id, goal, ts, ts),
            )
        return self.get_project(pid)

    def get_project(self, project_id: str) -> dict | None:
        with self.cursor() as conn:
            row = conn.execute("SELECT * FROM projects WHERE id=?", (project_id,)).fetchone()
        return dict(row) if row else None

    def list_projects(self) -> list[dict]:
        with self.cursor() as conn:
            rows = conn.execute(
                "SELECT * FROM projects ORDER BY updated_at DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def update_project(self, project_id: str, **fields: Any) -> dict | None:
        allowed = {"name", "model_id", "goal", "status"}
        sets = {k: v for k, v in fields.items() if k in allowed}
        if not sets:
            return self.get_project(project_id)
        sets["updated_at"] = now()
        cols = ", ".join(f"{k}=?" for k in sets)
        with self.cursor() as conn:
            conn.execute(
                f"UPDATE projects SET {cols} WHERE id=?",
                (*sets.values(), project_id),
            )
        return self.get_project(project_id)

    def delete_project(self, project_id: str) -> None:
        with self.cursor() as conn:
            conn.execute("DELETE FROM projects WHERE id=?", (project_id,))

    def touch_project(self, project_id: str) -> None:
        with self.cursor() as conn:
            conn.execute(
                "UPDATE projects SET updated_at=? WHERE id=?", (now(), project_id)
            )

    # ---------------- conversations ----------------

    def create_conversation(self, project_id: str, title: str = "") -> dict:
        cid = new_id("conv")
        ts = now()
        with self.cursor() as conn:
            conn.execute(
                "INSERT INTO conversations(id,project_id,title,created_at,updated_at)"
                " VALUES(?,?,?,?,?)",
                (cid, project_id, title or "新对话", ts, ts),
            )
        self.touch_project(project_id)
        return self.get_conversation(cid)

    def get_conversation(self, conv_id: str) -> dict | None:
        with self.cursor() as conn:
            row = conn.execute(
                "SELECT * FROM conversations WHERE id=?", (conv_id,)
            ).fetchone()
        return dict(row) if row else None

    def list_conversations(self, project_id: str) -> list[dict]:
        with self.cursor() as conn:
            rows = conn.execute(
                "SELECT * FROM conversations WHERE project_id=? ORDER BY updated_at DESC",
                (project_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    # ---------------- messages ----------------

    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str = "",
        tool_calls: list | None = None,
        tool_call_id: str | None = None,
    ) -> dict:
        mid = new_id("msg")
        ts = now()
        with self.cursor() as conn:
            conn.execute(
                "INSERT INTO messages(id,conversation_id,role,content,tool_calls,"
                "tool_call_id,created_at) VALUES(?,?,?,?,?,?,?)",
                (
                    mid,
                    conversation_id,
                    role,
                    content,
                    json.dumps(tool_calls, ensure_ascii=False) if tool_calls else None,
                    tool_call_id,
                    ts,
                ),
            )
            conn.execute(
                "UPDATE conversations SET updated_at=? WHERE id=?",
                (ts, conversation_id),
            )
        return self.get_message(mid)

    def get_message(self, message_id: str) -> dict | None:
        with self.cursor() as conn:
            row = conn.execute(
                "SELECT * FROM messages WHERE id=?", (message_id,)
            ).fetchone()
        return self._message_row(row) if row else None

    def list_messages(self, conversation_id: str, limit: int = 200) -> list[dict]:
        with self.cursor() as conn:
            rows = conn.execute(
                "SELECT * FROM messages WHERE conversation_id=?"
                " ORDER BY created_at ASC LIMIT ?",
                (conversation_id, limit),
            ).fetchall()
        return [self._message_row(r) for r in rows]

    @staticmethod
    def _message_row(row: sqlite3.Row) -> dict:
        d = dict(row)
        if d.get("tool_calls"):
            try:
                d["tool_calls"] = json.loads(d["tool_calls"])
            except json.JSONDecodeError:
                d["tool_calls"] = None
        return d

    # ---------------- plans ----------------

    def create_plan(
        self,
        project_id: str,
        model_id: str,
        plan: dict,
        conversation_id: str | None = None,
        goal: str = "",
    ) -> dict:
        pid = new_id("plan")
        ts = now()
        with self.cursor() as conn:
            conn.execute(
                "INSERT INTO plans(id,conversation_id,project_id,model_id,goal,"
                "plan_json,status,created_at) VALUES(?,?,?,?,?,?,'proposed',?)",
                (
                    pid,
                    conversation_id,
                    project_id,
                    model_id,
                    goal,
                    json.dumps(plan, ensure_ascii=False),
                    ts,
                ),
            )
        return self.get_plan(pid)

    def get_plan(self, plan_id: str) -> dict | None:
        with self.cursor() as conn:
            row = conn.execute("SELECT * FROM plans WHERE id=?", (plan_id,)).fetchone()
        if not row:
            return None
        d = dict(row)
        d["plan"] = json.loads(d.pop("plan_json"))
        return d

    def list_plans(self, project_id: str) -> list[dict]:
        with self.cursor() as conn:
            rows = conn.execute(
                "SELECT * FROM plans WHERE project_id=? ORDER BY created_at DESC",
                (project_id,),
            ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["plan"] = json.loads(d.pop("plan_json"))
            out.append(d)
        return out

    def set_plan_status(self, plan_id: str, status: str) -> None:
        with self.cursor() as conn:
            conn.execute("UPDATE plans SET status=? WHERE id=?", (status, plan_id))

    # ---------------- edits ----------------

    def create_edit(
        self,
        project_id: str,
        model_id: str,
        *,
        plan_id: str | None = None,
        dry_run: bool = True,
        base_revision: str = "",
        edit_dir: str = "",
        diff: dict | None = None,
        status: str = "preview",
        reason: str = "",
    ) -> dict:
        eid = new_id("edit")
        ts = now()
        with self.cursor() as conn:
            conn.execute(
                "INSERT INTO edits(id,plan_id,project_id,model_id,dry_run,status,"
                "base_revision,edit_dir,diff_json,reason,created_at)"
                " VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (
                    eid,
                    plan_id,
                    project_id,
                    model_id,
                    1 if dry_run else 0,
                    status,
                    base_revision,
                    edit_dir,
                    json.dumps(diff or {}, ensure_ascii=False),
                    reason,
                    ts,
                ),
            )
        return self.get_edit(eid)

    def get_edit(self, edit_id: str) -> dict | None:
        with self.cursor() as conn:
            row = conn.execute("SELECT * FROM edits WHERE id=?", (edit_id,)).fetchone()
        return self._edit_row(row) if row else None

    def list_edits(self, project_id: str | None = None) -> list[dict]:
        with self.cursor() as conn:
            if project_id:
                rows = conn.execute(
                    "SELECT * FROM edits WHERE project_id=? ORDER BY created_at DESC",
                    (project_id,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM edits ORDER BY created_at DESC LIMIT 200"
                ).fetchall()
        return [self._edit_row(r) for r in rows]

    @staticmethod
    def _edit_row(row: sqlite3.Row) -> dict:
        d = dict(row)
        try:
            d["diff"] = json.loads(d.pop("diff_json"))
        except json.JSONDecodeError:
            d["diff"] = {}
        d["dry_run"] = bool(d["dry_run"])
        return d

    def set_edit_status(self, edit_id: str, status: str, reason: str = "") -> None:
        with self.cursor() as conn:
            if reason:
                conn.execute(
                    "UPDATE edits SET status=?, reason=? WHERE id=?",
                    (status, reason, edit_id),
                )
            else:
                conn.execute("UPDATE edits SET status=? WHERE id=?", (status, edit_id))

    # ---------------- evals ----------------

    def create_eval(
        self, edit_id: str, project_id: str, passed: bool, result: dict
    ) -> dict:
        vid = new_id("eval")
        ts = now()
        with self.cursor() as conn:
            conn.execute(
                "INSERT INTO evals(id,edit_id,project_id,passed,result_json,created_at)"
                " VALUES(?,?,?,?,?,?)",
                (vid, edit_id, project_id, 1 if passed else 0,
                 json.dumps(result, ensure_ascii=False), ts),
            )
        return self.get_eval(vid)

    def get_eval(self, eval_id: str) -> dict | None:
        with self.cursor() as conn:
            row = conn.execute("SELECT * FROM evals WHERE id=?", (eval_id,)).fetchone()
        if not row:
            return None
        d = dict(row)
        d["result"] = json.loads(d.pop("result_json"))
        d["passed"] = bool(d["passed"])
        return d

    def list_evals(self, edit_id: str) -> list[dict]:
        with self.cursor() as conn:
            rows = conn.execute(
                "SELECT * FROM evals WHERE edit_id=? ORDER BY created_at DESC",
                (edit_id,),
            ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["result"] = json.loads(d.pop("result_json"))
            d["passed"] = bool(d["passed"])
            out.append(d)
        return out

    # ---------------- settings ----------------

    def set_setting(self, key: str, value: str) -> None:
        with self.cursor() as conn:
            conn.execute(
                "INSERT INTO settings(key,value,updated_at) VALUES(?,?,?)"
                " ON CONFLICT(key) DO UPDATE SET value=excluded.value,"
                " updated_at=excluded.updated_at",
                (key, value, now()),
            )

    def get_setting(self, key: str, default: str | None = None) -> str | None:
        with self.cursor() as conn:
            row = conn.execute(
                "SELECT value FROM settings WHERE key=?", (key,)
            ).fetchone()
        return row["value"] if row else default

    def get_settings(self) -> dict[str, str]:
        with self.cursor() as conn:
            rows = conn.execute("SELECT key,value FROM settings").fetchall()
        return {r["key"]: r["value"] for r in rows}
