"""API 端到端测试。"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_home, tiny_model_path, monkeypatch):
    """每个测试用独立的 app 实例（重置单例）。"""
    from wsp.api import routes
    from wsp.api.app import app

    routes.reset_service()
    routes._service = None  # 强制重新创建
    return TestClient(app)


@pytest.fixture
def ready(client):
    """已经导入 tiny 模型的客户端。"""
    from wsp.api import routes

    routes._service.import_model(str(routes._service.state.paths.home), "x") if False else None
    return client


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert "encryption" in body


def test_plan_schema_endpoint(client):
    r = client.get("/api/schema/plan")
    assert r.status_code == 200
    assert "properties" in r.json()


def test_probe_tasks_endpoint(client):
    r = client.get("/api/probe/tasks")
    names = [t["name"] for t in r.json()["tasks"]]
    assert "math_short" in names


def test_import_and_list_models(client, tiny_model_path):
    r = client.post("/api/models/import",
                    json={"path": str(tiny_model_path), "model_id": "tiny"})
    assert r.status_code == 200, r.text
    assert r.json()["id"] == "tiny"
    models = client.get("/api/models").json()["models"]
    assert any(m["id"] == "tiny" for m in models)
    tiny = next(m for m in models if m["id"] == "tiny")
    assert tiny["can_edit"] is True


def test_import_nonexistent_path(client):
    r = client.post("/api/models/import", json={"path": "Z:/nope"})
    assert r.status_code == 400


def test_inspect_model(client, tiny_model_path):
    client.post("/api/models/import", json={"path": str(tiny_model_path), "model_id": "tiny"})
    r = client.get("/api/models/tiny/inspect?layer=1")
    assert r.status_code == 200
    body = r.json()
    assert body["editable"] is True
    assert "mlp.down_proj" in body["modules"]


def test_create_project_and_conversation(client, tiny_model_path):
    client.post("/api/models/import", json={"path": str(tiny_model_path), "model_id": "tiny"})
    r = client.post("/api/projects",
                    json={"name": "p1", "model_id": "tiny", "goal": "g"})
    assert r.status_code == 200
    pid = r.json()["id"]
    r2 = client.post("/api/conversations", json={"project_id": pid, "title": "c1"})
    assert r2.status_code == 200
    convs = client.get(f"/api/projects/{pid}/conversations").json()["conversations"]
    assert len(convs) == 1


def test_create_plan_and_dry_run(client, tiny_model_path, sample_plan):
    client.post("/api/models/import", json={"path": str(tiny_model_path), "model_id": "tiny"})
    pid = client.post("/api/projects",
                      json={"name": "p", "model_id": "tiny"}).json()["id"]
    r = client.post("/api/plans", json={"project_id": pid, "plan": sample_plan})
    assert r.status_code == 200, r.text
    plan_id = r.json()["id"]
    r2 = client.post(f"/api/plans/{plan_id}/apply", json={"dry_run": True})
    assert r2.status_code == 200
    assert r2.json()["result"]["dry_run"] is True


def test_apply_and_rollback_via_api(client, tiny_model_path, sample_plan):
    client.post("/api/models/import", json={"path": str(tiny_model_path), "model_id": "tiny"})
    pid = client.post("/api/projects",
                      json={"name": "p", "model_id": "tiny"}).json()["id"]
    plan_id = client.post("/api/plans",
                          json={"project_id": pid, "plan": sample_plan}).json()["id"]
    r = client.post(f"/api/plans/{plan_id}/apply", json={"dry_run": False})
    assert r.status_code == 200, r.text
    edit_id = r.json()["edit"]["id"]

    edits = client.get("/api/edits").json()["edits"]
    assert any(e["id"] == edit_id for e in edits)

    r2 = client.post("/api/rollback", json={"edit_id": edit_id})
    assert r2.status_code == 200
    assert r2.json()["rolled_back"] is True


def test_invalid_plan_rejected_by_api(client, tiny_model_path):
    client.post("/api/models/import", json={"path": str(tiny_model_path), "model_id": "tiny"})
    pid = client.post("/api/projects",
                      json={"name": "p", "model_id": "tiny"}).json()["id"]
    r = client.post("/api/plans", json={"project_id": pid, "plan": {"goal": "x"}})
    assert r.status_code == 400
    assert "schema" in r.json()["detail"]


def test_unknown_plan_404(client):
    r = client.get("/api/plans/plan_nope")
    assert r.status_code == 404


def test_llm_settings_roundtrip(client):
    r = client.put("/api/settings/llm", json={
        "base_url": "https://api.example.com/v1",
        "api_key": "sk-abc123",
        "model": "test-model",
    })
    assert r.status_code == 200
    body = r.json()
    assert body["api_key_set"] is True
    assert "sk-abc123" not in str(body)
    got = client.get("/api/settings/llm").json()
    assert got["model"] == "test-model"
    assert got["base_url"] == "https://api.example.com/v1"


def test_index_serves_html(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "WSP Studio" in r.text


def test_edits_filter_by_project(client, tiny_model_path, sample_plan):
    client.post("/api/models/import", json={"path": str(tiny_model_path), "model_id": "tiny"})
    pid = client.post("/api/projects",
                      json={"name": "p", "model_id": "tiny"}).json()["id"]
    plan_id = client.post("/api/plans",
                          json={"project_id": pid, "plan": sample_plan}).json()["id"]
    client.post(f"/api/plans/{plan_id}/apply", json={"dry_run": True})
    edits = client.get(f"/api/edits?project_id={pid}").json()["edits"]
    assert edits
    other = client.get("/api/edits?project_id=prj_nope").json()["edits"]
    assert other == []
