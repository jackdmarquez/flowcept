"""Webservice API tests with a mocked DBAPI dependency."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from flowcept.commons.flowcept_dataclasses.blob_object import BlobObject
from flowcept.commons.flowcept_dataclasses.workflow_object import WorkflowObject
from flowcept.webservice.deps import get_db_api
from flowcept.webservice.main import create_app


class FakeDB:
    """Simple fake DBAPI for endpoint tests."""

    def __init__(self):
        self.workflows = [
            {"workflow_id": "wf-1", "user": "alice", "campaign_id": "c1", "name": "run-a", "utc_timestamp": 200},
            {"workflow_id": "wf-2", "user": "bob", "campaign_id": "c2", "name": "run-b", "utc_timestamp": 100},
        ]
        self.tasks = [
            {"task_id": "t2", "workflow_id": "wf-1", "status": "running", "started_at": 20},
            {"task_id": "t1", "workflow_id": "wf-1", "status": "finished", "started_at": 10},
            {"task_id": "t3", "workflow_id": "wf-2", "status": "finished", "started_at": 30},
        ]
        self.objects = [
            {
                "object_id": "o1",
                "workflow_id": "wf-1",
                "task_id": "t1",
                "type": "dataset",
                "version": 1,
                "custom_metadata": {"k": "v1"},
                "data": b"payload-1",
                "created_at": "2025-01-02T00:00:00",
            },
            {
                "object_id": "o2",
                "workflow_id": "wf-2",
                "task_id": "t3",
                "type": "ml_model",
                "version": 2,
                "custom_metadata": {"k": "v2"},
                "data": b"payload-2",
                "created_at": "2025-01-01T00:00:00",
            },
        ]

    def workflow_query(self, filter):
        return [wf for wf in self.workflows if all(wf.get(k) == v for k, v in filter.items())]

    def get_workflow_object(self, workflow_id):
        for wf in self.workflows:
            if wf["workflow_id"] == workflow_id:
                return WorkflowObject.from_dict(wf)
        return None

    def query(self, **kwargs):
        collection = kwargs.get("collection")
        filter_ = kwargs.get("filter") or {}
        limit = kwargs.get("limit", 0)

        if collection == "workflows":
            rs = [wf for wf in self.workflows if all(wf.get(k) == v for k, v in filter_.items())]
            return rs[:limit] if limit else rs

        if collection == "objects":
            rs = [obj for obj in self.objects if all(obj.get(k) == v for k, v in filter_.items())]
            return rs[:limit] if limit else rs

        return []

    def task_query(
        self,
        filter,
        projection=None,
        limit=0,
        sort=None,
        aggregation=None,
        remove_json_unserializables=True,
    ):
        rs = [task for task in self.tasks if all(task.get(k) == v for k, v in (filter or {}).items())]

        if sort:
            for field, order in reversed(sort):
                rs = sorted(rs, key=lambda item: item.get(field), reverse=(order == -1))

        if projection:
            rs = [{k: v for k, v in row.items() if k in projection} for row in rs]

        return rs[:limit] if limit else rs

    def blob_object_query(self, filter):
        return [obj for obj in self.objects if all(obj.get(k) == v for k, v in (filter or {}).items())]

    def get_blob_object(self, object_id, version=None):
        if version is None:
            for obj in self.objects:
                if obj["object_id"] == object_id:
                    return BlobObject.from_dict(obj)
            raise ValueError(f"Object not found for object_id={object_id}.")

        for obj in self.objects:
            if obj["object_id"] == object_id and obj["version"] == version:
                return BlobObject.from_dict(obj)

        raise ValueError(f"Object not found for object_id={object_id}, version={version}.")

    def get_object_history(self, object_id):
        return [
            {"object_id": object_id, "version": 2, "created_at": "2025-01-02T00:00:00"},
            {"object_id": object_id, "version": 1, "created_at": "2025-01-01T00:00:00"},
        ]


def build_client() -> tuple[TestClient, FakeDB]:
    app = create_app()
    fake_db = FakeDB()
    app.dependency_overrides[get_db_api] = lambda: fake_db
    return TestClient(app), fake_db


def test_root_and_openapi_endpoints():
    client, _ = build_client()

    root = client.get("/")
    assert root.status_code == 200
    assert root.json()["service"] == "flowcept-webservice"

    assert client.get("/openapi.json").status_code == 200
    assert client.get("/docs").status_code == 200
    assert client.get("/redoc").status_code == 200


def test_health_endpoints():
    client, _ = build_client()
    assert client.get("/api/v1/health/live").json() == {"status": "ok"}
    assert client.get("/api/v1/health/ready").json() == {"status": "ready"}


def test_workflows_list_get_and_query():
    client, _ = build_client()

    rs = client.get("/api/v1/workflows", params={"limit": 10})
    assert rs.status_code == 200
    items = rs.json()["items"]
    assert [item["workflow_id"] for item in items] == ["wf-2", "wf-1"]

    rs = client.get("/api/v1/workflows", params={"user": "alice", "limit": 5})
    assert rs.status_code == 200
    body = rs.json()
    assert body["count"] == 1
    assert body["items"][0]["workflow_id"] == "wf-1"

    rs = client.get("/api/v1/workflows/wf-1")
    assert rs.status_code == 200
    assert rs.json()["workflow_id"] == "wf-1"

    rs = client.post(
        "/api/v1/workflows/query",
        json={"filter": {"campaign_id": "c2"}, "limit": 10, "projection": ["workflow_id"]},
    )
    assert rs.status_code == 200
    assert rs.json()["count"] == 1


def test_workflow_provenance_card_download_route():
    client, _ = build_client()

    def _fake_generate_report(**kwargs):
        output = kwargs["output_path"]
        Path(output).write_text("# Provenance Card\n\nworkflow: wf-1\n", encoding="utf-8")
        return {"output": output}

    with patch("flowcept.webservice.routers.workflows.Flowcept.generate_report", side_effect=_fake_generate_report):
        rs = client.post("/api/v1/workflows/wf-1/reports/provenance-card/download")

    assert rs.status_code == 200
    assert rs.headers["content-type"].startswith("text/markdown")
    assert "attachment; filename=\"provenance_card_wf-1.md\"" == rs.headers["content-disposition"]
    assert "# Provenance Card" in rs.text


def test_workflows_errors():
    client, _ = build_client()

    rs = client.get("/api/v1/workflows/does-not-exist")
    assert rs.status_code == 404

    rs = client.get("/api/v1/workflows", params={"filter_json": "not-json"})
    assert rs.status_code == 400

    rs = client.post("/api/v1/workflows/does-not-exist/reports/provenance-card/download")
    assert rs.status_code == 404


def test_workflow_provenance_card_download_generation_error():
    client, _ = build_client()

    with patch(
        "flowcept.webservice.routers.workflows.Flowcept.generate_report",
        side_effect=RuntimeError("report generation failed"),
    ):
        rs = client.post("/api/v1/workflows/wf-1/reports/provenance-card/download")

    assert rs.status_code == 500
    assert "Could not generate provenance card" in rs.json()["detail"]


def test_tasks_list_get_by_workflow_and_query():
    client, _ = build_client()

    rs = client.get("/api/v1/tasks", params={"workflow_id": "wf-1", "limit": 10})
    assert rs.status_code == 200
    assert rs.json()["count"] == 2
    assert [item["task_id"] for item in rs.json()["items"]] == ["t1", "t2"]

    rs = client.get("/api/v1/tasks/t1")
    assert rs.status_code == 200
    assert rs.json()["task_id"] == "t1"

    rs = client.get("/api/v1/tasks/by_workflow/wf-2")
    assert rs.status_code == 200
    assert rs.json()["count"] == 1

    rs = client.post(
        "/api/v1/tasks/query",
        json={
            "filter": {"workflow_id": "wf-1"},
            "sort": [{"field": "started_at", "order": -1}],
            "projection": ["task_id", "started_at"],
            "limit": 10,
        },
    )
    assert rs.status_code == 200
    items = rs.json()["items"]
    assert items[0]["started_at"] >= items[1]["started_at"]


def test_tasks_errors_and_validation():
    client, _ = build_client()

    rs = client.get("/api/v1/tasks/missing")
    assert rs.status_code == 404

    rs = client.get("/api/v1/tasks", params={"filter_json": "[]"})
    assert rs.status_code == 400

    rs = client.post(
        "/api/v1/tasks/query",
        json={
            "filter": {},
            "projection": ["task_id", "workflow_id"],
            "aggregation": [{"operator": "max", "field": "started_at"}],
            "limit": 10,
        },
    )
    assert rs.status_code == 400


def test_objects_list_get_version_history_and_query():
    client, _ = build_client()

    rs = client.get("/api/v1/objects", params={"workflow_id": "wf-1", "limit": 10})
    assert rs.status_code == 200
    assert rs.json()["count"] == 1
    assert "data" not in rs.json()["items"][0]

    rs = client.get("/api/v1/objects", params={"limit": 10})
    assert rs.status_code == 200
    assert [item["object_id"] for item in rs.json()["items"]] == ["o2", "o1"]

    rs = client.get("/api/v1/objects/o1")
    assert rs.status_code == 200
    assert rs.json()["object_id"] == "o1"
    assert "data" not in rs.json()

    rs = client.get("/api/v1/objects/o1", params={"include_data": True})
    assert rs.status_code == 200
    assert isinstance(rs.json()["data"], str)

    rs = client.get("/api/v1/objects/o2/versions/2", params={"include_data": True})
    assert rs.status_code == 200
    assert rs.json()["version"] == 2

    rs = client.get("/api/v1/objects/o1/download")
    assert rs.status_code == 200
    assert rs.content == b"payload-1"

    rs = client.get("/api/v1/objects/o2/versions/2/download")
    assert rs.status_code == 200
    assert rs.content == b"payload-2"

    rs = client.get("/api/v1/objects/o2/history", params={"limit": 1})
    assert rs.status_code == 200
    assert rs.json()["count"] == 1

    rs = client.post(
        "/api/v1/objects/query",
        json={
            "filter": {},
            "projection": ["object_id", "version"],
            "sort": [{"field": "version", "order": -1}],
            "limit": 1,
            "include_data": False,
        },
    )
    assert rs.status_code == 200
    body = rs.json()
    assert body["count"] == 1
    assert set(body["items"][0].keys()) <= {"object_id", "version"}


def test_objects_errors_and_validation():
    client, _ = build_client()

    rs = client.get("/api/v1/objects/unknown")
    assert rs.status_code == 404

    rs = client.get("/api/v1/objects/o1/versions/99")
    assert rs.status_code == 404

    rs = client.get("/api/v1/objects", params={"filter_json": "not-json"})
    assert rs.status_code == 400

    rs = client.post("/api/v1/objects/query", json={"filter": {}, "limit": 5001})
    assert rs.status_code == 422


def test_datasets_routes():
    client, _ = build_client()

    rs = client.get("/api/v1/datasets")
    assert rs.status_code == 200
    assert rs.json()["count"] == 1
    assert rs.json()["items"][0]["type"] == "dataset"

    rs = client.get("/api/v1/datasets/o1")
    assert rs.status_code == 200
    assert rs.json()["type"] == "dataset"

    rs = client.get("/api/v1/datasets/o1/versions/1")
    assert rs.status_code == 200
    assert rs.json()["version"] == 1

    rs = client.get("/api/v1/datasets/o1/download")
    assert rs.status_code == 200
    assert rs.content == b"payload-1"

    rs = client.post("/api/v1/datasets/query", json={"filter": {}, "limit": 10})
    assert rs.status_code == 200
    assert rs.json()["count"] == 1
    assert rs.json()["items"][0]["type"] == "dataset"

    rs = client.get("/api/v1/datasets/o2")
    assert rs.status_code == 404


def test_models_routes():
    client, _ = build_client()

    rs = client.get("/api/v1/models")
    assert rs.status_code == 200
    assert rs.json()["count"] == 1
    assert rs.json()["items"][0]["type"] == "ml_model"

    rs = client.get("/api/v1/models/o2")
    assert rs.status_code == 200
    assert rs.json()["type"] == "ml_model"

    rs = client.get("/api/v1/models/o2/versions/2")
    assert rs.status_code == 200
    assert rs.json()["version"] == 2

    rs = client.get("/api/v1/models/o2/download")
    assert rs.status_code == 200
    assert rs.content == b"payload-2"

    rs = client.post("/api/v1/models/query", json={"filter": {}, "limit": 10})
    assert rs.status_code == 200
    assert rs.json()["count"] == 1
    assert rs.json()["items"][0]["type"] == "ml_model"

    rs = client.get("/api/v1/models/o1")
    assert rs.status_code == 404
