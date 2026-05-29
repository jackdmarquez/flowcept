"""Tests for provenance sanitization."""

# ruff: noqa: D103

from flowcept.commons.sanitization import REDACTED, sanitize_value
from flowcept.commons.flowcept_dataclasses.workflow_object import WorkflowObject
import flowcept.commons.flowcept_dataclasses.workflow_object as workflow_object_module


def test_sanitize_value_redacts_nested_secret_keys_and_uri_credentials():
    value = {
        "password": "pw",
        "nested": {
            "Authorization": "Bearer abc",
            "ok": "mongodb://user:secret@mongo:27017/flowcept",
        },
        "items": [{"api_key": "abc"}, "redis://:secret@redis:6379/0"],
    }

    sanitized = sanitize_value(value)

    assert sanitized["password"] == REDACTED
    assert sanitized["nested"]["Authorization"] == REDACTED
    assert sanitized["nested"]["ok"] == f"mongodb://{REDACTED}@mongo:27017/flowcept"
    assert sanitized["items"][0]["api_key"] == REDACTED
    assert sanitized["items"][1] == f"redis://{REDACTED}@redis:6379/0"


def test_workflow_enrich_redacts_flowcept_settings(monkeypatch):
    monkeypatch.setattr(
        workflow_object_module,
        "settings",
        {
            "adapters": {
                "broker_amqp": {
                    "username": "observer",
                    "password": "pw",
                    "uri": "amqp://user:secret@rabbit:5672/",
                }
            },
            "agent": {"api_key": "abc"},
        },
    )

    workflow = WorkflowObject(workflow_id="wf")
    workflow.enrich()

    amqp_settings = workflow.flowcept_settings["adapters"]["broker_amqp"]
    assert amqp_settings["password"] == REDACTED
    assert amqp_settings["uri"] == REDACTED
    assert workflow.flowcept_settings["agent"]["api_key"] == REDACTED
