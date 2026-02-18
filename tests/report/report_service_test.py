import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from flowcept import Flowcept


def _sample_records():
    return [
        {
            "type": "workflow",
            "workflow_id": "wf-1",
            "campaign_id": "camp-1",
            "name": "demo_wf",
        },
        {
            "type": "task",
            "workflow_id": "wf-1",
            "campaign_id": "camp-1",
            "task_id": "t1",
            "activity_id": "LoadData",
            "status": "FINISHED",
            "started_at": 10.0,
            "ended_at": 15.0,
            "custom_metadata": {"current_stage": "ingest"},
        },
        {
            "type": "task",
            "workflow_id": "wf-1",
            "campaign_id": "camp-1",
            "task_id": "t2",
            "activity_id": "LoadData",
            "status": "FINISHED",
            "started_at": 16.0,
            "ended_at": 22.0,
            "custom_metadata": {"current_stage": "ingest"},
        },
        {
            "type": "task",
            "workflow_id": "wf-1",
            "campaign_id": "camp-1",
            "task_id": "t3",
            "activity_id": "Transform",
            "status": "FINISHED",
            "started_at": 23.0,
            "ended_at": 25.0,
            "custom_metadata": {"current_stage": "transform"},
        },
        {
            "object_id": "obj-1",
            "workflow_id": "wf-1",
            "task_id": "t3",
            "type": "ml_model",
            "version": 1,
            "grid_fs_file_id": "abc123",
            "object_size_bytes": 4096,
            "data": b"should-not-be-rendered",
        },
    ]


def _sample_records_with_telemetry_and_io():
    return [
        {
            "type": "workflow",
            "workflow_id": "wf-tele-1",
            "campaign_id": "camp-tele-1",
            "name": "telemetry_demo",
            "sys_name": "Darwin",
            "environment_id": "laptop",
            "code_repository": {
                "branch": "main",
                "short_sha": "abc123",
                "dirty": "clean",
                "remote": "git@example/repo.git",
            },
        },
        {
            "type": "task",
            "workflow_id": "wf-tele-1",
            "campaign_id": "camp-tele-1",
            "task_id": "t1",
            "activity_id": "LoadData",
            "status": "FINISHED",
            "started_at": 10.0,
            "ended_at": 20.0,
            "used": {"input_path": "data/a.json", "batch": 16},
            "generated": {"rows": 1000},
            "telemetry_at_start": {
                "cpu": {"times_avg": {"user": 1.0, "system": 0.5}, "percent_all": 10.0},
                "disk": {"io_sum": {"read_bytes": 1000, "write_bytes": 200, "read_count": 10, "write_count": 2}},
                "memory": {"virtual": {"used": 10000}},
            },
            "telemetry_at_end": {
                "cpu": {"times_avg": {"user": 2.5, "system": 1.0}, "percent_all": 35.0},
                "disk": {"io_sum": {"read_bytes": 7000, "write_bytes": 1200, "read_count": 25, "write_count": 6}},
                "memory": {"virtual": {"used": 18000}},
            },
        },
        {
            "type": "task",
            "workflow_id": "wf-tele-1",
            "campaign_id": "camp-tele-1",
            "task_id": "t2",
            "activity_id": "Transform",
            "status": "FINISHED",
            "started_at": 21.0,
            "ended_at": 23.0,
            "used": {"output_dir": "out/demo", "batch": 32},
            "generated": {"rows": 500},
            "telemetry_at_start": {
                "cpu": {"times_avg": {"user": 2.0, "system": 1.0}, "percent_all": 20.0},
                "disk": {"io_sum": {"read_bytes": 7000, "write_bytes": 1200, "read_count": 25, "write_count": 6}},
                "memory": {"virtual": {"used": 18000}},
            },
            "telemetry_at_end": {
                "cpu": {"times_avg": {"user": 2.8, "system": 1.5}, "percent_all": 45.0},
                "disk": {"io_sum": {"read_bytes": 9000, "write_bytes": 4200, "read_count": 35, "write_count": 18}},
                "memory": {"virtual": {"used": 24000}},
            },
            "custom_metadata": {"metadata": {"saved_files": ["out/demo/data.pt"]}},
        },
    ]


class ReportServiceTests(unittest.TestCase):
    def test_generate_report_from_records(self):
        with tempfile.TemporaryDirectory() as td:
            output = Path(td) / "PROVENANCE_CARD.md"
            stats = Flowcept.generate_report(records=_sample_records(), output_path=str(output))
            assert output.exists()
            assert stats["input_mode"] == "records"
            content = output.read_text(encoding="utf-8")
            assert "## Aggregation Method" in content
            assert "## Object Artifacts Summary" in content
            assert "## Per Activity Details" in content
            assert "- **Total Activities:** `2`" in content
            assert "- **Status Counts:** `{'FINISHED': 3}`" in content
            assert "| Total Size | 4.00 KB |" in content
            assert "| Average Size | 4.00 KB |" in content

    def test_generate_report_from_jsonl(self):
        with tempfile.TemporaryDirectory() as td:
            jsonl_path = Path(td) / "flowcept_buffer.jsonl"
            with jsonl_path.open("w", encoding="utf-8") as f:
                for rec in _sample_records():
                    rec_copy = dict(rec)
                    if isinstance(rec_copy.get("data"), (bytes, bytearray)):
                        rec_copy["data"] = "bytes-redacted"
                    f.write(json.dumps(rec_copy) + "\n")
            output = Path(td) / "card.md"
            stats = Flowcept.generate_report(input_jsonl_path=str(jsonl_path), output_path=str(output))
            assert output.exists()
            assert stats["input_mode"] == "jsonl"

    def test_generate_report_input_validation(self):
        with self.assertRaises(ValueError):
            Flowcept.generate_report(
                input_jsonl_path="/tmp/a.jsonl",
                records=[],
            )

    def test_generate_report_from_db_mode_dispatch(self):
        with tempfile.TemporaryDirectory() as td:
            output = Path(td) / "db_card.md"
            mocked_dataset = {
                "workflow": {"workflow_id": "wf-1", "campaign_id": "camp-1", "name": "demo_wf"},
                "tasks": _sample_records()[1:4],
                "objects": [_sample_records()[4]],
            }
            with patch("flowcept.report.service.load_records_from_db", return_value=mocked_dataset) as mocked:
                stats = Flowcept.generate_report(workflow_id="wf-1", output_path=str(output))
                assert output.exists()
                assert stats["input_mode"] == "db"
                mocked.assert_called_once()

    def test_generate_report_contains_insights_sections(self):
        with tempfile.TemporaryDirectory() as td:
            output = Path(td) / "PROVENANCE_CARD.md"
            stats = Flowcept.generate_report(records=_sample_records_with_telemetry_and_io(), output_path=str(output))
            assert output.exists()
            assert stats["input_mode"] == "records"
            content = output.read_text(encoding="utf-8")
            assert "## Timing Report" in content
            assert "### Interpretation & Insights" in content
            assert "Slowest activities:" in content
            assert "Fastest activities:" in content
            assert "## Per Activity Details" in content
            assert "Activities with richest **used** metadata" in content
            assert "Activities with richest **generated** metadata" in content
            assert "## Workflow-level Resource Usage" in content
            assert "## Per-activity Resource Usage" in content
            assert "Most IO-heavy Activities (Read + Write):" in content
            assert "Most CPU-active Activities:" in content
