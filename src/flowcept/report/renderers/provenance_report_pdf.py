"""PDF renderer for provenance reports with executive summary charts."""

from __future__ import annotations

import html
from pathlib import Path
import re
from tempfile import TemporaryDirectory
import textwrap
from typing import Any, Dict, Iterable, List, Tuple

from flowcept.report.aggregations import as_float
from flowcept.report.renderers.provenance_card_markdown import (
    _deep_get,
    _extract_telemetry_overview,
    _flatten_numeric,
    _fmt_bytes,
    _fmt_count,
    _fmt_percent,
    _fmt_seconds,
    render_provenance_card_markdown,
)


def _extract_io_bytes(task: Dict[str, Any]) -> float:
    """Return IO byte delta for a task from telemetry start/end snapshots."""
    start = task.get("telemetry_at_start", {}) if isinstance(task.get("telemetry_at_start"), dict) else {}
    end = task.get("telemetry_at_end", {}) if isinstance(task.get("telemetry_at_end"), dict) else {}
    start_io = start.get("disk", {}).get("io_sum", {}) if isinstance(start.get("disk", {}), dict) else {}
    end_io = end.get("disk", {}).get("io_sum", {}) if isinstance(end.get("disk", {}), dict) else {}
    read_delta = (as_float(end_io.get("read_bytes")) or 0.0) - (as_float(start_io.get("read_bytes")) or 0.0)
    write_delta = (as_float(end_io.get("write_bytes")) or 0.0) - (as_float(start_io.get("write_bytes")) or 0.0)
    return max(0.0, read_delta) + max(0.0, write_delta)


def _build_telemetry_chart_data(tasks: Iterable[Dict[str, Any]]) -> List[Tuple[str, List[str], List[float], str]]:
    """Create optional telemetry charts when corresponding metrics are available."""
    cpu_by_activity: Dict[str, List[float]] = {}
    memory_by_activity: Dict[str, float] = {}
    network_by_activity: Dict[str, float] = {}
    gpu_used_by_activity: Dict[str, float] = {}

    for task in tasks:
        activity = str(task.get("activity_id", "unknown"))
        start = task.get("telemetry_at_start", {}) if isinstance(task.get("telemetry_at_start"), dict) else {}
        end = task.get("telemetry_at_end", {}) if isinstance(task.get("telemetry_at_end"), dict) else {}
        if not start and not end:
            continue

        cpu_delta = (
            (as_float(_deep_get(end, ["cpu", "percent_all"])) or 0.0)
            - (as_float(_deep_get(start, ["cpu", "percent_all"])) or 0.0)
        )
        if cpu_delta > 0:
            cpu_by_activity.setdefault(activity, []).append(cpu_delta)

        memory_delta = (
            (as_float(_deep_get(end, ["memory", "virtual", "used"])) or 0.0)
            - (as_float(_deep_get(start, ["memory", "virtual", "used"])) or 0.0)
        )
        if memory_delta > 0:
            memory_by_activity[activity] = memory_by_activity.get(activity, 0.0) + memory_delta

        sent_delta = (
            (as_float(_deep_get(end, ["network", "netio_sum", "bytes_sent"])) or 0.0)
            - (as_float(_deep_get(start, ["network", "netio_sum", "bytes_sent"])) or 0.0)
        )
        recv_delta = (
            (as_float(_deep_get(end, ["network", "netio_sum", "bytes_recv"])) or 0.0)
            - (as_float(_deep_get(start, ["network", "netio_sum", "bytes_recv"])) or 0.0)
        )
        network_total = max(0.0, sent_delta) + max(0.0, recv_delta)
        if network_total > 0:
            network_by_activity[activity] = network_by_activity.get(activity, 0.0) + network_total

        start_gpu = start.get("gpu", {}) if isinstance(start.get("gpu"), dict) else {}
        end_gpu = end.get("gpu", {}) if isinstance(end.get("gpu"), dict) else {}
        gpu_delta = 0.0
        for gpu_key, gpu_end in end_gpu.items():
            if not isinstance(gpu_end, dict):
                continue
            flat_end: Dict[str, float] = {}
            flat_start: Dict[str, float] = {}
            _flatten_numeric("", gpu_end, flat_end)
            gpu_start = start_gpu.get(gpu_key, {}) if isinstance(start_gpu.get(gpu_key), dict) else {}
            _flatten_numeric("", gpu_start, flat_start)
            for metric, val_end in flat_end.items():
                metric_l = metric.lower()
                if "used" not in metric_l or "gpu" in metric_l:
                    continue
                val_start = flat_start.get(metric, 0.0)
                gpu_delta += val_end - val_start if val_end >= val_start else val_end
        if gpu_delta > 0:
            gpu_used_by_activity[activity] = gpu_used_by_activity.get(activity, 0.0) + gpu_delta

    charts: List[Tuple[str, List[str], List[float], str]] = []
    cpu_rows = sorted(
        ((k, sum(v) / len(v)) for k, v in cpu_by_activity.items() if v),
        key=lambda x: x[1],
        reverse=True,
    )[:5]
    if cpu_rows:
        charts.append(
            (
                "Most CPU-Active Activities (Average CPU % Delta)",
                [n for n, _ in cpu_rows],
                [v for _, v in cpu_rows],
                "%",
            )
        )

    mem_rows = sorted(memory_by_activity.items(), key=lambda x: x[1], reverse=True)[:5]
    if mem_rows:
        charts.append(
            ("Largest Memory Growth Activities", [n for n, _ in mem_rows], [v for _, v in mem_rows], "Bytes")
        )

    net_rows = sorted(network_by_activity.items(), key=lambda x: x[1], reverse=True)[:5]
    if net_rows:
        charts.append(
            ("Most Network-Active Activities (Bytes Moved)", [n for n, _ in net_rows], [v for _, v in net_rows], "Bytes")
        )

    gpu_rows = sorted(gpu_used_by_activity.items(), key=lambda x: x[1], reverse=True)[:5]
    if gpu_rows:
        charts.append(
            ("Highest GPU Memory Delta Activities", [n for n, _ in gpu_rows], [v for _, v in gpu_rows], "Bytes")
        )
    return charts


def _build_plot_data(
    transformations: List[Dict[str, Any]],
    tasks: List[Dict[str, Any]],
) -> List[Tuple[str, List[str], List[float], str]]:
    """Build chart definitions for timing/resource summaries."""
    slowest = [r for r in transformations if r.get("elapsed_avg") is not None]
    slowest = sorted(slowest, key=lambda r: float(r.get("elapsed_avg") or 0.0), reverse=True)[:5]
    fastest = [r for r in transformations if r.get("elapsed_avg") is not None and float(r.get("elapsed_avg") or 0.0) > 0]
    fastest = sorted(fastest, key=lambda r: float(r.get("elapsed_avg") or 0.0))[:5]

    io_by_activity: Dict[str, float] = {}
    for task in tasks:
        activity = str(task.get("activity_id", "unknown"))
        io_by_activity[activity] = io_by_activity.get(activity, 0.0) + _extract_io_bytes(task)
    most_resource = sorted(io_by_activity.items(), key=lambda x: x[1], reverse=True)[:5]

    charts: List[Tuple[str, List[str], List[float], str]] = []
    if slowest:
        charts.append(
            (
                "Top Slowest Activities (Average Elapsed Seconds)",
                [str(r.get("activity_id", "unknown")) for r in slowest],
                [float(r.get("elapsed_avg") or 0.0) for r in slowest],
                "Seconds",
            )
        )
    if fastest:
        charts.append(
            (
                "Top Fastest Activities (Average Elapsed Seconds)",
                [str(r.get("activity_id", "unknown")) for r in fastest],
                [float(r.get("elapsed_avg") or 0.0) for r in fastest],
                "Seconds",
            )
        )
    if most_resource:
        charts.append(
            (
                "Most Resource-Demanding Activities (Total IO Bytes)",
                [name for name, _ in most_resource],
                [float(v) for _, v in most_resource],
                "IO Bytes",
            )
        )
    charts.extend(_build_telemetry_chart_data(tasks))
    return charts


def _render_bar_plot(title: str, labels: List[str], values: List[float], y_label: str, output_path: Path) -> None:
    """Render a polished bar chart into a PNG file."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#f8fafc")
    wrapped_labels = [_wrap_text(label, width=14) for label in labels]
    bars = ax.bar(wrapped_labels, values, color="#0ea5e9", edgecolor="#0369a1", linewidth=1.0)
    ax.set_title(title, fontsize=13, fontweight="bold", color="#0f172a")
    ax.set_ylabel(y_label, fontsize=10, color="#334155")
    ax.tick_params(axis="x", labelrotation=0, labelsize=8)
    ax.tick_params(axis="y", labelsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    is_bytes = y_label.lower() in {"bytes", "io bytes"}
    is_percent = y_label.strip() == "%"
    for bar, value in zip(bars, values):
        if is_bytes:
            text_value = _fmt_bytes(value)
        elif is_percent:
            text_value = _fmt_percent(value)
        else:
            text_value = f"{value:.2f}"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            text_value,
            ha="center",
            va="bottom",
            fontsize=8,
            color="#0f172a",
        )
    fig.tight_layout()
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _build_telemetry_table(telemetry_overview: Dict[str, Any]):
    """Return rows (header + values) for workflow-level telemetry table."""
    rows = [
        ["Metric", "Value"],
        ["CPU User Delta (s)", _fmt_seconds(telemetry_overview.get("cpu_user"))],
        ["CPU System Delta (s)", _fmt_seconds(telemetry_overview.get("cpu_system"))],
        ["Average CPU (%)", _fmt_percent(telemetry_overview.get("cpu_percent_avg"))],
        ["Memory Used Delta", _fmt_bytes(telemetry_overview.get("memory_used"))],
        ["Read Bytes", _fmt_bytes(telemetry_overview.get("read_bytes"))],
        ["Write Bytes", _fmt_bytes(telemetry_overview.get("write_bytes"))],
        ["Read Ops", _fmt_count(telemetry_overview.get("read_count"))],
        ["Write Ops", _fmt_count(telemetry_overview.get("write_count"))],
    ]
    network = telemetry_overview.get("network", {}) if isinstance(telemetry_overview.get("network"), dict) else {}
    if network.get("net_bytes_sent") or network.get("net_bytes_recv"):
        rows.extend(
            [
                ["Network Sent", _fmt_bytes(network.get("net_bytes_sent"))],
                ["Network Received", _fmt_bytes(network.get("net_bytes_recv"))],
            ]
        )
    gpu_devices = len(telemetry_overview.get("gpu_names", [])) or len(telemetry_overview.get("gpu_ids", []))
    if gpu_devices:
        rows.append(["GPU Devices Seen", _fmt_count(gpu_devices)])
    return [rows[0]] + [r for r in rows[1:] if r[1] not in {"-", "unknown"}]


def _truncate_text(value: str, max_len: int = 180) -> str:
    """Truncate long strings to keep PDF table cells readable."""
    text = value.strip()
    return text if len(text) <= max_len else text[: max_len - 3] + "..."


def _wrap_text(value: str, width: int = 20) -> str:
    """Wrap long labels/cells so they fit into PDF column widths."""
    text = value.strip()
    if not text:
        return text
    return "\n".join(textwrap.wrap(text, width=width, break_long_words=True, break_on_hyphens=True))


def _convert_inline_markup(text: str) -> str:
    """Convert markdown-ish inline markup to ReportLab-friendly rich text."""
    cleaned = text.replace("<br>", " ").replace("<br/>", " ").strip()
    cleaned = cleaned.replace("**", "")
    cleaned = cleaned.replace("`", "")

    parts = []
    last = 0
    for match in re.finditer(r"\[([^\]]+)\]\(([^)]+)\)", cleaned):
        start, end = match.span()
        if start > last:
            parts.append(html.escape(cleaned[last:start]))
        link_text = html.escape(match.group(1))
        href = html.escape(match.group(2), quote=True)
        parts.append(f'<link href="{href}" color="blue">{link_text}</link>')
        last = end
    if last < len(cleaned):
        parts.append(html.escape(cleaned[last:]))
    return "".join(parts)


def _render_per_activity_details_table(lines: List[str], styles: Dict[str, Any]):
    """Render 'Per Activity Details' section as a compact table."""
    from reportlab.lib import colors
    from reportlab.lib.units import inch
    from reportlab.platypus import Paragraph, Spacer, Table, TableStyle

    rows = [["Activity", "Group", "Field", "Summary"]]
    current_activity = ""
    current_group = ""

    for raw in lines:
        stripped = raw.lstrip()
        if not stripped.startswith("- "):
            continue
        bullet = _convert_inline_markup(stripped[2:].strip())
        if not bullet:
            continue
        if "(n=" in bullet and ":" not in bullet:
            current_activity = bullet
            current_group = ""
            continue
        if bullet.endswith("(aggregated):"):
            current_group = bullet[:-1]
            continue
        if ":" in bullet:
            field, summary = bullet.split(":", 1)
            rows.append(
                [
                    _wrap_text(_truncate_text(current_activity, 80), width=20),
                    _truncate_text(current_group, 28),
                    _truncate_text(field, 34),
                    _truncate_text(summary.strip(), 180),
                ]
            )

    story = [Paragraph("Per Activity Details", styles["h2"])]
    if len(rows) == 1:
        story.append(Paragraph("No activity detail rows available.", styles["body"]))
        return story

    table = Table(rows, colWidths=[1.5 * inch, 1.2 * inch, 1.2 * inch, 2.9 * inch])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f8fafc")),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#cbd5e1")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    story.append(table)
    story.append(Spacer(1, 0.08 * inch))
    return story


def _render_object_details_table(lines: List[str], styles: Dict[str, Any]):
    """Render 'Object Details by Type' section as a structured table."""
    from reportlab.lib import colors
    from reportlab.lib.units import inch
    from reportlab.platypus import Paragraph, Spacer, Table, TableStyle

    rows = [["Type", "Object", "Version", "Storage", "Size", "Task", "Workflow", "Custom Metadata"]]
    current_type = ""
    current_obj: Dict[str, str] | None = None

    for raw in lines:
        stripped = raw.lstrip()
        if not stripped.startswith("- "):
            continue
        bullet = _convert_inline_markup(stripped[2:].strip())
        if not bullet:
            continue

        # Type header bullets, e.g., "Datasets:"
        if bullet.endswith(":") and "(" not in bullet:
            current_type = bullet[:-1]
            continue

        # Object row, e.g., "<id> (version=2, storage=gridfs, size=1.9 KB)"
        if "(version=" in bullet and "storage=" in bullet:
            object_id = bullet.split(" (", 1)[0]
            attrs_text = bullet.split("(", 1)[1].rstrip(")")
            attrs = {"version": "", "storage": "", "size": ""}
            for part in attrs_text.split(","):
                if "=" in part:
                    k, v = part.split("=", 1)
                    attrs[k.strip()] = v.strip()
            current_obj = {
                "type": current_type,
                "object_id": object_id,
                "version": attrs.get("version", ""),
                "storage": attrs.get("storage", ""),
                "size": attrs.get("size", ""),
                "task_id": "",
                "workflow_id": "",
                "custom_metadata": "",
            }
            rows.append(
                [
                    _truncate_text(current_obj["type"], 20),
                    _truncate_text(current_obj["object_id"], 38),
                    current_obj["version"],
                    current_obj["storage"],
                    current_obj["size"],
                    "",
                    "",
                    "",
                ]
            )
            continue

        if current_obj is None or len(rows) <= 1:
            continue

        # Detail lines
        if bullet.startswith("task_id:"):
            details = [p.strip() for p in bullet.split(";")]
            for p in details:
                if p.startswith("task_id:"):
                    rows[-1][5] = _truncate_text(p.split(":", 1)[1].strip(), 24)
                elif p.startswith("workflow_id:"):
                    rows[-1][6] = _truncate_text(p.split(":", 1)[1].strip(), 24)
            continue
        if bullet.startswith("custom_metadata:"):
            rows[-1][7] = _truncate_text(bullet.split(":", 1)[1].strip(), 120)

    story = [Paragraph("Object Details by Type", styles["h2"])]
    if len(rows) == 1:
        story.append(Paragraph("No object detail rows available.", styles["body"]))
        return story

    table = Table(rows, colWidths=[0.65 * inch, 1.35 * inch, 0.45 * inch, 0.6 * inch, 0.6 * inch, 0.75 * inch, 1.0 * inch, 1.4 * inch])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f8fafc")),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#cbd5e1")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 7),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    story.append(table)
    story.append(Spacer(1, 0.08 * inch))
    return story


def _markdown_to_story(
    markdown_text: str,
    styles: Dict[str, Any],
    telemetry_table,
):
    """Convert markdown-like text to reportlab story elements."""
    from reportlab.lib.units import inch
    from reportlab.platypus import Paragraph, Preformatted, Spacer, Table, TableStyle
    from reportlab.lib import colors

    story = []
    lines = markdown_text.splitlines()
    telemetry_inserted = False

    in_workflow_args = False

    idx = 0
    while idx < len(lines):
        raw = lines[idx].rstrip()
        line = _convert_inline_markup(raw)
        stripped = raw.lstrip()
        indent = len(raw) - len(stripped)

        if not line:
            story.append(Spacer(1, 0.06 * inch))
            idx += 1
            continue
        if line.startswith("### "):
            story.append(Paragraph(line[4:].strip(), styles["h3"]))
            idx += 1
            continue
        if line.startswith("## "):
            heading = line[3:].strip()
            if (not telemetry_inserted) and heading == "Workflow-level Resource Usage":
                story.append(Paragraph("Workflow-level Telemetry Summary", styles["h2"]))
                story.append(telemetry_table)
                story.append(Spacer(1, 0.12 * inch))
                telemetry_inserted = True
            in_workflow_args = False
            if heading == "Per Activity Details":
                section_lines = []
                idx += 1
                while idx < len(lines) and not lines[idx].startswith("## "):
                    section_lines.append(lines[idx])
                    idx += 1
                story.extend(_render_per_activity_details_table(section_lines, styles))
                continue
            if heading == "Object Details by Type":
                section_lines = []
                idx += 1
                while idx < len(lines) and not lines[idx].startswith("## "):
                    section_lines.append(lines[idx])
                    idx += 1
                story.extend(_render_object_details_table(section_lines, styles))
                continue
            story.append(Paragraph(heading, styles["h2"]))
            idx += 1
            continue
        if line.startswith("# "):
            idx += 1
            continue

        if stripped.startswith("- "):
            bullet_text = _convert_inline_markup(stripped[2:])
            if bullet_text.startswith("Workflow args:"):
                in_workflow_args = True
            else:
                in_workflow_args = False
            if indent >= 4:
                style = styles["b3"]
            elif indent >= 2:
                style = styles["b2"]
            else:
                style = styles["b1"]
            story.append(Paragraph(f"• {bullet_text}", style))
            idx += 1
            continue

        if stripped.startswith("| "):
            table_lines = []
            while idx < len(lines) and lines[idx].lstrip().startswith("| "):
                table_lines.append(lines[idx].lstrip())
                idx += 1
            rows = []
            for row_line in table_lines:
                cells = [c.strip() for c in row_line.strip().strip("|").split("|")]
                if all(set(c) <= {"-"} for c in cells):
                    continue
                rows.append(cells)
            if rows:
                col_count = max(len(r) for r in rows)
                for r in rows:
                    if len(r) < col_count:
                        r.extend([""] * (col_count - len(r)))
                width = 6.8 * inch / max(1, col_count)
                table = Table(rows, colWidths=[width] * col_count)
                table.setStyle(
                    TableStyle(
                        [
                            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
                            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                            ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f8fafc")),
                            ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#cbd5e1")),
                            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                            ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                            ("FONTSIZE", (0, 0), (-1, -1), 8),
                        ]
                    )
                )
                story.append(table)
                story.append(Spacer(1, 0.06 * inch))
            continue

        if stripped.startswith("```"):
            code_lines = []
            idx += 1
            while idx < len(lines) and not lines[idx].lstrip().startswith("```"):
                code_lines.append(lines[idx])
                idx += 1
            if idx < len(lines):
                idx += 1
            story.append(Preformatted("\n".join(code_lines), styles["mono"]))
            story.append(Spacer(1, 0.06 * inch))
            continue

        if in_workflow_args and re.match(r"^[A-Za-z0-9_]+:\s", line):
            story.append(Paragraph(f"• {_truncate_text(line, 120)}", styles["b2"]))
        else:
            story.append(Paragraph(line, styles["body"]))
        idx += 1

    if not telemetry_inserted:
        story.append(Paragraph("Workflow-level Telemetry Summary", styles["h2"]))
        story.append(telemetry_table)
        story.append(Spacer(1, 0.12 * inch))

    return story


def _build_pdf_document(
    markdown_text: str,
    plot_paths: List[Path],
    telemetry_overview: Dict[str, Any],
    workflow_title: str,
    output_path: Path,
) -> None:
    """Create the final PDF with markdown content and ending executive plots."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import LETTER
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=LETTER,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36,
        title=f"{workflow_title} - Workflow Provenance Report",
    )
    base = getSampleStyleSheet()
    styles = {
        "title": ParagraphStyle("title", parent=base["Heading1"], fontSize=18, textColor=colors.HexColor("#0f172a")),
        "subtitle": ParagraphStyle("subtitle", parent=base["Normal"], fontSize=10, textColor=colors.HexColor("#334155")),
        "h2": ParagraphStyle("h2", parent=base["Heading2"], fontSize=13, textColor=colors.HexColor("#0f172a")),
        "h3": ParagraphStyle("h3", parent=base["Heading3"], fontSize=11, textColor=colors.HexColor("#0f172a")),
        "body": ParagraphStyle("body", parent=base["Normal"], fontSize=9, leading=12, textColor=colors.HexColor("#111827")),
        "b1": ParagraphStyle("b1", parent=base["Normal"], fontSize=9, leading=12, leftIndent=10, bulletIndent=0),
        "b2": ParagraphStyle("b2", parent=base["Normal"], fontSize=9, leading=12, leftIndent=24, bulletIndent=12),
        "b3": ParagraphStyle("b3", parent=base["Normal"], fontSize=9, leading=12, leftIndent=38, bulletIndent=24),
        "mono": ParagraphStyle(
            "mono",
            parent=base["Code"],
            fontName="Courier",
            fontSize=7.5,
            leading=9,
            textColor=colors.HexColor("#111827"),
        ),
    }

    telemetry_rows = _build_telemetry_table(telemetry_overview)
    telemetry_table = Table(telemetry_rows, colWidths=[2.8 * inch, 3.8 * inch])
    telemetry_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f8fafc")),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e1")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ]
        )
    )

    story = [
        Paragraph(workflow_title, styles["title"]),
        Paragraph("Workflow Provenance Report", styles["subtitle"]),
        Spacer(1, 0.08 * inch),
    ]
    story.extend(_markdown_to_story(markdown_text, styles, telemetry_table))

    if plot_paths:
        story.append(PageBreak())
        story.append(Paragraph("Executive Plots", styles["h2"]))
        story.append(Spacer(1, 0.06 * inch))
        for idx, plot_path in enumerate(plot_paths):
            story.append(Image(str(plot_path), width=6.8 * inch, height=2.7 * inch))
            story.append(Spacer(1, 0.14 * inch))
            if idx < len(plot_paths) - 1:
                story.append(Spacer(1, 0.06 * inch))

    doc.build(story)


def render_provenance_report_pdf(
    dataset: Dict[str, Any],
    transformations: List[Dict[str, Any]],
    object_summary: Dict[str, Any],
    output_path: Path,
) -> Dict[str, Any]:
    """Render a provenance report PDF and include executive plots."""
    try:
        import matplotlib  # noqa: F401
        import reportlab  # noqa: F401
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "PDF report generation requires optional dependencies. Install with: pip install flowcept[report_pdf]"
        ) from e

    with TemporaryDirectory(prefix="flowcept_report_pdf_") as tmp_dir:
        tmp = Path(tmp_dir)
        md_path = tmp / "PROVENANCE_CARD.md"
        markdown_stats = render_provenance_card_markdown(
            dataset=dataset,
            transformations=transformations,
            object_summary=object_summary,
            output_path=md_path,
        )
        markdown_text = md_path.read_text(encoding="utf-8")
        workflow = dataset.get("workflow", {}) if isinstance(dataset.get("workflow"), dict) else {}
        workflow_title = str(workflow.get("name") or workflow.get("workflow_id") or "Workflow")

        charts = _build_plot_data(transformations=transformations, tasks=dataset.get("tasks", []))
        plot_paths: List[Path] = []
        for idx, (title, labels, values, y_label) in enumerate(charts):
            plot_path = tmp / f"plot_{idx + 1}.png"
            _render_bar_plot(title=title, labels=labels, values=values, y_label=y_label, output_path=plot_path)
            plot_paths.append(plot_path)

        telemetry_overview = _extract_telemetry_overview(dataset.get("tasks", []))
        _build_pdf_document(
            markdown_text=markdown_text,
            plot_paths=plot_paths,
            telemetry_overview=telemetry_overview,
            workflow_title=workflow_title,
            output_path=output_path,
        )
        return {**markdown_stats, "plots": len(plot_paths)}
