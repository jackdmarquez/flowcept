"""PDF renderer for provenance reports with executive summary charts."""

from __future__ import annotations

import html
from pathlib import Path
import re
from tempfile import TemporaryDirectory
import textwrap
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Tuple

from flowcept.commons.vocabulary import ML_Types
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

        cpu_delta = (as_float(_deep_get(end, ["cpu", "percent_all"])) or 0.0) - (
            as_float(_deep_get(start, ["cpu", "percent_all"])) or 0.0
        )
        if cpu_delta > 0:
            cpu_by_activity.setdefault(activity, []).append(cpu_delta)

        memory_delta = (as_float(_deep_get(end, ["memory", "virtual", "used"])) or 0.0) - (
            as_float(_deep_get(start, ["memory", "virtual", "used"])) or 0.0
        )
        if memory_delta > 0:
            memory_by_activity[activity] = memory_by_activity.get(activity, 0.0) + memory_delta

        sent_delta = (as_float(_deep_get(end, ["network", "netio_sum", "bytes_sent"])) or 0.0) - (
            as_float(_deep_get(start, ["network", "netio_sum", "bytes_sent"])) or 0.0
        )
        recv_delta = (as_float(_deep_get(end, ["network", "netio_sum", "bytes_recv"])) or 0.0) - (
            as_float(_deep_get(start, ["network", "netio_sum", "bytes_recv"])) or 0.0
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
        charts.append(("Largest Memory Growth Activities", [n for n, _ in mem_rows], [v for _, v in mem_rows], "Bytes"))

    net_rows = sorted(network_by_activity.items(), key=lambda x: x[1], reverse=True)[:5]
    if net_rows:
        charts.append(
            (
                "Most Network-Active Activities (Bytes Moved)",
                [n for n, _ in net_rows],
                [v for _, v in net_rows],
                "Bytes",
            )
        )

    gpu_rows = sorted(gpu_used_by_activity.items(), key=lambda x: x[1], reverse=True)[:5]
    if gpu_rows:
        charts.append(
            ("Highest GPU Memory Delta Activities", [n for n, _ in gpu_rows], [v for _, v in gpu_rows], "Bytes")
        )
    return charts


def _build_plot_data(
    activities: List[Dict[str, Any]],
    tasks: List[Dict[str, Any]],
) -> List[Tuple[str, List[str], List[float], str]]:
    """Build chart definitions for timing/resource summaries."""
    slowest = [r for r in activities if r.get("elapsed_median") is not None]
    slowest = sorted(slowest, key=lambda r: float(r.get("elapsed_median") or 0.0), reverse=True)[:5]
    fastest = [
        r for r in activities if r.get("elapsed_median") is not None and float(r.get("elapsed_median") or 0.0) > 0
    ]
    fastest = sorted(fastest, key=lambda r: float(r.get("elapsed_median") or 0.0))[:5]

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
                [float(r.get("elapsed_median") or 0.0) for r in slowest],
                "Seconds",
            )
        )
    if fastest:
        charts.append(
            (
                "Top Fastest Activities (Average Elapsed Seconds)",
                [str(r.get("activity_id", "unknown")) for r in fastest],
                [float(r.get("elapsed_median") or 0.0) for r in fastest],
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


def _build_ml_learning_plot_spec(dataset: Dict[str, Any]) -> Dict[str, Any] | None:
    """Build a line-plot spec for ML learning metrics over task end time."""
    workflow = dataset.get("workflow", {}) if isinstance(dataset.get("workflow"), dict) else {}
    workflow_subtype = str(workflow.get("subtype", "")).strip()
    if workflow_subtype != ML_Types.WORKFLOW:
        return None

    tasks = dataset.get("tasks", [])
    if not isinstance(tasks, list):
        return None
    learning_tasks = [t for t in tasks if str(t.get("subtype", "")).strip() == ML_Types.LEARNING]
    if len(learning_tasks) <= 2:
        return None

    metric_candidates = ["val_loss", "loss", "best_val_loss", "val_accuracy", "accuracy"]
    chosen_metric = None
    chosen_points: List[Tuple[float, float]] = []
    for metric in metric_candidates:
        points: List[Tuple[float, float]] = []
        for task in learning_tasks:
            ended = as_float(task.get("ended_at"))
            generated = task.get("generated", {}) if isinstance(task.get("generated"), dict) else {}
            val = as_float(generated.get(metric))
            if ended is not None and val is not None:
                points.append((ended, float(val)))
        if len(points) > 2:
            points.sort(key=lambda p: p[0])
            chosen_metric = metric
            chosen_points = points
            break

    if not chosen_metric or not chosen_points:
        return None

    optimize = "min" if "loss" in chosen_metric else "max"
    y_vals = [p[1] for p in chosen_points]
    best_idx = y_vals.index(min(y_vals) if optimize == "min" else max(y_vals))
    x_dt = [datetime.fromtimestamp(p[0], tz=timezone.utc) for p in chosen_points]
    return {
        "title": f"Learning Trend Over Time ({chosen_metric})",
        "x_dt": x_dt,
        "y_vals": y_vals,
        "y_label": chosen_metric,
        "best_idx": best_idx,
        "optimize": optimize,
    }


def _render_ml_line_plot(spec: Dict[str, Any], output_path: Path) -> None:
    """Render an ML metric trend line plot with optimum marker."""
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    x_dt = spec.get("x_dt", [])
    y_vals = spec.get("y_vals", [])
    if not x_dt or not y_vals:
        return

    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#f8fafc")
    ax.plot(x_dt, y_vals, color="#0ea5e9", marker="o", linewidth=1.8, markersize=4)

    best_idx = int(spec.get("best_idx", 0))
    if 0 <= best_idx < len(x_dt):
        ax.plot(
            [x_dt[best_idx]],
            [y_vals[best_idx]],
            marker="x",
            color="#dc2626",
            markersize=10,
            markeredgewidth=2.2,
            linestyle="None",
            label="best",
        )

    ax.set_title(str(spec.get("title", "Learning Trend")), fontsize=13, fontweight="bold", color="#0f172a")
    ax.set_ylabel(str(spec.get("y_label", "metric")), fontsize=10, color="#334155")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.tick_params(axis="x", labelrotation=25, labelsize=8)
    ax.tick_params(axis="y", labelsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=6))
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


def _convert_inline_markup_with_code(text: str) -> str:
    """Convert inline markup and render backticked text as gray code chips."""
    cleaned = text.replace("<br>", " ").replace("<br/>", " ").strip()
    cleaned = cleaned.replace("**", "")

    def _render_code_spans(chunk: str) -> str:
        items: List[str] = []
        last_idx = 0
        for match in re.finditer(r"`([^`]+)`", chunk):
            start, end = match.span()
            if start > last_idx:
                items.append(html.escape(chunk[last_idx:start]))
            code_text = html.escape(match.group(1))
            items.append(f'<font name="Courier" backcolor="#f8fafc">{code_text}</font>')
            last_idx = end
        if last_idx < len(chunk):
            items.append(html.escape(chunk[last_idx:]))
        return "".join(items)

    parts: List[str] = []
    last = 0
    for match in re.finditer(r"\[([^\]]+)\]\(([^)]+)\)", cleaned):
        start, end = match.span()
        if start > last:
            parts.append(_render_code_spans(cleaned[last:start]))
        link_text = _render_code_spans(match.group(1))
        href = html.escape(match.group(2), quote=True)
        parts.append(f'<link href="{href}" color="blue">{link_text}</link>')
        last = end
    if last < len(cleaned):
        parts.append(_render_code_spans(cleaned[last:]))
    return "".join(parts)


def _wrap_long_token_text(text: str, chunk: int = 18) -> str:
    """Insert line-break opportunities for long unbroken tokens."""
    if not text:
        return ""

    def _split_token(match: re.Match) -> str:
        token = match.group(0)
        return "<br/>".join(token[i : i + chunk] for i in range(0, len(token), chunk))

    # Break very long non-whitespace runs so PDF tables never overflow cell width.
    return re.sub(r"\S{30,}", _split_token, text)


def _build_wrapped_table(rows: List[List[Any]], col_widths: List[float], styles: Dict[str, Any], font_size: int = 8):
    """Build a reportlab table whose cells are wrapped Paragraphs."""
    from reportlab.lib import colors
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.platypus import Paragraph, Table, TableStyle

    header_style = ParagraphStyle(
        "tbl_header",
        parent=styles["body"],
        fontName="Helvetica-Bold",
        fontSize=font_size,
        leading=font_size + 2,
        textColor=colors.white,
        wordWrap="CJK",
    )
    body_style = ParagraphStyle(
        "tbl_body",
        parent=styles["body"],
        fontName="Helvetica",
        fontSize=font_size,
        leading=font_size + 2,
        textColor=colors.HexColor("#111827"),
        wordWrap="CJK",
    )

    rendered_rows: List[List[Any]] = []
    for row_idx, row in enumerate(rows):
        row_style = header_style if row_idx == 0 else body_style
        rendered = []
        for cell in row:
            raw = _wrap_long_token_text(str(cell).strip())
            rich = _convert_inline_markup_with_code(raw)
            rendered.append(Paragraph(rich, row_style))
        rendered_rows.append(rendered)

    table = Table(rendered_rows, colWidths=col_widths)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f8fafc")),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#cbd5e1")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ]
        )
    )
    return table


def _render_per_activity_details_table(lines: List[str], styles: Dict[str, Any]):
    """Render 'Per Activity Details' section with markdown-like bullet styling."""
    from reportlab.lib.units import inch
    from reportlab.platypus import Paragraph, Spacer

    story = [Paragraph("Per Activity Details", styles["h2"])]
    if not lines:
        story.append(Paragraph("No activity detail rows available.", styles["body"]))
        return story
    for raw in lines:
        raw_line = raw.rstrip()
        stripped = raw_line.lstrip()
        indent = len(raw_line) - len(stripped)
        if not stripped:
            story.append(Spacer(1, 0.04 * inch))
            continue
        if stripped.startswith("### "):
            story.append(Paragraph(_convert_inline_markup_with_code(stripped[4:].strip()), styles["h3"]))
            continue
        if stripped.startswith("- "):
            bullet = _convert_inline_markup_with_code(stripped[2:].strip())
            if indent >= 4:
                style = styles["b3"]
            elif indent >= 2:
                style = styles["b2"]
            else:
                style = styles["b1"]
            story.append(Paragraph(f"• {bullet}", style))
            continue
        story.append(Paragraph(_convert_inline_markup_with_code(stripped), styles["body"]))
    story.append(Spacer(1, 0.08 * inch))
    return story


def _render_object_details_table(lines: List[str], styles: Dict[str, Any]):
    """Render 'Object Details by Type' in a structured, readable PDF layout."""
    from reportlab.lib.units import inch
    from reportlab.platypus import Paragraph, Preformatted, Spacer

    story = [Paragraph("Object Details by Type", styles["h2"])]
    if not lines:
        story.append(Paragraph("No object detail rows available.", styles["body"]))
        return story

    idx = 0
    while idx < len(lines):
        raw_line = lines[idx].rstrip()
        stripped = raw_line.lstrip()
        indent = len(raw_line) - len(stripped)
        if not stripped:
            story.append(Spacer(1, 0.04 * inch))
            idx += 1
            continue
        if stripped.startswith("- **") and stripped.endswith(":**"):
            label = stripped[2:].strip().strip("*").rstrip(":")
            story.append(Paragraph(f"• <b>{_convert_inline_markup_with_code(label)}</b>:", styles["b2"]))
            story.append(Spacer(1, 0.015 * inch))
            idx += 1
            continue
        if stripped.startswith("- "):
            bullet = stripped[2:].strip()
            if indent <= 2:
                # Object headline row
                story.append(Paragraph(f"• {_convert_inline_markup_with_code(bullet)}", styles["b2"]))
                idx += 1
                # Consume object detail lines until next object/type section
                while idx < len(lines):
                    detail_raw = lines[idx].rstrip()
                    detail_stripped = detail_raw.lstrip()
                    detail_indent = len(detail_raw) - len(detail_stripped)
                    if detail_stripped.startswith("- ") and detail_indent <= 2:
                        break
                    if detail_stripped.startswith("```"):
                        code_lines = []
                        idx += 1
                        while idx < len(lines) and not lines[idx].lstrip().startswith("```"):
                            code_lines.append(lines[idx])
                            idx += 1
                        if idx < len(lines):
                            idx += 1
                        if code_lines:
                            story.append(Preformatted("\n".join(code_lines), styles["mono_indent"]))
                            story.append(Spacer(1, 0.03 * inch))
                        continue
                    if detail_stripped:
                        detail_text = detail_stripped.replace("<br>", "").replace("<br/>", "").strip()
                        if detail_text:
                            if "task_id" in detail_text and ";" in detail_text:
                                parts = [p.strip() for p in detail_text.split(";") if p.strip()]
                                normalized_parts = []
                                for part in parts:
                                    clean = part.replace("`", "").strip()
                                    if ":" in clean:
                                        key, value = clean.split(":", 1)
                                        key = key.strip().lower()
                                        value = value.strip()
                                        if key in {"task_id", "workflow_id", "timestamp"}:
                                            normalized_parts.append(f"{key}: {value}")
                                        else:
                                            normalized_parts.append(clean)
                                    else:
                                        # Fallback: if timestamp label was lost upstream, keep value on timestamp key.
                                        normalized_parts.append(f"timestamp: {clean}")
                                for part in normalized_parts:
                                    story.append(Paragraph(_convert_inline_markup_with_code(part), styles["obj_detail"]))
                            else:
                                story.append(Paragraph(_convert_inline_markup_with_code(detail_text), styles["obj_detail"]))
                    idx += 1
                story.append(Spacer(1, 0.04 * inch))
                continue
            story.append(Paragraph(_convert_inline_markup_with_code(bullet), styles["obj_detail"]))
            idx += 1
            continue
        story.append(Paragraph(_convert_inline_markup_with_code(stripped), styles["obj_detail"]))
        idx += 1

    story.append(Spacer(1, 0.08 * inch))
    return story


def _render_object_details_from_records(objects: List[Dict[str, Any]], styles: Dict[str, Any]):
    """Render object details directly from object records with deterministic indentation."""
    from collections import defaultdict
    from reportlab.platypus import Paragraph, Preformatted, Spacer

    story = [Paragraph("Object Details by Type", styles["h2"])]
    if not objects:
        story.append(Paragraph("No object detail rows available.", styles["body"]))
        return story

    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for obj in objects:
        obj_type = str(obj.get("type", "unknown"))
        grouped[obj_type].append(obj)

    for obj_type in sorted(grouped.keys()):
        header = "Models" if obj_type in {"ml_model", "model"} else ("Datasets" if obj_type == "dataset" else obj_type)
        story.append(Paragraph(f"• <b>{_convert_inline_markup_with_code(header)}</b>:", styles["b2"]))
        story.append(Spacer(1, 0.015 * inch))

        for obj in grouped[obj_type]:
            object_id = str(obj.get("object_id", "-"))
            version = str(obj.get("version", "-"))
            storage = str(obj.get("storage_type", "-"))
            size = _fmt_bytes(as_float(obj.get("object_size_bytes")))
            story.append(
                Paragraph(
                    f"• {_convert_inline_markup_with_code(f'`{object_id}` (version=`{version}`, storage=`{storage}`, size=`{size}`)')}",
                    styles["b2"],
                )
            )

            task_id = str(obj.get("task_id", "-"))
            workflow_id = str(obj.get("workflow_id", "-"))
            timestamp = str(obj.get("updated_at") or obj.get("created_at") or "-")
            story.append(Paragraph(_convert_inline_markup_with_code(f"task_id: `{task_id}`"), styles["obj_detail"]))
            story.append(
                Paragraph(_convert_inline_markup_with_code(f"workflow_id: `{workflow_id}`"), styles["obj_detail"])
            )
            story.append(Paragraph(_convert_inline_markup_with_code(f"timestamp: `{timestamp}`"), styles["obj_detail"]))

            sha = str(obj.get("data_sha256", "-"))
            story.append(Paragraph(_convert_inline_markup_with_code(f"sha256: `{sha}`"), styles["obj_detail"]))

            tags = obj.get("tags")
            if isinstance(tags, list) and tags:
                story.append(
                    Paragraph(
                        _convert_inline_markup_with_code(f"tags: `{', '.join(str(t) for t in tags)}`"),
                        styles["obj_detail"],
                    )
                )

            story.append(Paragraph(_convert_inline_markup_with_code("custom_metadata:"), styles["obj_detail"]))
            metadata = obj.get("custom_metadata", {})
            meta_text = str(metadata) if metadata is not None else "{}"
            story.append(Preformatted(meta_text, styles["mono_indent"]))
            story.append(Spacer(1, 0.03 * inch))

    story.append(Spacer(1, 0.08 * inch))
    return story


def _markdown_to_story(
    markdown_text: str,
    styles: Dict[str, Any],
    telemetry_table,
    plot_paths: List[Path],
    object_records: List[Dict[str, Any]] | None = None,
):
    """Convert markdown-like text to reportlab story elements."""
    from reportlab.lib.units import inch
    from reportlab.platypus import Image, PageBreak, Paragraph, Preformatted, Spacer, Table, TableStyle
    from reportlab.lib import colors

    story = []
    lines = markdown_text.splitlines()
    telemetry_inserted = False
    executive_plots_inserted = False

    in_workflow_args = False

    idx = 0
    while idx < len(lines):
        raw = lines[idx].rstrip()
        line = _convert_inline_markup_with_code(raw)
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
            if (not executive_plots_inserted) and heading == "Aggregation Method" and plot_paths:
                story.append(PageBreak())
                story.append(Paragraph("Plots", styles["h2_tight"] if "h2_tight" in styles else styles["h2"]))
                for plot_idx, plot_path in enumerate(plot_paths):
                    story.append(Image(str(plot_path), width=6.8 * inch, height=2.7 * inch))
                    story.append(Spacer(1, 0.14 * inch))
                    if plot_idx < len(plot_paths) - 1:
                        story.append(Spacer(1, 0.06 * inch))
                executive_plots_inserted = True
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
                if object_records is not None:
                    story.extend(_render_object_details_from_records(object_records, styles))
                else:
                    story.extend(_render_object_details_table(section_lines, styles))
                continue
            story.append(Paragraph(heading, styles["h2"]))
            idx += 1
            continue
        if line.startswith("# "):
            idx += 1
            continue

        if stripped.startswith("- "):
            bullet_text = _convert_inline_markup_with_code(stripped[2:])
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
                table = _build_wrapped_table(rows, [width] * col_count, styles, font_size=8)
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

        if in_workflow_args:
            normalized = stripped.replace("<br>", "").replace("<br/>", "").strip()
            if normalized:
                rendered = _convert_inline_markup_with_code(normalized)
                story.append(Paragraph(f"• {_truncate_text(rendered, 220)}", styles["b3"]))
                idx += 1
                continue
        if line.startswith("Provenance card generated by"):
            line = line.replace("Provenance card generated by", "Provenance report generated by", 1)
        story.append(Paragraph(line, styles["body"]))
        idx += 1

    if not telemetry_inserted:
        story.append(Paragraph("Workflow-level Telemetry Summary", styles["h2"]))
        story.append(telemetry_table)
        story.append(Spacer(1, 0.12 * inch))
    if not executive_plots_inserted and plot_paths:
        story.append(PageBreak())
        story.append(Paragraph("Plots", styles["h2_tight"] if "h2_tight" in styles else styles["h2"]))
        for plot_idx, plot_path in enumerate(plot_paths):
            story.append(Image(str(plot_path), width=6.8 * inch, height=2.7 * inch))
            story.append(Spacer(1, 0.14 * inch))
            if plot_idx < len(plot_paths) - 1:
                story.append(Spacer(1, 0.06 * inch))

    return story


def _build_pdf_document(
    markdown_text: str,
    plot_paths: List[Path],
    telemetry_overview: Dict[str, Any],
    workflow_title: str,
    output_path: Path,
    object_records: List[Dict[str, Any]] | None = None,
) -> None:
    """Create the final PDF with markdown content and ending executive plots."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import LETTER
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

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
        "subtitle": ParagraphStyle(
            "subtitle", parent=base["Normal"], fontSize=10, textColor=colors.HexColor("#334155")
        ),
        "h2": ParagraphStyle("h2", parent=base["Heading2"], fontSize=13, textColor=colors.HexColor("#0f172a")),
        "h2_tight": ParagraphStyle(
            "h2_tight",
            parent=base["Heading2"],
            fontSize=13,
            textColor=colors.HexColor("#0f172a"),
            spaceBefore=0,
            spaceAfter=4,
        ),
        "h3": ParagraphStyle("h3", parent=base["Heading3"], fontSize=11, textColor=colors.HexColor("#0f172a")),
        "body": ParagraphStyle(
            "body", parent=base["Normal"], fontSize=9, leading=12, textColor=colors.HexColor("#111827")
        ),
        "b1": ParagraphStyle("b1", parent=base["Normal"], fontSize=9, leading=12, leftIndent=10, bulletIndent=0),
        "b2": ParagraphStyle("b2", parent=base["Normal"], fontSize=9, leading=12, leftIndent=24, bulletIndent=12),
        "b3": ParagraphStyle("b3", parent=base["Normal"], fontSize=9, leading=12, leftIndent=38, bulletIndent=24),
        "obj_detail": ParagraphStyle(
            "obj_detail",
            parent=base["Normal"],
            fontSize=9,
            leading=12,
            leftIndent=62,
            bulletIndent=0,
            textColor=colors.HexColor("#111827"),
        ),
        "mono": ParagraphStyle(
            "mono",
            parent=base["Code"],
            fontName="Courier",
            fontSize=7.5,
            leading=9,
            textColor=colors.HexColor("#111827"),
        ),
        "mono_indent": ParagraphStyle(
            "mono_indent",
            parent=base["Code"],
            fontName="Courier",
            fontSize=7.5,
            leading=9,
            leftIndent=62,
            textColor=colors.HexColor("#111827"),
        ),
    }

    telemetry_rows = _build_telemetry_table(telemetry_overview)
    telemetry_table = _build_wrapped_table(telemetry_rows, [2.8 * inch, 3.8 * inch], styles, font_size=9)

    story = [
        Paragraph(workflow_title, styles["title"]),
        Paragraph("Workflow Provenance Report", styles["subtitle"]),
        Spacer(1, 0.08 * inch),
    ]
    story.extend(_markdown_to_story(markdown_text, styles, telemetry_table, plot_paths, object_records=object_records))

    doc.build(story)


def render_provenance_report_pdf(
    dataset: Dict[str, Any],
    activities: List[Dict[str, Any]],
    object_summary: Dict[str, Any],
    output_path: Path,
) -> Dict[str, Any]:
    """Render a provenance report PDF and include executive plots."""
    try:
        import matplotlib
        import reportlab  # noqa: F401
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "PDF report generation requires optional dependencies. Install with: pip install flowcept[report_pdf]"
        ) from e
    # Force non-interactive backend for deterministic headless rendering in CI/batch environments.
    matplotlib.use("Agg", force=True)

    with TemporaryDirectory(prefix="flowcept_report_pdf_") as tmp_dir:
        tmp = Path(tmp_dir)
        md_path = tmp / "PROVENANCE_CARD.md"
        markdown_stats = render_provenance_card_markdown(
            dataset=dataset,
            activities=activities,
            object_summary=object_summary,
            output_path=md_path,
        )
        markdown_text = md_path.read_text(encoding="utf-8")
        workflow = dataset.get("workflow", {}) if isinstance(dataset.get("workflow"), dict) else {}
        workflow_title = str(workflow.get("name") or workflow.get("workflow_id") or "Workflow")

        charts = _build_plot_data(activities=activities, tasks=dataset.get("tasks", []))
        plot_paths: List[Path] = []
        ml_plot_spec = _build_ml_learning_plot_spec(dataset)
        if ml_plot_spec is not None:
            ml_plot_path = tmp / "plot_1_ml_line.png"
            _render_ml_line_plot(ml_plot_spec, ml_plot_path)
            plot_paths.append(ml_plot_path)
        for idx, (title, labels, values, y_label) in enumerate(charts):
            plot_path = tmp / f"plot_{idx + 2 if ml_plot_spec is not None else idx + 1}.png"
            _render_bar_plot(title=title, labels=labels, values=values, y_label=y_label, output_path=plot_path)
            plot_paths.append(plot_path)

        telemetry_overview = _extract_telemetry_overview(dataset.get("tasks", []))
        _build_pdf_document(
            markdown_text=markdown_text,
            plot_paths=plot_paths,
            telemetry_overview=telemetry_overview,
            workflow_title=workflow_title,
            output_path=output_path,
            object_records=dataset.get("objects", []),
        )
        return {**markdown_stats, "plots": len(plot_paths)}
