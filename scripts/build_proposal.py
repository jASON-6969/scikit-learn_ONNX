"""Render docs/proposal.md to a repeatable proposal PDF using ReportLab.

Run from any directory:
    python scripts/build_proposal.py
Requires reportlab; this document-only tool is separate from the benchmark.
"""

from __future__ import annotations

import argparse
import html
import re
from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Flowable,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
INK = colors.HexColor("#182432")
BLUE = colors.HexColor("#225F86")
PALE = colors.HexColor("#EEF4F8")


def inline(text: str) -> str:
    """Escape Markdown text and implement the small inline subset we use."""
    escaped = html.escape(text, quote=False)
    escaped = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", escaped)
    escaped = re.sub(r"(?<!\*)\*([^*]+?)\*(?!\*)", r"<i>\1</i>", escaped)
    escaped = re.sub(r"`([^`]+)`", r'<font name="Courier" size="8.4">\1</font>', escaped)
    return escaped


def make_styles() -> dict[str, ParagraphStyle]:
    """Provide consistent, compact styles for the four-page document."""
    base = getSampleStyleSheet()
    body = ParagraphStyle(
        "ProposalBody", parent=base["BodyText"], fontName="Helvetica",
        fontSize=9.35, leading=13.15, textColor=INK, spaceAfter=7.0,
        splitLongWords=True, alignment=TA_LEFT,
    )
    return {
        "body": body,
        "title": ParagraphStyle(
            "ProposalTitle", parent=body, fontName="Helvetica-Bold",
            fontSize=19, leading=23, textColor=INK, spaceAfter=12,
        ),
        "h2": ParagraphStyle(
            "ProposalH2", parent=body, fontName="Helvetica-Bold",
            fontSize=12.5, leading=16, textColor=BLUE,
            spaceBefore=8, spaceAfter=9, keepWithNext=True,
        ),
        "h3": ParagraphStyle(
            "ProposalH3", parent=body, fontName="Helvetica-Bold",
            fontSize=10.1, leading=14, spaceBefore=5, spaceAfter=6,
            keepWithNext=True,
        ),
        "bullet": ParagraphStyle(
            "ProposalBullet", parent=body, leftIndent=10,
            firstLineIndent=-8, spaceAfter=5,
        ),
        "table": ParagraphStyle(
            "ProposalTable", parent=body, fontSize=8.65,
            leading=11.6, spaceAfter=0,
        ),
        "reference": ParagraphStyle(
            "ProposalReference", parent=body, fontSize=8.5,
            leading=11.6, spaceAfter=9,
        ),
    }


def make_table(lines: list[str], styles: dict[str, ParagraphStyle], width: float) -> Table:
    """Render a Markdown table with wrapping and repeating column headings."""
    rows = [[cell.strip() for cell in line.strip().strip("|").split("|")] for line in lines]
    rows = [row for row in rows if not all(re.fullmatch(r":?-+:?", cell) for cell in row)]
    count = len(rows[0])
    ratios = [0.23, 0.49, 0.28] if count == 3 else [0.29, 0.71]
    if "Main responsibility" in rows[0]:
        ratios = [0.22, 0.42, 0.36]
    if "Internal target" in rows[0]:
        ratios = [0.24, 0.32, 0.44]
    data = [[Paragraph(inline(cell), styles["table"]) for cell in row] for row in rows]
    table = Table(data, colWidths=[width * ratio for ratio in ratios], repeatRows=1, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), PALE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LINEBELOW", (0, 0), (-1, 0), 0.7, BLUE),
        ("LINEBELOW", (0, 1), (-1, -1), 0.3, colors.HexColor("#CED8E0")),
    ]))
    return table


def parse_markdown(source: str, styles: dict[str, ParagraphStyle], width: float) -> list[Flowable]:
    """Translate controlled headings, paragraphs, bullets and tables."""
    flowables: list[Flowable] = []
    lines = source.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        if line == "<!-- pagebreak -->":
            flowables.append(PageBreak())
            i += 1
            continue
        if line.startswith("|"):
            table_lines = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                table_lines.append(lines[i])
                i += 1
            flowables.extend([make_table(table_lines, styles, width), Spacer(1, 9)])
            continue
        heading = re.match(r"^(#{1,3}) (.+)$", line)
        if heading:
            kind = {1: "title", 2: "h2", 3: "h3"}[len(heading[1])]
            flowables.append(Paragraph(inline(heading[2]), styles[kind]))
            i += 1
            continue
        if line.startswith("- "):
            flowables.append(Paragraph("&#8226; " + inline(line[2:]), styles["bullet"]))
            i += 1
            continue
        paragraph = [line]
        i += 1
        while i < len(lines) and lines[i].strip() and not lines[i].startswith(("#", "|", "- ", "<!--")):
            paragraph.append(lines[i].strip())
            i += 1
        text = " ".join(paragraph)
        style = styles["reference"] if re.match(r"^\[\d+\]", text) else styles["body"]
        flowables.append(Paragraph(inline(text), style))
    return flowables


def decorate_page(canvas: Any, doc: Any) -> None:
    """Draw restrained running headers and page numbers inside margins."""
    canvas.saveState()
    page_width, page_height = A4
    canvas.setFillColor(BLUE)
    canvas.setFont("Helvetica-Bold", 8)
    canvas.drawString(18 * mm, page_height - 13 * mm, "COMP 4330SEF  |  PROJECT PROPOSAL")
    canvas.setStrokeColor(colors.HexColor("#CED8E0"))
    canvas.setLineWidth(0.4)
    canvas.line(18 * mm, 15 * mm, page_width - 18 * mm, 15 * mm)
    canvas.setFillColor(colors.HexColor("#617182"))
    canvas.setFont("Helvetica", 8)
    canvas.drawString(18 * mm, 10 * mm, "14 October 2026  |  CPU inference study")
    canvas.drawRightString(page_width - 18 * mm, 10 * mm, str(doc.page))
    canvas.restoreState()


def build(source: Path, output: Path) -> Path:
    """Build the PDF from the editable Markdown source."""
    output.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(output), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
        topMargin=22 * mm, bottomMargin=20 * mm,
        title="Evaluating CPU Inference Performance of scikit-learn and ONNX Runtime",
        author="COMP 4330SEF Project Team",
        subject="Project proposal and controlled CPU inference benchmark methodology",
    )
    styles = make_styles()
    story = parse_markdown(source.read_text(encoding="utf-8"), styles, doc.width)
    doc.build(story, onFirstPage=decorate_page, onLaterPages=decorate_page)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "docs" / "proposal.md")
    parser.add_argument("--output", type=Path, default=ROOT / "output" / "pdf" / "project_proposal.pdf")
    args = parser.parse_args()
    print(build(args.source.resolve(), args.output.resolve()))


if __name__ == "__main__":
    main()
