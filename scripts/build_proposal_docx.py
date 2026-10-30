"""Build an editable Word proposal from its maintained Markdown source."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Mm, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]


def add_inline(paragraph: Any, text: str) -> None:
    """Preserve emphasis while keeping all content as native editable runs."""
    for token in re.split(r"(\*\*.*?\*\*|`[^`]+`|\*[^*]+\*)", text):
        if not token:
            continue
        if token.startswith("**") and token.endswith("**"):
            paragraph.add_run(token[2:-2]).bold = True
        elif token.startswith("`") and token.endswith("`"):
            run = paragraph.add_run(token[1:-1])
            run.font.name = "Consolas"
            run.font.size = Pt(9)
        elif token.startswith("*") and token.endswith("*"):
            paragraph.add_run(token[1:-1]).italic = True
        else:
            paragraph.add_run(token)


def configure_document(document: Any) -> None:
    section = document.sections[0]
    section.page_width = Mm(210)
    section.page_height = Mm(297)
    section.top_margin = section.bottom_margin = Mm(18)
    section.left_margin = section.right_margin = Mm(18)
    section.footer_distance = Mm(8)
    # The default Word template carries a blue border in its Title style.
    # Remove paragraph borders before applying the document's own styles.
    for style in document.styles:
        for border in style.element.findall(".//" + qn("w:pBdr")):
            border.getparent().remove(border)
    for name in ("Normal", "Title", "Subtitle", "Heading 1", "Heading 2", "List Bullet"):
        style = document.styles[name]
        style.font.name = "Calibri"
        style.font.color.rgb = RGBColor(0, 0, 0)
        style.paragraph_format.line_spacing = 1.07
        style.paragraph_format.space_after = Pt(5)
        style.paragraph_format.widow_control = True
    document.styles["Normal"].font.size = Pt(10)
    document.styles["Title"].font.size = Pt(20)
    document.styles["Title"].paragraph_format.space_after = Pt(10)
    for name, size in (("Heading 1", 13), ("Heading 2", 10.5)):
        style = document.styles[name]
        style.font.size = Pt(size)
        style.font.bold = True
        style.paragraph_format.space_before = Pt(7)
        style.paragraph_format.space_after = Pt(6)
        style.paragraph_format.keep_with_next = True
    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    field = OxmlElement("w:fldSimple")
    field.set(qn("w:instr"), "PAGE")
    footer._p.append(field)
    document.core_properties.title = "Evaluating CPU Inference Performance of scikit-learn and ONNX Runtime"
    document.core_properties.subject = "COMP 4330SEF project proposal"
    document.core_properties.author = ""


def add_table(document: Any, lines: list[str]) -> None:
    rows = [[cell.strip() for cell in line.strip().strip("|").split("|")] for line in lines]
    rows = [row for row in rows if not all(re.fullmatch(r":?-+:?", cell) for cell in row)]
    widths = [0.29, 0.71] if len(rows[0]) == 2 else [0.23, 0.49, 0.28]
    if "Main responsibility" in rows[0]:
        widths = [0.22, 0.42, 0.36]
    if "Internal target" in rows[0]:
        widths = [0.24, 0.32, 0.44]
    table = document.add_table(rows=len(rows), cols=len(rows[0]))
    table.autofit = False
    properties = table._tbl.tblPr
    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        border = OxmlElement("w:" + edge)
        for key, value in (("val", "single"), ("sz", "4"), ("color", "D9D9D9")):
            border.set(qn("w:" + key), value)
        borders.append(border)
    properties.append(borders)
    margins = OxmlElement("w:tblCellMar")
    for edge, width in (("top", 80), ("bottom", 80), ("left", 110), ("right", 110)):
        margin = OxmlElement("w:" + edge)
        margin.set(qn("w:w"), str(width))
        margin.set(qn("w:type"), "dxa")
        margins.append(margin)
    properties.append(margins)
    for column, fraction in zip(table.columns, widths):
        column.width = Mm(174 * fraction)
    for row_index, values in enumerate(rows):
        row = table.rows[row_index]
        cant_split = OxmlElement("w:cantSplit")
        row._tr.get_or_add_trPr().append(cant_split)
        if row_index == 0:
            row._tr.get_or_add_trPr().append(OxmlElement("w:tblHeader"))
        for column_index, value in enumerate(values):
            cell = row.cells[column_index]
            cell.width = Mm(174 * widths[column_index])
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            paragraph = cell.paragraphs[0]
            paragraph.paragraph_format.space_after = Pt(0)
            paragraph.paragraph_format.line_spacing = 1.05
            add_inline(paragraph, value)
            for run in paragraph.runs:
                run.font.size = Pt(9)
                run.bold = row_index == 0 or run.bold
                run.font.color.rgb = RGBColor(0, 0, 0)
            if row_index == 0:
                shading = OxmlElement("w:shd")
                shading.set(qn("w:fill"), "E7EEF4")
                cell._tc.get_or_add_tcPr().append(shading)
    space = document.add_paragraph()
    space.paragraph_format.space_after = Pt(2)
    space.paragraph_format.line_spacing = Pt(3)
    space.paragraph_format.space_before = Pt(0)


def build(source: Path, output: Path) -> Path:
    document = Document()
    configure_document(document)
    lines = source.read_text(encoding="utf-8").splitlines()
    index = 0
    pending_break = False
    while index < len(lines):
        text = lines[index].strip()
        index += 1
        if not text:
            continue
        if text == "<!-- pagebreak -->":
            pending_break = True
            continue
        if text.startswith("|"):
            table_lines = [text]
            while index < len(lines) and lines[index].strip().startswith("|"):
                table_lines.append(lines[index].strip())
                index += 1
            add_table(document, table_lines)
            continue
        heading = re.match(r"^(#{1,3}) (.+)$", text)
        if heading:
            style = {1: "Title", 2: "Heading 1", 3: "Heading 2"}[len(heading[1])]
            paragraph = document.add_paragraph(style=style)
            add_inline(paragraph, heading[2])
        elif text.startswith("- "):
            paragraph = document.add_paragraph(style="List Bullet")
            add_inline(paragraph, text[2:])
        else:
            parts = [text]
            while index < len(lines) and lines[index].strip() and not lines[index].startswith(("#", "|", "- ", "<!--")):
                parts.append(lines[index].strip())
                index += 1
            paragraph = document.add_paragraph()
            add_inline(paragraph, " ".join(parts))
            if re.match(r"^\[\d+\]", text):
                paragraph.paragraph_format.line_spacing = 1.05
                for run in paragraph.runs:
                    run.font.size = Pt(9)
        if pending_break:
            paragraph.paragraph_format.page_break_before = True
            pending_break = False
    output.parent.mkdir(parents=True, exist_ok=True)
    document.save(output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "docs" / "proposal.md")
    parser.add_argument("--output", type=Path, default=ROOT / "output" / "docx" / "project_proposal.docx")
    args = parser.parse_args()
    print(build(args.source.resolve(), args.output.resolve()))


if __name__ == "__main__":
    main()

