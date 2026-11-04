from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path

import pypdfium2 as pdfium
from docx import Document
from docx.oxml.ns import qn
from PIL import Image, ImageDraw
from pypdf import PdfReader


ROOT = Path(
    r"E:\桌面\博士课题\Data\data\processed\训练数据汇总"
    r"\城市水生态四科学问题_尺度响应分析_2026-07-23"
)
DOCX = ROOT / (
    r"10_结果解读报告"
    r"\0727_Scale-dependent divergence of water-quality grades and aquatic biotic "
    r"communities across rapidly urbanizing river–lake systems_13个递增range修订版_2026-07-29.docx"
)
XLSX = ROOT / (
    r"14_三类环境子变量×生物五家族尺度分解_2026-07-27"
    r"\环境子变量×生物五家族_13个递增range核心结果_2026-07-29.xlsx"
)
MD = ROOT / (
    r"10_结果解读报告"
    r"\城市水生态四科学问题_尺度响应分析与论文级解读报告.md"
)
PDF = ROOT / (
    r"14_三类环境子变量×生物五家族尺度分解_2026-07-27"
    r"\06_QA\Word_13range修订版\report.pdf"
)
QA_DIR = PDF.parent
PAGES_DIR = QA_DIR / "pages"
CONTACT_DIR = QA_DIR / "contact_sheets"
QA_JSON = QA_DIR / "13range_revision_final_qa.json"

RANGES = [
    200,
    500,
    1000,
    1500,
    2000,
    2500,
    3000,
    3500,
    4000,
    4500,
    5000,
    5500,
    6000,
]
FORBIDDEN = [
    "滨水区尺度",
    "中尺度",
    "景观尺度",
    "尺度域",
    "三尺度域",
    "三种背景色",
    "riparian zone",
    "intermediate and landscape extents",
    "spatial domains",
]


def iter_table_text(table):
    for row in table.rows:
        for cell in row.cells:
            yield cell.text


def docx_text_and_font_audit():
    doc = Document(DOCX)
    body_parts = [p.text for p in doc.paragraphs]
    for table in doc.tables:
        body_parts.extend(iter_table_text(table))
    body = "\n".join(body_parts)

    total_runs = 0
    latin_violations = []
    east_asia_violations = []
    paragraphs = list(doc.paragraphs)
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                paragraphs.extend(cell.paragraphs)
    for p_idx, paragraph in enumerate(paragraphs):
        for run in paragraph.runs:
            if not run.text.strip():
                continue
            total_runs += 1
            latin_font = run.font.name
            east_asia_font = run._element.rPr.rFonts.get(qn("w:eastAsia")) if run._element.rPr is not None else None
            if latin_font not in (None, "Times New Roman"):
                latin_violations.append({"paragraph": p_idx, "text": run.text[:80], "font": latin_font})
            if re.search(r"[\u3400-\u9fff]", run.text) and east_asia_font not in (None, "宋体"):
                east_asia_violations.append(
                    {"paragraph": p_idx, "text": run.text[:80], "font": east_asia_font}
                )

    return {
        "paragraphs": len(doc.paragraphs),
        "tables": len(doc.tables),
        "inline_shapes": len(doc.inline_shapes),
        "forbidden_hits": {term: body.lower().count(term.lower()) for term in FORBIDDEN if term.lower() in body.lower()},
        "all_13_ranges_present": all(str(value) in body for value in RANGES),
        "nonblank_runs": total_runs,
        "latin_font_violations": latin_violations[:20],
        "east_asia_font_violations": east_asia_violations[:20],
    }


def zipped_xml_text(path: Path) -> str:
    chunks = []
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            if name.endswith(".xml"):
                chunks.append(archive.read(name).decode("utf-8", errors="ignore"))
    return "\n".join(chunks)


def render_pdf():
    PAGES_DIR.mkdir(parents=True, exist_ok=True)
    CONTACT_DIR.mkdir(parents=True, exist_ok=True)
    for old_file in PAGES_DIR.glob("page-*.png"):
        old_file.unlink()
    for old_file in CONTACT_DIR.glob("contact-*.png"):
        old_file.unlink()

    pdf = pdfium.PdfDocument(str(PDF))
    page_paths = []
    for page_index in range(len(pdf)):
        page = pdf[page_index]
        bitmap = page.render(scale=1.35)
        image = bitmap.to_pil().convert("RGB")
        output = PAGES_DIR / f"page-{page_index + 1:02d}.png"
        image.save(output, quality=92)
        page_paths.append(output)

    thumb_width = 520
    rows_per_sheet = 2
    columns_per_sheet = 4
    pages_per_sheet = rows_per_sheet * columns_per_sheet
    contact_paths = []
    for start in range(0, len(page_paths), pages_per_sheet):
        batch = page_paths[start : start + pages_per_sheet]
        thumbs = []
        for page_number, page_path in enumerate(batch, start=start + 1):
            image = Image.open(page_path).convert("RGB")
            ratio = thumb_width / image.width
            thumb = image.resize((thumb_width, round(image.height * ratio)), Image.Resampling.LANCZOS)
            thumbs.append((page_number, thumb))
        cell_height = max(image.height for _, image in thumbs) + 38
        canvas = Image.new(
            "RGB",
            (columns_per_sheet * thumb_width, rows_per_sheet * cell_height),
            "white",
        )
        draw = ImageDraw.Draw(canvas)
        for batch_index, (page_number, thumb) in enumerate(thumbs):
            x = (batch_index % columns_per_sheet) * thumb_width
            y = (batch_index // columns_per_sheet) * cell_height
            canvas.paste(thumb, (x, y + 30))
            draw.text((x + 12, y + 8), f"Page {page_number}", fill="black")
        output = CONTACT_DIR / f"contact-{start // pages_per_sheet + 1:02d}.png"
        canvas.save(output)
        contact_paths.append(output)
    return len(page_paths), len(contact_paths)


def main():
    report = {
        "docx": str(DOCX),
        "xlsx": str(XLSX),
        "pdf": str(PDF),
        "files_exist": {
            "docx": DOCX.exists(),
            "xlsx": XLSX.exists(),
            "md": MD.exists(),
            "pdf": PDF.exists(),
        },
    }
    report["word"] = docx_text_and_font_audit()

    workbook_xml = zipped_xml_text(XLSX)
    report["workbook"] = {
        "forbidden_hits": {
            term: workbook_xml.lower().count(term.lower())
            for term in FORBIDDEN
            if term.lower() in workbook_xml.lower()
        },
        "all_13_ranges_present": all(str(value) in workbook_xml for value in RANGES),
    }

    md_text = MD.read_text(encoding="utf-8")
    report["markdown"] = {
        "forbidden_hits": {
            term: md_text.lower().count(term.lower())
            for term in FORBIDDEN
            if term.lower() in md_text.lower()
        },
        "all_13_ranges_present": all(str(value) in md_text for value in RANGES),
    }

    figure_scripts = [
        ROOT / r"11_脚本与QA\05_publication_figures.py",
        ROOT / r"11_脚本与QA\08_taxon_family_publication_figures.py",
        ROOT / r"11_脚本与QA\14_environment_subvariable_figures.py",
    ]
    script_hits = {}
    for script in figure_scripts:
        text = script.read_text(encoding="utf-8")
        hits = [term for term in ["axvspan", "DOMAIN_COLORS", *FORBIDDEN] if term.lower() in text.lower()]
        if hits:
            script_hits[script.name] = hits
    report["figure_scripts"] = {"checked": len(figure_scripts), "forbidden_hits": script_hits}

    report["pdf_page_count"] = len(PdfReader(str(PDF)).pages)
    rendered_pages, contact_sheets = render_pdf()
    report["rendered_pages"] = rendered_pages
    report["contact_sheets"] = contact_sheets

    checks = [
        all(report["files_exist"].values()),
        not report["word"]["forbidden_hits"],
        report["word"]["inline_shapes"] == 19,
        report["word"]["tables"] == 7,
        report["word"]["all_13_ranges_present"],
        not report["workbook"]["forbidden_hits"],
        report["workbook"]["all_13_ranges_present"],
        not report["markdown"]["forbidden_hits"],
        report["markdown"]["all_13_ranges_present"],
        not report["figure_scripts"]["forbidden_hits"],
        report["pdf_page_count"] == 41,
        report["rendered_pages"] == 41,
    ]
    report["status"] = "PASS" if all(checks) else "FAIL"
    QA_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
