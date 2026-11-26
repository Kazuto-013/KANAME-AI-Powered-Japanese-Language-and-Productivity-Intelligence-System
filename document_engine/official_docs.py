from __future__ import annotations

import datetime
import io
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt, RGBColor
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from document_engine.report_generator import GeneratedReport

_PDF_FONT = None
def _pdf_font() -> str:
    global _PDF_FONT
    if _PDF_FONT is None:
        try:
            pdfmetrics.registerFont(UnicodeCIDFont("HeiseiKakuGo-W5"))
            _PDF_FONT = "HeiseiKakuGo-W5"
        except Exception:
            _PDF_FONT = "Helvetica"
    return _PDF_FONT

def _today_str() -> str:
    return datetime.date.today().strftime("%Y-%m-%d")

def _sanitize_filename(s: str) -> str:
    return "".join(c for c in s if c.isalnum() or c in "-._") or "document"

def _to_dict(data: Any, **kwargs) -> dict[str, Any]:
    if isinstance(data, dict):
        d = dict(data)
    else:
        d = {}
    d.update(kwargs)
    return d

def _fmt_date_reiwa(date_str: str | None) -> str:
    if not date_str:
        return ""
    # Also accept slash-formatted dates e.g. 2026/04/01
    cleaned = str(date_str).strip().replace("/", "-").replace(".", "-")
    try:
        d = datetime.date.fromisoformat(cleaned)
    except Exception:
        return str(date_str) if date_str else ""
    reiwa_start = datetime.date(2019, 5, 1)
    if d >= reiwa_start:
        y = d.year - 2018
        return f"令和{y}年{d.month}月{d.day}日"
    return f"{d.year}年{d.month}月{d.day}日"

def _age_from_dob(dob: str | None) -> str:
    if not dob:
        return ""
    cleaned = str(dob).strip().replace("/", "-").replace(".", "-")
    try:
        d = datetime.date.fromisoformat(cleaned)
    except Exception:
        return ""
    today = datetime.date.today()
    age = today.year - d.year
    if (today.month, today.day) < (d.month, d.day):
        age -= 1
    age = max(age, 0)
    return f"{age}歳"
    today = datetime.date.today()
    age = today.year - d.year
    if (today.month, today.day) < (d.month, d.day):
        age -= 1
    return f"{age}歳"

def _gender_jp(g: str | None) -> str:
    return {"male": "男", "female": "女"}.get(g or "", "")

def _wrap_report(filename: str, fmt: str, payload: bytes) -> GeneratedReport:
    return GeneratedReport(
        filename=filename,
        format=fmt,
        bytes=payload,
        size_bytes=len(payload),
    )


# ============================================================
# 1. 履歴書 (Resume / Rirekisho)
# ============================================================

def _docx_set_cell_shading(cell, color_hex: str) -> None:
    """Apply shading (background colour) to a DOCX table cell."""
    try:
        tc_pr = cell._tc.get_or_add_tcPr()
        shd = tc_pr.makeelement(qn("w:shd"), {
            qn("w:fill"): color_hex.lstrip("#").upper(),
            qn("w:val"): "clear",
        })
        # Remove any existing shd children first to avoid duplicates
        for old in tc_pr.findall(qn("w:shd")):
            tc_pr.remove(old)
        tc_pr.append(shd)
    except Exception:
        pass


def _docx_set_cell_borders(cell, color_hex: str = "000000", sz: int = 4) -> None:
    """Fix merged-cell double-border artefacts: set a single unified border on all 4 sides."""
    try:
        tc_pr = cell._tc.get_or_add_tcPr()
        tc_borders = tc_pr.find(qn("w:tcBorders"))
        if tc_borders is None:
            tc_borders = tc_pr.makeelement(qn("w:tcBorders"), {})
            tc_pr.append(tc_borders)
        fill = color_hex.lstrip("#").upper()
        for side in ("top", "left", "bottom", "right"):
            tag = qn(f"w:{side}")
            child = tc_borders.find(tag)
            if child is None:
                child = tc_borders.makeelement(tag, {})
                tc_borders.append(child)
            child.set(qn("w:val"), "single")
            child.set(qn("w:sz"), str(int(sz)))
            child.set(qn("w:color"), fill)
            child.set(qn("w:space"), "0")
    except Exception:
        pass


def _rlh(color_hex: str) -> str:
    """Normalize a 6-char hex colour string for ReportLab colors.HexColor (requires # prefix)."""
    c = (color_hex or "FFFFFF").strip()
    if c.startswith("#"):
        return c
    return "#" + c


def _docx_insert_image_centered(cell, img_path: str, width_inches: float = 1.2, height_inches: float = 1.6):
    """Embed an image into a DOCX table cell; return True if image was inserted, False if fallback."""
    try:
        if not img_path:
            return False
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run()
        r.add_picture(img_path, width=Inches(width_inches), height=Inches(height_inches))
        return True
    except Exception:
        return False


def _docx_add_header_footer(section, header_left: str, header_center: str, header_right: str):
    """Add a simple DOCX header with left/center/right text. Footer: page number."""
    try:
        hp = section.header.paragraphs[0] if section.header.paragraphs else section.header.add_paragraph()
        hp.text = ""
        hp.alignment = WD_ALIGN_PARAGRAPH.LEFT
        if header_left or header_center or header_right:
            tab_stops = hp.paragraph_format.tab_stops
            try:
                from docx.enum.text import WD_TAB_ALIGNMENT
                try:
                    tab_stops.add_tab_stop(Cm(8.5), alignment=WD_TAB_ALIGNMENT.CENTER)
                    tab_stops.add_tab_stop(Cm(17.0), alignment=WD_TAB_ALIGNMENT.RIGHT)
                except Exception:
                    pass
            except Exception:
                pass
            run = hp.add_run(f"{header_left or ''}\t{header_center or ''}\t{header_right or ''}")
            run.font.size = Pt(8.5)
            run.font.color.rgb = RGBColor(0x6B, 0x72, 0x80)
        # Footer: page number marker
        fp = section.footer.paragraphs[0] if section.footer.paragraphs else section.footer.add_paragraph()
        fp.text = ""
        fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r2 = fp.add_run()
        r2.font.size = Pt(8.5)
        r2.font.color.rgb = RGBColor(0x6B, 0x72, 0x80)
        try:
            from docx.oxml import OxmlElement
            fld_begin = OxmlElement("w:fldChar")
            fld_begin.set(qn("w:fldCharType"), "begin")
            instr_text = OxmlElement("w:instrText")
            instr_text.set(qn("xml:space"), "preserve")
            instr_text.text = "PAGE   \\* MERGEFORMAT"
            fld_sep = OxmlElement("w:fldChar")
            fld_sep.set(qn("w:fldCharType"), "separate")
            fld_end = OxmlElement("w:fldChar")
            fld_end.set(qn("w:fldCharType"), "end")
            r2._r.append(fld_begin)
            r2._r.append(instr_text)
            r2._r.append(fld_sep)
            r2._r.append(fld_end)
        except Exception:
            r2.add_text("- 1 / 1 -")
    except Exception:
        pass


_EMPLOYMENT_LABEL = {
    "full_time": "正社員",
    "contract": "契約社員",
    "temp": "派遣社員",
    "outsourcing": "業務委託",
    "part_time": "アルバイト",
}
_MARITAL_LABEL = {
    "single": "未婚",
    "married": "既婚",
    "other": "その他",
}


def generate_resume_docx(data: dict[str, Any]) -> GeneratedReport:
    d = _to_dict(data)
    doc = Document()

    for section in doc.sections:
        section.top_margin = Cm(1.5)
        section.bottom_margin = Cm(1.5)
        section.left_margin = Cm(1.8)
        section.right_margin = Cm(1.8)

    style = doc.styles["Normal"]
    for cand in ("ＭＳ 明朝", "Yu Mincho", "Meiryo", "Arial Unicode MS"):
        try:
            style.font.name = cand
            style._element.rPr.rFonts.set(qn("w:eastAsia"), cand)
            break
        except Exception:
            continue
    style.font.size = Pt(10.5)

    name = d.get("full_name_kanji", "")
    name_kana = d.get("full_name_kana", "")
    today_ymd = d.get("issue_date") or _today_str()
    _docx_add_header_footer(
        doc.sections[0],
        header_left=name or "履歴書",
        header_center="履歴書",
        header_right=_fmt_date_reiwa(today_ymd),
    )

    # Title row
    title_table = doc.add_table(rows=1, cols=2)
    title_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    lc, rc = title_table.rows[0].cells
    lc.width = Cm(13)
    rc.width = Cm(4)
    p = lc.paragraphs[0]
    r = p.add_run("履歴書")
    r.bold = True
    r.font.size = Pt(22)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER

    rp = rc.paragraphs[0]
    rr = rp.add_run(_fmt_date_reiwa(today_ymd) + "\n提出 / 作成日")
    rr.font.size = Pt(10)
    rp.alignment = WD_ALIGN_PARAGRAPH.RIGHT

    doc.add_paragraph()

    # Personal info + photo grid: 6 rows × 3 cols (new row 5 = 家族構成)
    info_table = doc.add_table(rows=6, cols=3)
    info_table.style = "Table Grid"
    info_table.alignment = WD_TABLE_ALIGNMENT.CENTER

    for row in info_table.rows:
        row.cells[0].width = Cm(3.5)
        row.cells[1].width = Cm(9.5)
        row.cells[2].width = Cm(4)

    # Photo cell: merge rows 0..5 (6 rows)
    pc = info_table.rows[0].cells[2]
    for i in range(1, 6):
        pc = pc.merge(info_table.rows[i].cells[2])
    # Fix border artefacts on merged cell
    _docx_set_cell_borders(pc)
    _docx_set_cell_shading(pc, "FAFAFA")
    photo_ok = _docx_insert_image_centered(pc, d.get("photo_path"), 1.2, 1.6)
    if not photo_ok:
        # Clean fallback: rectangle placeholder with border + centered label
        pc.text = ""
        ph = pc.paragraphs[0]
        ph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        pr = ph.add_run("\n\n顔写真\n貼付欄\n")
        pr.font.size = Pt(9)
        pr.font.color.rgb = RGBColor(0x94, 0xA3, 0xB8)
        pr.bold = True

    # Row 0: Furigana
    info_table.rows[0].cells[0].text = "ふりがな"
    info_table.rows[0].cells[1].text = name_kana or ""

    # Row 1: 氏名
    info_table.rows[1].cells[0].text = "氏名"
    p = info_table.rows[1].cells[1].paragraphs[0]
    r = p.add_run(name or "")
    r.bold = True
    r.font.size = Pt(16)

    # Row 2: 生年月日 年齢 性別 婚姻
    dob = d.get("date_of_birth")
    age = _age_from_dob(dob)
    gender = _gender_jp(d.get("gender"))
    marital = _MARITAL_LABEL.get(d.get("marital_status"), "")
    row2_txt = f"{_fmt_date_reiwa(dob)}  年齢：{age}  性別：{gender}"
    if marital:
        row2_txt = row2_txt + f"  婚姻：{marital}"
    info_table.rows[2].cells[0].text = "生年月日"
    info_table.rows[2].cells[1].text = row2_txt

    # Row 3: 〒住所
    pc_full = f"〒{d.get('postal_code', '')}\n{d.get('address_prefecture', '')}{d.get('address_city', '')}{d.get('address_detail', '')}"
    info_table.rows[3].cells[0].text = "〒住所"
    info_table.rows[3].cells[1].text = pc_full.strip()

    # Row 4: 連絡先
    info_table.rows[4].cells[0].text = "連絡先"
    phone = d.get("phone", "") or ""
    email = d.get("email", "") or ""
    info_table.rows[4].cells[1].text = f"電話：{phone}\nメール：{email}"

    # Row 5: 家族構成 (NEW)
    info_table.rows[5].cells[0].text = "家族構成"
    fam_rows = []
    fam_list = d.get("family_members") or []
    if fam_list:
        for m in fam_list:
            rel = m.get("relationship") if isinstance(m, dict) else str(m)
            mn = m.get("name") if isinstance(m, dict) else ""
            mage = m.get("age") if isinstance(m, dict) else None
            dep = m.get("is_dependent") if isinstance(m, dict) else False
            parts = [rel]
            if mn:
                parts.append(str(mn))
            if mage is not None and str(mage) != "":
                parts.append(f"{mage}歳")
            if dep:
                parts.append("扶養")
            fam_rows.append("・".join(parts))
    else:
        fam_rows.append("（なし）")
    info_table.rows[5].cells[1].text = "\n".join(fam_rows)

    doc.add_paragraph()

    def _section_table(header_jp: str, rows_data: list[tuple[str, str]], append_ijou: bool = True):
        doc.add_paragraph(header_jp)
        t = doc.add_table(rows=1, cols=2)
        t.style = "Table Grid"
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        t.rows[0].cells[0].width = Cm(4)
        t.rows[0].cells[1].width = Cm(13)
        t.rows[0].cells[0].text = "年月"
        t.rows[0].cells[1].text = header_jp
        _docx_set_cell_shading(t.rows[0].cells[0], "F0F0F0")
        _docx_set_cell_shading(t.rows[0].cells[1], "F0F0F0")
        for y, c in rows_data:
            r = t.add_row().cells
            r[0].text = y
            r[1].text = c
        if append_ijou:
            r = t.add_row().cells
            r[0].text = "以上"
            r[1].text = "以上"
            _docx_set_cell_shading(r[0], "FCFCFC")
            _docx_set_cell_shading(r[1], "FCFCFC")

    # 学歴
    edu_rows = []
    for e in d.get("education_list", []) or []:
        sd = _fmt_date_reiwa(e.get("start_date"))
        ed = _fmt_date_reiwa(e.get("end_date"))
        school = e.get("school_name", "")
        dept = e.get("department") or ""
        deg = e.get("degree") or ""
        grad = e.get("is_graduated", True)
        text_start = f"{sd}  {school}{' ' + dept if dept else ''}{' ' + deg if deg else ''} 入学"
        edu_rows.append((sd.split("日")[0] if "日" in sd else sd, text_start))
        if ed or grad:
            text_end = f"{ed}  {school}{' ' + dept if dept else ''}{' ' + deg if deg else ''} 卒業" if grad else f"{ed or '現在に至る'}  在学中"
            edu_rows.append((ed.split("日")[0] if "日" in ed else (ed or "現在"), text_end))
    _section_table("学歴", edu_rows, append_ijou=True)

    doc.add_paragraph()

    # 職歴
    wh_rows = []
    for w in d.get("work_history_list", []) or []:
        sd = _fmt_date_reiwa(w.get("start_date"))
        ed = _fmt_date_reiwa(w.get("end_date"))
        comp = w.get("company_name", "")
        pos = w.get("position") or ""
        desc = w.get("description") or ""
        emp = _EMPLOYMENT_LABEL.get(str(w.get("employment_type") or "").strip()) or ""
        cur = w.get("is_current", False)
        resign = w.get("resign_reason") or ""
        start_txt_parts = [comp]
        if pos:
            start_txt_parts.append(pos)
        if emp:
            start_txt_parts.append(f"（{emp}）")
        if desc:
            start_txt_parts.append(desc)
        start_txt = f"{sd}  {'  '.join(start_txt_parts)} 入社"
        wh_rows.append((sd.split("日")[0] if "日" in sd else sd, start_txt))
        if cur:
            wh_rows.append(("現在に至る", f"{'  '.join(start_txt_parts[0:2] or [comp])} 在職中"))
        elif ed:
            end_txt = f"{ed}  {comp} 退社"
            if resign:
                end_txt = end_txt + f"  （退職理由：{resign}）"
            wh_rows.append((ed.split("日")[0] if "日" in ed else ed, end_txt))
    _section_table("職歴", wh_rows, append_ijou=True)

    doc.add_paragraph()

    # 免許・資格
    lic_rows = []
    for l in d.get("licenses_list", []) or []:
        ad = _fmt_date_reiwa(l.get("acquisition_date"))
        ln = l.get("license_name", "")
        iss = l.get("issuer") or ""
        lic_rows.append((ad.split("日")[0] if "日" in ad else ad, f"{ln}{'（' + iss + '）' if iss else ''}"))
    _section_table("免許・資格", lic_rows, append_ijou=True)

    doc.add_paragraph()

    # 志望動機 etc text boxes
    def _text_box(label: str, content: str | None, lines: int = 6):
        p = doc.add_paragraph()
        p.add_run(label).bold = True
        t = doc.add_table(rows=1, cols=1)
        t.style = "Table Grid"
        cell = t.rows[0].cells[0]
        cell.text = content or ""
        _docx_set_cell_borders(cell)
        text_len = len(cell.text.split("\n")) if cell.text else 0
        for _ in range(max(0, lines - text_len)):
            cell.add_paragraph("")

    _text_box("志望の動機", d.get("motivation"), 8)
    _text_box("趣味・特技", d.get("hobbies"), 4)
    _text_box("自己PR", d.get("self_pr"), 8)
    _text_box("本人希望記入欄（特に給料・職種・勤務時間・勤務地・その他についての希望などがあれば記入）", d.get("requests"), 6)

    buff = io.BytesIO()
    doc.save(buff)
    payload = buff.getvalue()

    fname = f"履歴書_{_sanitize_filename(name or _today_str())}.docx"
    return _wrap_report(fname, "docx", payload)


def generate_resume_pdf(data: dict[str, Any]) -> GeneratedReport:
    d = _to_dict(data)
    buff = io.BytesIO()
    doc = SimpleDocTemplate(
        buff, pagesize=A4,
        leftMargin=18*mm, rightMargin=18*mm,
        topMargin=20*mm, bottomMargin=20*mm,
        title="履歴書",
    )
    font = _pdf_font()
    styles = getSampleStyleSheet()

    sTitle = ParagraphStyle("sTitle", parent=styles["Title"], fontName=font, fontSize=22, leading=28, alignment=TA_CENTER)
    sNormal = ParagraphStyle("sNormal", parent=styles["BodyText"], fontName=font, fontSize=10.5, leading=15)
    sBold = ParagraphStyle("sBold", parent=sNormal, fontName=font, fontSize=10.5, leading=15)
    sLabel = ParagraphStyle("sLabel", parent=sNormal, fontName=font, fontSize=9, leading=13, textColor=colors.HexColor("#333333"))
    sNameBig = ParagraphStyle("sNameBig", parent=sNormal, fontName=font, fontSize=16, leading=20)
    sRight = ParagraphStyle("sRight", parent=sNormal, fontName=font, fontSize=10, leading=13, alignment=TA_RIGHT)
    sSec = ParagraphStyle("sSec", parent=sBold, fontSize=11, leading=16, spaceBefore=3*mm, spaceAfter=1*mm)
    sCell = ParagraphStyle("sCell", parent=sNormal, fontSize=10, leading=13)
    sPhoto = ParagraphStyle("sPhoto", parent=sCell, fontSize=9, leading=13, textColor=colors.HexColor("#94A3B8"), alignment=TA_CENTER)
    sGrayCell = ParagraphStyle("sGrayCell", parent=sCell, fontName=font, fontSize=10, leading=13)

    story = []

    name = d.get("full_name_kanji", "")
    name_kana = d.get("full_name_kana", "")
    today_ymd = d.get("issue_date") or _today_str()
    date_jp = _fmt_date_reiwa(today_ymd)

    # Header on every page (including photo rect placeholder drawing)
    def _on_page(canvas, doc_build):
        canvas.saveState()
        try:
            w, h = A4
            # Header line (top)
            canvas.setFont(font, 8)
            canvas.setFillColor(colors.HexColor("#6B7280"))
            canvas.drawString(18*mm, h - (10*mm), (name or "履歴書"))
            canvas.drawCentredString(w / 2.0, h - (10*mm), "履歴書")
            canvas.drawRightString(w - 18*mm, h - (10*mm), date_jp)
            # Footer: page x / n (use PAGE field-like via canvas.pageNumber)
            try:
                from reportlab.pdfgen import canvas as _c
                _ = _c
                canvas.drawCentredString(w / 2.0, 10*mm, f"- {doc_build.page} -")
            except Exception:
                pass
            # Photo placeholder rectangle (fixed location on page 1 if no image)
            if doc_build.page == 1 and (not d.get("photo_path") or not os.path.exists(str(d.get("photo_path") or ""))):
                # Draw rectangle at right-hand photo area coords
                x_photo = w - 18*mm - 40*mm
                y_photo_top = h - 20*mm - 10*mm  # right below header
                # photo box 40mm wide x 50mm tall
                canvas.setStrokeColor(colors.HexColor("#94A3B8"))
                canvas.setFillColor(colors.HexColor("#FAFAFA"))
                canvas.setLineWidth(0.5)
                canvas.rect(x_photo, y_photo_top - 50*mm, 40*mm, 50*mm, stroke=1, fill=1)
                canvas.setFillColor(colors.HexColor("#94A3B8"))
                canvas.setFont(font, 9)
                canvas.drawCentredString(x_photo + 20*mm, y_photo_top - 25*mm - 3, "顔写真")
                canvas.drawCentredString(x_photo + 20*mm, y_photo_top - 25*mm - 15, "貼付欄")
        except Exception:
            pass
        canvas.restoreState()

    title_row = [
        Paragraph("履歴書", sTitle),
        Paragraph(f"{date_jp}<br/>提出 / 作成日", sRight),
    ]
    tt = Table([title_row], colWidths=[130*mm, 40*mm])
    tt.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), font),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
    ]))
    story.append(tt)
    story.append(Spacer(1, 3*mm))

    # Personal info grid — 6 rows (NEW: row 5 = 家族構成). Row heights adjusted accordingly.
    # Photo goes into (2, 0) column — we either insert an Image flowable or a Paragraph placeholder.
    photo_cell_flowable = ""
    try:
        ph_path = d.get("photo_path") or None
        if ph_path and os.path.exists(str(ph_path)):
            from reportlab.platypus import Image as RLImage  # local import: optional
            try:
                photo_cell_flowable = RLImage(ph_path, width=30*mm, height=40*mm, hAlign="CENTER")
            except Exception:
                photo_cell_flowable = ""
    except Exception:
        photo_cell_flowable = ""
    if not photo_cell_flowable:
        # Leave cell empty; on-page handler will draw the placeholder rectangle
        photo_cell_flowable = Paragraph("", sPhoto)

    rows = []
    # row 0: furigana + name_kana + photo
    rows.append([
        Paragraph("ふりがな", sLabel),
        Paragraph(name_kana or "", sCell),
        photo_cell_flowable,
    ])
    rows.append([
        Paragraph("氏名", sSec),
        Paragraph(name or "", sNameBig),
        "",
    ])
    dob = d.get("date_of_birth")
    age = _age_from_dob(dob)
    gender = _gender_jp(d.get("gender"))
    marital = _MARITAL_LABEL.get(d.get("marital_status"), "")
    row2 = f"{_fmt_date_reiwa(dob)}  年齢：{age}  性別：{gender}"
    if marital:
        row2 = row2 + f"  婚姻：{marital}"
    rows.append([
        Paragraph("生年月日", sLabel),
        Paragraph(row2, sCell),
        "",
    ])
    pc_full = f"〒{d.get('postal_code', '')} {d.get('address_prefecture', '')}{d.get('address_city', '')}{d.get('address_detail', '')}"
    rows.append([
        Paragraph("〒住所", sLabel),
        Paragraph(pc_full.strip(), sCell),
        "",
    ])
    phone = d.get("phone", "") or ""
    email = d.get("email", "") or ""
    rows.append([
        Paragraph("連絡先", sLabel),
        Paragraph(f"電話：{phone}<br/>メール：{email}", sCell),
        "",
    ])
    # Family row (NEW)
    fam_parts = []
    fam_list = d.get("family_members") or []
    if fam_list:
        for m in fam_list:
            rel = m.get("relationship") if isinstance(m, dict) else str(m)
            mn = m.get("name") if isinstance(m, dict) else ""
            mage = m.get("age") if isinstance(m, dict) else None
            dep = m.get("is_dependent") if isinstance(m, dict) else False
            p_parts = [str(rel)]
            if mn:
                p_parts.append(str(mn))
            if mage is not None and str(mage) != "":
                p_parts.append(f"{mage}歳")
            if dep:
                p_parts.append("扶養")
            fam_parts.append("・".join(p_parts))
    if not fam_parts:
        fam_parts.append("（なし）")
    rows.append([
        Paragraph("家族構成", sLabel),
        Paragraph("<br/>".join(fam_parts), sCell),
        "",
    ])

    # row heights: furigana 8mm, name 14mm, dob 10mm, address 20mm, contact 14mm, family 20mm = 86mm total
    grid = Table(rows, colWidths=[35*mm, 95*mm, 40*mm], rowHeights=[8*mm, 14*mm, 10*mm, 20*mm, 14*mm, 20*mm])
    grid.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), font),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#111111")),
        ("SPAN", (2, 0), (2, 5)),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BACKGROUND", (2, 0), (2, 5), colors.HexColor("#FAFAFA")),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("ALIGN", (2, 0), (2, 5), "CENTER"),
        ("VALIGN", (2, 0), (2, 5), "MIDDLE"),
    ]))
    story.append(grid)
    story.append(Spacer(1, 3*mm))

    def _section_jp(title_jp: str, data_rows: list[tuple[str, str]], append_ijou: bool = True):
        story.append(Paragraph(title_jp, sSec))
        t_rows = [[Paragraph("年月", sBold), Paragraph(title_jp, sBold)]]
        for y, c in data_rows:
            t_rows.append([Paragraph(y or "", sCell), Paragraph(c or "", sCell)])
        if append_ijou:
            t_rows.append([Paragraph("以上", sGrayCell), Paragraph("以上", sGrayCell)])
        t = Table(t_rows, colWidths=[40*mm, 130*mm], repeatRows=1)
        style_commands = [
            ("FONTNAME", (0, 0), (-1, -1), font),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F0F0F0")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]
        if append_ijou:
            style_commands.append(("BACKGROUND", (0, -1), (-1, -1), colors.HexColor("#FCFCFC")))
        t.setStyle(TableStyle(style_commands))
        story.append(t)

    edu_rows = []
    for e in d.get("education_list", []) or []:
        sd = _fmt_date_reiwa(e.get("start_date"))
        ed = _fmt_date_reiwa(e.get("end_date"))
        school = e.get("school_name", "")
        dept = e.get("department") or ""
        deg = e.get("degree") or ""
        grad = e.get("is_graduated", True)
        edu_rows.append((sd.split("日")[0] if "日" in sd else sd, f"{sd}  {school}{' ' + dept if dept else ''}{' ' + deg if deg else ''} 入学"))
        if ed or grad:
            end_key = ed.split("日")[0] if "日" in ed else (ed or "現在")
            end_txt = f"{ed}  {school}{' ' + dept if dept else ''}{' ' + deg if deg else ''} 卒業" if grad else f"{ed or '現在に至る'}  在学中"
            edu_rows.append((end_key, end_txt))
    _section_jp("学歴", edu_rows, append_ijou=True)

    wh_rows = []
    for w in d.get("work_history_list", []) or []:
        sd = _fmt_date_reiwa(w.get("start_date"))
        ed = _fmt_date_reiwa(w.get("end_date"))
        comp = w.get("company_name", "")
        pos = w.get("position") or ""
        desc = w.get("description") or ""
        emp = _EMPLOYMENT_LABEL.get(str(w.get("employment_type") or "").strip()) or ""
        cur = w.get("is_current", False)
        resign = w.get("resign_reason") or ""
        parts = [comp]
        if pos:
            parts.append(pos)
        if emp:
            parts.append(f"（{emp}）")
        if desc:
            parts.append(desc)
        start_txt = f"{sd}  {'  '.join(parts)} 入社"
        wh_rows.append((sd.split("日")[0] if "日" in sd else sd, start_txt))
        if cur:
            wh_rows.append(("現在に至る", f"{'  '.join(parts[0:2] or [comp])} 在職中"))
        elif ed:
            end_txt = f"{ed}  {comp} 退社"
            if resign:
                end_txt = end_txt + f"  （退職理由：{resign}）"
            wh_rows.append((ed.split("日")[0] if "日" in ed else ed, end_txt))
    _section_jp("職歴", wh_rows, append_ijou=True)

    lic_rows = []
    for l in d.get("licenses_list", []) or []:
        ad = _fmt_date_reiwa(l.get("acquisition_date"))
        ln = l.get("license_name", "")
        iss = l.get("issuer") or ""
        lic_rows.append((ad.split("日")[0] if "日" in ad else ad, f"{ln}{'（' + iss + '）' if iss else ''}"))
    _section_jp("免許・資格", lic_rows, append_ijou=True)

    def _text_box_pdf(label: str, content: str | None, height_mm: float = 30):
        story.append(Paragraph(label, sSec))
        box = Table([[Paragraph(content or "", sCell)]], colWidths=[170*mm], rowHeights=[height_mm*mm])
        box.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (-1, -1), font),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#000000")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(box)

    _text_box_pdf("志望の動機", d.get("motivation"), 40)
    _text_box_pdf("趣味・特技", d.get("hobbies"), 20)
    _text_box_pdf("自己PR", d.get("self_pr"), 40)
    _text_box_pdf("本人希望記入欄（給料・職種・勤務時間・勤務地 他）", d.get("requests"), 30)

    doc.build(story, onFirstPage=_on_page, onLaterPages=_on_page)
    payload = buff.getvalue()
    fname = f"履歴書_{_sanitize_filename(name or _today_str())}.pdf"
    return _wrap_report(fname, "pdf", payload)


# ============================================================
# 2. 職務経歴書 (Career History)
# ============================================================

def generate_career_history_docx(data: dict[str, Any]) -> GeneratedReport:
    d = _to_dict(data)
    doc = Document()

    for section in doc.sections:
        section.top_margin = Cm(2)
        section.bottom_margin = Cm(2)
        section.left_margin = Cm(2)
        section.right_margin = Cm(2)

    style = doc.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(11)
    style._element.rPr.rFonts.set(qn("w:eastAsia"), "ＭＳ ゴシック")

    name = d.get("full_name_kanji", "")
    name_kana = d.get("full_name_kana", "")
    phone = d.get("phone", "") or ""
    email = d.get("email", "") or ""

    # Title
    tp = doc.add_heading("職務経歴書", level=0)
    tp.alignment = WD_ALIGN_PARAGRAPH.CENTER

    # Header block
    header_t = doc.add_table(rows=1, cols=2)
    header_t.alignment = WD_TABLE_ALIGNMENT.CENTER
    lc, rc = header_t.rows[0].cells
    lc.width = Cm(9)
    rc.width = Cm(8)
    lp = lc.paragraphs[0]
    lr = lp.add_run(f"{name}  {name_kana}")
    lr.bold = True
    lp.add_run(f"\n連絡先：{phone}\nメール：{email}")

    rp = rc.paragraphs[0]
    rp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    rp.add_run(f"作成日：{d.get('issue_date') or _today_str()}")

    doc.add_paragraph()

    # Career summary
    doc.add_heading("職務要約", level=1)
    summary = d.get("career_summary") or ""
    doc.add_paragraph(summary or "(職務要約はありません)")
    doc.add_paragraph()

    # Roles
    roles = d.get("roles", []) or []
    doc.add_heading("職務経歴", level=1)
    for idx, role in enumerate(roles, start=1):
        period_from = role.get("period_from", "")
        period_to = role.get("period_to") or "現在"
        company = role.get("company_name", "")
        position = role.get("position", "")
        dept = role.get("department") or ""
        overview = role.get("overview") or ""
        projects = role.get("projects", []) or []

        doc.add_heading(f"{idx}. {period_from} ～ {period_to}  {company}", level=2)
        p_info = doc.add_paragraph()
        p_info.add_run(f"部署：{dept}  役職：{position}").italic = True
        if overview:
            doc.add_paragraph(f"概要：{overview}")

        if projects:
            doc.add_paragraph("主なプロジェクト / 実績：")
            proj_table = doc.add_table(rows=1, cols=6)
            proj_table.style = "Light Grid Accent 1"
            hdr = proj_table.rows[0].cells
            hdr[0].text = "期間"
            hdr[1].text = "プロジェクト名"
            hdr[2].text = "役割"
            hdr[3].text = "規模"
            hdr[4].text = "技術"
            hdr[5].text = "成果"
            for pr in projects:
                row = proj_table.add_row().cells
                pf = pr.get("period_from") or ""
                pt = pr.get("period_to") or ""
                row[0].text = f"{pf} ～ {pt}".strip(" ～")
                row[1].text = pr.get("project_name", "")
                row[2].text = pr.get("role") or ""
                ts = pr.get("team_size")
                row[3].text = f"{ts}名" if ts else ""
                techs = pr.get("technologies") or []
                row[4].text = ", ".join(techs)
                achs = pr.get("achievements") or []
                ov = pr.get("overview") or ""
                all_txt = ([ov] if ov else []) + achs
                row[5].text = "\n".join(all_txt)
        doc.add_paragraph()

    # Skills
    doc.add_heading("保有スキル / 技術スタック", level=1)
    skills = d.get("skills", []) or []
    tech_stack = d.get("tech_stack", {}) or {}
    if skills:
        doc.add_paragraph("スキル：" + ", ".join(skills))
    if tech_stack:
        st_table = doc.add_table(rows=1, cols=2)
        st_table.style = "Light Grid Accent 1"
        st_table.rows[0].cells[0].text = "カテゴリ"
        st_table.rows[0].cells[1].text = "技術"
        for cat, items in tech_stack.items():
            r = st_table.add_row().cells
            r[0].text = cat
            r[1].text = ", ".join(items or [])
    doc.add_paragraph()

    # Education / Certifications
    edu = d.get("education")
    certs = d.get("certifications", []) or []
    if edu:
        doc.add_heading("学歴", level=1)
        doc.add_paragraph(edu)
    if certs:
        doc.add_heading("資格", level=1)
        for c in certs:
            doc.add_paragraph(f"・ {c}", style="List Bullet")

    buff = io.BytesIO()
    doc.save(buff)
    payload = buff.getvalue()
    fname = f"職務経歴書_{_sanitize_filename(name or _today_str())}.docx"
    return _wrap_report(fname, "docx", payload)


def generate_career_history_pdf(data: dict[str, Any]) -> GeneratedReport:
    d = _to_dict(data)
    buff = io.BytesIO()
    doc = SimpleDocTemplate(buff, pagesize=A4, leftMargin=20*mm, rightMargin=20*mm, topMargin=20*mm, bottomMargin=20*mm)
    font = _pdf_font()
    styles = getSampleStyleSheet()

    sTitle = ParagraphStyle("sTitle", parent=styles["Title"], fontName=font, fontSize=22, leading=28, alignment=TA_CENTER, spaceAfter=6*mm)
    sH1 = ParagraphStyle("sH1", parent=styles["Heading1"], fontName=font, fontSize=14, leading=18, spaceBefore=4*mm, spaceAfter=2*mm)
    sH2 = ParagraphStyle("sH2", parent=styles["Heading2"], fontName=font, fontSize=12, leading=16, spaceBefore=3*mm, spaceAfter=1*mm)
    sBody = ParagraphStyle("sBody", parent=styles["BodyText"], fontName=font, fontSize=10.5, leading=15)
    sItalic = ParagraphStyle("sItalic", parent=sBody, fontName=font, fontSize=10, leading=14, textColor=colors.HexColor("#555555"))
    sRight = ParagraphStyle("sRight", parent=sBody, alignment=TA_RIGHT)
    sCell = ParagraphStyle("sCell", parent=sBody, fontSize=9, leading=12)
    sHdr = ParagraphStyle("sHdr", parent=sCell, fontName=font, fontSize=9, leading=12, textColor=colors.white)

    story = []
    name = d.get("full_name_kanji", "")
    name_kana = d.get("full_name_kana", "")
    phone = d.get("phone", "") or ""
    email = d.get("email", "") or ""

    story.append(Paragraph("職務経歴書", sTitle))

    header = [
        [
            Paragraph(f"<b>{name}</b>  {name_kana}<br/>連絡先：{phone}<br/>メール：{email}", sBody),
            Paragraph(f"作成日：{d.get('issue_date') or _today_str()}", sRight),
        ]
    ]
    ht = Table(header, colWidths=[90*mm, 80*mm])
    ht.setStyle(TableStyle([("FONTNAME", (0, 0), (-1, -1), font), ("VALIGN", (0,0), (-1,-1), "TOP")]))
    story.append(ht)
    story.append(Spacer(1, 4*mm))

    story.append(Paragraph("職務要約", sH1))
    story.append(Paragraph(d.get("career_summary") or "(職務要約はありません)", sBody))
    story.append(Spacer(1, 3*mm))

    roles = d.get("roles", []) or []
    story.append(Paragraph("職務経歴", sH1))
    for idx, role in enumerate(roles, start=1):
        period_from = role.get("period_from", "")
        period_to = role.get("period_to") or "現在"
        company = role.get("company_name", "")
        position = role.get("position", "")
        dept = role.get("department") or ""
        overview = role.get("overview") or ""
        projects = role.get("projects", []) or []

        story.append(Paragraph(f"{idx}. {period_from} ～ {period_to}  {company}", sH2))
        story.append(Paragraph(f"部署：{dept}  役職：{position}", sItalic))
        if overview:
            story.append(Paragraph(f"概要：{overview}", sBody))
        if projects:
            story.append(Paragraph("主なプロジェクト / 実績：", sBody))
            rows_p = [[
                Paragraph("期間", sHdr),
                Paragraph("プロジェクト名", sHdr),
                Paragraph("役割", sHdr),
                Paragraph("規模", sHdr),
                Paragraph("技術", sHdr),
                Paragraph("成果", sHdr),
            ]]
            for pr in projects:
                pf = pr.get("period_from") or ""
                pt = pr.get("period_to") or ""
                ts = pr.get("team_size")
                techs = pr.get("technologies") or []
                achs = pr.get("achievements") or []
                ov = pr.get("overview") or ""
                all_txt = ([ov] if ov else []) + achs
                rows_p.append([
                    Paragraph(f"{pf} ～ {pt}".strip(" ～"), sCell),
                    Paragraph(pr.get("project_name", ""), sCell),
                    Paragraph(pr.get("role") or "", sCell),
                    Paragraph(f"{ts}名" if ts else "", sCell),
                    Paragraph(", ".join(techs), sCell),
                    Paragraph("<br/>".join(all_txt), sCell),
                ])
            pj = Table(rows_p, colWidths=[22*mm, 32*mm, 25*mm, 16*mm, 30*mm, 45*mm], repeatRows=1)
            pj.setStyle(TableStyle([
                ("FONTNAME", (0, 0), (-1, -1), font),
                ("GRID", (0,0), (-1,-1), 0.4, colors.grey),
                ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#2c3e50")),
                ("VALIGN", (0,0), (-1,-1), "TOP"),
                ("LEFTPADDING", (0,0), (-1,-1), 4),
                ("RIGHTPADDING", (0,0), (-1,-1), 4),
                ("TOPPADDING", (0,0), (-1,-1), 3),
                ("BOTTOMPADDING", (0,0), (-1,-1), 3),
            ]))
            story.append(pj)
        story.append(Spacer(1, 2*mm))

    # Skills
    story.append(Paragraph("保有スキル / 技術スタック", sH1))
    skills = d.get("skills", []) or []
    tech_stack = d.get("tech_stack", {}) or {}
    if skills:
        story.append(Paragraph("スキル：" + ", ".join(skills), sBody))
    if tech_stack:
        st_rows = [[Paragraph("カテゴリ", sHdr), Paragraph("技術", sHdr)]]
        for cat, items in tech_stack.items():
            st_rows.append([Paragraph(cat, sCell), Paragraph(", ".join(items or []), sCell)])
        st_t = Table(st_rows, colWidths=[40*mm, 130*mm], repeatRows=1)
        st_t.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (-1, -1), font),
            ("GRID", (0,0), (-1,-1), 0.4, colors.grey),
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#34495e")),
            ("VALIGN", (0,0), (-1,-1), "TOP"),
            ("LEFTPADDING", (0,0), (-1,-1), 4),
            ("TOPPADDING", (0,0), (-1,-1), 3),
        ]))
        story.append(st_t)

    edu = d.get("education")
    certs = d.get("certifications", []) or []
    if edu:
        story.append(Paragraph("学歴", sH1))
        story.append(Paragraph(edu, sBody))
    if certs:
        story.append(Paragraph("資格", sH1))
        for c in certs:
            story.append(Paragraph(f"・ {c}", sBody))

    doc.build(story)
    payload = buff.getvalue()
    fname = f"職務経歴書_{_sanitize_filename(name or _today_str())}.pdf"
    return _wrap_report(fname, "pdf", payload)


# ============================================================
# 3. 請求書 (Invoice)
# ============================================================

def _compute_invoice_totals(d: dict[str, Any]) -> tuple[float, float, float]:
    subtotal = d.get("subtotal")
    total_tax = d.get("total_tax")
    total = d.get("total_amount")
    if subtotal is None or total_tax is None or total is None:
        items = d.get("items", []) or []
        sub = 0.0
        tax_agg = 0.0
        for it in items:
            q = float(it.get("quantity", 1) or 0)
            u = float(it.get("unit_price", 0) or 0)
            tr = float(it.get("tax_rate", 0.10) or 0)
            amt = q * u
            sub += amt
            tax_agg += amt * tr
        if subtotal is None:
            subtotal = sub
        if total_tax is None:
            total_tax = tax_agg
        if total is None:
            total = subtotal + total_tax
    return float(subtotal or 0), float(total_tax or 0), float(total or 0)


def generate_invoice_docx(data: dict[str, Any]) -> GeneratedReport:
    d = _to_dict(data)
    doc = Document()
    for section in doc.sections:
        section.top_margin = Cm(2)
        section.bottom_margin = Cm(2)
        section.left_margin = Cm(2)
        section.right_margin = Cm(2)

    style = doc.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(11)
    style._element.rPr.rFonts.set(qn("w:eastAsia"), "ＭＳ ゴシック")

    # Top-right invoice meta
    meta_t = doc.add_table(rows=1, cols=2)
    lc, rc = meta_t.rows[0].cells
    lc.width = Cm(10)
    rc.width = Cm(7)

    # Left: Buyer info
    buyer_name = d.get("buyer_name", "")
    buyer_dept = d.get("buyer_department") or ""
    buyer_addr = d.get("buyer_address") or ""
    p = lc.paragraphs[0]
    p.add_run(f"{buyer_name}  ")
    r = p.add_run("御中" if not d.get("recipient_name") else f"{d.get('recipient_name', '')} 様")
    r.bold = True
    if buyer_dept:
        p.add_run(f"\n{buyer_dept}")
    if buyer_addr:
        p.add_run(f"\n{buyer_addr}")

    # Right: Invoice meta
    rp = rc.paragraphs[0]
    rp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    rp.add_run(f"請求書番号：{d.get('invoice_number', '')}\n")
    rp.add_run(f"発行日：{d.get('issue_date') or _today_str()}\n")
    rp.add_run(f"支払期日：{d.get('due_date') or ''}")

    doc.add_paragraph()

    # Seller block
    sell_t = doc.add_table(rows=1, cols=2)
    lc2, rc2 = sell_t.rows[0].cells
    lc2.width = Cm(7)
    rc2.width = Cm(10)

    # Title centered-ish (below seller/buyer blocks)
    seller_name = d.get("seller_name", "")
    seller_dept = d.get("seller_department") or ""
    seller_addr = d.get("seller_address") or ""
    seller_phone = d.get("seller_phone") or ""
    tax_id = d.get("seller_tax_id") or ""
    sp = lc2.paragraphs[0]
    sp.alignment = WD_ALIGN_PARAGRAPH.LEFT
    sp.add_run(f"登録番号：{tax_id}\n" if tax_id else "")
    sp.add_run("適格請求書発行事業者" if tax_id else "")

    rp2 = rc2.paragraphs[0]
    rp2.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    rp2.add_run(f"{seller_name}\n")
    if seller_dept:
        rp2.add_run(f"{seller_dept}\n")
    if seller_addr:
        rp2.add_run(f"{seller_addr}\n")
    if seller_phone:
        rp2.add_run(f"TEL：{seller_phone}")

    doc.add_paragraph()
    tp = doc.add_heading(d.get("title") or "請求書", level=0)
    tp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.add_paragraph()

    # Items table
    items = d.get("items", []) or []
    it_t = doc.add_table(rows=1, cols=6)
    it_t.style = "Light Grid Accent 1"
    it_t.alignment = WD_TABLE_ALIGNMENT.CENTER
    hdr = it_t.rows[0].cells
    hdr[0].text = "No."
    hdr[1].text = "品名"
    hdr[2].text = "数量"
    hdr[3].text = "単価"
    hdr[4].text = "税率"
    hdr[5].text = "金額"
    for i, it in enumerate(items, start=1):
        r = it_t.add_row().cells
        q = float(it.get("quantity", 1) or 0)
        u = float(it.get("unit_price", 0) or 0)
        tr = float(it.get("tax_rate", 0.10) or 0)
        amt = q * u
        r[0].text = str(i)
        r[1].text = it.get("description", "")
        r[2].text = f"{q:g}"
        r[3].text = f"\\{u:,.0f}"
        r[4].text = f"{int(tr*100)}%"
        r[5].text = f"\\{amt:,.0f}"

    doc.add_paragraph()

    subtotal, total_tax, total = _compute_invoice_totals(d)

    # Summary block right
    sum_t = doc.add_table(rows=3, cols=2)
    sum_t.alignment = WD_TABLE_ALIGNMENT.RIGHT
    rows_d = [
        ("小計", f"\\{subtotal:,.0f}"),
        ("消費税", f"\\{total_tax:,.0f}"),
        ("合計金額", f"\\{total:,.0f}"),
    ]
    for i, (lbl, val) in enumerate(rows_d):
        row = sum_t.rows[i].cells
        row[0].width = Cm(3)
        row[1].width = Cm(5)
        row[0].text = lbl
        p = row[1].paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        rr = p.add_run(val)
        if i == 2:
            rr.bold = True
            rr.font.size = Pt(13)
    doc.add_paragraph()

    notes = d.get("notes")
    if notes:
        doc.add_heading("特記事項", level=2)
        doc.add_paragraph(notes)
        doc.add_paragraph()

    # Bank transfer info
    if d.get("payment_method") == "bank_transfer" or d.get("bank_name"):
        doc.add_heading("支払方法（銀行振込）", level=2)
        bk = doc.add_table(rows=2, cols=5)
        bk.style = "Light Grid Accent 1"
        hbk = bk.rows[0].cells
        for j, lbl in enumerate(["銀行名", "支店", "口座種別", "口座番号", "名義"]):
            hbk[j].text = lbl
        rb = bk.rows[1].cells
        atypes = {"ordinary": "普通", "current": "当座"}
        rb[0].text = d.get("bank_name") or ""
        rb[1].text = d.get("bank_branch") or ""
        rb[2].text = atypes.get(d.get("bank_account_type") or "", d.get("bank_account_type") or "")
        rb[3].text = d.get("bank_account_number") or ""
        rb[4].text = d.get("bank_account_holder") or ""

    buff = io.BytesIO()
    doc.save(buff)
    payload = buff.getvalue()
    inv_no = d.get("invoice_number") or _today_str()
    fname = f"請求書_{_sanitize_filename(inv_no)}.docx"
    return _wrap_report(fname, "docx", payload)


def generate_invoice_pdf(data: dict[str, Any]) -> GeneratedReport:
    d = _to_dict(data)
    buff = io.BytesIO()
    doc = SimpleDocTemplate(buff, pagesize=A4, leftMargin=20*mm, rightMargin=20*mm, topMargin=20*mm, bottomMargin=20*mm)
    font = _pdf_font()
    styles = getSampleStyleSheet()

    sTitle = ParagraphStyle("sTitle", parent=styles["Title"], fontName=font, fontSize=22, leading=28, alignment=TA_CENTER, spaceAfter=6*mm)
    sH2 = ParagraphStyle("sH2", parent=styles["Heading2"], fontName=font, fontSize=12, leading=16, spaceBefore=3*mm, spaceAfter=1*mm)
    sBody = ParagraphStyle("sBody", parent=styles["BodyText"], fontName=font, fontSize=10.5, leading=15)
    sRight = ParagraphStyle("sRight", parent=sBody, alignment=TA_RIGHT)
    sBold = ParagraphStyle("sBold", parent=sBody, fontName=font, fontSize=10.5, leading=15)
    sCell = ParagraphStyle("sCell", parent=sBody, fontSize=10, leading=13)
    sHdr = ParagraphStyle("sHdr", parent=sCell, fontName=font, fontSize=10, leading=13, textColor=colors.white)
    sSum = ParagraphStyle("sSum", parent=sRight, fontName=font, fontSize=13, leading=17)

    story = []

    buyer_name = d.get("buyer_name", "")
    buyer_dept = d.get("buyer_department") or ""
    buyer_addr = d.get("buyer_address") or ""
    left_txt = f"<b>{buyer_name}</b>  {'御中' if not d.get('recipient_name') else d.get('recipient_name', '') + ' 様'}"
    if buyer_dept:
        left_txt += f"<br/>{buyer_dept}"
    if buyer_addr:
        left_txt += f"<br/>{buyer_addr}"

    right_txt = (
        f"請求書番号：{d.get('invoice_number', '')}<br/>"
        f"発行日：{d.get('issue_date') or _today_str()}<br/>"
        f"支払期日：{d.get('due_date') or ''}"
    )

    top = Table([[Paragraph(left_txt, sBody), Paragraph(right_txt, sRight)]], colWidths=[100*mm, 70*mm])
    top.setStyle(TableStyle([("FONTNAME", (0, 0), (-1, -1), font), ("VALIGN", (0,0), (-1,-1), "TOP")]))
    story.append(top)
    story.append(Spacer(1, 3*mm))

    seller_name = d.get("seller_name", "")
    seller_dept = d.get("seller_department") or ""
    seller_addr = d.get("seller_address") or ""
    seller_phone = d.get("seller_phone") or ""
    tax_id = d.get("seller_tax_id") or ""

    sell_left = ""
    if tax_id:
        sell_left += f"登録番号：{tax_id}<br/>適格請求書発行事業者"
    sell_right = f"{seller_name}"
    if seller_dept:
        sell_right += f"<br/>{seller_dept}"
    if seller_addr:
        sell_right += f"<br/>{seller_addr}"
    if seller_phone:
        sell_right += f"<br/>TEL：{seller_phone}"

    sell = Table([[Paragraph(sell_left, sBody), Paragraph(sell_right, sRight)]], colWidths=[70*mm, 100*mm])
    sell.setStyle(TableStyle([("VALIGN", (0,0), (-1,-1), "TOP")]))
    story.append(sell)
    story.append(Spacer(1, 2*mm))

    story.append(Paragraph(d.get("title") or "請求書", sTitle))

    items = d.get("items", []) or []
    rows_p = [[
        Paragraph("No.", sHdr),
        Paragraph("品名", sHdr),
        Paragraph("数量", sHdr),
        Paragraph("単価", sHdr),
        Paragraph("税率", sHdr),
        Paragraph("金額", sHdr),
    ]]
    for i, it in enumerate(items, start=1):
        q = float(it.get("quantity", 1) or 0)
        u = float(it.get("unit_price", 0) or 0)
        tr = float(it.get("tax_rate", 0.10) or 0)
        amt = q * u
        rows_p.append([
            Paragraph(str(i), sCell),
            Paragraph(it.get("description", ""), sCell),
            Paragraph(f"{q:g}", sCell),
            Paragraph(f"\\{u:,.0f}", sCell),
            Paragraph(f"{int(tr*100)}%", sCell),
            Paragraph(f"\\{amt:,.0f}", sCell),
        ])
    it_t = Table(rows_p, colWidths=[12*mm, 63*mm, 18*mm, 25*mm, 16*mm, 36*mm], repeatRows=1)
    it_t.setStyle(TableStyle([
        ("GRID", (0,0), (-1,-1), 0.4, colors.grey),
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#2c3e50")),
        ("ALIGN", (2,1), (5,-1), "RIGHT"),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("LEFTPADDING", (0,0), (-1,-1), 4),
        ("RIGHTPADDING", (0,0), (-1,-1), 4),
        ("TOPPADDING", (0,0), (-1,-1), 3),
        ("BOTTOMPADDING", (0,0), (-1,-1), 3),
    ]))
    story.append(it_t)
    story.append(Spacer(1, 4*mm))

    subtotal, total_tax, total = _compute_invoice_totals(d)
    sum_rows = [
        [Paragraph("小計", sBold), Paragraph(f"\\{subtotal:,.0f}", sRight)],
        [Paragraph("消費税", sBold), Paragraph(f"\\{total_tax:,.0f}", sRight)],
        [Paragraph("合計金額", sBold), Paragraph(f"<b>\\{total:,.0f}</b>", sSum)],
    ]
    st = Table(sum_rows, colWidths=[30*mm, 50*mm], hAlign="RIGHT")
    st.setStyle(TableStyle([
        ("GRID", (0,0), (-1,-1), 0.5, colors.black),
        ("BACKGROUND", (0,2), (-1,2), colors.HexColor("#f8f9fa")),
        ("SPAN", (0,0), (-1,-1)),
        ("LEFTPADDING", (0,0), (-1,-1), 6),
        ("RIGHTPADDING", (0,0), (-1,-1), 6),
        ("TOPPADDING", (0,0), (-1,-1), 4),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
    ]))
    story.append(st)
    story.append(Spacer(1, 3*mm))

    notes = d.get("notes")
    if notes:
        story.append(Paragraph("特記事項", sH2))
        story.append(Paragraph(notes, sBody))

    if d.get("payment_method") == "bank_transfer" or d.get("bank_name"):
        story.append(Paragraph("支払方法（銀行振込）", sH2))
        atypes = {"ordinary": "普通", "current": "当座"}
        bk_rows = [[
            Paragraph("銀行名", sHdr),
            Paragraph("支店", sHdr),
            Paragraph("口座種別", sHdr),
            Paragraph("口座番号", sHdr),
            Paragraph("名義", sHdr),
        ], [
            Paragraph(d.get("bank_name") or "", sCell),
            Paragraph(d.get("bank_branch") or "", sCell),
            Paragraph(atypes.get(d.get("bank_account_type") or "", d.get("bank_account_type") or ""), sCell),
            Paragraph(d.get("bank_account_number") or "", sCell),
            Paragraph(d.get("bank_account_holder") or "", sCell),
        ]]
        bk = Table(bk_rows, colWidths=[34*mm, 34*mm, 28*mm, 34*mm, 40*mm], repeatRows=1)
        bk.setStyle(TableStyle([
            ("GRID", (0,0), (-1,-1), 0.4, colors.grey),
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#34495e")),
            ("VALIGN", (0,0), (-1,-1), "TOP"),
            ("LEFTPADDING", (0,0), (-1,-1), 4),
            ("TOPPADDING", (0,0), (-1,-1), 3),
        ]))
        story.append(bk)

    doc.build(story)
    payload = buff.getvalue()
    inv_no = d.get("invoice_number") or _today_str()
    fname = f"請求書_{_sanitize_filename(inv_no)}.pdf"
    return _wrap_report(fname, "pdf", payload)


# ============================================================
# 4. 一般報告書 (Business Report)
# ============================================================

_BR_SECTIONS = [
    ("executive_summary", "概要", False),
    ("background", "背景", False),
    ("objectives", "目的", True),
    ("methods", "調査・実施方法", False),
    ("results", "結果", False),
    ("discussion", "考察", False),
    ("issues", "課題", True),
    ("conclusion", "結論", False),
    ("recommendations", "提言・今後の方針", True),
    ("references", "参考文献", True),
    ("attachments", "添付資料", True),
]

def _br_populated(d: dict[str, Any], key: str, is_list: bool) -> bool:
    v = d.get(key)
    if is_list:
        return bool(v and len(v) > 0)
    return bool(v and str(v).strip() != "")


def generate_business_report_docx(data: dict[str, Any]) -> GeneratedReport:
    d = _to_dict(data)
    doc = Document()
    for section in doc.sections:
        section.top_margin = Cm(2.5)
        section.bottom_margin = Cm(2.5)
        section.left_margin = Cm(2.5)
        section.right_margin = Cm(2.5)

    style = doc.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(11)
    style._element.rPr.rFonts.set(qn("w:eastAsia"), "ＭＳ ゴシック")

    title = d.get("title", "")
    subtitle = d.get("subtitle") or ""
    author = d.get("author", "")
    author_dept = d.get("author_department") or ""
    report_date = d.get("report_date") or _today_str()
    recipient = d.get("recipient") or ""
    recipient_dept = d.get("recipient_department") or ""

    # Cover page
    for _ in range(4):
        doc.add_paragraph()
    t_c = doc.add_heading(title, level=0)
    t_c.alignment = WD_ALIGN_PARAGRAPH.CENTER
    if subtitle:
        sp = doc.add_paragraph()
        sp.alignment = WD_ALIGN_PARAGRAPH.CENTER
        sr = sp.add_run(subtitle)
        sr.font.size = Pt(14)
        sr.italic = True
    for _ in range(4):
        doc.add_paragraph()

    meta_t = doc.add_table(rows=4, cols=2)
    meta_t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for row in meta_t.rows:
        row.cells[0].width = Cm(5)
        row.cells[1].width = Cm(8)
    meta_t.rows[0].cells[0].text = "提出先"
    meta_t.rows[0].cells[1].text = f"{recipient_dept}  {recipient}".strip()
    meta_t.rows[1].cells[0].text = "報告日"
    meta_t.rows[1].cells[1].text = report_date
    meta_t.rows[2].cells[0].text = "作成者"
    meta_t.rows[2].cells[1].text = f"{author_dept}  {author}".strip()
    meta_t.rows[3].cells[0].text = "所属"
    meta_t.rows[3].cells[1].text = author_dept

    doc.add_page_break()

    # TOC
    doc.add_heading("目次", level=1)
    toc_t = doc.add_table(rows=0, cols=2)
    toc_t.style = "Table Grid"
    idx = 1
    for key, label, is_list in _BR_SECTIONS:
        if _br_populated(d, key, is_list):
            r = toc_t.add_row().cells
            r[0].text = f"{idx}."
            r[1].text = f"{idx}. {label}"
            idx += 1
    doc.add_page_break()

    # Content sections
    idx = 1
    for key, label, is_list in _BR_SECTIONS:
        if not _br_populated(d, key, is_list):
            continue
        doc.add_heading(f"{idx}. {label}", level=1)
        idx += 1
        val = d.get(key)
        if is_list:
            for item in (val or []):
                if key == "references":
                    doc.add_paragraph(f"[{val.index(item)+1}] {item}")
                elif key == "attachments":
                    doc.add_paragraph(f"添付{val.index(item)+1}：{item}")
                else:
                    doc.add_paragraph(f"・ {item}", style="List Bullet")
        else:
            paragraphs = (val or "").split("\n") if isinstance(val, str) else [str(val)]
            for pg in paragraphs:
                if pg.strip():
                    doc.add_paragraph(pg)

    buff = io.BytesIO()
    doc.save(buff)
    payload = buff.getvalue()
    fname = f"報告書_{_sanitize_filename(title or _today_str())}.docx"
    return _wrap_report(fname, "docx", payload)


def generate_business_report_pdf(data: dict[str, Any]) -> GeneratedReport:
    d = _to_dict(data)
    buff = io.BytesIO()
    doc = SimpleDocTemplate(buff, pagesize=A4, leftMargin=25*mm, rightMargin=25*mm, topMargin=25*mm, bottomMargin=25*mm)
    font = _pdf_font()
    styles = getSampleStyleSheet()

    sTitle = ParagraphStyle("sTitle", parent=styles["Title"], fontName=font, fontSize=24, leading=30, alignment=TA_CENTER)
    sSub = ParagraphStyle("sSub", parent=styles["BodyText"], fontName=font, fontSize=14, leading=18, alignment=TA_CENTER, textColor=colors.HexColor("#555555"))
    sH1 = ParagraphStyle("sH1", parent=styles["Heading1"], fontName=font, fontSize=14, leading=18, spaceBefore=4*mm, spaceAfter=2*mm)
    sBody = ParagraphStyle("sBody", parent=styles["BodyText"], fontName=font, fontSize=10.5, leading=16, alignment=TA_JUSTIFY)
    sCell = ParagraphStyle("sCell", parent=sBody, fontSize=10, leading=14)

    story = []
    title = d.get("title", "")
    subtitle = d.get("subtitle") or ""
    author = d.get("author", "")
    author_dept = d.get("author_department") or ""
    report_date = d.get("report_date") or _today_str()
    recipient = d.get("recipient") or ""
    recipient_dept = d.get("recipient_department") or ""

    story.append(Spacer(1, 50*mm))
    story.append(Paragraph(title, sTitle))
    if subtitle:
        story.append(Spacer(1, 4*mm))
        story.append(Paragraph(subtitle, sSub))
    story.append(Spacer(1, 50*mm))

    meta_rows = [
        ["提出先", f"{recipient_dept}  {recipient}".strip()],
        ["報告日", report_date],
        ["作成者", f"{author_dept}  {author}".strip()],
        ["所属", author_dept],
    ]
    meta_rows_p = [[Paragraph(a, sBody), Paragraph(b, sBody)] for a, b in meta_rows]
    mt = Table(meta_rows_p, colWidths=[40*mm, 100*mm], hAlign="CENTER")
    mt.setStyle(TableStyle([
        ("GRID", (0,0), (-1,-1), 0.3, colors.grey),
        ("BACKGROUND", (0,0), (0,-1), colors.HexColor("#f5f5f5")),
        ("LEFTPADDING", (0,0), (-1,-1), 6),
        ("TOPPADDING", (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
    ]))
    story.append(mt)
    story.append(PageBreak())

    story.append(Paragraph("目次", sH1))
    toc_rows = []
    idx = 1
    for key, label, is_list in _BR_SECTIONS:
        if _br_populated(d, key, is_list):
            toc_rows.append([Paragraph(f"{idx}.", sCell), Paragraph(f"{idx}. {label}", sCell)])
            idx += 1
    if toc_rows:
        toct = Table(toc_rows, colWidths=[15*mm, 125*mm])
        toct.setStyle(TableStyle([
            ("GRID", (0,0), (-1,-1), 0.3, colors.grey),
            ("LEFTPADDING", (0,0), (-1,-1), 6),
            ("TOPPADDING", (0,0), (-1,-1), 4),
            ("BOTTOMPADDING", (0,0), (-1,-1), 4),
        ]))
        story.append(toct)
    story.append(PageBreak())

    idx = 1
    for key, label, is_list in _BR_SECTIONS:
        if not _br_populated(d, key, is_list):
            continue
        story.append(Paragraph(f"{idx}. {label}", sH1))
        idx += 1
        val = d.get(key)
        if is_list:
            lst = val or []
            for i, item in enumerate(lst, start=1):
                if key == "references":
                    story.append(Paragraph(f"[{i}] {item}", sBody))
                elif key == "attachments":
                    story.append(Paragraph(f"添付{i}：{item}", sBody))
                else:
                    story.append(Paragraph(f"・ {item}", sBody))
        else:
            paragraphs = (val or "").split("\n") if isinstance(val, str) else [str(val)]
            for pg in paragraphs:
                if pg.strip():
                    story.append(Paragraph(pg, sBody))
        story.append(Spacer(1, 2*mm))

    doc.build(story)
    payload = buff.getvalue()
    fname = f"報告書_{_sanitize_filename(title or _today_str())}.pdf"
    return _wrap_report(fname, "pdf", payload)


# ============================================================
# 5. 送付状 (Cover Letter / Soufu-Jo)
# ============================================================

def generate_cover_letter_docx(data: dict[str, Any]) -> GeneratedReport:
    d = _to_dict(data)
    doc = Document()
    for section in doc.sections:
        section.top_margin = Cm(2.5)
        section.bottom_margin = Cm(2.5)
        section.left_margin = Cm(2.5)
        section.right_margin = Cm(2.5)

    style = doc.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(11)
    style._element.rPr.rFonts.set(qn("w:eastAsia"), "ＭＳ 明朝")

    letter_date = d.get("letter_date") or _today_str()

    # Top-right date
    dp = doc.add_paragraph()
    dp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    dp.add_run(letter_date)

    # Top-left recipient block
    rec_p = doc.add_paragraph()
    rec_company = d.get("recipient_company", "")
    rec_dept = d.get("recipient_department") or ""
    rec_title = d.get("recipient_title") or ""
    rec_name = d.get("recipient_name") or "御中"
    rec_p.add_run(f"{rec_company}\n")
    if rec_dept:
        rec_p.add_run(f"{rec_dept}\n")
    if rec_title:
        rec_p.add_run(f"{rec_title}  ")
    r = rec_p.add_run(rec_name if rec_name == "御中" else f"{rec_name} 様")
    r.bold = True
    doc.add_paragraph()

    # Subject
    sub = d.get("subject", "")
    sp = doc.add_paragraph()
    sp.alignment = WD_ALIGN_PARAGRAPH.LEFT
    sr = sp.add_run(f"【件名】{sub}")
    sr.bold = True
    sr.underline = True
    doc.add_paragraph()

    # Greeting
    doc.add_paragraph(d.get("greeting") or "お世話になっております。")

    # Body paragraphs
    for pg in d.get("body_paragraphs", []) or []:
        doc.add_paragraph(pg)

    # Closing
    doc.add_paragraph()
    doc.add_paragraph(d.get("closing") or "ご確認の程、何卒よろしくお願い申し上げます。")
    doc.add_paragraph()

    # Enclosures table
    enclosures = d.get("enclosures", []) or []
    enc_title = d.get("enclosures_title") or "添付書類一覧"
    if enclosures:
        doc.add_heading(enc_title, level=2)
        enc_t = doc.add_table(rows=1, cols=3)
        enc_t.style = "Light Grid Accent 1"
        enc_t.rows[0].cells[0].text = "No."
        enc_t.rows[0].cells[1].text = "書類名"
        enc_t.rows[0].cells[2].text = "部数"
        for i, enc in enumerate(enclosures, start=1):
            r = enc_t.add_row().cells
            r[0].text = str(i)
            r[1].text = enc
            r[2].text = "1"
        doc.add_paragraph()

    # Sender block bottom-left
    send_p = doc.add_paragraph()
    send_p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    sp_code = d.get("sender_postal_code") or ""
    sp_addr = d.get("sender_address") or ""
    sp_company = d.get("sender_company") or ""
    sp_dept = d.get("sender_department") or ""
    sp_name = d.get("sender_name", "")
    sp_phone = d.get("sender_phone") or ""
    sp_email = d.get("sender_email") or ""

    lines = []
    if sp_code:
        lines.append(f"〒{sp_code}")
    if sp_addr:
        lines.append(sp_addr)
    if sp_company:
        lines.append(sp_company)
    if sp_dept:
        lines.append(sp_dept)
    lines.append(f"{sp_name}  印")
    contact_line_parts = []
    if sp_phone:
        contact_line_parts.append(f"TEL：{sp_phone}")
    if sp_email:
        contact_line_parts.append(f"EMAIL：{sp_email}")
    if contact_line_parts:
        lines.append("  ".join(contact_line_parts))
    send_p.add_run("\n".join(lines))

    buff = io.BytesIO()
    doc.save(buff)
    payload = buff.getvalue()
    subject_key = sub or _today_str()
    fname = f"送付状_{_sanitize_filename(subject_key)}.docx"
    return _wrap_report(fname, "docx", payload)


def generate_cover_letter_pdf(data: dict[str, Any]) -> GeneratedReport:
    d = _to_dict(data)
    buff = io.BytesIO()
    doc = SimpleDocTemplate(buff, pagesize=A4, leftMargin=25*mm, rightMargin=25*mm, topMargin=25*mm, bottomMargin=25*mm)
    font = _pdf_font()
    styles = getSampleStyleSheet()

    sH2 = ParagraphStyle("sH2", parent=styles["Heading2"], fontName=font, fontSize=12, leading=16, spaceBefore=3*mm, spaceAfter=1*mm)
    sBody = ParagraphStyle("sBody", parent=styles["BodyText"], fontName=font, fontSize=11, leading=18)
    sRight = ParagraphStyle("sRight", parent=sBody, alignment=TA_RIGHT)
    sBold = ParagraphStyle("sBold", parent=sBody, fontName=font, fontSize=11, leading=18)
    sSubj = ParagraphStyle("sSubj", parent=sBold, fontSize=13, leading=18)
    sCell = ParagraphStyle("sCell", parent=sBody, fontSize=10.5, leading=15)
    sHdr = ParagraphStyle("sHdr", parent=sCell, fontName=font, fontSize=10.5, leading=15, textColor=colors.white)

    story = []
    letter_date = d.get("letter_date") or _today_str()
    story.append(Paragraph(letter_date, sRight))
    story.append(Spacer(1, 6*mm))

    rec_company = d.get("recipient_company", "")
    rec_dept = d.get("recipient_department") or ""
    rec_title = d.get("recipient_title") or ""
    rec_name = d.get("recipient_name") or "御中"
    rec_txt = f"{rec_company}"
    if rec_dept:
        rec_txt += f"<br/>{rec_dept}"
    if rec_title:
        rec_txt += f"<br/>{rec_title}"
    honorific = "御中" if rec_name == "御中" else f"{rec_name} 様"
    rec_txt += f"<br/><b>{honorific}</b>"
    story.append(Paragraph(rec_txt, sBody))
    story.append(Spacer(1, 6*mm))

    sub = d.get("subject", "")
    story.append(Paragraph(f"【件名】{sub}", sSubj))
    story.append(Spacer(1, 4*mm))

    story.append(Paragraph(d.get("greeting") or "お世話になっております。", sBody))
    story.append(Spacer(1, 2*mm))

    for pg in d.get("body_paragraphs", []) or []:
        story.append(Paragraph(pg, sBody))
        story.append(Spacer(1, 1.5*mm))

    story.append(Spacer(1, 3*mm))
    story.append(Paragraph(d.get("closing") or "ご確認の程、何卒よろしくお願い申し上げます。", sBody))
    story.append(Spacer(1, 4*mm))

    enclosures = d.get("enclosures", []) or []
    enc_title = d.get("enclosures_title") or "添付書類一覧"
    if enclosures:
        story.append(Paragraph(enc_title, sH2))
        enc_rows = [[
            Paragraph("No.", sHdr),
            Paragraph("書類名", sHdr),
            Paragraph("部数", sHdr),
        ]]
        for i, enc in enumerate(enclosures, start=1):
            enc_rows.append([
                Paragraph(str(i), sCell),
                Paragraph(enc, sCell),
                Paragraph("1", sCell),
            ])
        et = Table(enc_rows, colWidths=[15*mm, 110*mm, 15*mm], repeatRows=1)
        et.setStyle(TableStyle([
            ("GRID", (0,0), (-1,-1), 0.4, colors.grey),
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#2c3e50")),
            ("ALIGN", (0,0), (0,-1), "CENTER"),
            ("ALIGN", (2,0), (2,-1), "CENTER"),
            ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
            ("LEFTPADDING", (0,0), (-1,-1), 6),
            ("TOPPADDING", (0,0), (-1,-1), 4),
            ("BOTTOMPADDING", (0,0), (-1,-1), 4),
        ]))
        story.append(et)
        story.append(Spacer(1, 5*mm))

    sp_code = d.get("sender_postal_code") or ""
    sp_addr = d.get("sender_address") or ""
    sp_company = d.get("sender_company") or ""
    sp_dept = d.get("sender_department") or ""
    sp_name = d.get("sender_name", "")
    sp_phone = d.get("sender_phone") or ""
    sp_email = d.get("sender_email") or ""

    lines = []
    if sp_code:
        lines.append(f"〒{sp_code}")
    if sp_addr:
        lines.append(sp_addr)
    if sp_company:
        lines.append(sp_company)
    if sp_dept:
        lines.append(sp_dept)
    lines.append(f"{sp_name}  印")
    contact_line_parts = []
    if sp_phone:
        contact_line_parts.append(f"TEL：{sp_phone}")
    if sp_email:
        contact_line_parts.append(f"EMAIL：{sp_email}")
    if contact_line_parts:
        lines.append("  ".join(contact_line_parts))
    story.append(Paragraph("<br/>".join(lines), sBody))

    doc.build(story)
    payload = buff.getvalue()
    subject_key = sub or _today_str()
    fname = f"送付状_{_sanitize_filename(subject_key)}.pdf"
    return _wrap_report(fname, "pdf", payload)


# ============================================================
# 6. 会社データ分析報告書 (Company Dataset Analysis Report)
# ============================================================

try:
    from ai_engine.data_intelligence import analyzer as _analyzer
    _ANALYZER_AVAILABLE = True
except Exception:
    _analyzer = None  # type: ignore
    _ANALYZER_AVAILABLE = False

try:
    import numpy as _np
    _NUMPY_AVAILABLE = True
except Exception:
    _np = None  # type: ignore
    _NUMPY_AVAILABLE = False


def _corr_color_hex(r: float | None) -> str:
    if r is None:
        return "FFFFFF"
    try:
        v = float(r)
        if _np is not None and (_np.isnan(v) or _np.isinf(v)):
            return "FFFFFF"
    except Exception:
        return "FFFFFF"
    if v >= 0:
        t = min(max(v, 0.0), 1.0)
        r_comp = int(255 * (1 - t))
        g_comp = 255
        b_comp = int(255 * (1 - t * 0.4))
    else:
        t = min(max(-v, 0.0), 1.0)
        r_comp = 255
        g_comp = int(255 * (1 - t))
        b_comp = int(255 * (1 - t * 0.4))
    return f"{r_comp:02X}{g_comp:02X}{b_comp:02X}"


def _missing_color_hex(pct: float) -> str:
    if pct <= 0:
        return "FFFFFF"
    if pct < 5:
        return "E8F5E9"
    if pct < 20:
        return "FFF8E1"
    return "FFCDD2"


def _outlier_color_hex(pct: float) -> str:
    if pct <= 0:
        return "F5F5F5"
    if pct < 2:
        return "E3F2FD"
    if pct < 10:
        return "FFF3E0"
    return "FFCDD2"


def _auto_recommendations(
    analysis: Any,
    corr: dict[str, Any],
    outliers_per_col: dict[str, dict[str, Any]],
    missing_per_col: dict[str, float],
    iqr_k: float,
    corr_method: str,
) -> tuple[list[str], list[str]]:
    suggestions: list[str] = []
    action_plan: list[str] = []

    summary = getattr(analysis, "summary", None)
    summary_dict = summary.to_dict() if summary is not None and hasattr(summary, "to_dict") else (summary or {})
    rows = int(summary_dict.get("rows", 0) or 0)
    cols = int(summary_dict.get("columns", 0) or 0)

    suggestions.append(f"データセット概要：{rows} 行 × {cols} 列。IQR k-factor={iqr_k}、相関係数手法={corr_method} で分析を実施。")

    high_missing = [(c, p) for c, p in missing_per_col.items() if p > 0]
    if high_missing:
        suggestions.append(f"欠損値が検出された列数：{len(high_missing)} / {cols}")
        for c, p in sorted(high_missing, key=lambda x: -x[1]):
            if p > 20:
                action_plan.append(f"列「{c}」の欠損率 {p:.1f}% は 20% を超過。列の除外または再収集を推奨。")
            elif p >= 5:
                action_plan.append(f"列「{c}」の欠損率 {p:.1f}%。数値列は中央値、カテゴリ列は最頻値での補完を推奨。")
            else:
                action_plan.append(f"列「{c}」の欠損率 {p:.1f}%。行数が少ないため、該当行を削除で対応可能。")
    else:
        suggestions.append("全列で欠損値なし（クリーンなデータセット）。")

    high_outlier_cols = []
    for c, info in outliers_per_col.items():
        pct = float(info.get("pct", 0) or 0)
        if pct > 10:
            high_outlier_cols.append((c, pct))
            action_plan.append(f"列「{c}」の外れ値率 {pct:.1f}%（>10%）。ウィンザライズ処理（{iqr_k}×IQR で clipping）を検討。")
        elif pct > 2:
            high_outlier_cols.append((c, pct))
            suggestions.append(f"列「{c}」に外れ値率 {pct:.1f}% を検出。分布を可視化した上での判断を推奨。")
    if not high_outlier_cols:
        suggestions.append(f"IQR k={iqr_k} 基準では、外れ値率はすべての数値列で 2% 以下。")

    matrix = corr.get("matrix") or []
    corr_cols = corr.get("columns") or []
    multi_pairs: list[tuple[str, str, float]] = []
    if corr_cols and matrix:
        n = len(corr_cols)
        for i in range(n):
            for j in range(i + 1, n):
                try:
                    v = float(matrix[i][j])
                    if _np is not None and (_np.isnan(v) or _np.isinf(v)):
                        continue
                    if abs(v) >= 0.85:
                        multi_pairs.append((corr_cols[i], corr_cols[j], v))
                except Exception:
                    continue
    if multi_pairs:
        for a, b, v in multi_pairs:
            action_plan.append(f"多重共線性の疑い：「{a}」 と 「{b}」 の相関 r = {v:+.3f}（|r| >= 0.85）。片方を除外するか、主成分分析を検討。")
    else:
        suggestions.append("|r| >= 0.85 の強い相関ペアは検出されず。多重共線性のリスクは低い。")

    numeric_cols = list(getattr(analysis, "numeric_columns", []) or [])
    cat_cols = list(getattr(analysis, "categorical_columns", []) or [])
    if not numeric_cols:
        suggestions.append("数値列が存在しないため、カテゴリ特徴のみの分析となっています。")
        action_plan.append("数値化可能な列（金額・件数など）があれば追加し、再分析を推奨。")
    else:
        suggestions.append(f"数値列 {len(numeric_cols)} 件、カテゴリ列 {len(cat_cols)} 件を検出。")

    if rows > 0 and rows < 30:
        action_plan.append(f"サンプルサイズ {rows} 行は統計的に少ないため、今後のデータ蓄積またはブートストラップ法による頑健化を推奨。")

    if not action_plan:
        action_plan.append("現時点で大きな品質リスクは検出されませんでした。定期的に同報告書を出力し、トレンドを監視することを推奨します。")

    return suggestions, action_plan


def build_company_analysis_report_from_dataset(
    data_or_filepath,
    report_meta: dict[str, Any] | None = None,
    iqr_k: float = 1.5,
    corr_method: str = "pearson",
    include_raw_appendix: bool = True,
):
    if not _ANALYZER_AVAILABLE or _analyzer is None:
        raise RuntimeError("ai_engine.data_intelligence.analyzer を import できませんでした。")

    report_meta = report_meta or {}

    import pandas as pd

    if isinstance(data_or_filepath, pd.DataFrame):
        df = data_or_filepath.copy()
        source_name = report_meta.get("source_filename") or "DataFrame"
    elif isinstance(data_or_filepath, (str, os.PathLike)):
        df = _analyzer.load_dataframe_from_path(str(data_or_filepath))
        source_name = os.path.basename(str(data_or_filepath))
    elif isinstance(data_or_filepath, bytes):
        fn = report_meta.get("source_filename") or "dataset.csv"
        df = _analyzer.load_dataframe_from_file(data_or_filepath, fn)
        source_name = fn
    else:
        raise TypeError(f"Unsupported input type: {type(data_or_filepath)}")

    rows_n = len(df)
    cols_n = len(df.columns) if rows_n > 0 else 0
    if rows_n == 0:
        analysis_data = {
            "summary": {"rows": 0, "columns": cols_n, "column_names": list(df.columns) if cols_n else [],
                        "dtypes": {}, "memory_bytes": 0, "missing_total": 0, "missing_by_column": {}},
            "numeric_stats": [], "categorical_stats": [],
            "numeric_columns": [], "categorical_columns": [], "datetime_columns": [],
            "sample_records": [],
        }
        corr_info = {"columns": [], "index": [], "matrix": [], "method": corr_method}
        outliers_per_col: dict[str, dict[str, Any]] = {}
    else:
        analysis = _analyzer.analyze_dataframe(df, sample_rows=min(10, rows_n))
        analysis_data = analysis.to_dict()
        corr_info = _analyzer.correlation_matrix(df, method=corr_method)
        outliers_per_col = {}
        for c in list(getattr(analysis, "numeric_columns", []) or []):
            try:
                o = _analyzer.iqr_outliers(df, c, k=iqr_k)
                n_total = int(o.get("n_total", rows_n) or rows_n)
                n_out = len(list(o.get("outliers") or []))
                pct = (100.0 * n_out / n_total) if n_total > 0 else 0.0
                outliers_per_col[c] = {
                    "count": n_out,
                    "pct": pct,
                    "lower": o.get("lower_bound"),
                    "upper": o.get("upper_bound"),
                    "iqr": o.get("iqr"),
                }
            except Exception as _e:
                outliers_per_col[c] = {"count": 0, "pct": 0.0, "lower": None, "upper": None, "iqr": None}

    missing_by_col = (
        dict(analysis_data.get("summary", {}).get("missing_by_column", {}))
        if rows_n > 0
        else {c: 0 for c in (analysis_data.get("summary", {}).get("column_names") or [])}
    )
    missing_per_col_pct: dict[str, float] = {}
    for c, m in missing_by_col.items():
        missing_per_col_pct[c] = (100.0 * int(m or 0) / rows_n) if rows_n > 0 else 0.0

    if rows_n == 0:
        suggestions = ["データセットが空（0行）です。データを登録の上、再実行してください。"]
        action_plan = ["CSV/Excel ファイルの内容を確認し、最低 1 行以上のデータを用意してください。"]
    else:
        suggestions, action_plan = _auto_recommendations(
            analysis if rows_n > 0 else analysis_data,
            corr_info, outliers_per_col, missing_per_col_pct, iqr_k, corr_method
        )

    numeric_stats = analysis_data.get("numeric_stats") or []
    categorical_stats = analysis_data.get("categorical_stats") or []
    kpis: list[dict[str, Any]] = [
        {"name": "総行数 (Rows)", "value": rows_n, "unit": "行", "color": "1565C0"},
        {"name": "総列数 (Columns)", "value": cols_n, "unit": "列", "color": "2E7D32"},
        {"name": "数値列数", "value": len(analysis_data.get("numeric_columns") or []), "unit": "列", "color": "6A1B9A"},
        {"name": "カテゴリ列数", "value": len(analysis_data.get("categorical_columns") or []), "unit": "列", "color": "EF6C00"},
        {"name": "日付 / 時間列数", "value": len(analysis_data.get("datetime_columns") or []), "unit": "列", "color": "00838F"},
        {"name": "欠損値合計", "value": int(analysis_data.get("summary", {}).get("missing_total", 0) or 0), "unit": "件", "color": "C62828"},
    ]

    top_categories: list[dict[str, Any]] = []
    for cs in categorical_stats:
        uniq_count = int(cs.get("unique") or 0)
        top_val = cs.get("top")
        top_freq = int(cs.get("freq") or 0)
        top_pct = (100.0 * top_freq / rows_n) if rows_n > 0 else 0.0
        top_categories.append({
            "column": cs.get("column"),
            "unique": uniq_count,
            "top_value": "" if top_val is None else str(top_val),
            "top_frequency": top_freq,
            "top_pct": top_pct,
        })
    top_categories.sort(key=lambda x: -x["unique"])

    data_payload: dict[str, Any] = {
        "report_title": report_meta.get("report_title") or f"会社データ分析報告書 ({source_name})",
        "report_date": report_meta.get("report_date") or _today_str(),
        "period_covered": report_meta.get("period_covered") or "",
        "company_name": report_meta.get("company_name") or "",
        "department": report_meta.get("department") or "",
        "author_name": report_meta.get("author_name") or report_meta.get("author") or "",
        "analyst_name": report_meta.get("analyst_name") or report_meta.get("analyst") or "",
        "recipient_name": report_meta.get("recipient_name") or report_meta.get("recipient") or "",
        "source_filename": source_name,
        "iqr_k": iqr_k,
        "corr_method": corr_method,
        "kpis": kpis,
        "dataset_overview": analysis_data.get("summary") or {},
        "numeric_stats": numeric_stats,
        "categorical_stats": categorical_stats,
        "top_categories": top_categories[:10],
        "correlation": corr_info,
        "outliers_per_column": outliers_per_col,
        "missing_by_column": missing_by_col,
        "missing_pct_by_column": missing_per_col_pct,
        "discussion": "\n".join(suggestions) if suggestions else "",
        "action_plan": action_plan,
        "appendix_sample_rows": (analysis_data.get("sample_records") or []) if include_raw_appendix else [],
        "appendix_raw_stats": {
            "numeric": numeric_stats,
            "categorical": categorical_stats,
        } if include_raw_appendix else {},
        "include_appendix": include_raw_appendix,
    }

    for k, v in (report_meta.get("overrides") or {}).items():
        if v not in (None, "", []):
            data_payload[k] = v

    return data_payload


def _car_sections() -> list[tuple[str, str, bool]]:
    return [
        ("dataset_overview", "データセット概要 / 分析サマリ", False),
        ("kpis", "主要 KPI", True),
        ("numeric_stats", "記述統計（数値列）", True),
        ("top_categories", "カテゴリ列 Top-10 分布", True),
        ("correlation", "相関行列（heatmap）", False),
        ("outliers_per_column", "IQR 外れ値分析 / 列別", False),
        ("missing_pct_by_column", "欠損値分析 / 列別", False),
        ("discussion", "考察・示唆（自動生成）", False),
        ("action_plan", "アクションプラン（自動生成）", True),
        ("appendix_sample_rows", "付録A: サンプルレコード", True),
        ("appendix_raw_stats", "付録B: 生の統計量一覧", False),
    ]


def _car_populated(d: dict[str, Any], key: str, _is_list: bool) -> bool:
    v = d.get(key)
    if v is None:
        return False
    if isinstance(v, (list, dict)):
        return len(v) > 0
    if isinstance(v, str):
        return v.strip() != ""
    return True


def generate_company_analysis_report_docx(data: dict[str, Any]) -> GeneratedReport:
    d = _to_dict(data)
    doc = Document()
    for section in doc.sections:
        section.top_margin = Cm(2.2)
        section.bottom_margin = Cm(2.2)
        section.left_margin = Cm(2)
        section.right_margin = Cm(2)

    style = doc.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(10.5)
    try:
        style._element.rPr.rFonts.set(qn("w:eastAsia"), "ＭＳ 明朝")
    except Exception:
        try:
            style._element.rPr.rFonts.set(qn("w:eastAsia"), "Yu Mincho")
        except Exception:
            style._element.rPr.rFonts.set(qn("w:eastAsia"), "Meiryo")

    title = d.get("report_title") or "会社データ分析報告書"
    report_date = d.get("report_date") or _today_str()
    period = d.get("period_covered") or ""
    company = d.get("company_name") or ""
    dept = d.get("department") or ""
    author = d.get("author_name") or ""
    analyst = d.get("analyst_name") or ""
    recipient = d.get("recipient_name") or ""
    source = d.get("source_filename") or ""

    doc.add_paragraph()
    doc.add_paragraph()
    tp = doc.add_heading(title, level=0)
    tp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    if period:
        pp = doc.add_paragraph()
        pp.alignment = WD_ALIGN_PARAGRAPH.CENTER
        pr = pp.add_run(f"【集計期間】{period}")
        pr.italic = True
    doc.add_paragraph()
    doc.add_paragraph()
    doc.add_paragraph()

    meta_t = doc.add_table(rows=6, cols=2)
    meta_t.alignment = WD_TABLE_ALIGNMENT.CENTER
    meta_t.style = "Table Grid"
    meta_labels = [
        ("報告日", report_date),
        ("会社名", company),
        ("部署", dept),
        ("作成者 / 担当", f"{author}  {analyst}".strip()),
        ("提出先", recipient),
        ("データソース", source),
    ]
    for i, (lb, val) in enumerate(meta_labels):
        lc, rc = meta_t.rows[i].cells
        lc.text = lb
        _docx_set_cell_shading(lc, "ECEFF1")
        rc.text = val or "-"

    doc.add_page_break()

    doc.add_heading("目次", level=1)
    toc_t = doc.add_table(rows=0, cols=2)
    toc_t.style = "Light Grid Accent 1"
    idx = 1
    for key, label, _ in _car_sections():
        if key == "kpis":
            if _car_populated(d, key, True):
                r = toc_t.add_row().cells
                r[0].text = f"{idx}."
                r[1].text = f"{idx}. {label}"
                idx += 1
        else:
            if _car_populated(d, key, False):
                r = toc_t.add_row().cells
                r[0].text = f"{idx}."
                r[1].text = f"{idx}. {label}"
                idx += 1

    doc.add_page_break()

    idx = 1

    overview = d.get("dataset_overview") or {}
    if overview:
        doc.add_heading(f"{idx}. データセット概要 / 分析サマリ", level=1)
        idx += 1
        rows_n = int(overview.get("rows", 0) or 0)
        cols_n = int(overview.get("columns", 0) or 0)
        mem_bytes = int(overview.get("memory_bytes", 0) or 0)
        missing_total = int(overview.get("missing_total", 0) or 0)
        col_names = overview.get("column_names") or []
        dtypes = overview.get("dtypes") or {}
        ov_t = doc.add_table(rows=5, cols=2)
        ov_t.style = "Table Grid"
        ov_rows = [
            ("サイズ（行 × 列）", f"{rows_n} 行 × {cols_n} 列"),
            ("メモリサイズ", f"{mem_bytes:,} bytes  (~{mem_bytes/1024:.1f} KB)"),
            ("欠損値 合計", f"{missing_total} 件  ({(100*missing_total/(rows_n*cols_n)):.2f}% of cells)" if rows_n and cols_n else f"{missing_total} 件"),
            ("列名一覧", ", ".join([str(c) for c in col_names]) if col_names else "-"),
            ("列型 (dtypes)", ", ".join([f"{c}: {dtypes.get(c,'?')}" for c in (col_names or [])]) if col_names else "-"),
        ]
        for i, (lb, val) in enumerate(ov_rows):
            lc, rc = ov_t.rows[i].cells
            lc.text = lb
            _docx_set_cell_shading(lc, "ECEFF1")
            rc.text = val
        doc.add_paragraph()

    kpis = d.get("kpis") or []
    if kpis:
        doc.add_heading(f"{idx}. 主要 KPI", level=1)
        idx += 1
        kpi_t = doc.add_table(rows=2, cols=max(1, len(kpis)))
        kpi_t.alignment = WD_TABLE_ALIGNMENT.CENTER
        for i, k in enumerate(kpis):
            hc = kpi_t.rows[0].cells[i]
            hc.text = str(k.get("name", ""))
            _docx_set_cell_shading(hc, k.get("color") or "1565C0")
            for p in hc.paragraphs:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                for run in p.runs:
                    run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
                    run.bold = True
            vc = kpi_t.rows[1].cells[i]
            val = k.get("value")
            unit = k.get("unit") or ""
            try:
                if isinstance(val, float):
                    s = f"{val:,.2f} {unit}".strip()
                else:
                    s = f"{val:,} {unit}".strip() if isinstance(val, int) else f"{val} {unit}".strip()
            except Exception:
                s = f"{val} {unit}".strip()
            vc.text = s
            _docx_set_cell_shading(vc, "F5F5F5")
            for p in vc.paragraphs:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                for run in p.runs:
                    run.bold = True
                    run.font.size = Pt(14)
        doc.add_paragraph()

    numeric_stats = d.get("numeric_stats") or []
    if numeric_stats:
        doc.add_heading(f"{idx}. 記述統計（数値列）", level=1)
        idx += 1
        headers = ["列名", "件数", "平均", "標準偏差", "最小値", "25%", "中央値(50%)", "75%", "最大値"]
        ns_t = doc.add_table(rows=1 + len(numeric_stats), cols=len(headers))
        ns_t.style = "Table Grid"
        for j, h in enumerate(headers):
            c = ns_t.rows[0].cells[j]
            c.text = h
            _docx_set_cell_shading(c, "37474F")
            for p in c.paragraphs:
                for run in p.runs:
                    run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
                    run.bold = True
        for i, s in enumerate(numeric_stats, start=1):
            row = ns_t.rows[i].cells
            vals = [
                s.get("column") or "",
                int(s.get("count") or 0),
                s.get("mean"),
                s.get("std"),
                s.get("min"),
                s.get("q25"),
                s.get("q50"),
                s.get("q75"),
                s.get("max"),
            ]
            for j, v in enumerate(vals):
                if isinstance(v, float):
                    row[j].text = f"{v:,.4f}" if abs(v) < 1000 else f"{v:,.2f}"
                elif v is None:
                    row[j].text = "-"
                else:
                    row[j].text = str(v)
                if j == 0:
                    _docx_set_cell_shading(row[j], "ECEFF1")
        doc.add_paragraph()

    top_categories = d.get("top_categories") or []
    if top_categories:
        doc.add_heading(f"{idx}. カテゴリ列 Top-10 分布", level=1)
        idx += 1
        headers = ["列名", "ユニーク数", "最頻値", "最頻値 度数", "最頻値 割合"]
        cat_t = doc.add_table(rows=1 + len(top_categories), cols=len(headers))
        cat_t.style = "Table Grid"
        for j, h in enumerate(headers):
            c = cat_t.rows[0].cells[j]
            c.text = h
            _docx_set_cell_shading(c, "37474F")
            for p in c.paragraphs:
                for run in p.runs:
                    run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
                    run.bold = True
        for i, tc in enumerate(top_categories, start=1):
            row = cat_t.rows[i].cells
            row[0].text = str(tc.get("column", ""))
            _docx_set_cell_shading(row[0], "ECEFF1")
            row[1].text = str(tc.get("unique", 0))
            row[2].text = str(tc.get("top_value", "") or "-")
            row[3].text = str(tc.get("top_frequency", 0))
            pct = float(tc.get("top_pct") or 0)
            row[4].text = f"{pct:.2f}%"
            if pct >= 80:
                _docx_set_cell_shading(row[4], "FFCDD2")
            elif pct >= 50:
                _docx_set_cell_shading(row[4], "FFF8E1")
            else:
                _docx_set_cell_shading(row[4], "E8F5E9")
        doc.add_paragraph()

    corr = d.get("correlation") or {}
    corr_cols = corr.get("columns") or []
    corr_matrix = corr.get("matrix") or []
    if corr_cols and corr_matrix:
        doc.add_heading(f"{idx}. 相関行列（{d.get('corr_method') or 'pearson'}法 / heatmap）", level=1)
        idx += 1
        n = len(corr_cols)
        ct = doc.add_table(rows=n + 1, cols=n + 1)
        ct.style = "Table Grid"
        ct.rows[0].cells[0].text = ""
        _docx_set_cell_shading(ct.rows[0].cells[0], "263238")
        for j, cc in enumerate(corr_cols):
            top = ct.rows[0].cells[j + 1]
            top.text = str(cc)
            _docx_set_cell_shading(top, "37474F")
            for p in top.paragraphs:
                for run in p.runs:
                    run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
                    run.bold = True
            left = ct.rows[j + 1].cells[0]
            left.text = str(cc)
            _docx_set_cell_shading(left, "37474F")
            for p in left.paragraphs:
                for run in p.runs:
                    run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
                    run.bold = True
        for i in range(n):
            for j in range(n):
                try:
                    v = corr_matrix[i][j]
                except Exception:
                    v = None
                cell = ct.rows[i + 1].cells[j + 1]
                if v is None:
                    cell.text = ""
                else:
                    cell.text = f"{float(v):+.3f}"
                _docx_set_cell_shading(cell, _corr_color_hex(v))
        doc.add_paragraph()

    outliers_per_col = d.get("outliers_per_column") or {}
    if outliers_per_col:
        doc.add_heading(f"{idx}. IQR 外れ値分析 / 列別 (k={d.get('iqr_k') or 1.5})", level=1)
        idx += 1
        headers = ["列名", "外れ値数", "外れ値率", "IQR", "下限 (Q1 - k×IQR)", "上限 (Q3 + k×IQR)"]
        ot = doc.add_table(rows=1 + len(outliers_per_col), cols=len(headers))
        ot.style = "Table Grid"
        for j, h in enumerate(headers):
            c = ot.rows[0].cells[j]
            c.text = h
            _docx_set_cell_shading(c, "37474F")
            for p in c.paragraphs:
                for run in p.runs:
                    run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
                    run.bold = True
        for i, (col, info) in enumerate(sorted(outliers_per_col.items()), start=1):
            row = ot.rows[i].cells
            row[0].text = str(col)
            _docx_set_cell_shading(row[0], "ECEFF1")
            cnt = int(info.get("count") or 0)
            pct = float(info.get("pct") or 0)
            row[1].text = str(cnt)
            row[2].text = f"{pct:.2f}%"
            _docx_set_cell_shading(row[2], _outlier_color_hex(pct))
            row[3].text = f"{info.get('iqr'):,.4f}" if isinstance(info.get("iqr"), float) else str(info.get("iqr") or "-")
            row[4].text = f"{info.get('lower'):,.4f}" if isinstance(info.get("lower"), float) else str(info.get("lower") or "-")
            row[5].text = f"{info.get('upper'):,.4f}" if isinstance(info.get("upper"), float) else str(info.get("upper") or "-")
        doc.add_paragraph()

    missing_pct = d.get("missing_pct_by_column") or {}
    missing_by_col = d.get("missing_by_column") or {}
    if missing_pct or missing_by_col:
        doc.add_heading(f"{idx}. 欠損値分析 / 列別", level=1)
        idx += 1
        cols_m = list(set(list(missing_pct.keys()) + list(missing_by_col.keys())))
        headers = ["列名", "欠損数", "欠損率", "補完推奨手法"]
        mt = doc.add_table(rows=1 + len(cols_m), cols=len(headers))
        mt.style = "Table Grid"
        for j, h in enumerate(headers):
            c = mt.rows[0].cells[j]
            c.text = h
            _docx_set_cell_shading(c, "37474F")
            for p in c.paragraphs:
                for run in p.runs:
                    run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
                    run.bold = True
        for i, col in enumerate(sorted(cols_m), start=1):
            row = mt.rows[i].cells
            row[0].text = str(col)
            _docx_set_cell_shading(row[0], "ECEFF1")
            cnt = int(missing_by_col.get(col, 0) or 0)
            pct = float(missing_pct.get(col, 0) or 0)
            row[1].text = str(cnt)
            row[2].text = f"{pct:.2f}%"
            _docx_set_cell_shading(row[2], _missing_color_hex(pct))
            if pct > 20:
                row[3].text = "列の除外 / 再収集"
            elif pct >= 5:
                row[3].text = "数値列: 中央値  /  カテゴリ列: 最頻値 補完"
            elif pct > 0:
                row[3].text = "該当行の削除（件数少ない）"
            else:
                row[3].text = "不要"
        doc.add_paragraph()

    discussion = d.get("discussion") or ""
    if discussion.strip():
        doc.add_heading(f"{idx}. 考察・示唆（自動生成）", level=1)
        idx += 1
        for pg in str(discussion).split("\n"):
            pg = pg.strip()
            if pg:
                p = doc.add_paragraph(pg, style="List Bullet")
        doc.add_paragraph()

    action_plan = d.get("action_plan") or []
    if action_plan:
        doc.add_heading(f"{idx}. アクションプラン（自動生成）", level=1)
        idx += 1
        for item in action_plan:
            doc.add_paragraph(f"・ {item}", style="List Bullet")
        doc.add_paragraph()

    sample_rows = d.get("appendix_sample_rows") or []
    if sample_rows and d.get("include_appendix", True):
        doc.add_heading(f"{idx}. 付録A: サンプルレコード", level=1)
        idx += 1
        columns = list(sample_rows[0].keys()) if sample_rows else []
        if columns:
            at = doc.add_table(rows=1 + len(sample_rows), cols=len(columns))
            at.style = "Table Grid"
            for j, cc in enumerate(columns):
                c = at.rows[0].cells[j]
                c.text = str(cc)
                _docx_set_cell_shading(c, "263238")
                for p in c.paragraphs:
                    for run in p.runs:
                        run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
                        run.bold = True
            for i, rec in enumerate(sample_rows, start=1):
                for j, cc in enumerate(columns):
                    v = rec.get(cc)
                    at.rows[i].cells[j].text = "" if v is None else str(v)
        doc.add_paragraph()

    raw_stats = d.get("appendix_raw_stats") or {}
    if raw_stats and d.get("include_appendix", True):
        doc.add_heading(f"{idx}. 付録B: 生の統計量一覧", level=1)
        idx += 1
        doc.add_paragraph()
        try:
            import json as _json
            raw_txt = _json.dumps(raw_stats, ensure_ascii=False, indent=2, default=str)
            doc.add_paragraph(raw_txt)
        except Exception:
            doc.add_paragraph(str(raw_stats))

    buff = io.BytesIO()
    doc.save(buff)
    payload = buff.getvalue()
    fname = f"会社データ分析報告書_{_sanitize_filename(source or report_date or _today_str())}.docx"
    return _wrap_report(fname, "docx", payload)


def generate_company_analysis_report_pdf(data: dict[str, Any]) -> GeneratedReport:
    d = _to_dict(data)
    buff = io.BytesIO()
    doc = SimpleDocTemplate(buff, pagesize=A4, leftMargin=18*mm, rightMargin=18*mm, topMargin=20*mm, bottomMargin=20*mm)
    font = _pdf_font()
    styles = getSampleStyleSheet()

    sTitle = ParagraphStyle("carTitle", parent=styles["Title"], fontName=font, fontSize=24, leading=30, alignment=TA_CENTER, spaceAfter=4*mm, textColor=colors.HexColor("#0D47A1"))
    sSub = ParagraphStyle("carSub", parent=styles["BodyText"], fontName=font, fontSize=11, leading=15, alignment=TA_CENTER, textColor=colors.HexColor("#455A64"))
    sH1 = ParagraphStyle("carH1", parent=styles["Heading1"], fontName=font, fontSize=14, leading=18, spaceBefore=5*mm, spaceAfter=2*mm, textColor=colors.HexColor("#0D47A1"))
    sBody = ParagraphStyle("carBody", parent=styles["BodyText"], fontName=font, fontSize=10, leading=14.5)
    sCell = ParagraphStyle("carCell", parent=sBody, fontSize=8.5, leading=11)
    sCellSmall = ParagraphStyle("carCellSm", parent=sBody, fontSize=7.5, leading=10)
    sHdr = ParagraphStyle("carHdr", parent=sCell, fontName=font, textColor=colors.white, fontSize=9, leading=11)
    sKPIHdr = ParagraphStyle("carKPIH", parent=sCell, fontName=font, textColor=colors.white, fontSize=9, leading=12, alignment=TA_CENTER)
    sKPIVal = ParagraphStyle("carKPIV", parent=sCell, fontName=font, fontSize=14, leading=18, alignment=TA_CENTER, textColor=colors.HexColor("#0D47A1"))
    sMono = ParagraphStyle("carMono", parent=sBody, fontSize=7.5, leading=10, fontName="Courier" if False else font)

    def P(txt: Any, st=sBody) -> Paragraph:
        return Paragraph("" if txt is None else str(txt), st)

    def num(v: Any, decimals: int = 2) -> str:
        if v is None:
            return "-"
        try:
            if isinstance(v, bool):
                return str(v)
            if isinstance(v, int):
                return f"{v:,}"
            if isinstance(v, float):
                if _np is not None and (_np.isnan(v) or _np.isinf(v)):
                    return "-"
                if abs(v) >= 1000:
                    return f"{v:,.{decimals}f}"
                return f"{v:.4f}" if abs(v) < 1 else f"{v:,.{decimals}f}"
        except Exception:
            pass
        return str(v)

    story = []

    title = d.get("report_title") or "会社データ分析報告書"
    report_date = d.get("report_date") or _today_str()
    period = d.get("period_covered") or ""
    company = d.get("company_name") or ""
    dept = d.get("department") or ""
    author = d.get("author_name") or ""
    analyst = d.get("analyst_name") or ""
    recipient = d.get("recipient_name") or ""
    source = d.get("source_filename") or ""

    story.append(Spacer(1, 50*mm))
    story.append(P(title, sTitle))
    if period:
        story.append(Spacer(1, 3*mm))
        story.append(P(f"【集計期間】{period}", sSub))
    story.append(Spacer(1, 50*mm))

    meta_rows = [
        ["報告日", report_date or "-"],
        ["会社名", company or "-"],
        ["部署", dept or "-"],
        ["作成者 / 担当", f"{author}  {analyst}".strip() or "-"],
        ["提出先", recipient or "-"],
        ["データソース", source or "-"],
    ]
    meta_p = [[P(a, sBody), P(b, sBody)] for a, b in meta_rows]
    mt = Table(meta_p, colWidths=[40*mm, 120*mm])
    mt.setStyle(TableStyle([
        ("GRID", (0,0), (-1,-1), 0.3, colors.grey),
        ("BACKGROUND", (0,0), (0,-1), colors.HexColor("#ECEFF1")),
        ("FONTNAME", (0,0), (-1,-1), font),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("LEFTPADDING", (0,0), (-1,-1), 6),
        ("TOPPADDING", (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
    ]))
    story.append(mt)
    story.append(PageBreak())

    story.append(P("目次", sH1))
    toc_rows = []
    idx = 1
    for key, label, _ in _car_sections():
        if _car_populated(d, key, False):
            toc_rows.append([P(f"{idx}.", sCell), P(f"{idx}. {label}", sCell)])
            idx += 1
    if toc_rows:
        tt = Table(toc_rows, colWidths=[12*mm, 148*mm])
        tt.setStyle(TableStyle([
            ("GRID", (0,0), (-1,-1), 0.2, colors.HexColor("#CFD8DC")),
            ("LEFTPADDING", (0,0), (-1,-1), 5),
            ("TOPPADDING", (0,0), (-1,-1), 3),
            ("BOTTOMPADDING", (0,0), (-1,-1), 3),
        ]))
        story.append(tt)
    story.append(PageBreak())

    idx = 1

    overview = d.get("dataset_overview") or {}
    if overview:
        story.append(P(f"{idx}. データセット概要 / 分析サマリ", sH1))
        idx += 1
        rows_n = int(overview.get("rows", 0) or 0)
        cols_n = int(overview.get("columns", 0) or 0)
        mem_bytes = int(overview.get("memory_bytes", 0) or 0)
        missing_total = int(overview.get("missing_total", 0) or 0)
        col_names = overview.get("column_names") or []
        dtypes = overview.get("dtypes") or {}
        cell_pct = ""
        if rows_n and cols_n:
            cell_pct = f"  ({100*missing_total/(rows_n*cols_n):.2f}% of cells)"
        ov_rows = [
            [P("サイズ（行 × 列）", sHdr), P(f"{rows_n} 行 × {cols_n} 列", sCell)],
            [P("メモリサイズ", sHdr), P(f"{mem_bytes:,} bytes  (~{mem_bytes/1024:.1f} KB)", sCell)],
            [P("欠損値 合計", sHdr), P(f"{missing_total} 件{cell_pct}", sCell)],
            [P("列名一覧", sHdr), P(", ".join([str(c) for c in col_names]) if col_names else "-", sCell)],
            [P("列型 (dtypes)", sHdr), P(", ".join([f"{c}: {dtypes.get(c,'?')}" for c in (col_names or [])]) if col_names else "-", sCellSmall)],
        ]
        ovt = Table(ov_rows, colWidths=[45*mm, 115*mm], repeatRows=1)
        ov_style_cmds = [
            ("GRID", (0,0), (-1,-1), 0.3, colors.grey),
            ("BACKGROUND", (0,0), (0,-1), colors.HexColor("#37474F")),
            ("VALIGN", (0,0), (-1,-1), "TOP"),
            ("LEFTPADDING", (0,0), (-1,-1), 5),
            ("TOPPADDING", (0,0), (-1,-1), 4),
            ("BOTTOMPADDING", (0,0), (-1,-1), 4),
        ]
        ovt.setStyle(TableStyle(ov_style_cmds))
        story.append(ovt)
        story.append(Spacer(1, 3*mm))

    kpis = d.get("kpis") or []
    if kpis:
        story.append(P(f"{idx}. 主要 KPI", sH1))
        idx += 1
        n_kpi = len(kpis)
        widths_total = 160
        col_w = [widths_total * mm / n_kpi for _ in range(n_kpi)]
        hdr_row = []
        val_row = []
        for k in kpis:
            hdr_row.append(P(k.get("name", ""), sKPIHdr))
            v = k.get("value")
            u = k.get("unit") or ""
            try:
                if isinstance(v, float):
                    vs = f"{v:,.2f} {u}".strip()
                else:
                    vs = f"{num(v)} {u}".strip()
            except Exception:
                vs = f"{v} {u}".strip()
            val_row.append(P(f"<b>{vs}</b>", sKPIVal))
        kt = Table([hdr_row, val_row], colWidths=col_w)
        kt_cmds = [
            ("GRID", (0,0), (-1,-1), 0.3, colors.HexColor("#B0BEC5")),
            ("ALIGN", (0,0), (-1,-1), "CENTER"),
            ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
            ("TOPPADDING", (0,0), (-1,0), 6),
            ("BOTTOMPADDING", (0,0), (-1,0), 6),
            ("TOPPADDING", (0,1), (-1,1), 7),
            ("BOTTOMPADDING", (0,1), (-1,1), 7),
            ("BACKGROUND", (0,1), (-1,1), colors.HexColor("#F5F5F5")),
        ]
        for j, k in enumerate(kpis):
            kt_cmds.append(("BACKGROUND", (j, 0), (j, 0), colors.HexColor(_rlh(k.get("color") or "1565C0"))))
        kt.setStyle(TableStyle(kt_cmds))
        story.append(kt)
        story.append(Spacer(1, 3*mm))

    numeric_stats = d.get("numeric_stats") or []
    if numeric_stats:
        story.append(P(f"{idx}. 記述統計（数値列）", sH1))
        idx += 1
        headers = ["列名", "件数", "平均", "標準偏差", "最小値", "25%", "中央値", "75%", "最大値"]
        ns_rows = [[P(h, sHdr) for h in headers]]
        for s in numeric_stats:
            ns_rows.append([
                P(s.get("column") or "", sCell),
                P(num(int(s.get("count") or 0)), sCell),
                P(num(s.get("mean")), sCell),
                P(num(s.get("std")), sCell),
                P(num(s.get("min")), sCell),
                P(num(s.get("q25")), sCell),
                P(num(s.get("q50")), sCell),
                P(num(s.get("q75")), sCell),
                P(num(s.get("max")), sCell),
            ])
        ns_cols = [28*mm, 17*mm, 17*mm, 17*mm, 17*mm, 17*mm, 17*mm, 17*mm, 17*mm]
        nst = Table(ns_rows, colWidths=ns_cols, repeatRows=1)
        cmds = [
            ("GRID", (0,0), (-1,-1), 0.25, colors.grey),
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#37474F")),
            ("ALIGN", (1,0), (-1,-1), "RIGHT"),
            ("FONTNAME", (0,0), (-1,-1), font),
            ("LEFTPADDING", (0,0), (-1,-1), 4),
            ("TOPPADDING", (0,0), (-1,-1), 3),
            ("BOTTOMPADDING", (0,0), (-1,-1), 3),
        ]
        for i in range(1, len(ns_rows)):
            cmds.append(("BACKGROUND", (0, i), (0, i), colors.HexColor("#ECEFF1")))
        nst.setStyle(TableStyle(cmds))
        story.append(nst)
        story.append(Spacer(1, 3*mm))

    top_categories = d.get("top_categories") or []
    if top_categories:
        story.append(P(f"{idx}. カテゴリ列 Top-10 分布", sH1))
        idx += 1
        headers = ["列名", "ユニーク数", "最頻値", "度数", "割合"]
        ct_rows = [[P(h, sHdr) for h in headers]]
        color_cmds = []
        for i, tc in enumerate(top_categories, start=1):
            pct = float(tc.get("top_pct") or 0)
            if pct >= 80:
                cell_c = "FFCDD2"
            elif pct >= 50:
                cell_c = "FFF8E1"
            else:
                cell_c = "E8F5E9"
            ct_rows.append([
                P(tc.get("column") or "", sCell),
                P(num(tc.get("unique")), sCell),
                P(tc.get("top_value") or "-", sCell),
                P(num(tc.get("top_frequency")), sCell),
                P(f"{pct:.2f}%", sCell),
            ])
            color_cmds.append(("BACKGROUND", (4, i), (4, i), colors.HexColor(_rlh(cell_c))))
        ct = Table(ct_rows, colWidths=[32*mm, 20*mm, 48*mm, 20*mm, 20*mm], repeatRows=1)
        cmds = [
            ("GRID", (0,0), (-1,-1), 0.25, colors.grey),
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#37474F")),
            ("FONTNAME", (0,0), (-1,-1), font),
            ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
            ("LEFTPADDING", (0,0), (-1,-1), 4),
            ("TOPPADDING", (0,0), (-1,-1), 3),
            ("BOTTOMPADDING", (0,0), (-1,-1), 3),
        ]
        for i in range(1, len(ct_rows)):
            cmds.append(("BACKGROUND", (0, i), (0, i), colors.HexColor("#ECEFF1")))
        cmds.extend(color_cmds)
        ct.setStyle(TableStyle(cmds))
        story.append(ct)
        story.append(Spacer(1, 3*mm))

    corr = d.get("correlation") or {}
    corr_cols = corr.get("columns") or []
    corr_matrix = corr.get("matrix") or []
    if corr_cols and corr_matrix:
        story.append(P(f"{idx}. 相関行列（{d.get('corr_method') or 'pearson'}法 / heatmap）", sH1))
        idx += 1
        n = len(corr_cols)
        available_w = 160 * mm
        cell_w = min(18*mm, available_w / (n + 1))
        col_w = [cell_w] + [cell_w] * n
        corr_rows = []
        hdr_r = [P("", sHdr)]
        for cc in corr_cols:
            hdr_r.append(P(cc, sHdr))
        corr_rows.append(hdr_r)
        for i in range(n):
            row = [P(corr_cols[i], sHdr)]
            for j in range(n):
                try:
                    v = corr_matrix[i][j]
                except Exception:
                    v = None
                txt = "" if v is None else f"{float(v):+.3f}"
                row.append(P(txt, sCellSmall))
            corr_rows.append(row)
        ct = Table(corr_rows, colWidths=col_w, repeatRows=1)
        cmds = [
            ("GRID", (0,0), (-1,-1), 0.2, colors.HexColor("#90A4AE")),
            ("FONTNAME", (0,0), (-1,-1), font),
            ("ALIGN", (0,0), (-1,-1), "CENTER"),
            ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
            ("TOPPADDING", (0,0), (-1,-1), 3),
            ("BOTTOMPADDING", (0,0), (-1,-1), 3),
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#37474F")),
            ("BACKGROUND", (0,0), (0,-1), colors.HexColor("#37474F")),
            ("BACKGROUND", (0,0), (0,0), colors.HexColor("#263238")),
        ]
        for i in range(n):
            for j in range(n):
                try:
                    v = corr_matrix[i][j]
                except Exception:
                    v = None
                cmds.append(("BACKGROUND", (j + 1, i + 1), (j + 1, i + 1), colors.HexColor(_rlh(_corr_color_hex(v)))))
        ct.setStyle(TableStyle(cmds))
        story.append(ct)
        story.append(Spacer(1, 3*mm))

    outliers_per_col = d.get("outliers_per_column") or {}
    if outliers_per_col:
        story.append(P(f"{idx}. IQR 外れ値分析 / 列別 (k={d.get('iqr_k') or 1.5})", sH1))
        idx += 1
        headers = ["列名", "外れ値数", "外れ値率", "IQR", "下限", "上限"]
        o_rows = [[P(h, sHdr) for h in headers]]
        color_cmds = []
        for i, (col, info) in enumerate(sorted(outliers_per_col.items()), start=1):
            pct = float(info.get("pct") or 0)
            color_cmds.append(("BACKGROUND", (2, i), (2, i), colors.HexColor(_rlh(_outlier_color_hex(pct)))))
            o_rows.append([
                P(col, sCell),
                P(num(info.get("count")), sCell),
                P(f"{pct:.2f}%", sCell),
                P(num(info.get("iqr")), sCell),
                P(num(info.get("lower")), sCell),
                P(num(info.get("upper")), sCell),
            ])
        ot = Table(o_rows, colWidths=[30*mm, 18*mm, 18*mm, 28*mm, 28*mm, 28*mm], repeatRows=1)
        cmds = [
            ("GRID", (0,0), (-1,-1), 0.25, colors.grey),
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#37474F")),
            ("ALIGN", (1,0), (-1,-1), "RIGHT"),
            ("FONTNAME", (0,0), (-1,-1), font),
            ("LEFTPADDING", (0,0), (-1,-1), 4),
            ("TOPPADDING", (0,0), (-1,-1), 3),
            ("BOTTOMPADDING", (0,0), (-1,-1), 3),
        ]
        for i in range(1, len(o_rows)):
            cmds.append(("BACKGROUND", (0, i), (0, i), colors.HexColor("#ECEFF1")))
        cmds.extend(color_cmds)
        ot.setStyle(TableStyle(cmds))
        story.append(ot)
        story.append(Spacer(1, 3*mm))

    missing_pct = d.get("missing_pct_by_column") or {}
    missing_by_col = d.get("missing_by_column") or {}
    cols_m = list(set(list(missing_pct.keys()) + list(missing_by_col.keys())))
    if cols_m:
        story.append(P(f"{idx}. 欠損値分析 / 列別", sH1))
        idx += 1
        headers = ["列名", "欠損数", "欠損率", "補完推奨手法"]
        m_rows = [[P(h, sHdr) for h in headers]]
        color_cmds = []
        for i, col in enumerate(sorted(cols_m), start=1):
            cnt = int(missing_by_col.get(col, 0) or 0)
            pct = float(missing_pct.get(col, 0) or 0)
            color_cmds.append(("BACKGROUND", (2, i), (2, i), colors.HexColor(_rlh(_missing_color_hex(pct)))))
            if pct > 20:
                rec = "列の除外 / 再収集"
            elif pct >= 5:
                rec = "数値列:中央値 / カテゴリ:最頻値"
            elif pct > 0:
                rec = "該当行削除"
            else:
                rec = "不要"
            m_rows.append([
                P(col, sCell),
                P(num(cnt), sCell),
                P(f"{pct:.2f}%", sCell),
                P(rec, sCell),
            ])
        mt2 = Table(m_rows, colWidths=[35*mm, 18*mm, 18*mm, 80*mm], repeatRows=1)
        cmds = [
            ("GRID", (0,0), (-1,-1), 0.25, colors.grey),
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#37474F")),
            ("ALIGN", (1,0), (2,-1), "RIGHT"),
            ("FONTNAME", (0,0), (-1,-1), font),
            ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
            ("LEFTPADDING", (0,0), (-1,-1), 4),
            ("TOPPADDING", (0,0), (-1,-1), 3),
            ("BOTTOMPADDING", (0,0), (-1,-1), 3),
        ]
        for i in range(1, len(m_rows)):
            cmds.append(("BACKGROUND", (0, i), (0, i), colors.HexColor("#ECEFF1")))
        cmds.extend(color_cmds)
        mt2.setStyle(TableStyle(cmds))
        story.append(mt2)
        story.append(Spacer(1, 3*mm))

    discussion = d.get("discussion") or ""
    if discussion.strip():
        story.append(P(f"{idx}. 考察・示唆（自動生成）", sH1))
        idx += 1
        for pg in str(discussion).split("\n"):
            pg = pg.strip()
            if pg:
                story.append(P(f"・ {pg}", sBody))
        story.append(Spacer(1, 2*mm))

    action_plan = d.get("action_plan") or []
    if action_plan:
        story.append(P(f"{idx}. アクションプラン（自動生成）", sH1))
        idx += 1
        for item in action_plan:
            story.append(P(f"・ {item}", sBody))
        story.append(Spacer(1, 2*mm))

    sample_rows = d.get("appendix_sample_rows") or []
    if sample_rows and d.get("include_appendix", True):
        story.append(P(f"{idx}. 付録A: サンプルレコード", sH1))
        idx += 1
        columns = list(sample_rows[0].keys()) if sample_rows else []
        if columns:
            n_cols = len(columns)
            avail = 160 * mm
            col_w_a = [max(18*mm, avail / n_cols) for _ in columns]
            a_rows = [[P(c, sHdr) for c in columns]]
            for rec in sample_rows:
                a_rows.append([P("" if rec.get(c) is None else str(rec.get(c)), sCellSmall) for c in columns])
            at = Table(a_rows, colWidths=col_w_a, repeatRows=1)
            cmds = [
                ("GRID", (0,0), (-1,-1), 0.2, colors.grey),
                ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#263238")),
                ("FONTNAME", (0,0), (-1,-1), font),
                ("VALIGN", (0,0), (-1,-1), "TOP"),
                ("LEFTPADDING", (0,0), (-1,-1), 3),
                ("TOPPADDING", (0,0), (-1,-1), 2),
                ("BOTTOMPADDING", (0,0), (-1,-1), 2),
            ]
            at.setStyle(TableStyle(cmds))
            story.append(at)
        story.append(Spacer(1, 2*mm))

    raw_stats = d.get("appendix_raw_stats") or {}
    if raw_stats and d.get("include_appendix", True):
        story.append(P(f"{idx}. 付録B: 生の統計量一覧", sH1))
        idx += 1
        try:
            import json as _json
            raw_txt = _json.dumps(raw_stats, ensure_ascii=False, indent=2, default=str)
        except Exception:
            raw_txt = str(raw_stats)
        lines = raw_txt.splitlines()
        for line in lines[:200]:
            story.append(P(line, sMono))

    doc.build(story)
    payload = buff.getvalue()
    fname = f"会社データ分析報告書_{_sanitize_filename(source or report_date or _today_str())}.pdf"
    return _wrap_report(fname, "pdf", payload)


# ============================================================
# 7. 議事録 (Meeting Minutes)
# ============================================================

_MEETING_TYPE_LABEL = {
    "regular": "定例会議",
    "extraordinary": "臨時会議",
    "emergency": "緊急会議",
}


def _mm_meta_rows(d: dict[str, Any]) -> list[tuple[str, str]]:
    """Build (label, value) rows for the meeting-minutes metadata table, guarding empties with '—'."""
    def _val(v: Any) -> str:
        return str(v) if (v is not None and str(v).strip() != "") else "—"

    start = d.get("start_time") or ""
    end = d.get("end_time") or ""
    if start or end:
        time_val = f"{start} ～ {end}"
    else:
        time_val = "—"
    mt = d.get("meeting_type")
    mt_label = _MEETING_TYPE_LABEL.get(mt or "", _val(mt))
    return [
        ("日時", f"{_fmt_date_reiwa(d.get('meeting_date')) or '—'}   {time_val}"),
        ("場所", _val(d.get("location"))),
        ("種別", mt_label),
        ("議長", _val(d.get("chairperson"))),
        ("書記", _val(d.get("recorder"))),
        ("次回開催", _fmt_date_reiwa(d.get("next_meeting_date")) or "—"),
    ]


def generate_meeting_minutes_docx(data: dict[str, Any]) -> GeneratedReport:
    d = _to_dict(data)
    doc = Document()
    for section in doc.sections:
        section.top_margin = Cm(2.5)
        section.bottom_margin = Cm(2.5)
        section.left_margin = Cm(2.5)
        section.right_margin = Cm(2.5)

    style = doc.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(11)
    style._element.rPr.rFonts.set(qn("w:eastAsia"), "ＭＳ ゴシック")

    title = d.get("title") or "議事録"

    tp = doc.add_heading(title, level=0)
    tp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.add_paragraph()

    # Metadata table
    meta_rows = _mm_meta_rows(d)
    mt = doc.add_table(rows=len(meta_rows), cols=2)
    mt.style = "Light Grid Accent 1"
    for i, (lbl, val) in enumerate(meta_rows):
        row = mt.rows[i].cells
        row[0].width = Cm(4)
        row[1].width = Cm(12)
        row[0].text = lbl
        row[1].text = val
    doc.add_paragraph()

    # Attendees / absentees
    attendees = d.get("attendees") or []
    absentees = d.get("absentees") or []
    doc.add_heading("出席者", level=2)
    doc.add_paragraph("、".join(attendees) if attendees else "—")
    doc.add_heading("欠席者", level=2)
    doc.add_paragraph("、".join(absentees) if absentees else "—")

    # Agenda (numbered)
    agenda = d.get("agenda") or []
    doc.add_heading("議題", level=2)
    if agenda:
        for i, item in enumerate(agenda, start=1):
            doc.add_paragraph(f"{i}. {item}")
    else:
        doc.add_paragraph("—")

    # Discussion (free text)
    doc.add_heading("議事内容", level=2)
    discussion = d.get("discussion")
    if discussion and str(discussion).strip():
        for pg in str(discussion).split("\n"):
            if pg.strip():
                doc.add_paragraph(pg)
    else:
        doc.add_paragraph("—")

    # Decisions (numbered)
    decisions = d.get("decisions") or []
    doc.add_heading("決定事項", level=2)
    if decisions:
        for i, item in enumerate(decisions, start=1):
            doc.add_paragraph(f"{i}. {item}")
    else:
        doc.add_paragraph("—")

    # Action items table
    action_items = d.get("action_items") or []
    doc.add_heading("アクションアイテム", level=2)
    at = doc.add_table(rows=1, cols=4)
    at.style = "Light Grid Accent 1"
    hdr = at.rows[0].cells
    for j, lbl in enumerate(["No.", "タスク", "担当", "期限"]):
        hdr[j].text = lbl
    if action_items:
        for i, it in enumerate(action_items, start=1):
            r = at.add_row().cells
            r[0].text = str(i)
            r[1].text = str(it.get("task") or "—")
            r[2].text = str(it.get("owner") or "—")
            r[3].text = _fmt_date_reiwa(it.get("due_date")) or "—"
    else:
        r = at.add_row().cells
        r[0].text = "—"
        r[1].text = "—"
        r[2].text = "—"
        r[3].text = "—"

    buff = io.BytesIO()
    doc.save(buff)
    payload = buff.getvalue()
    fname = f"議事録_{_sanitize_filename(title)}.docx"
    return _wrap_report(fname, "docx", payload)


def generate_meeting_minutes_pdf(data: dict[str, Any]) -> GeneratedReport:
    d = _to_dict(data)
    buff = io.BytesIO()
    doc = SimpleDocTemplate(buff, pagesize=A4, leftMargin=22*mm, rightMargin=22*mm, topMargin=22*mm, bottomMargin=22*mm)
    font = _pdf_font()
    styles = getSampleStyleSheet()

    sTitle = ParagraphStyle("sTitle", parent=styles["Title"], fontName=font, fontSize=20, leading=26, alignment=TA_CENTER, spaceAfter=4*mm)
    sH2 = ParagraphStyle("sH2", parent=styles["Heading2"], fontName=font, fontSize=12, leading=16, spaceBefore=3*mm, spaceAfter=1.5*mm)
    sBody = ParagraphStyle("sBody", parent=styles["BodyText"], fontName=font, fontSize=10.5, leading=16)
    sCell = ParagraphStyle("sCell", parent=sBody, fontSize=10, leading=13)
    sHdr = ParagraphStyle("sHdr", parent=sCell, fontName=font, fontSize=10, leading=13, textColor=colors.white)

    story = []
    title = d.get("title") or "議事録"
    story.append(Paragraph(title, sTitle))

    # Metadata table
    meta_rows = _mm_meta_rows(d)
    meta_rows_p = [[Paragraph(lbl, sCell), Paragraph(val, sCell)] for lbl, val in meta_rows]
    mt = Table(meta_rows_p, colWidths=[30*mm, 130*mm])
    mt.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), font),
        ("GRID", (0,0), (-1,-1), 0.4, colors.grey),
        ("BACKGROUND", (0,0), (0,-1), colors.HexColor("#f0f2f5")),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("LEFTPADDING", (0,0), (-1,-1), 6),
        ("RIGHTPADDING", (0,0), (-1,-1), 6),
        ("TOPPADDING", (0,0), (-1,-1), 4),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
    ]))
    story.append(mt)
    story.append(Spacer(1, 3*mm))

    attendees = d.get("attendees") or []
    absentees = d.get("absentees") or []
    story.append(Paragraph("出席者", sH2))
    story.append(Paragraph("、".join(attendees) if attendees else "—", sBody))
    story.append(Paragraph("欠席者", sH2))
    story.append(Paragraph("、".join(absentees) if absentees else "—", sBody))

    agenda = d.get("agenda") or []
    story.append(Paragraph("議題", sH2))
    if agenda:
        for i, item in enumerate(agenda, start=1):
            story.append(Paragraph(f"{i}. {item}", sBody))
    else:
        story.append(Paragraph("—", sBody))

    story.append(Paragraph("議事内容", sH2))
    discussion = d.get("discussion")
    if discussion and str(discussion).strip():
        for pg in str(discussion).split("\n"):
            if pg.strip():
                story.append(Paragraph(pg, sBody))
    else:
        story.append(Paragraph("—", sBody))

    decisions = d.get("decisions") or []
    story.append(Paragraph("決定事項", sH2))
    if decisions:
        for i, item in enumerate(decisions, start=1):
            story.append(Paragraph(f"{i}. {item}", sBody))
    else:
        story.append(Paragraph("—", sBody))

    action_items = d.get("action_items") or []
    story.append(Paragraph("アクションアイテム", sH2))
    action_rows = [[
        Paragraph("No.", sHdr),
        Paragraph("タスク", sHdr),
        Paragraph("担当", sHdr),
        Paragraph("期限", sHdr),
    ]]
    if action_items:
        for i, it in enumerate(action_items, start=1):
            action_rows.append([
                Paragraph(str(i), sCell),
                Paragraph(str(it.get("task") or "—"), sCell),
                Paragraph(str(it.get("owner") or "—"), sCell),
                Paragraph(_fmt_date_reiwa(it.get("due_date")) or "—", sCell),
            ])
    else:
        action_rows.append([
            Paragraph("—", sCell),
            Paragraph("—", sCell),
            Paragraph("—", sCell),
            Paragraph("—", sCell),
        ])
    at = Table(action_rows, colWidths=[12*mm, 88*mm, 30*mm, 30*mm], repeatRows=1)
    at.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), font),
        ("GRID", (0,0), (-1,-1), 0.4, colors.grey),
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#2c3e50")),
        ("ALIGN", (0,0), (0,-1), "CENTER"),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("LEFTPADDING", (0,0), (-1,-1), 5),
        ("RIGHTPADDING", (0,0), (-1,-1), 5),
        ("TOPPADDING", (0,0), (-1,-1), 3),
        ("BOTTOMPADDING", (0,0), (-1,-1), 3),
    ]))
    story.append(at)

    doc.build(story)
    payload = buff.getvalue()
    fname = f"議事録_{_sanitize_filename(title)}.pdf"
    return _wrap_report(fname, "pdf", payload)


# ============================================================
# 8. 自由書式文書 (Custom / User-defined Document)
# ============================================================

def generate_custom_document_docx(data: dict[str, Any]) -> GeneratedReport:
    d = _to_dict(data)
    doc = Document()
    for section in doc.sections:
        section.top_margin = Cm(2.5)
        section.bottom_margin = Cm(2.5)
        section.left_margin = Cm(2.5)
        section.right_margin = Cm(2.5)

    style = doc.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(11)
    style._element.rPr.rFonts.set(qn("w:eastAsia"), "ＭＳ ゴシック")

    title = d.get("title") or "文書"

    tp = doc.add_heading(title, level=0)
    tp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.add_paragraph()

    # Optional header block (recipient / author / date)
    recipient = d.get("recipient")
    author = d.get("author")
    document_date = d.get("document_date") or _today_str()
    header_lines = []
    if recipient:
        header_lines.append(f"宛先：{recipient}")
    if author:
        header_lines.append(f"作成者：{author}")
    if document_date:
        header_lines.append(f"日付：{_fmt_date_reiwa(document_date) or document_date}")
    if header_lines:
        hp = doc.add_paragraph()
        hp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        hp.add_run("\n".join(header_lines))
        doc.add_paragraph()

    # Optional intro paragraph
    intro = d.get("intro")
    if intro and str(intro).strip():
        doc.add_paragraph(intro)
        doc.add_paragraph()

    # Custom key-value fields table
    fields = d.get("fields") or []
    doc.add_heading("基本情報", level=2)
    ft = doc.add_table(rows=1, cols=2)
    ft.style = "Light Grid Accent 1"
    ft.rows[0].cells[0].text = "項目"
    ft.rows[0].cells[1].text = "内容"
    if fields:
        for f in fields:
            r = ft.add_row().cells
            r[0].text = str(f.get("label") or "—")
            r[1].text = str(f.get("value") or "") or "—"
    else:
        r = ft.add_row().cells
        r[0].text = "—"
        r[1].text = "—"
    doc.add_paragraph()

    # Custom sections
    sections = d.get("sections") or []
    if sections:
        for sec in sections:
            doc.add_heading(str(sec.get("heading") or "—"), level=1)
            body = sec.get("body") or ""
            if str(body).strip():
                for pg in str(body).split("\n"):
                    if pg.strip():
                        doc.add_paragraph(pg)
            else:
                doc.add_paragraph("—")
            doc.add_paragraph()
    else:
        doc.add_paragraph("—")
        doc.add_paragraph()

    # Optional closing
    closing = d.get("closing")
    if closing and str(closing).strip():
        doc.add_paragraph(closing)

    buff = io.BytesIO()
    doc.save(buff)
    payload = buff.getvalue()
    fname = f"文書_{_sanitize_filename(title)}.docx"
    return _wrap_report(fname, "docx", payload)


def generate_custom_document_pdf(data: dict[str, Any]) -> GeneratedReport:
    d = _to_dict(data)
    buff = io.BytesIO()
    doc = SimpleDocTemplate(buff, pagesize=A4, leftMargin=22*mm, rightMargin=22*mm, topMargin=22*mm, bottomMargin=22*mm)
    font = _pdf_font()
    styles = getSampleStyleSheet()

    sTitle = ParagraphStyle("sTitle", parent=styles["Title"], fontName=font, fontSize=20, leading=26, alignment=TA_CENTER, spaceAfter=4*mm)
    sH1 = ParagraphStyle("sH1", parent=styles["Heading1"], fontName=font, fontSize=14, leading=18, spaceBefore=4*mm, spaceAfter=2*mm)
    sH2 = ParagraphStyle("sH2", parent=styles["Heading2"], fontName=font, fontSize=12, leading=16, spaceBefore=3*mm, spaceAfter=1.5*mm)
    sBody = ParagraphStyle("sBody", parent=styles["BodyText"], fontName=font, fontSize=10.5, leading=16, alignment=TA_JUSTIFY)
    sRight = ParagraphStyle("sRight", parent=sBody, alignment=TA_RIGHT)
    sCell = ParagraphStyle("sCell", parent=sBody, fontSize=10, leading=13)
    sHdr = ParagraphStyle("sHdr", parent=sCell, fontName=font, fontSize=10, leading=13, textColor=colors.white)

    story = []
    title = d.get("title") or "文書"
    story.append(Paragraph(title, sTitle))

    # Optional header block (recipient / author / date)
    recipient = d.get("recipient")
    author = d.get("author")
    document_date = d.get("document_date") or _today_str()
    header_lines = []
    if recipient:
        header_lines.append(f"宛先：{recipient}")
    if author:
        header_lines.append(f"作成者：{author}")
    if document_date:
        header_lines.append(f"日付：{_fmt_date_reiwa(document_date) or document_date}")
    if header_lines:
        story.append(Paragraph("<br/>".join(header_lines), sRight))
        story.append(Spacer(1, 4*mm))

    # Optional intro paragraph
    intro = d.get("intro")
    if intro and str(intro).strip():
        story.append(Paragraph(intro, sBody))
        story.append(Spacer(1, 3*mm))

    # Custom key-value fields table
    fields = d.get("fields") or []
    story.append(Paragraph("基本情報", sH2))
    field_rows = [[
        Paragraph("項目", sHdr),
        Paragraph("内容", sHdr),
    ]]
    if fields:
        for f in fields:
            field_rows.append([
                Paragraph(str(f.get("label") or "—"), sCell),
                Paragraph(str(f.get("value") or "") or "—", sCell),
            ])
    else:
        field_rows.append([
            Paragraph("—", sCell),
            Paragraph("—", sCell),
        ])
    ft = Table(field_rows, colWidths=[45*mm, 115*mm], repeatRows=1)
    ft.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), font),
        ("GRID", (0,0), (-1,-1), 0.4, colors.grey),
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#2c3e50")),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("LEFTPADDING", (0,0), (-1,-1), 6),
        ("RIGHTPADDING", (0,0), (-1,-1), 6),
        ("TOPPADDING", (0,0), (-1,-1), 4),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
    ]))
    story.append(ft)
    story.append(Spacer(1, 3*mm))

    # Custom sections
    sections = d.get("sections") or []
    if sections:
        for sec in sections:
            story.append(Paragraph(str(sec.get("heading") or "—"), sH1))
            body = sec.get("body") or ""
            if str(body).strip():
                for pg in str(body).split("\n"):
                    if pg.strip():
                        story.append(Paragraph(pg, sBody))
            else:
                story.append(Paragraph("—", sBody))
            story.append(Spacer(1, 2*mm))
    else:
        story.append(Paragraph("—", sBody))
        story.append(Spacer(1, 2*mm))

    # Optional closing
    closing = d.get("closing")
    if closing and str(closing).strip():
        story.append(Spacer(1, 2*mm))
        story.append(Paragraph(closing, sBody))

    doc.build(story)
    payload = buff.getvalue()
    fname = f"文書_{_sanitize_filename(title)}.pdf"
    return _wrap_report(fname, "pdf", payload)


# ============================================================
# Dispatcher
# ============================================================

GENERATOR_MAP: dict[str, dict[str, callable]] = {
    "japanese-resume": {"docx": generate_resume_docx, "pdf": generate_resume_pdf},
    "career-history": {"docx": generate_career_history_docx, "pdf": generate_career_history_pdf},
    "invoice": {"docx": generate_invoice_docx, "pdf": generate_invoice_pdf},
    "business-report": {"docx": generate_business_report_docx, "pdf": generate_business_report_pdf},
    "cover-letter": {"docx": generate_cover_letter_docx, "pdf": generate_cover_letter_pdf},
    "company-analysis-report": {"docx": generate_company_analysis_report_docx, "pdf": generate_company_analysis_report_pdf},
    "meeting-minutes": {"docx": generate_meeting_minutes_docx, "pdf": generate_meeting_minutes_pdf},
    "custom-document": {"docx": generate_custom_document_docx, "pdf": generate_custom_document_pdf},
}


def generate_official_document(template_id: str, doc_format: str, data: dict[str, Any]) -> GeneratedReport:
    if template_id not in GENERATOR_MAP:
        raise ValueError(f"Unknown template_id: {template_id}. Available: {list(GENERATOR_MAP.keys())}")
    fmt = (doc_format or "docx").lower()
    if fmt not in GENERATOR_MAP[template_id]:
        raise ValueError(f"Unsupported format {fmt} for template {template_id}")
    return GENERATOR_MAP[template_id][fmt](data)


# ============================================================
# Self-test (parse-time only; runs only if executed directly)
# ============================================================

# ============================================================
# Verification / Self-test (all 10 generators, edge cases)
# ============================================================

@dataclass
class DocVerificationResult:
    template_id: str
    doc_format: str
    ok: bool
    filename: str = ""
    size_bytes: int = 0
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "template_id": self.template_id,
            "format": self.doc_format,
            "ok": self.ok,
            "filename": self.filename,
            "size_bytes": self.size_bytes,
            "error": self.error,
        }


# ---- Sample data for each template (valid, realistic) ----

_SAMPLE_RESUME: dict[str, Any] = {
    "full_name_kanji": "佐藤健一",
    "full_name_kana": "さとうけんいち",
    "issue_date": "2026-04-01",
    "date_of_birth": "1995-03-15",
    "gender": "male",
    "postal_code": "150-0001",
    "address_prefecture": "東京都",
    "address_city": "渋谷区",
    "address_detail": "神宮前1-2-3 神宮前マンション405",
    "phone": "090-1234-5678",
    "email": "kenichi.sato@example.co.jp",
    "education_list": [
        {"start_date": "2013-04-01", "end_date": "2017-03-31",
         "school_name": "東京都立○○高等学校", "department": "普通科",
         "degree": "", "is_graduated": True},
        {"start_date": "2017-04-01", "end_date": "2021-03-31",
         "school_name": "○○大学", "department": "工学部 情報工学科",
         "degree": "学士（工学）", "is_graduated": True},
    ],
    "work_history_list": [
        {"start_date": "2021-04-01", "end_date": "2024-06-30",
         "company_name": "株式会社ABCテクノロジー", "position": "シニアエンジニア",
         "description": "基幹システムの設計・開発を担当", "is_current": False},
        {"start_date": "2024-07-01", "end_date": "",
         "company_name": "株式会社XYZソリューションズ", "position": "プロジェクトリーダー",
         "description": "AIプロジェクトを統括", "is_current": True},
    ],
    "licenses_list": [
        {"acquisition_date": "2019-06-01", "license_name": "基本情報技術者試験", "issuer": "IPA"},
        {"acquisition_date": "2022-09-01", "license_name": "応用情報技術者試験", "issuer": "IPA"},
        {"acquisition_date": "2023-12-01", "license_name": "AWS Certified Solutions Architect – Associate", "issuer": "Amazon Web Services"},
    ],
    "motivation": "貴社のAI事業を技術面から牽引し、事業成長に貢献したいと考えております。前職で培った大規模システム開発経験とチームマネジメントの経験を活かし、即戦力として活躍できます。",
    "hobbies": "読書（主に技術書・経営書）、週末はサイクリングで体を鍛えています。",
    "self_pr": "Python・FastAPI・PostgreSQLを用いたシステム設計からリリースまで一貫して担当しました。AI / MLプロジェクトでは、データ基盤構築からモデル配備までリードし、業務効率を35%改善した実績があります。",
}

_SAMPLE_CAREER: dict[str, Any] = {
    "full_name_kanji": "佐藤健一",
    "full_name_kana": "さとうけんいち",
    "phone": "090-1234-5678",
    "email": "kenichi.sato@example.co.jp",
    "issue_date": "2026-04-01",
    "career_summary": "情報システム開発一筋で5年間の実務経験があります。Webアプリ・API開発を中心に、要件定義～設計～開発～運用保守まで一貫して担当可能。直近2年はAI・機械学習プロジェクトをリードし、業務改善・生産性向上に寄与してきました。",
    "roles": [
        {
            "period_from": "2024年7月", "period_to": "現在",
            "company_name": "株式会社XYZソリューションズ",
            "position": "プロジェクトリーダー", "department": "AI事業部",
            "overview": "企業向けAIソリューション開発チームの統括（5名）",
            "projects": [
                {"period_from": "2024年9月", "period_to": "2026年2月",
                 "project_name": "社内業務自動化AI基盤構築", "role": "PL / アーキテクト",
                 "team_size": 5,
                 "technologies": ["Python 3.10", "FastAPI", "PostgreSQL", "scikit-learn", "AWS"],
                 "achievements": ["業務効率35%改善", "月間工数120時間削減", "コード品質向上 +42%"],
                 "overview": "RPA+MLの社内基盤を設計・構築"},
                {"period_from": "2025年3月", "period_to": "2026年3月",
                 "project_name": "自然言語処理・文書生成システム",
                 "role": "開発リーダー", "team_size": 4,
                 "technologies": ["SudachiPy", "FastAPI", "Streamlit", "faster-whisper", "PostgreSQL"],
                 "achievements": ["5種公式文書テンプレート実装", "処理性能2.4倍に向上"],
                 "overview": "日本語NLP + 文書自動生成エンジン"},
            ],
        },
        {
            "period_from": "2021年4月", "period_to": "2024年6月",
            "company_name": "株式会社ABCテクノロジー",
            "position": "シニアエンジニア", "department": "基幹システム事業部",
            "overview": "顧客企業向け基幹システムのフルスタック開発",
            "projects": [
                {"period_from": "2022年1月", "period_to": "2023年12月",
                 "project_name": "生産管理システムのフルリプレース",
                 "role": "バックエンド担当", "team_size": 10,
                 "technologies": ["Java 17", "Spring Boot", "PostgreSQL", "Vue.js"],
                 "achievements": ["月次決算処理を65%高速化", "障害発生率を -70%"],
                 "overview": "在庫・生産・会計を統合する基幹系のマイグレーション"},
            ],
        },
    ],
    "skills": ["プロジェクトマネジメント", "アーキテクチャ設計", "コードレビュー", "チーム育成", "顧客折衝"],
    "tech_stack": {
        "言語": ["Python 3.10", "Java", "TypeScript", "SQL"],
        "FW / ライブラリ": ["FastAPI", "Streamlit", "Gradio", "scikit-learn", "pandas", "Spring Boot"],
        "DB / インフラ": ["PostgreSQL", "Redis", "AWS", "Docker"],
        "その他": ["SudachiPy", "faster-whisper", "ReportLab", "python-docx"],
    },
    "education": "2017年4月 ～ 2021年3月  ○○大学 工学部 情報工学科  学士（工学）卒業",
    "certifications": ["基本情報技術者試験 (2019/06)", "応用情報技術者試験 (2022/09)", "AWS Certified Solutions Architect – Associate (2023/12)"],
}

_SAMPLE_INVOICE: dict[str, Any] = {
    "invoice_number": "INV-2026-1001",
    "issue_date": "2026-04-01",
    "due_date": "2026-04-30",
    "buyer_name": "株式会社ABC商事",
    "buyer_department": "購買部",
    "buyer_address": "東京都千代田区丸の内1-9-2",
    "recipient_name": "購買部長",
    "seller_name": "株式会社XYZソリューションズ",
    "seller_department": "経営管理部",
    "seller_address": "東京都渋谷区神宮前1-2-3",
    "seller_phone": "03-1234-5678",
    "seller_tax_id": "T1234567890123",
    "title": "請求書",
    "items": [
        {"description": "AIソリューション開発費（2026年3月分）",
         "quantity": 1, "unit_price": 850000, "tax_rate": 0.10},
        {"description": "月次サポート保守費",
         "quantity": 1, "unit_price": 150000, "tax_rate": 0.10},
        {"description": "追加モジュール開発：文書自動生成",
         "quantity": 20, "unit_price": 25000, "tax_rate": 0.10},
        {"description": "サーバー利用料",
         "quantity": 1, "unit_price": 80000, "tax_rate": 0.10},
    ],
    "notes": "本請求書は、2026年3月実績に基づくものです。期日までに指定口座へお振込みください。",
    "payment_method": "bank_transfer",
    "bank_name": "みずほ銀行",
    "bank_branch": "渋谷支店",
    "bank_account_type": "ordinary",
    "bank_account_number": "1234567",
    "bank_account_holder": "カブシキガイシャXYZ",
}

_SAMPLE_REPORT: dict[str, Any] = {
    "title": "2026年第1四半期 AI事業部 業務実績報告書",
    "subtitle": "～ 生産性向上・新規プロジェクトの推進状況 ～",
    "report_date": "2026-04-05",
    "author": "佐藤健一",
    "author_department": "AI事業部 プロジェクト1課",
    "recipient": "取締役社長",
    "recipient_department": "経営陣",
    "executive_summary": "2026年第1四半期、AI事業部は売上高前年同期比 +18%、利益率 +2.3pt を達成しました。文書自動生成・音声認識基盤の2プロダクトが正式リリースされ、稼働開始しています。",
    "background": "2025年度よりスタートした中期計画「DX2027」に基づき、社内のコア業務へAIを横断展開することが経営課題となっていた。特に文書業務・コールセンター業務の効率化は、従業員満足度と顧客満足度の両面から早期実現が求められていた。",
    "objectives": [
        "売上高 前年同期比 +15% 以上を達成する",
        "文書自動生成システムを4月に本番稼働させ、経理・人事部門の工数を30%削減する",
        "コールセンター向け音声認識・通話要約ソリューションをリリースする",
        "新規機械学習プロジェクトを3件以上開始する",
    ],
    "methods": "文書自動生成: python-docx + ReportLab による日本語公式文書5種類のテンプレート実装、FastAPI の REST API 化。音声認識: faster-whisper (CTranslate2) を採用し、日本語バイアス + VAD フィルタによる高精度化。ダッシュボード: Streamlit + Plotly によるインタラクティブな KPI 可視化基盤の構築。",
    "results": "・売上高 前年比 +18%（目標 +15% を達成）。・営業利益率 12.8%（前年同期 10.5% → +2.3pt）。・文書自動生成システム: 5テンプレート DOCX/PDF 両対応。4/1本番稼働開始。・音声認識ソリューション: 文字起こし精度 96.2%（従来比 +8.7pt）。・新規MLプロジェクト: 4件開始（分類・クラスタリング・意図検出・翻訳品質推定）。",
    "discussion": "音声認識の精度向上は larger-v3 モデルと日本語 initial_prompt の相乗効果が大きかった。文書生成はテンプレート毎に罫線・フォントの調整に工数を要したが、QAフェーズで連続稼働テストを実施し、99.6%の生成成功率を確認した。",
    "issues": [
        "モデル推論のレイテンシがピーク時に一時的に増加（+20%）する事象が2回発生。",
        "日本語PDF 生成時のマルチバイトフォント描画について、追加の外字対応が今後必要。",
        "データセット品質のばらつきにより、一部業務での ML 精度目標達成に追加チューニングが必要。",
    ],
    "conclusion": "Q1は数値目標（売上・利益率）をいずれも上回り、主要プロダクトの本番リリースも達成。Q2以降は本番利用から得られるフィードバックを反映し、更なる品質改善・コスト最適化に取り組む。",
    "recommendations": [
        "Q2推論基盤のオートスケーリング強化とキャッシュ戦略の見直しを実施し、レイテンシを安定化させる。",
        "ReportLab の外字/縦書き対応を強化し、より多様な公式文書（機密保持契約書など）へ展開する。",
        "データ品質パイプラインを整備し、ラベル付けプロセスとデータレビューをチーム内で標準化する。",
        "KANAMEダッシュボードへ ML 品質メトリクスを常時表示し、モデルドリフトを検知可能な体制を構築する。",
    ],
    "references": [
        "中期経営計画『DX2027』（2025年4月 経営陣決定）",
        "文書自動生成システム 内部設計書 v2.1（2026年3月）",
        "faster-whisper ベンチマーク結果 社内資料（2026年1月）",
    ],
    "attachments": [
        "資料1: Q1 決算数値の内訳（エクセル）",
        "資料2: 文書生成システム 稼働モニタリング結果",
        "資料3: 音声認識ソリューション 精度レポート",
    ],
}

_SAMPLE_COVER_LETTER: dict[str, Any] = {
    "letter_date": "2026-04-01",
    "recipient_company": "株式会社ABC商事",
    "recipient_department": "購買部",
    "recipient_title": "購買部長",
    "recipient_name": "",
    "subject": "御社発注書 №PO-2026-0389 納品のご案内",
    "greeting": "お世話になっております。",
    "body_paragraphs": [
        "この度は御社発注書№PO-2026-0389に基づき、下記の通り納品させていただきます。",
        "納品物は、商品セット一式＋付属ドキュメント＋導入支援ガイドとなります。インストール作業、動作確認は、担当者が貴社指定日時に伺い実施いたします。",
        "なお、契約書・見積書に記載の通り、導入後30日間の無償保守をご提供いたします。不具合・ご不明点がございましたら、いつでも担当窓口までご連絡ください。",
        "何卒よろしくお願い申し上げます。",
    ],
    "closing": "ご確認の程、何卒よろしくお願い申し上げます。",
    "enclosures": [
        "納品書 №ND-2026-0417",
        "請求書 №INV-2026-1001",
        "導入支援ガイド v1.3",
        "製品仕様書（ハードコピー）",
        "保守契約書（控え）",
    ],
    "sender_postal_code": "150-0001",
    "sender_address": "東京都渋谷区神宮前1-2-3",
    "sender_company": "株式会社XYZソリューションズ",
    "sender_department": "AI事業部 営業グループ",
    "sender_name": "佐藤健一",
    "sender_phone": "03-1234-5678",
    "sender_email": "sales-support@xyz-solutions.example.co.jp",
}

_SAMPLE_MEETING_MINUTES: dict[str, Any] = {
    "title": "2026年度 第1回 定例プロジェクト会議 議事録",
    "meeting_date": "2026-04-10",
    "start_time": "14:00",
    "end_time": "15:30",
    "location": "本社 3階 会議室A",
    "meeting_type": "regular",
    "chairperson": "鈴木一郎",
    "recorder": "佐藤花子",
    "attendees": ["鈴木一郎", "佐藤花子", "山田太郎", "田中次郎", "高橋三郎"],
    "absentees": ["伊藤四郎"],
    "agenda": [
        "前回議事録の確認",
        "第1四半期の進捗報告",
        "文書自動生成システムのリリース判定",
        "次期開発計画の協議",
    ],
    "discussion": "前回議事録の内容を確認し、全項目について異議なく承認された。第1四半期の進捗は売上・品質の両面で目標を上回り、順調に推移していることが報告された。文書自動生成システムは DOCX / PDF の両形式で生成成功率99.6%を達成しており、リリース判定基準を満たしていることが確認された。次期開発計画については、音声認識ソリューションの横展開とデータ品質パイプラインの整備を優先する方針で合意した。",
    "decisions": [
        "文書自動生成システムを4月15日に本番リリースする。",
        "次期開発は音声認識の横展開を最優先とする。",
        "データ品質パイプライン整備の担当をデータ戦略室に決定する。",
    ],
    "action_items": [
        {"task": "リリース判定資料の最終化", "owner": "佐藤花子", "due_date": "2026-04-12"},
        {"task": "本番環境のスモークテスト実施", "owner": "山田太郎", "due_date": "2026-04-14"},
        {"task": "次期開発計画書のドラフト作成", "owner": "田中次郎", "due_date": "2026-04-22"},
    ],
    "next_meeting_date": "2026-05-08",
}

_SAMPLE_CUSTOM_DOCUMENT: dict[str, Any] = {
    "title": "新規システム導入 ご提案書",
    "author": "佐藤健一",
    "document_date": "2026-04-15",
    "recipient": "株式会社ABC商事 御中",
    "intro": "平素より大変お世話になっております。この度、貴社の業務効率化を目的とした新規システム導入について、下記のとおりご提案申し上げます。",
    "fields": [
        {"label": "案件名", "value": "KANAME 文書自動生成システム導入"},
        {"label": "提案金額", "value": "3,500,000円（税別）"},
        {"label": "導入予定期間", "value": "2026年5月 ～ 2026年7月"},
        {"label": "担当者", "value": "佐藤健一（AI事業部）"},
    ],
    "sections": [
        {"heading": "1. 提案の背景", "body": "貴社では経理・人事部門を中心に定型文書の作成業務が多く、月間で約120時間の工数が発生しています。手作業による作成は属人化や表記ゆれのリスクも伴います。"},
        {"heading": "2. 提案内容", "body": "KANAME の文書自動生成システムを導入し、履歴書・請求書・報告書などの定型文書をテンプレート化します。API 経由で既存システムとも連携可能です。"},
        {"heading": "3. 期待される効果", "body": "文書作成工数を約30%削減し、表記ゆれの解消と監査対応の効率化を実現します。加えて、帳票の電子化によりペーパーレス化にも貢献します。"},
        {"heading": "4. 導入スケジュール", "body": "5月に要件定義、6月に構築・テスト、7月上旬に本番稼働を予定しています。導入後は30日間の無償保守を提供いたします。"},
    ],
    "closing": "ご検討の程、何卒よろしくお願い申し上げます。",
}

def _build_sample_company_analysis_report() -> dict[str, Any]:
    import pandas as pd
    rows_20 = [
        {"date": "2026-01-05", "region": "東京", "product": "A", "sales_units": 120, "revenue": 2880000, "cost": 2160000},
        {"date": "2026-01-12", "region": "大阪", "product": "B", "sales_units": 85,  "revenue": 1870000, "cost": 1496000},
        {"date": "2026-01-19", "region": "東京", "product": "A", "sales_units": 135, "revenue": 3240000, "cost": 2430000},
        {"date": "2026-01-26", "region": "愛知", "product": "C", "sales_units": 52,  "revenue": 1560000, "cost": 1248000},
        {"date": "2026-02-02", "region": "東京", "product": "B", "sales_units": 98,  "revenue": 2156000, "cost": 1724800},
        {"date": "2026-02-09", "region": "大阪", "product": "A", "sales_units": 110, "revenue": 2640000, "cost": 1980000},
        {"date": "2026-02-16", "region": "福岡", "product": "C", "sales_units": 41,  "revenue": 1230000, "cost": 984000},
        {"date": "2026-02-23", "region": "東京", "product": "A", "sales_units": 148, "revenue": 3552000, "cost": 2664000},
        {"date": "2026-03-02", "region": "愛知", "product": "B", "sales_units": 72,  "revenue": 1584000, "cost": 1267200},
        {"date": "2026-03-09", "region": "大阪", "product": "C", "sales_units": 60,  "revenue": 1800000, "cost": 1440000},
        {"date": "2026-03-16", "region": "東京", "product": "B", "sales_units": 102, "revenue": 2244000, "cost": 1795200},
        {"date": "2026-03-23", "region": "福岡", "product": "A", "sales_units": 88,  "revenue": 2112000, "cost": 1584000},
        {"date": "2026-03-30", "region": "東京", "product": "C", "sales_units": 77,  "revenue": 2310000, "cost": 1848000},
        {"date": "2026-04-06", "region": "大阪", "product": "B", "sales_units": 95,  "revenue": 2090000, "cost": 1672000},
        {"date": "2026-04-13", "region": "愛知", "product": "A", "sales_units": 125, "revenue": 3000000, "cost": 2250000},
        {"date": "2026-04-20", "region": "東京", "product": "A", "sales_units": 140, "revenue": 3360000, "cost": 2520000},
        {"date": "2026-04-27", "region": "大阪", "product": "C", "sales_units": 55,  "revenue": 1650000, "cost": 1320000},
        {"date": "2026-05-04", "region": "福岡", "product": "B", "sales_units": 70,  "revenue": 1540000, "cost": 1232000},
        {"date": "2026-05-11", "region": "東京", "product": "C", "sales_units": 82,  "revenue": 2460000, "cost": 1968000},
        {"date": "2026-05-18", "region": "愛知", "product": "A", "sales_units": 130, "revenue": 3120000, "cost": 2340000},
    ]
    df = pd.DataFrame(rows_20)
    meta = {
        "report_title": "2026年 上期 売上データ分析報告書",
        "report_date": "2026-05-20",
        "period_covered": "2026年1月5日 ～ 2026年5月18日",
        "company_name": "株式会社XYZソリューションズ",
        "department": "データ戦略室 AI事業部",
        "author_name": "佐藤健一",
        "analyst_name": "山田太郎",
        "recipient_name": "経営会議",
        "source_filename": "sales_q1q2_2026.csv",
    }
    if _ANALYZER_AVAILABLE:
        return build_company_analysis_report_from_dataset(df, meta, iqr_k=1.5, corr_method="pearson", include_raw_appendix=True)
    return {
        "report_title": meta["report_title"], "report_date": meta["report_date"],
        "period_covered": meta["period_covered"], "company_name": meta["company_name"],
        "department": meta["department"], "author_name": meta["author_name"],
        "analyst_name": meta["analyst_name"], "recipient_name": meta["recipient_name"],
        "source_filename": meta["source_filename"], "iqr_k": 1.5, "corr_method": "pearson",
        "kpis": [{"name": "総行数", "value": len(rows_20), "unit": "行", "color": "1565C0"}],
        "dataset_overview": {"rows": len(rows_20), "columns": len(rows_20[0]),
                             "column_names": list(rows_20[0].keys()),
                             "dtypes": {k: "object" for k in rows_20[0]},
                             "memory_bytes": 0, "missing_total": 0, "missing_by_column": {k: 0 for k in rows_20[0]}},
        "numeric_stats": [], "categorical_stats": [], "top_categories": [],
        "correlation": {"columns": [], "index": [], "matrix": [], "method": "pearson"},
        "outliers_per_column": {}, "missing_by_column": {}, "missing_pct_by_column": {},
        "discussion": "データ分析ライブラリが利用できないため、サンプルデータでのみレンダリングしています。",
        "action_plan": ["ai_engine.data_intelligence.analyzer をインポート可能であることを確認してください。"],
        "appendix_sample_rows": rows_20[:5], "appendix_raw_stats": {}, "include_appendix": True,
    }


_SAMPLE_COMPANY_ANALYSIS_REPORT: dict[str, Any] = _build_sample_company_analysis_report()


_SAMPLE_MAP: dict[str, dict[str, Any]] = {
    "japanese-resume": _SAMPLE_RESUME,
    "career-history": _SAMPLE_CAREER,
    "invoice": _SAMPLE_INVOICE,
    "business-report": _SAMPLE_REPORT,
    "cover-letter": _SAMPLE_COVER_LETTER,
    "company-analysis-report": _SAMPLE_COMPANY_ANALYSIS_REPORT,
    "meeting-minutes": _SAMPLE_MEETING_MINUTES,
    "custom-document": _SAMPLE_CUSTOM_DOCUMENT,
}


# ---- Edge-case data (empty lists, missing optional fields) ----

_EDGE_COUNTER = [0]


def _edge_case_data(template_id: str) -> dict[str, Any]:
    """Minimal / edge-case data to ensure generators don't crash on empty inputs."""
    if template_id == "japanese-resume":
        return {
            "full_name_kanji": "",
            "full_name_kana": "",
            "date_of_birth": "",
            "education_list": [],
            "work_history_list": [],
            "licenses_list": [],
        }
    if template_id == "career-history":
        return {"full_name_kanji": "", "roles": [], "skills": []}
    if template_id == "invoice":
        return {"items": [], "seller_name": ""}
    if template_id == "business-report":
        return {"title": "タイトルのみ", "background": "背景のみ", "objectives": [], "issues": [], "references": [], "attachments": []}
    if template_id == "cover-letter":
        return {"body_paragraphs": [], "enclosures": []}
    if template_id == "meeting-minutes":
        return {
            "title": "タイトルのみ",
            "attendees": [],
            "absentees": [],
            "agenda": [],
            "decisions": [],
            "action_items": [],
            "discussion": "",
        }
    if template_id == "custom-document":
        return {"title": "タイトルのみ", "sections": [], "fields": []}
    if template_id == "company-analysis-report":
        import pandas as pd
        _EDGE_COUNTER[0] = (_EDGE_COUNTER[0] + 1) % 4
        variant = _EDGE_COUNTER[0]
        meta_fallback = {
            "report_title": "エッジケース分析報告書",
            "report_date": _today_str(),
            "company_name": "(株)EdgeCase",
            "department": "QA",
            "author_name": "自動テスト",
            "source_filename": "edge_case.csv",
        }
        if variant == 0:
            return build_company_analysis_report_from_dataset(pd.DataFrame(), meta_fallback)
        if variant == 1:
            df_single = pd.DataFrame({"only_category": ["X", "Y", "X", "Z", "Y", "X"]})
            return build_company_analysis_report_from_dataset(df_single, meta_fallback)
        if variant == 2:
            df_all_nan = pd.DataFrame({"a": [None, None, None], "b": [float("nan"), float("nan"), float("nan")]})
            return build_company_analysis_report_from_dataset(df_all_nan, meta_fallback)
        df_cat = pd.DataFrame({
            "region": ["東京", "大阪", "東京", "愛知", "福岡", "大阪", "東京", "東京", "愛知", "福岡"],
            "product": ["A", "B", "C", "A", "B", "C", "A", "B", "C", "A"],
            "channel": ["Web", "店舗", "Web", "代理店", "Web", "店舗", "Web", "代理店", "店舗", "Web"],
        })
        return build_company_analysis_report_from_dataset(df_cat, meta_fallback)
    return {}


def verify_all_documents(
    output_dir: str | None = None,
    include_edge_cases: bool = True,
) -> dict[str, Any]:
    """Runs all 10 document generators with sample (and optionally edge-case) data.

    Returns a verification report with:
      - per-template per-format success + size_bytes
      - count of pass / fail
      - total bytes generated
      - any error messages

    If output_dir is provided, generated bytes are also written to disk there
    (useful for manual QA inspection).
    """
    if output_dir:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
    else:
        out_path = None

    results: list[DocVerificationResult] = []
    total_bytes = 0
    passes = 0
    fails = 0

    for template_id, sample in _SAMPLE_MAP.items():
        for fmt in ("docx", "pdf"):
            try:
                rep = generate_official_document(template_id, fmt, sample)
                assert rep.bytes and len(rep.bytes) > 0, "empty payload"
                results.append(DocVerificationResult(
                    template_id=template_id, doc_format=fmt, ok=True,
                    filename=rep.filename, size_bytes=rep.size_bytes,
                ))
                total_bytes += rep.size_bytes
                passes += 1
                if out_path is not None:
                    (out_path / rep.filename).write_bytes(rep.bytes)
            except Exception as e:
                fails += 1
                results.append(DocVerificationResult(
                    template_id=template_id, doc_format=fmt, ok=False,
                    error=f"sample data — {type(e).__name__}: {e}",
                ))

    if include_edge_cases:
        for template_id in GENERATOR_MAP:
            edge = _edge_case_data(template_id)
            for fmt in ("docx", "pdf"):
                try:
                    rep = generate_official_document(template_id, fmt, edge)
                    assert rep.bytes and len(rep.bytes) > 0, "empty payload (edge-case)"
                    results.append(DocVerificationResult(
                        template_id=f"{template_id}[edge]", doc_format=fmt, ok=True,
                        filename=rep.filename, size_bytes=rep.size_bytes,
                    ))
                    total_bytes += rep.size_bytes
                    passes += 1
                    if out_path is not None:
                        edge_fn = f"edge_{template_id}_{fmt}_{rep.filename}"
                        (out_path / edge_fn).write_bytes(rep.bytes)
                except Exception as e:
                    fails += 1
                    results.append(DocVerificationResult(
                        template_id=f"{template_id}[edge]", doc_format=fmt, ok=False,
                        error=f"edge-case data — {type(e).__name__}: {e}",
                    ))

    errors = [r.to_dict() for r in results if not r.ok]
    return {
        "total_checks": len(results),
        "passed": passes,
        "failed": fails,
        "total_bytes_generated": total_bytes,
        "all_ok": fails == 0,
        "results": [r.to_dict() for r in results],
        "errors": errors,
    }


if __name__ == "__main__":
    public_fns = [
        generate_resume_docx, generate_resume_pdf,
        generate_career_history_docx, generate_career_history_pdf,
        generate_invoice_docx, generate_invoice_pdf,
        generate_business_report_docx, generate_business_report_pdf,
        generate_cover_letter_docx, generate_cover_letter_pdf,
        generate_company_analysis_report_docx, generate_company_analysis_report_pdf,
        generate_meeting_minutes_docx, generate_meeting_minutes_pdf,
        generate_custom_document_docx, generate_custom_document_pdf,
        build_company_analysis_report_from_dataset,
        generate_official_document,
    ]
    for fn in public_fns:
        assert callable(fn), f"Not callable: {fn}"
        print(fn.__name__)
    print("GENERATOR_MAP keys:", list(GENERATOR_MAP.keys()))
    n_gen = len(GENERATOR_MAP)
    print(f"All imports OK, all {n_gen*2} generators + bridge + 1 dispatcher present.")
    print("\n--- Running verify_all_documents() ---")
    report = verify_all_documents(output_dir=None, include_edge_cases=True)
    print(f"  Checks : {report['total_checks']}  (expected: sample {n_gen*2} + edge {n_gen*2} = {n_gen*4})")
    print(f"  Passed : {report['passed']}")
    print(f"  Failed : {report['failed']}")
    print(f"  Total bytes : {report['total_bytes_generated']:,}")
    print(f"  All OK ? : {report['all_ok']}")
    for err in report["errors"]:
        print("   FAIL:", err["template_id"], err["format"], "→", err["error"])
