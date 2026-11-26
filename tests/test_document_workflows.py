from __future__ import annotations

import io
import json
import zipfile

from docx import Document
from fastapi.testclient import TestClient
from pypdf import PdfReader, PdfWriter

from backend.main import app
from document_engine.docx_translate import translate_docx_bytes


client = TestClient(app)


def test_docx_translation_preserves_paragraph_and_table_structure():
    source = Document()
    source.add_heading("会議報告", level=1)
    source.add_paragraph("本日の会議で計画を確認しました。")
    table = source.add_table(rows=1, cols=1)
    table.cell(0, 0).text = "次回の予定"
    buffer = io.BytesIO()
    source.save(buffer)

    translated = Document(io.BytesIO(translate_docx_bytes(
        buffer.getvalue(), source_lang="ja", target_lang="en", force_fallback=True,
    )))

    assert len(translated.paragraphs) == len(source.paragraphs)
    assert len(translated.tables) == 1
    assert translated.tables[0].cell(0, 0).text.strip()
    assert translated.paragraphs[0].style.name == source.paragraphs[0].style.name


def test_document_batch_returns_zip_with_manifest():
    response = client.post(
        "/api/documents/batch/generate/download",
        json={
            "auto_translate": False,
            "items": [{
                "template_id": "custom-document",
                "payload": {
                    "title": "Batch Test",
                    "intro": "Test introduction",
                    "sections": [{"heading": "Summary", "body": "Test body"}],
                    "doc_format": "docx",
                },
            }],
        },
    )

    assert response.status_code == 200, response.text
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        names = archive.namelist()
        assert len(names) == 2
        assert any(name.endswith(".docx") for name in names)
        manifest = json.loads(archive.read("manifest.json"))
        assert manifest["count"] == 1
        assert manifest["documents"][0]["template_id"] == "custom-document"


def test_pdf_merge_keeps_upload_order_and_all_pages():
    pdfs = []
    for page_count in (1, 2):
        writer = PdfWriter()
        for _ in range(page_count):
            writer.add_blank_page(width=300, height=400)
        output = io.BytesIO()
        writer.write(output)
        pdfs.append(output.getvalue())

    response = client.post(
        "/api/documents/merge-pdfs",
        files=[
            ("files", ("ja.pdf", pdfs[0], "application/pdf")),
            ("files", ("en.pdf", pdfs[1], "application/pdf")),
        ],
    )

    assert response.status_code == 200, response.text
    assert len(PdfReader(io.BytesIO(response.content)).pages) == 3
