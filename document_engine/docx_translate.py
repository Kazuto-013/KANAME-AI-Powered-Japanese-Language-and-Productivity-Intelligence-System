"""Translate DOCX text while retaining its paragraph, table, and section layout."""
from __future__ import annotations

import io

from docx import Document

from ai_engine.translation.translator import translate_text


def _iter_paragraphs(container, seen: set[int]):
    for paragraph in container.paragraphs:
        marker = id(paragraph._p)
        if marker not in seen:
            seen.add(marker)
            yield paragraph
    for table in container.tables:
        for row in table.rows:
            for cell in row.cells:
                yield from _iter_paragraphs(cell, seen)


def translate_docx_bytes(
    payload: bytes,
    source_lang: str = "auto",
    target_lang: str = "ja",
    domain: str | None = None,
    force_fallback: bool = False,
) -> bytes:
    """Translate text paragraph-by-paragraph without flattening DOCX structure.

    Paragraph and table structure, section settings, and paragraph styles are
    retained. Inline run styling within a translated paragraph follows its first
    non-empty run because translation may change word and sentence boundaries.
    """
    if target_lang not in ("ja", "en") or source_lang not in ("auto", "ja", "en"):
        raise ValueError("source_lang must be auto/ja/en and target_lang must be ja/en")
    document = Document(io.BytesIO(payload))
    seen: set[int] = set()
    containers = [document]
    for section in document.sections:
        containers.extend([section.header, section.footer, section.first_page_header, section.first_page_footer])

    count = 0
    for container in containers:
        for paragraph in _iter_paragraphs(container, seen):
            original = paragraph.text
            if not original.strip():
                continue
            if count >= 1000:
                raise ValueError("DOCX contains more than 1000 translatable paragraphs.")
            result = translate_text(
                original,
                source_lang=source_lang,
                target_lang=target_lang,
                domain=domain,
                force_fallback=force_fallback,
            )
            runs = paragraph.runs
            if runs:
                first_nonempty = next((run for run in runs if run.text.strip()), runs[0])
                first_nonempty.text = result.translated_text
                for run in runs:
                    if run is not first_nonempty:
                        run.text = ""
            else:
                paragraph.add_run(result.translated_text)
            count += 1

    output = io.BytesIO()
    document.save(output)
    return output.getvalue()
