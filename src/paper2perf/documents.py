from pathlib import Path
import re


def pdf_page_text(page):
    """Preserve raised digits without joining table values to footnote numbers.

    Apply substitutions only when raw glyph geometry and the word stream agree
    exactly. Ordinary baseline numbers (including 42) are never rewritten.
    """
    import pymupdf
    flags = pymupdf.TEXTFLAGS_TEXT
    text = page.get_text(sort=True, flags=flags)
    chars = []
    raised = []
    for block in page.get_text('rawdict')['blocks']:
        for line in block.get('lines', []):
            for span in line['spans']:
                for char in span['chars']:
                    item = (char['c'], char['bbox'], bool(span['flags'] & 1))
                    chars.append(item)
                    if item[2] and item[0] in '0123456789':
                        raised.append(item)
    if not raised:
        return text
    tokens = list(re.finditer(r'\S+', text))
    words = page.get_text('words', sort=True, flags=flags)
    # Do not guess at offsets if the library's layout and word order disagree.
    if [m.group() for m in tokens] != [w[4] for w in words]:
        return page.get_text(sort=True, flags=flags | pymupdf.TEXT_PRESERVE_SPANS)
    translation = str.maketrans('0123456789', '⁰¹²³⁴⁵⁶⁷⁸⁹')
    replacements = []
    def inside(bounds, glyph):
        x = (glyph[0] + glyph[2]) / 2
        y = (glyph[1] + glyph[3]) / 2
        return bounds[0] - .01 <= x <= bounds[2] + .01 and bounds[1] - .01 <= y <= bounds[3] + .01
    for token, word in zip(tokens, words):
        if not any(inside(word, box) for _, box, _ in raised):
            continue
        glyphs = [(value, superscript) for value, box, superscript in chars if inside(word, box)]
        if ''.join(value for value, _ in glyphs) != word[4]:
            return page.get_text(sort=True, flags=flags | pymupdf.TEXT_PRESERVE_SPANS)
        replacement = ''.join(value.translate(translation) if superscript else value
                              for value, superscript in glyphs)
        replacements.append((token.start(), token.end(), replacement))
    for start, end, replacement in reversed(replacements):
        text = text[:start] + replacement + text[end:]
    return text


def ingest_documents(settings, store, sid, paths):
    ids = []
    for source in paths:
        source = Path(source).expanduser()
        with source.open("rb") as handle:
            raw = handle.read(settings.max_document_bytes + 1)
        if len(raw) > settings.max_document_bytes:
            raise ValueError(f"Document exceeds the {settings.max_document_bytes}-byte limit: {source}")
        suffix, pages = extract_pages(source.name, raw, settings.max_pages)
        ids.append(store.add_document(sid, source.name, suffix, raw, pages))
    return list(dict.fromkeys(ids))


def extract_pages(name: str, data: bytes, max_pages=400) -> tuple[str, list[str]]:
    """Read original documents; page numbers always refer to the uploaded file."""
    suffix = Path(name).suffix.lower()
    if suffix == ".pdf":
        import pymupdf
        try:
            with pymupdf.open(stream=data, filetype="pdf") as doc:
                if doc.needs_pass:
                    raise ValueError("Encrypted PDFs are unsupported; upload an unlocked copy.")
                if len(doc) > max_pages:
                    raise ValueError(f"Document exceeds the {max_pages}-page limit.")
                pages = [pdf_page_text(page) for page in doc]
        except (RuntimeError, pymupdf.FileDataError) as exc:
            raise ValueError("Could not parse this PDF.") from exc
    elif suffix in {".txt", ".md"}:
        try:
            pages = data.decode("utf-8").split("\f")
        except UnicodeDecodeError as exc:
            raise ValueError("Text documents must use UTF-8.") from exc
        if len(pages) > max_pages:
            raise ValueError("Too many pages.")
    else:
        raise ValueError("Upload PDF, TXT, or Markdown documents.")
    if not any(p.strip() for p in pages):
        raise ValueError("No text found. Scanned PDFs require OCR before upload.")
    if sum(len(p) for p in pages) > 4_000_000:
        raise ValueError("Extracted document text exceeds the size limit.")
    return suffix, pages
