"""
loader.py - reads every supported file in a document folder.

Supported now:
    .pdf        text extracted page by page with pypdf
    .txt, .md   read as plain text (one "page")
    .csv        recorded as present, NOT chunked (handled by the CSV tool in a later module)

Anything else, or any file that fails to read, goes into `skipped` with a reason.
The loader never crashes on one bad file: an unseen folder at the review may
contain surprises, and one broken file should not stop the whole run.
"""

import logging
from pathlib import Path

from pypdf import PdfReader

from forgeai.retrieval.models import LoadedDocument, LoadResult, PageText, SkippedFile

# pypdf prints its own warnings for damaged PDFs. We report problems ourselves
# (in `skipped` or a document note), so hide its duplicate messages.
logging.getLogger("pypdf").setLevel(logging.ERROR)

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".csv"}

# Encodings tried, in order, for text files. Files saved on Windows are often cp1252, not UTF-8.
# "utf-8-sig" is UTF-8 that also removes the invisible marker some Windows editors add at the start.
TEXT_ENCODINGS = ("utf-8-sig", "cp1252")


def read_text_file(path: Path) -> str:
    """Read a text file, trying UTF-8 first and the common Windows encoding second."""
    for encoding in TEXT_ENCODINGS:
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"could not decode text (tried {', '.join(TEXT_ENCODINGS)})")


def read_pdf(path: Path) -> tuple[list[PageText], str]:
    """
    Extract text from each page of a PDF.

    Returns (pages, note). Pages with no extractable text are left out and listed
    in the note, because they usually mean a scanned image, which needs OCR.
    """
    reader = PdfReader(str(path))
    if reader.is_encrypted:
        # Many "encrypted" PDFs only have an empty owner password; try that before giving up.
        try:
            reader.decrypt("")
        except Exception as exc:
            raise ValueError("PDF is password-protected") from exc

    pages: list[PageText] = []
    empty_pages: list[int] = []
    for number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append(PageText(page=number, text=text))
        else:
            empty_pages.append(number)

    note = ""
    if empty_pages:
        note = f"no extractable text on page(s) {empty_pages} (possibly scanned images)"
    return pages, note


def load_documents(folder: str | Path) -> LoadResult:
    """
    Load every supported file in `folder` (sub-folders included), sorted by path
    so the result is the same on every run.
    """
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"Document folder not found: {folder}")

    result = LoadResult(folder=str(folder.resolve()))

    for path in sorted(p for p in folder.rglob("*") if p.is_file()):
        # Skip hidden/system files such as .gitkeep or desktop.ini-style dotfiles.
        if path.name.startswith("."):
            continue

        # Name = path relative to the folder, with "/" separators, e.g. "hr/leave_policy.pdf".
        # Two files called "policy.pdf" in different sub-folders then get different names.
        name = path.relative_to(folder).as_posix()

        extension = path.suffix.lower()
        if extension not in SUPPORTED_EXTENSIONS:
            result.skipped.append(
                SkippedFile(name=name, path=str(path), reason=f"unsupported file type '{extension}'")
            )
            continue

        try:
            if extension == ".pdf":
                pages, note = read_pdf(path)
                if not pages:
                    raise ValueError("no extractable text in any page (possibly a scanned PDF)")
                doc = LoadedDocument(name=name, path=str(path), file_type="pdf", pages=pages, note=note)

            elif extension == ".csv":
                # Decision: CSVs are recorded but not chunked until we build the CSV lookup tool.
                read_text_file(path)  # still check that the file is readable
                doc = LoadedDocument(
                    name=name, path=str(path), file_type="csv", chunked=False,
                    note="CSV recorded; not chunked (to be handled by the CSV lookup tool)",
                )

            else:  # .txt or .md
                text = read_text_file(path).strip()
                if not text:
                    raise ValueError("file is empty")
                doc = LoadedDocument(
                    name=name, path=str(path), file_type=extension.lstrip("."),
                    pages=[PageText(page=1, text=text)],
                )

        except Exception as exc:  # one bad file must not stop the run
            result.skipped.append(SkippedFile(name=name, path=str(path), reason=str(exc)))
            continue

        result.documents.append(doc)

    return result
