#!/usr/bin/env python3
"""
Convert a folder of veterinary textbooks (PDF / EPUB / folder-of-PDFs)
into a structured Obsidian vault.

Usage:
    python3 convert_to_obsidian.py --input ./Books --output ./Obsidian/VetVault
    python3 convert_to_obsidian.py --input ./Books --output ./Obsidian/VetVault --only "John B. West"
"""

import argparse
import glob
import os
import re
import shutil
import sys
import zipfile
from datetime import date
from pathlib import Path
from typing import List, Tuple


# ---------- helpers ----------

def slugify(text: str) -> str:
    text = re.sub(r"[^\w\s-]", "", text).strip().lower()
    text = re.sub(r"[-\s]+", "-", text)
    return text[:80] or "untitled"


def chapter_title_from_filename(filename: str) -> str:
    name = os.path.splitext(os.path.basename(filename))[0]
    name = re.sub(r"^\d+---", "", name)
    name = re.sub(r"[-_]+", " ", name)
    name = re.sub(r"\s+\d{4}\s+.*$", "", name)
    return name.strip() or "Untitled"


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _normalize_heading(h: str) -> str:
    """Strip leading/trailing page-number cruft so running headers compare equal."""
    h = re.sub(r"^\s*\d+\s*", "", h)
    h = re.sub(r"\s*\d+\s*$", "", h)
    return re.sub(r"\s+", " ", h).strip().lower()


def _page_group_chapters(reader, total_pages: int, group_size: int = 20):
    pages_text = []
    for page in reader.pages:
        try:
            pages_text.append(page.extract_text() or "")
        except Exception:
            pages_text.append("")
    out = []
    for i in range(0, len(pages_text), group_size):
        start_pg = i + 1
        end_pg = min(i + group_size, total_pages)
        out.append(
            (f"Pages {start_pg}-{end_pg}", " ".join(pages_text[i:i + group_size]))
        )
    return out


# ---------- extractors ----------
# Each extractor returns List[(chapter_title, full_chapter_text)].

def extract_single_pdf(pdf_path: Path) -> List[Tuple[str, str]]:
    """Heading-line chapter detection with fallback to 20-page groups."""
    from pypdf import PdfReader

    reader = PdfReader(str(pdf_path))
    total_pages = len(reader.pages)
    print(f"    Reading {total_pages} pages...")

    chapters: List[Tuple[str, str]] = []
    current_title = "Introduction"
    current_pages: List[str] = []

    for page in reader.pages:
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        if not lines:
            continue

        first = lines[0]
        is_heading = (
            len(first) < 80
            and not first.endswith(".")
            and len(first.split()) >= 2
            and (
                re.match(r"^(Chapter|CHAPTER|Section|SECTION|Part|PART)\s", first)
                or (first.isupper() and len(first) > 4)
                or (first.istitle() and len(first.split()) <= 8)
            )
        )
        if is_heading and current_pages:
            # Suppress running headers — same heading text (modulo page numbers)
            # as the current chapter is just the running header on a new page.
            if _normalize_heading(first) == _normalize_heading(current_title):
                current_pages.append(text)
            else:
                chapters.append((current_title, " ".join(current_pages)))
                current_title = first
                current_pages = [text]
        else:
            current_pages.append(text)

    if current_pages:
        chapters.append((current_title, " ".join(current_pages)))

    # Sanity-check the result. Two ways heading detection commonly fails:
    #   (a) it found almost nothing — fall back to page groups
    #   (b) it found way too many tiny "chapters" (running headers slipped through)
    too_few = len(chapters) <= 2 and total_pages > 50
    too_many = total_pages >= 40 and len(chapters) > max(20, total_pages // 4)
    if too_few or too_many:
        reason = "few" if too_few else "too many"
        print(
            f"    Heading detection found {len(chapters)} chapters in {total_pages} pages "
            f"({reason}) — falling back to 20-page groups."
        )
        chapters = _page_group_chapters(reader, total_pages)

    cleaned: List[Tuple[str, str]] = []
    for title, text in chapters:
        text = clean_text(text)
        if len(text) >= 100:
            cleaned.append((title, text))
    return cleaned


def _natural_sort_key(s: str):
    """Sort filenames so 2 < 10 < Index, by splitting digit/non-digit runs."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def extract_pdf_folder(folder_path: Path) -> List[Tuple[str, str]]:
    """Each PDF in the folder (recursively) is one chapter. Title comes from the filename."""
    from pypdf import PdfReader

    pdf_files = sorted(
        (str(p) for p in folder_path.rglob("*.pdf")),
        key=_natural_sort_key,
    )
    if not pdf_files:
        print(f"    No PDFs in {folder_path}")
        return []

    chapters: List[Tuple[str, str]] = []
    for i, pdf_path in enumerate(pdf_files, 1):
        title = chapter_title_from_filename(pdf_path)
        try:
            reader = PdfReader(pdf_path)
            pages_text = []
            for page in reader.pages:
                try:
                    t = page.extract_text()
                except Exception:
                    t = ""
                if t:
                    pages_text.append(t)
            text = clean_text(" ".join(pages_text))
            if len(text) < 100:
                print(f"    [{i}/{len(pdf_files)}] '{title}' — skipped (too short)")
                continue
            chapters.append((title, text))
            print(f"    [{i}/{len(pdf_files)}] '{title}'")
        except Exception as e:
            print(f"    [{i}/{len(pdf_files)}] '{title}' — ERROR: {e}")
    return chapters


def extract_epub(epub_path: Path) -> List[Tuple[str, str]]:
    """Each HTML document in the EPUB becomes a chapter."""
    import ebooklib
    from ebooklib import epub
    from bs4 import BeautifulSoup

    book = epub.read_epub(str(epub_path))
    chapters: List[Tuple[str, str]] = []
    section_count = 0

    for item in book.get_items():
        if item.get_type() != ebooklib.ITEM_DOCUMENT:
            continue
        html = item.get_content().decode("utf-8", errors="ignore")
        soup = BeautifulSoup(html, "html.parser")
        text = clean_text(soup.get_text(separator=" "))
        if len(text) < 100:
            continue
        title_tag = soup.find(["h1", "h2", "h3", "title"])
        title = title_tag.get_text().strip() if title_tag else f"Section {section_count + 1}"
        title = clean_text(title)[:120] or f"Section {section_count + 1}"
        chapters.append((title, text))
        section_count += 1
        print(f"    [{section_count}] '{title}'")
    return chapters


# ---------- writers ----------

def make_frontmatter(book: str, chapter: str, chapter_num: int, tags: List[str]) -> str:
    safe_book = book.replace('"', "'")
    safe_chapter = chapter.replace('"', "'")
    tag_lines = "\n".join(f"  - {t}" for t in tags)
    return (
        "---\n"
        f'book: "{safe_book}"\n'
        f'chapter: "{safe_chapter}"\n'
        f"chapter_number: {chapter_num}\n"
        f"date_added: {date.today().isoformat()}\n"
        "tags:\n"
        f"{tag_lines}\n"
        "---\n\n"
    )


def write_book(
    vault_root: Path,
    book_name: str,
    chapters: List[Tuple[str, str]],
) -> Path:
    book_slug = slugify(book_name)
    book_folder = vault_root / "_Sources" / book_slug
    chapters_folder = book_folder / "Chapters"
    chapters_folder.mkdir(parents=True, exist_ok=True)

    chapter_links = []
    for i, (title, text) in enumerate(chapters, start=1):
        chapter_slug = f"{i:03d}_{slugify(title)}"
        front = make_frontmatter(
            book=book_name,
            chapter=title,
            chapter_num=i,
            tags=[book_slug, "veterinary", "textbook"],
        )
        body = f"# {title}\n\n{text}\n"
        (chapters_folder / f"{chapter_slug}.md").write_text(front + body, encoding="utf-8")
        chapter_links.append(f"- [[{chapter_slug}|{title}]]")

    moc = (
        "---\n"
        "type: book-moc\n"
        f'book: "{book_name.replace(chr(34), chr(39))}"\n'
        f"date_added: {date.today().isoformat()}\n"
        "tags:\n"
        "  - moc\n"
        f"  - {book_slug}\n"
        "---\n\n"
        f"# {book_name}\n\n"
        f"Map of Content. Auto-generated on {date.today().isoformat()}.\n\n"
        "## Chapters\n\n"
        + "\n".join(chapter_links)
        + "\n"
    )
    (book_folder / f"_MOC_{book_slug}.md").write_text(moc, encoding="utf-8")
    return book_folder


def build_master_index(vault_root: Path) -> None:
    sources = vault_root / "_Sources"
    if not sources.exists():
        return

    moc_links = []
    for book_dir in sorted(sources.iterdir()):
        if not book_dir.is_dir():
            continue
        for moc in sorted(book_dir.glob("_MOC_*.md")):
            moc_links.append(f"- [[{moc.stem}]]")

    index = (
        "---\n"
        "type: vault-index\n"
        f"date_added: {date.today().isoformat()}\n"
        "tags:\n"
        "  - moc\n"
        "  - index\n"
        "---\n\n"
        "# Veterinary Reference Vault\n\n"
        "Top-level index. Each entry links to a textbook Map of Content.\n\n"
        "## Books\n\n"
        + "\n".join(moc_links)
        + "\n\n"
        "## Suggested Domain Folders\n\n"
        "- Cardiology\n"
        "- Dermatology\n"
        "- Internal Medicine\n"
        "- Surgery\n"
        "- Pharmacology\n"
        "- Diagnostics\n"
        "- Emergency and Critical Care\n"
        "- Anesthesia\n"
        "- Oncology\n"
        "- Neurology\n"
    )
    (vault_root / "_INDEX.md").write_text(index, encoding="utf-8")
    print(f"\nMaster index written to: {vault_root / '_INDEX.md'}")


# ---------- discovery ----------

def book_name_from_entry(entry: Path) -> str:
    """Strip suffix; for files use the stem, for dirs use the directory name."""
    if entry.is_dir():
        return entry.name
    return entry.stem


def auto_unzip(zip_path: Path, books_root: Path) -> Path:
    """Extract a ZIP into Books/<name>/ if no same-named folder already exists."""
    target = books_root / zip_path.stem
    if target.exists() and target.is_dir():
        return target
    print(f"  Auto-unzipping {zip_path.name} -> {target.name}/")
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(target)
    return target


def discover_books(input_dir: Path) -> List[Path]:
    """
    Return a list of Path entries to process. Each entry is either a single PDF,
    a single EPUB, or a directory of PDFs.
    """
    entries: List[Path] = []
    for entry in sorted(input_dir.iterdir()):
        name = entry.name
        if name.startswith(".") or name == "__MACOSX":
            continue

        if entry.is_dir():
            # Auto-unzip any nested zips first (e.g. the radiology folder).
            for nested_zip in sorted(entry.glob("*.zip")):
                target = entry / nested_zip.stem
                if target.exists() and target.is_dir():
                    continue
                print(f"  Auto-unzipping nested {nested_zip.name} -> {target.relative_to(input_dir)}/")
                target.mkdir(parents=True, exist_ok=True)
                try:
                    with zipfile.ZipFile(nested_zip) as zf:
                        zf.extractall(target)
                except zipfile.BadZipFile as e:
                    print(f"    Bad zip, skipping: {e}")
            # Recursively look for PDFs.
            if any(entry.rglob("*.pdf")):
                entries.append(entry)
            else:
                print(f"  Skipping directory (no PDFs inside): {name}")
            continue

        suffix = entry.suffix.lower()
        if suffix in (".pdf", ".epub"):
            entries.append(entry)
        elif suffix == ".zip":
            unzipped = auto_unzip(entry, input_dir)
            if list(unzipped.glob("*.pdf")):
                entries.append(unzipped)
            else:
                print(f"  Unzipped {name} but found no PDFs in {unzipped.name}/")
        else:
            print(f"  Skipping unsupported file type: {name}")

    # Deduplicate: an unzipped folder may now be both auto-unzipped and already discovered.
    seen = set()
    unique: List[Path] = []
    for p in entries:
        key = str(p.resolve())
        if key in seen:
            continue
        seen.add(key)
        unique.append(p)
    return unique


# ---------- per-book pipeline ----------

def process_entry(entry: Path, vault_root: Path) -> str:
    """Return a status string: 'converted', 'skipped', or 'empty'."""
    book_name = book_name_from_entry(entry)
    book_slug = slugify(book_name)
    moc_path = vault_root / "_Sources" / book_slug / f"_MOC_{book_slug}.md"

    if moc_path.exists():
        print(f"\n[skip] {book_name} (already converted)")
        return "skipped"

    print(f"\n[convert] {book_name}")
    if entry.is_dir():
        chapters = extract_pdf_folder(entry)
    elif entry.suffix.lower() == ".pdf":
        chapters = extract_single_pdf(entry)
    elif entry.suffix.lower() == ".epub":
        chapters = extract_epub(entry)
    else:
        print(f"  Unknown entry type: {entry}")
        return "empty"

    if not chapters:
        print(f"  No usable text extracted from {book_name}")
        return "empty"

    out = write_book(vault_root, book_name, chapters)
    print(f"  -> {len(chapters)} chapters written to {out}")
    return "converted"


# ---------- main ----------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert veterinary textbooks (PDF/EPUB/folders) into an Obsidian vault."
    )
    parser.add_argument("--input", required=True, help="Folder containing books")
    parser.add_argument("--output", required=True, help="Output Obsidian vault folder")
    parser.add_argument(
        "--only",
        default=None,
        help="Optional substring filter: only process entries whose name contains this string",
    )
    args = parser.parse_args()

    input_dir = Path(args.input).expanduser().resolve()
    vault_root = Path(args.output).expanduser().resolve()

    if not input_dir.exists():
        print(f"Input folder not found: {input_dir}")
        sys.exit(1)
    vault_root.mkdir(parents=True, exist_ok=True)

    entries = discover_books(input_dir)
    if args.only:
        needle = args.only.lower()
        entries = [e for e in entries if needle in e.name.lower()]
    if not entries:
        print("No books matched.")
        sys.exit(0)

    print(f"\nFound {len(entries)} book(s) to consider.")
    counts = {"converted": 0, "skipped": 0, "empty": 0}
    for entry in entries:
        try:
            status = process_entry(entry, vault_root)
        except Exception as e:
            print(f"  ERROR processing {entry.name}: {e}")
            status = "empty"
        counts[status] = counts.get(status, 0) + 1

    build_master_index(vault_root)
    print(
        f"\nDone. converted={counts['converted']} "
        f"skipped={counts['skipped']} empty={counts['empty']}"
    )


if __name__ == "__main__":
    main()
