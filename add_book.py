"""
Add any epub or folder of PDFs to the clinical agent's database.

Usage:
    python3 add_book.py "path/to/book.epub" "Short Book Name"
    python3 add_book.py "path/to/pdf_folder/" "Short Book Name"

Examples:
    python3 add_book.py "de_lahuntas_veterinary_neuroanatomy_and_clinical_neurology/" "deLahunta Neuro"
    python3 add_book.py "Ettinger_ICTM.epub" "Ettinger ICTM"
    python3 add_book.py "Plumb_Drug_Handbook.epub" "Plumb Drugs"
"""

import sys
import os
import re
import glob

import chromadb

DB_FOLDER = "chroma_db"
CHUNK_SIZE = 800
OVERLAP = 100


def split_into_chunks(text, chapter_title, book_name, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunk_text = " ".join(chunk_words)
        if len(chunk_words) > 50:
            chunks.append({
                "book": book_name,
                "chapter": chapter_title,
                "text": chunk_text,
            })
        start += chunk_size - overlap
    return chunks


def chapter_title_from_filename(filename):
    """Turn a PDF filename into a readable chapter title."""
    name = os.path.splitext(os.path.basename(filename))[0]
    # Strip leading chapter numbers like "10---"
    name = re.sub(r'^\d+---', '', name)
    # Replace hyphens/underscores with spaces
    name = re.sub(r'[-_]+', ' ', name)
    # Remove trailing year/publisher junk like "_2021_de-Lahunta..."
    name = re.sub(r'\s+\d{4}\s+.*$', '', name)
    return name.strip()


def extract_single_pdf(pdf_path, book_name):
    """Extract a single multi-chapter PDF, detecting chapter boundaries by page headings."""
    from pypdf import PdfReader
    reader = PdfReader(pdf_path)
    total_pages = len(reader.pages)
    print(f"  Reading {total_pages} pages...")

    # Collect (chapter_title, text) groups
    chapters = []
    current_title = "Introduction"
    current_pages = []

    for page in reader.pages:
        text = page.extract_text() or ""
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        if not lines:
            continue

        # Detect a chapter heading: first non-empty line is short, title-like, not a sentence
        first = lines[0]
        is_heading = (
            len(first) < 80
            and not first.endswith(".")
            and len(first.split()) >= 2
            and (
                re.match(r'^(Chapter|CHAPTER|Section|SECTION|Part|PART)\s', first)
                or (first.isupper() and len(first) > 4)
                or (first.istitle() and len(first.split()) <= 8)
            )
        )
        if is_heading and current_pages:
            chapters.append((current_title, " ".join(current_pages)))
            current_title = first
            current_pages = [text]
        else:
            current_pages.append(text)

    if current_pages:
        chapters.append((current_title, " ".join(current_pages)))

    # If heading detection found almost nothing (≤2 chapters for >50 pages), fall back to page groups
    if len(chapters) <= 2 and total_pages > 50:
        print("  Heading detection found few chapters — falling back to 20-page groups.")
        chapters = []
        all_pages = []
        for page in reader.pages:
            t = page.extract_text() or ""
            all_pages.append(t)
        group_size = 20
        for i in range(0, len(all_pages), group_size):
            start_pg = i + 1
            end_pg = min(i + group_size, total_pages)
            chapters.append((f"Pages {start_pg}–{end_pg}", " ".join(all_pages[i:i + group_size])))

    all_chunks = []
    for idx, (title, text) in enumerate(chapters, 1):
        text = re.sub(r'\s+', ' ', text).strip()
        if len(text) < 100:
            continue
        chunks = split_into_chunks(text, title, book_name)
        all_chunks.extend(chunks)
        print(f"  [{idx}/{len(chapters)}] '{title}' -> {len(chunks)} chunks")

    return all_chunks


def extract_pdf_folder(folder_path, book_name):
    from pypdf import PdfReader
    pdf_files = sorted(glob.glob(os.path.join(folder_path, "*.pdf")))
    if not pdf_files:
        print(f"ERROR: No PDF files found in {folder_path}")
        return []

    all_chunks = []
    for i, pdf_path in enumerate(pdf_files, 1):
        chapter_title = chapter_title_from_filename(pdf_path)
        try:
            reader = PdfReader(pdf_path)
            pages_text = []
            for page in reader.pages:
                t = page.extract_text()
                if t:
                    pages_text.append(t)
            text = re.sub(r'\s+', ' ', " ".join(pages_text)).strip()
            if len(text) < 100:
                print(f"  [{i}/{len(pdf_files)}] '{chapter_title}' — skipped (too short)")
                continue
            chunks = split_into_chunks(text, chapter_title, book_name)
            all_chunks.extend(chunks)
            print(f"  [{i}/{len(pdf_files)}] '{chapter_title}' -> {len(chunks)} chunks")
        except Exception as e:
            print(f"  [{i}/{len(pdf_files)}] '{chapter_title}' — ERROR: {e}")

    return all_chunks


def extract_epub(epub_path, book_name):
    import ebooklib
    from ebooklib import epub
    from bs4 import BeautifulSoup

    print(f"Opening: {epub_path}")
    book = epub.read_epub(epub_path)
    all_chunks = []
    chapter_count = 0

    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            html = item.get_content().decode("utf-8", errors="ignore")
            soup = BeautifulSoup(html, "html.parser")
            text = re.sub(r'\s+', ' ', soup.get_text(separator=" ")).strip()
            if len(text) < 100:
                continue
            title_tag = soup.find(["h1", "h2", "h3", "title"])
            chapter_title = title_tag.get_text().strip() if title_tag else f"Section {chapter_count + 1}"
            chunks = split_into_chunks(text, chapter_title, book_name)
            all_chunks.extend(chunks)
            chapter_count += 1
            print(f"  [{chapter_count}] '{chapter_title}' -> {len(chunks)} chunks")

    return all_chunks


def get_existing_book_names(collection):
    """Check what books are already in the database."""
    results = collection.get(include=["metadatas"])
    books = set()
    for m in results["metadatas"]:
        if "book" in m:
            books.add(m["book"])
    return books


def main():
    if len(sys.argv) < 3:
        print("Usage: python3 add_book.py \"path/to/book.epub\" \"Short Book Name\"")
        print("\nExample:")
        print('  python3 add_book.py "deLahunta_Neuro.epub" "deLahunta Neuro"')
        return

    epub_path = sys.argv[1]
    book_name = sys.argv[2]

    if not os.path.exists(epub_path):
        print(f"ERROR: File not found: {epub_path}")
        return

    if not os.path.exists(DB_FOLDER):
        print(f"ERROR: Database folder '{DB_FOLDER}' not found.")
        print("Run 2_build_database.py first to set up the initial database.")
        return

    # Connect to existing database
    client = chromadb.PersistentClient(path=DB_FOLDER)
    collection = client.get_collection("critical_care")

    # Check if this book is already loaded
    existing_books = get_existing_book_names(collection)
    if book_name in existing_books:
        print(f"WARNING: '{book_name}' is already in the database.")
        answer = input("Re-add it anyway? This will create duplicate entries. (y/N): ").strip().lower()
        if answer != "y":
            print("Aborted.")
            return

    print(f"\nExtracting text from '{book_name}'...\n")
    if os.path.isdir(epub_path):
        chunks = extract_pdf_folder(epub_path, book_name)
    elif epub_path.lower().endswith(".pdf"):
        chunks = extract_single_pdf(epub_path, book_name)
    else:
        chunks = extract_epub(epub_path, book_name)
    print(f"\nTotal chunks extracted: {len(chunks)}")

    # Get current count to create unique IDs
    current_count = collection.count()
    print(f"Current database size: {current_count} chunks")
    print(f"Adding {len(chunks)} new chunks...")

    BATCH_SIZE = 50
    safe_book_prefix = re.sub(r'[^a-zA-Z0-9]', '_', book_name)

    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        ids = [f"{safe_book_prefix}_chunk_{i + j}" for j in range(len(batch))]
        documents = [c["text"] for c in batch]
        metadatas = [{"book": c["book"], "chapter": c["chapter"]} for c in batch]

        collection.add(ids=ids, documents=documents, metadatas=metadatas)

        done = min(i + BATCH_SIZE, len(chunks))
        pct = int(done / len(chunks) * 100)
        bar = "#" * (pct // 5) + "-" * (20 - pct // 5)
        print(f"  [{bar}] {pct}% ({done}/{len(chunks)})", end="\r")

    print(f"\n\nDone! '{book_name}' has been added.")
    print(f"Database now contains {collection.count()} total chunks.")
    print("\nBooks currently in database:")
    all_books = get_existing_book_names(collection)
    # Also count original critical care chunks that may not have book metadata
    for b in sorted(all_books):
        print(f"  - {b}")


if __name__ == "__main__":
    main()
