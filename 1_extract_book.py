"""
Step 1: Extract text from the Small Animal Critical Care Medicine epub.
This script pulls all the text out of the book and saves it as clean chunks
that can later be searched by the AI agent.
"""

import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
import json
import os
import re

EPUB_PATH = "3-s2.0-C20170041654-9780323764711.epub"
OUTPUT_PATH = "book_chunks.json"
CHUNK_SIZE = 800  # words per chunk (good size for AI to reason over)
OVERLAP = 100     # words of overlap between chunks (preserves context at boundaries)


def clean_text(html_content):
    """Strip HTML tags and clean up whitespace."""
    soup = BeautifulSoup(html_content, "html.parser")
    text = soup.get_text(separator=" ")
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def split_into_chunks(text, chapter_title, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
    """Split text into overlapping word-based chunks."""
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunk_text = " ".join(chunk_words)

        if len(chunk_words) > 50:  # skip tiny fragments
            chunks.append({
                "chapter": chapter_title,
                "text": chunk_text,
                "word_count": len(chunk_words),
                "chunk_index": len(chunks)
            })

        start += chunk_size - overlap

    return chunks


def extract_epub(epub_path):
    """Read the epub and extract all text into chunks."""
    print(f"Opening: {epub_path}")
    book = epub.read_epub(epub_path)

    all_chunks = []
    chapter_count = 0

    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            html = item.get_content().decode("utf-8", errors="ignore")
            text = clean_text(html)

            if len(text) < 100:
                continue  # skip nearly-empty pages

            # Try to extract a chapter title from the HTML
            soup = BeautifulSoup(html, "html.parser")
            title_tag = soup.find(["h1", "h2", "h3", "title"])
            if title_tag:
                chapter_title = title_tag.get_text().strip()
            else:
                chapter_title = f"Section {chapter_count + 1}"

            chunks = split_into_chunks(text, chapter_title)
            all_chunks.extend(chunks)
            chapter_count += 1

            print(f"  [{chapter_count}] '{chapter_title}' -> {len(chunks)} chunks")

    return all_chunks


def main():
    if not os.path.exists(EPUB_PATH):
        print(f"ERROR: Could not find '{EPUB_PATH}'")
        print("Make sure you are running this script from the 'Critical Care Flow' folder.")
        return

    print("Extracting text from epub... this may take a minute.\n")
    chunks = extract_epub(EPUB_PATH)

    print(f"\nTotal chunks extracted: {len(chunks)}")
    print(f"Saving to: {OUTPUT_PATH}")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    print("\nDone! book_chunks.json is ready.")
    print("Sample chunk:")
    print("-" * 60)
    if chunks:
        sample = chunks[10] if len(chunks) > 10 else chunks[0]
        print(f"Chapter: {sample['chapter']}")
        print(f"Text:    {sample['text'][:300]}...")


if __name__ == "__main__":
    main()
