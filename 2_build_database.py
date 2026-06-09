"""
Step 2: Build the vector database from book_chunks.json.
Run this ONCE. It embeds all 1743 book chunks so the agent can search them.
This will take a few minutes the first time.
"""

import json
import chromadb
import os

CHUNKS_FILE = "book_chunks.json"
DB_FOLDER = "chroma_db"


def main():
    if not os.path.exists(CHUNKS_FILE):
        print("ERROR: book_chunks.json not found.")
        print("Run 1_extract_book.py first.")
        return

    print("Loading book chunks...")
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"Loaded {len(chunks)} chunks.")

    print(f"\nConnecting to ChromaDB (stored in '{DB_FOLDER}/')...")
    client = chromadb.PersistentClient(path=DB_FOLDER)

    # Delete old collection if rebuilding
    try:
        client.delete_collection("critical_care")
        print("Removed old database collection.")
    except Exception:
        pass

    collection = client.create_collection(
        name="critical_care",
        metadata={"hnsw:space": "cosine"}
    )

    print("\nEmbedding and storing chunks (this takes a few minutes)...")
    BATCH_SIZE = 50

    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        ids = [f"chunk_{i + j}" for j in range(len(batch))]
        documents = [c["text"] for c in batch]
        metadatas = [{"chapter": c["chapter"], "chunk_index": c["chunk_index"]} for c in batch]

        collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )

        done = min(i + BATCH_SIZE, len(chunks))
        pct = int(done / len(chunks) * 100)
        bar = "#" * (pct // 5) + "-" * (20 - pct // 5)
        print(f"  [{bar}] {pct}% ({done}/{len(chunks)})", end="\r")

    print(f"\n\nDone! Database saved to '{DB_FOLDER}/'")
    print(f"Total chunks stored: {collection.count()}")
    print("\nYou can now run:  streamlit run 3_clinical_agent.py")


if __name__ == "__main__":
    main()
