#!/usr/bin/env python3
"""Small retrieval script for a named Chroma collection.

Reads configuration from `.env`:
- `PERSIST_DIR` (default: ../chroma_db)
- `MODEL_NAME` (default: sentence-transformers/all-MiniLM-L6-v2)
- `TOP_K` (optional, default: 5)

Interactive mode: Choose Query, Display info, or Quit. Then select collection.

Usage:
  python src/retrieve_chroma.py
Then follow prompts.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma.vectorstores import Chroma


load_dotenv()


def get_directory_size(path: str) -> str:
    """Calculate the total size of a directory in KB or MB."""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            try:
                total_size += os.path.getsize(fp)
            except OSError:
                pass  # Skip files that can't be accessed
    if total_size < 1024 * 1024:
        return f"{total_size / 1024:.2f} KB"
    else:
        return f"{total_size / (1024 * 1024):.2f} MB"


def select_collection() -> str:
    """Prompt user to select a collection from available options."""
    collections = [
        ("Sample", "sample.pdf"),
        ("Construction_Agreement", "Construction_Agreement.pdf"),
        ("Construction_Contract", "Construction_Contract-for-Major-Works.pdf")
    ]
    print("Available collections:")
    for i, (col_name, pdf_name) in enumerate(collections, start=1):
        print(f"{i}. {col_name} ({pdf_name})")

    while True:
        try:
            choice = int(input("Select collection (1-3): "))
            if 1 <= choice <= len(collections):
                return collections[choice - 1][0]
            else:
                print("Invalid choice. Please select 1, 2, or 3.")
        except ValueError:
            print("Please enter a number.")


def display_collection_info(db: Chroma, collection_name: str, persist_dir: str, model_name: str) -> None:
    """Display information about the selected collection."""
    try:
        count = db._collection.count()
        size = get_directory_size(persist_dir)
        print(f"\nCollection: {collection_name}")
        print(f"Number of chunks: {count}")
        print(f"Database size: {size} (shared across all collections)")
        print(f"Embedding model: {model_name}")
        print(f"Persist directory: {persist_dir}")
        # Additional metadata from collection
        if hasattr(db._collection, 'metadata') and db._collection.metadata:
            print(f"Collection metadata: {db._collection.metadata}")
    except Exception as e:
        print(f"Error retrieving collection info: {e}")


def main() -> None:
    persist_dir = os.getenv("PERSIST_DIR", "../chroma_db")
    model_name = os.getenv("MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
    try:
        top_k = int(os.getenv("TOP_K", "2"))
    except ValueError:
        top_k = 5

    persist_dir = str(Path(persist_dir).expanduser())
    if not Path(persist_dir).exists():
        raise SystemExit(f"Chroma persist directory not found: {persist_dir}")

    emb = HuggingFaceEmbeddings(model_name=model_name)

    while True:
        # Prompt for mode
        while True:
            print("Choose mode: \n 1. Query \n 2. Display info \n 3. Quit ")
            mode = input("Choose mode: ").strip()
            if mode in ['1', '2', '3']:
                break
            else:
                print("Invalid choice. Please select 1, 2, or 3.")

        if mode == '3':
            print("Exiting.")
            break

        # Prompt for collection selection
        collection_name = select_collection()

        # Load the collection
        db = Chroma(persist_directory=persist_dir, embedding_function=emb, collection_name=collection_name)

        if mode == '1':
            # Query mode
            # get query from argv or prompt
            if len(sys.argv) > 1:
                query = " ".join(sys.argv[1:])
            else:
                query = input("Enter your query: ")

            if not query:
                print("No query provided. Continuing.")
                continue

            # use similarity search with relevance scores if available
            try:
                results = db.similarity_search_with_score(query, k=top_k)
            except Exception:
                # fallback to similarity_search (older langchain versions)
                docs = db.similarity_search(query, k=top_k)
                results = [(d, None) for d in docs]

            for i, (doc, score) in enumerate(results, start=1):
                print(f"\nResult #{i}")
                if score is not None:
                    print(f"Score: {score}")
                src = doc.metadata.get("source") if getattr(doc, "metadata", None) else None
                if src:
                    print(f"Source: {src}")
                # Print a short snippet of text
                text = doc.page_content.strip()
                snippet = text if len(text) < 800 else text[:800] + "..."
                print(snippet)
        else:
            # Display info mode
            display_collection_info(db, collection_name, persist_dir, model_name)

        # Ask to continue
        cont = input("\nDo you want to perform another action? (y/n): ").strip().lower()
        if cont != 'y':
            print("Exiting.")
            break


if __name__ == "__main__":
    main()
