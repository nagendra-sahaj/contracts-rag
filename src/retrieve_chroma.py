#!/usr/bin/env python3
"""Small retrieval script for a named Chroma collection.

Reads configuration from `.env`:
- `PERSIST_DIR` (default: ./chroma_db)
- `MODEL_NAME` (default: sentence-transformers/all-MiniLM-L6-v2)
- `COLLECTION_NAME` (defaults to 'sample' or whatever you set)
- `TOP_K` (optional, default: 5)

Usage:
  python src/retrieve_chroma.py
Then type a query when prompted, or pass the query as the first argument.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma.vectorstores import Chroma


load_dotenv()


def main() -> None:
    persist_dir = os.getenv("PERSIST_DIR", "../chroma_db")
    model_name = os.getenv("MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
    collection_name = os.getenv("COLLECTION_NAME", "sample")
    try:
        top_k = int(os.getenv("TOP_K", "2"))
    except ValueError:
        top_k = 5

    persist_dir = str(Path(persist_dir).expanduser())
    if not Path(persist_dir).exists():
        raise SystemExit(f"Chroma persist directory not found: {persist_dir}")

    emb = HuggingFaceEmbeddings(model_name=model_name)

    # Load the collection for retrieval
    db = Chroma(persist_directory=persist_dir, embedding_function=emb, collection_name=collection_name)

    # get query from argv or prompt
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = input("Enter your query: ")

    if not query:
        print("No query provided. Exiting.")
        return

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


if __name__ == "__main__":
    main()
