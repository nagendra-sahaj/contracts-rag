# Contracts RAG: PDF -> Chroma + Retrieval + RAG

This project ingests contract PDFs into a Chroma vector store (HuggingFace embeddings), provides CLI and Streamlit UIs for retrieval, and supports RAG answers via Groq.

## Layout

- `src/config.py`: Centralized `.env` config and constants
- `src/core/`: Backend services
	- `vectorstore.py`: Embeddings/Chroma factories
	- `rag_service.py`: RAG chain setup (Groq)
	- `utils.py`: Display-agnostic helpers (retrieve, render results)
- `src/ingest/`: Ingestion tools
	- `build_chroma.py`: PDF -> chunks -> embeddings -> Chroma
- `src/cli/`: Command-line tools
	- `contracts_cli.py`: Unified CLI (Display, Retrieve, RAG)
- `src/ui/`: Streamlit UI
	- `app.py`: “Contract Documents Analysis” app

## Quick start

1) Install dependencies (uv recommended)

```bash
uv pip install -r requirements.txt
```

2) Configure `.env`

Set at minimum:
- `PDF_PATH`, `PERSIST_DIR`, `MODEL_NAME`, `CHUNK_SIZE`, `CHUNK_OVERLAP`
- `COLLECTION_NAME` (optional; defaults to PDF filename stem)
- `GROQ_API_KEY` and `GROQ_MODEL_NAME` for RAG
- `PDF_LOADER` (optional: `pymupdf` | `pdfplumber` | `pypdf`)

3) Ingest a PDF

```bash
uv run python src/ingest/build_chroma.py
```

4) Run the Streamlit UI

```bash
./run_streamlit.sh
```

If running manually:

```bash
PYTHONPATH="$(pwd)" uv run streamlit run src/ui/app.py
```

5) CLI tools

```bash
uv run python -m src.cli.contracts_cli
./run_cli.sh
```

## Notes

- Each chunk stores `source` metadata with the PDF filename for traceability.
- Use `PDF_LOADER=pymupdf` for better whitespace preservation with some PDFs.
- TOP_K, model names, and collections are controlled via `src/config.py` + `.env`.

## Example: query via code

```python
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma.vectorstores import Chroma

emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = Chroma(persist_directory="./chroma_db", embedding_function=emb, collection_name="Construction_Contract")
docs = db.similarity_search("your query", k=5)
```
