#!/usr/bin/env python3
"""
Streamlit UI for retrieving from Chroma collections.

Run with: streamlit run src/ui/app.py
"""
import streamlit as st
from pathlib import Path
import tempfile

from src.config import COLLECTIONS, PERSIST_DIR, MODEL_NAME, TOP_K, GROQ_API_KEY
from src.core.vectorstore import get_db
from src.core.rag_service import setup_rag_chain
from src.core.utils import (
    perform_retrieve,
    _display_collection_info,
    display_results,
    list_collections_with_stats,
)
from langchain_chroma.vectorstores import Chroma
from src.ingest.build_chroma import build_chroma_from_pdf


def display_collection_info(db: Chroma, collection_name: str, persist_dir: str, model_name: str, document_name: str = None):
    """Display information about the selected collection."""
    _display_collection_info(db, collection_name, persist_dir, model_name, document_name, st.subheader, st.write)


def main():
    st.set_page_config(page_title="Contract Documents Analysis")
    st.title("Contract Documents Analysis")

    persist_dir = str(Path(PERSIST_DIR).expanduser())
    if not Path(persist_dir).exists():
        st.error(f"Chroma persist directory not found: {persist_dir}")
        return

    # Sidebar for selections
    st.sidebar.header("Options")
    action = st.sidebar.selectbox("Choose action", ["List Collections", "Upload PDF", "Display", "Retrieve", "RAG"])
    collection_options = [f"{name} ({pdf})" for name, pdf in COLLECTIONS]
    selected_collection_display = st.sidebar.selectbox("Choose collection", collection_options)
    selected_collection_name = COLLECTIONS[collection_options.index(selected_collection_display)][0]
    pdf_name = COLLECTIONS[collection_options.index(selected_collection_display)][1]

    # Load the collection
    db = get_db(selected_collection_name)

    if action == "List Collections":
        st.header("All Collections")
        stats = list_collections_with_stats(persist_dir)
        if not stats:
            st.info("No collections found.")
        else:
            for s in stats:
                st.subheader(f"{s.get('name')}")
                st.write(f"Items: {s.get('count')}")
                sources = s.get('sample_sources') or []
                st.write(f"Sample sources: {sources}")

    elif action == "Upload PDF":
        st.header("Upload and Ingest PDF")
        new_collection = st.text_input("Enter new collection name")
        uploaded_pdf = st.file_uploader("Select a PDF", type=["pdf"])
        if st.button("Ingest"):
            if not new_collection:
                st.warning("Please enter a collection name.")
            elif not uploaded_pdf:
                st.warning("Please select a PDF file to upload.")
            else:
                with st.spinner("Ingesting PDF into Chroma..."):
                    try:
                        # Save uploaded file to a temporary path
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                            tmp.write(uploaded_pdf.getbuffer())
                            tmp_path = tmp.name

                        # Build Chroma with defaults from config
                        build_chroma_from_pdf(
                            pdf_path=tmp_path,
                            persist_dir=persist_dir,
                            model_name=MODEL_NAME,
                            chunk_size=1024,
                            chunk_overlap=100,
                            encoding_name="cl100k_base",
                            collection_name=new_collection,
                        )
                        st.success(f"Ingestion complete: collection '{new_collection}' created.")
                        st.caption("Note: To use this collection in Display/Retrieve/RAG selectors, add it to COLLECTIONS in src/config.py.")
                    except Exception as e:
                        st.error(f"Error during ingestion: {e}")

    elif action == "Retrieve":
        st.header("Retrieve from the Document")
        query = st.text_input("Enter your query:")
        if st.button("Search"):
            if not query:
                st.warning("Please enter a query.")
            else:
                results = perform_retrieve(db, query, TOP_K)
                display_results(results, st.subheader, st.write)

    elif action == "Display":
        st.header("Document Information")
        display_collection_info(db, selected_collection_name, persist_dir, MODEL_NAME, pdf_name)

    elif action == "RAG":
        st.header("RAG Query with Groq")
        if not GROQ_API_KEY:
            st.error("GROQ_API_KEY not set in .env")
        else:
            # Set up RAG chain (uses config defaults)
            qa_chain = setup_rag_chain(db, TOP_K)

            # Show which collection is in use for RAG
            st.caption(f"Collection: {selected_collection_name} ({pdf_name})")
            query = st.text_input("Enter your query for RAG:")
            if st.button("Generate Answer"):
                if not query:
                    st.warning("Please enter a query.")
                else:
                    with st.spinner("Generating answer..."):
                        try:
                            result = qa_chain.invoke(query)
                            st.markdown("**Answer:**")
                            st.write(result['result'])
                        except Exception as e:
                            st.error(f"Error: {e}")


if __name__ == "__main__":
    main()
