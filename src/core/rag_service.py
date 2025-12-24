"""RAG chain setup service."""
from __future__ import annotations

from langchain_classic.chains import RetrievalQA
from langchain_groq import ChatGroq

from src.config import GROQ_API_KEY, GROQ_MODEL_NAME


def setup_rag_chain(db, top_k: int, groq_api_key: str | None = None, groq_model: str | None = None):
    """Set up RAG chain with Groq LLM, using config defaults when not provided."""
    api_key = groq_api_key or GROQ_API_KEY
    model = groq_model or GROQ_MODEL_NAME
    if not api_key:
        raise ValueError("GROQ_API_KEY not set")

    llm = ChatGroq(model=model, api_key=api_key)
    retriever = db.as_retriever(search_kwargs={"k": top_k})
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    return qa_chain
