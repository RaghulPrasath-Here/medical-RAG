"""
Medical Document Q&A — RAG System
Streamlit web interface for querying medical research papers.

Run with: streamlit run app.py
"""

import streamlit as st
import requests
import json
import os
import chromadb
from sentence_transformers import SentenceTransformer


st.set_page_config(
    page_title="Medical RAG Q&A",
    page_icon="🏥",
    layout="wide"
)


# Initialize components (cached so they load once)

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True)

@st.cache_resource
def load_chroma():
    client = chromadb.PersistentClient(path="chroma_db")
    collection = client.get_or_create_collection(
        name="medical_docs",
        metadata={"hnsw:space": "cosine"}
    )
    return collection


embedding_model = load_embedding_model()
collection = load_chroma()

OLLAMA_URL = "http://localhost:11434"
SIMILARITY_THRESHOLD = 0.6


# Core functions 

def search_documents(query, n_results=5):
    query_embedding = embedding_model.encode(query).tolist()
    raw_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )
    results = []
    for i in range(len(raw_results['ids'][0])):
        similarity = 1 - raw_results['distances'][0][i]
        results.append({
            "text": raw_results['documents'][0][i],
            "source": raw_results['metadatas'][0][i]['source'],
            "page": raw_results['metadatas'][0][i]['page'],
            "similarity": round(similarity, 4)
        })
    return results


def build_prompt(query, retrieved_chunks):
    context_parts = []
    for chunk in retrieved_chunks:
        source = chunk['source'].replace('documents/', '')
        page = chunk['page']
        context_parts.append(
            f"[Source: {source}, Page {page + 1}]\n{chunk['text']}"
        )
    context = "\n\n---\n\n".join(context_parts)

    return f"""You are a medical research assistant. Answer the question based ONLY on the provided context from research papers. If the context does not contain enough information to answer the question, say "I don't have enough information in my documents to answer this question."

Keep your answer concise (3-5 sentences). Always mention which source the information comes from.

CONTEXT:
{context}

QUESTION: {query}

ANSWER:"""


def ask_llm(prompt, model="llama3"):
    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.3, "num_predict": 512}
            },
            timeout=120
        )
        if response.status_code == 200:
            return response.json()['response']
        else:
            return f"Error: {response.status_code}"
    except requests.ConnectionError:
        return "Error: Ollama is not running. Start it with `ollama serve` in your terminal."
    except requests.Timeout:
        return "Error: LLM took too long to respond. Try a shorter question."


def ask_rag_safe(query, n_results=5):
    retrieved = search_documents(query, n_results=n_results)
    best_score = retrieved[0]['similarity'] if retrieved else 0

    if best_score < SIMILARITY_THRESHOLD:
        return {
            "query": query,
            "answer": (
                "I don't have enough information in my documents to answer "
                "this question reliably."
            ),
            "citations": [],
            "confident": False,
            "best_score": best_score
        }

    prompt = build_prompt(query, retrieved)
    answer = ask_llm(prompt)

    citations = []
    seen = set()
    for r in retrieved:
        source = r['source'].replace('documents/', '')
        page = r['page'] + 1
        key = f"{source}_p{page}"
        if key not in seen:
            seen.add(key)
            citations.append({
                "source": source,
                "page": page,
                "similarity": r['similarity']
            })

    return {
        "query": query,
        "answer": answer,
        "citations": citations,
        "confident": True,
        "best_score": best_score
    }


# Sidebar 

with st.sidebar:
    st.title("System Info")
    st.markdown(f"**Documents in DB:** {collection.count()}")
    st.markdown(f"**Embedding model:** all-MiniLM-L6-v2")
    st.markdown(f"**LLM:** Llama 3 (via Ollama)")
    st.markdown(f"**Confidence threshold:** {SIMILARITY_THRESHOLD}")

    st.divider()

    st.markdown("### How it works")
    st.markdown(
        "1. Your question is converted into an embedding\n"
        "2. Most similar document chunks are retrieved\n"
        "3. An LLM generates an answer from those chunks\n"
        "4. Sources are cited with page numbers"
    )

    st.divider()

    st.markdown("### Sample questions")
    sample_questions = [
        "What are the symptoms of diabetes?",
        "What treatments exist for Alzheimer's?",
        "Is there a link between diabetes and Alzheimer's?",
        "What are the risk factors for Alzheimer's disease?",
        "How does insulin work in the body?",
    ]
    for q in sample_questions:
        if st.button(q, key=q):
            st.session_state.query = q


# Main interface 

st.title("🏥 Medical Document Q&A")
st.caption("Ask questions about your medical research papers. Answers are grounded in source documents with citations.")

# Input
query = st.text_input(
    "Ask a question:",
    value=st.session_state.get("query", ""),
    placeholder="e.g., What are the symptoms of diabetes?"
)

if query:
    with st.spinner("Searching documents and generating answer..."):
        result = ask_rag_safe(query)

    # Confidence indicator
    if result['confident']:
        st.success(f"Confident answer (best match: {result['best_score']:.2f})")
    else:
        st.warning(f"Low confidence (best match: {result['best_score']:.2f}) — answer blocked by hallucination guard")

    # Answer
    st.markdown("### Answer")
    st.markdown(result['answer'])

    # Citations
    if result['citations']:
        st.markdown("### Sources")
        for i, c in enumerate(result['citations']):
            score_pct = int(c['similarity'] * 100)
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"**[{i+1}]** {c['source']} — Page {c['page']}")
            with col2:
                st.progress(c['similarity'], text=f"{score_pct}%")