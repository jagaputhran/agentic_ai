from typing import Any, Dict, List, Optional

import os
import io

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse

from PyPDF2 import PdfReader

from src.constants import OPENSEARCH_INDEX, TEXT_CHUNK_SIZE, OLLAMA_MODEL_NAME
from src.opensearch import get_opensearch_client, hybrid_search
from src.ingestion import create_index, bulk_index_documents, delete_documents_by_document_name
from src.embeddings import get_embedding_model
from src.chat import prompt_template, run_llama_streaming, ensure_model_pulled
from src.utils import chunk_text


app = FastAPI(title="Local RAG API")

# Basic CORS for local React dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Initialize shared resources
client = get_opensearch_client()
create_index(client)
embedding_model = get_embedding_model()
ensure_model_pulled(OLLAMA_MODEL_NAME)


@app.get("/health")
def health() -> Dict[str, Any]:
    try:
        info = client.info()
        return {"opensearch": "ok", "version": info.get("version", {}), "ollama_model": OLLAMA_MODEL_NAME}
    except Exception as e:
        return JSONResponse(status_code=500, content={"opensearch": "error", "detail": str(e)})


@app.get("/documents")
def list_documents() -> Dict[str, Any]:
    query = {
        "size": 0,
        "aggs": {"unique_docs": {"terms": {"field": "document_name", "size": 1000}}},
    }
    resp = client.search(index=OPENSEARCH_INDEX, body=query)
    buckets = resp.get("aggregations", {}).get("unique_docs", {}).get("buckets", [])
    names = [b["key"] for b in buckets]
    return {"documents": names}


@app.post("/upload")
async def upload(file: UploadFile = File(...)) -> Dict[str, Any]:
    os.makedirs("uploaded_files", exist_ok=True)
    dest_path = os.path.join("uploaded_files", file.filename)
    content = await file.read()
    with open(dest_path, "wb") as f:
        f.write(content)

    reader = PdfReader(io.BytesIO(content))
    text = "".join([page.extract_text() or "" for page in reader.pages])

    chunks = chunk_text(text, chunk_size=TEXT_CHUNK_SIZE, overlap=100)
    embeddings = [embedding_model.encode(chunk) for chunk in chunks]

    docs = [
        {
            "doc_id": f"{file.filename}_{i}",
            "text": chunk,
            "embedding": emb,
            "document_name": file.filename,
        }
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings))
    ]
    success, errors = bulk_index_documents(docs)
    return {"indexed": success, "errors": errors, "chunks": len(docs)}


@app.delete("/documents/{document_name}")
def delete_document(document_name: str) -> Dict[str, Any]:
    resp = delete_documents_by_document_name(document_name)
    return {"deleted": resp}


@app.post("/search")
def search(payload: Dict[str, Any]) -> Dict[str, Any]:
    query: str = payload.get("query", "")
    top_k: int = int(payload.get("top_k", 5))
    use_hybrid: bool = bool(payload.get("use_hybrid", True))

    qemb = embedding_model.encode(query).tolist()
    if use_hybrid:
        hits = hybrid_search(query, qemb, top_k=top_k)
    else:
        # Fallback to pure BM25 match
        body = {"query": {"match": {"text": {"query": query}}}, "size": top_k}
        resp = client.search(index=OPENSEARCH_INDEX, body=body)
        hits = resp["hits"]["hits"]

    results = [
        {
            "_id": h.get("_id"),
            "_score": h.get("_score"),
            "text": h.get("_source", {}).get("text", ""),
            "document_name": h.get("_source", {}).get("document_name", ""),
        }
        for h in hits
    ]
    return {"hits": results}


@app.post("/chat/stream")
def chat_stream(payload: Dict[str, Any]):
    query: str = payload.get("query", "")
    use_hybrid: bool = bool(payload.get("use_hybrid", True))
    num_results: int = int(payload.get("num_results", 5))
    temperature: float = float(payload.get("temperature", 0.7))
    history: Optional[List[Dict[str, str]]] = payload.get("history", [])

    # Build context if hybrid enabled
    context = ""
    if use_hybrid:
        qemb = embedding_model.encode(query).tolist()
        hits = hybrid_search(query, qemb, top_k=num_results)
        for i, h in enumerate(hits):
            context += f"Document {i}:\n{h.get('_source', {}).get('text', '')}\n\n"

    prompt = prompt_template(query, context, history or [])

    stream = run_llama_streaming(prompt, temperature)
    if stream is None:
        return JSONResponse(status_code=500, content={"error": "LLM streaming failed"})

    def generator():
        for chunk in stream:
            if isinstance(chunk, dict) and "message" in chunk and "content" in chunk["message"]:
                yield chunk["message"]["content"]

    return StreamingResponse(generator(), media_type="text/plain")
