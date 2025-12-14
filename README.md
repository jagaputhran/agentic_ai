# 📝 Local RAG System — End-to-End Technical Guide

This repository implements a privacy-first, on-device Retrieval-Augmented Generation (RAG) system for personal documents. It combines Streamlit UI, document ingestion (PDF extraction + chunking), semantic embeddings, OpenSearch vector indexing with hybrid search, and local LLM inference via Ollama. Target users are engineers and power users who need fast, private querying over local PDFs without sending data to cloud services.

The system is composed of:
- Streamlit UI for chat and document upload
- Ingestion pipeline (PDF → text → chunks → embeddings → OpenSearch)
- Embedding model (`sentence-transformers/all-mpnet-base-v2`)
- OpenSearch index with KNN vectors + hybrid search
- LLM via Ollama (`llama3.2:latest`) with streaming responses


## 1. High-Level Goal
- Problem: Local, private question-answering over personal PDFs using semantic retrieval + LLMs, without cloud dependencies.
- Users: Engineers, analysts, and privacy-conscious individuals.
- Type: RAG system with hybrid search (keyword + vector) and a chatbot UI.
- Why components exist:
	- Streamlit: simple UX for chat and uploads.
	- Ingestion: parse PDFs, OCR if needed, chunk, embed, and index.
	- Embeddings: enable semantic search over chunks.
	- OpenSearch: scalable index for text + vectors, hybrid search.
	- LLM (Ollama): generate answers grounded in retrieved context.

## 2. Repository Structure (No Skipping)
/
 - LICENSE
	 - License metadata. Removing won’t break runtime.
 - README.md
	 - This guide. Removing won’t break runtime but hurts onboarding.
 - mypy.ini
	 - Type-checker config. Removing reduces static checks.
 - pyproject.toml
	 - Formatting configs for Black/isort. Safe to remove for runtime; reduces dev ergonomics.
 - requirements.txt
	 - Python dependencies. Removing breaks environment setup.
 - Welcome.py
	 - Streamlit landing page (welcome screen). If removed, `streamlit run welcome.py` fails, but core logic lives under `pages/`.
 - app.log
	 - Log file target; populated by `src/utils.setup_logging()`.
 - embedding_model/
	 - Placeholder if you choose local model storage. If removed, using HF hub still works.
	 - README.md: Optional instructions for local models.
 - images/
	 - UI assets (logo). Safe to remove; UI falls back to placeholder.
 - logs/
	 - Log directory used by `LOG_FILE_PATH` in `src/constants.py`. Removing breaks logging if path doesn’t exist.
 - notebooks/
	 - 01_Prerequisites_and_Environment_Setup.ipynb: Setup exploration.
	 - 02_OpenSearch_Index_and_Ingestion_standalone.ipynb: Index/ingestion experiments.
	 - 03_Hybrid_Search_and_Retrieval.ipynb: Hybrid search experiments.
	 - Removing notebooks doesn’t affect app runtime.
 - pages/
	 - 1_🤖_Chatbot.py
		 - Chat UI. Initializes OpenSearch client and index, loads embedding + Ollama, manages session state, streams responses.
		 - Removing breaks the chatbot page.
	 - 2_📄_Upload_Documents.py
		 - Upload UI. Loads embedding model, chunking, generates embeddings, bulk indexes into OpenSearch, and allows delete.
		 - Removing breaks document ingestion via UI.
 - src/
	 - __init__.py
		 - Package marker; safe to keep for imports.
	 - chat.py
		 - Backend chat logic: ensures Ollama model availability, builds prompts, runs streaming via `ollama.chat`, integrates hybrid search context.
		 - Removing breaks assistant responses.
	 - constants.py
		 - Central config: model names, index name, host/port. Removing breaks configuration across modules.
	 - embeddings.py
		 - Loads `SentenceTransformer` with caching, generates embeddings per chunk.
		 - Removing disables semantic indexing and query embedding.
	 - index_config.json
		 - OpenSearch index settings/mappings: `knn_vector` using FAISS HNSW; dimension populated at runtime.
		 - Removing prevents index creation.
	 - ingestion.py
		 - Index lifecycle (create/delete), bulk indexing of chunks, delete-by-query by `document_name`.
		 - Removing breaks ingestion and index management.
	 - ocr.py
		 - PDF text extraction with fallback OCR using Tesseract for image pages.
		 - Removing reduces robustness for scanned PDFs.
	 - opensearch.py
		 - Client init and hybrid search using OpenSearch `hybrid` query + `search_pipeline`.
		 - Removing breaks search.
	 - utils.py
		 - Logging, text cleanup, fixed-size token-ish chunking with overlap.
		 - Removing breaks logging and chunking.
 - uploaded_files/
	 - Storage for uploaded PDFs; referenced by upload page. Removing breaks local file persistence.

## 3. Execution Flow (Code → Runtime)
1. User runs `streamlit run welcome.py`.
2. Streamlit starts, renders the welcome page.
3. User navigates to Chatbot or Upload page.
4. On first visit, `embeddings.get_embedding_model()` downloads/caches `all-mpnet-base-v2` from HF; `ensure_model_pulled()` checks/pulls `OLLAMA_MODEL_NAME`.
5. OpenSearch client is initialized (`opensearch.get_opensearch_client()`), index ensured (`ingestion.create_index()`).
6. Upload flow: user uploads PDFs → `chunk_text()` → `generate_embeddings()` → `bulk_index_documents()` writes chunks to `documents` index.
7. Chat flow: user enters prompt → `chat.generate_response_streaming()` computes query embedding → `opensearch.hybrid_search()` retrieves top-K chunks.
8. Prompt constructed with context → `ollama.chat(stream=True)` streams tokens back to UI.
9. Streamlit renders streaming response; chat history updated.

## 4. Streamlit Chatbot Flow (UI → Backend)
- Initialization: `pages/1_🤖_Chatbot.py` calls `setup_logging()`, sets page config, CSS, and creates UI controls.
- Session state: `use_hybrid_search`, `num_results`, `temperature`, `embedding_models_loaded`, `chat_history`.
- Message capture: `st.chat_input()` returns `prompt`; added to `chat_history`.
- Backend invocation: `generate_response_streaming(prompt, use_hybrid_search, num_results, temperature, chat_history)`.
- Response rendering: streaming chunks are appended to `response_text` and displayed via `response_placeholder.markdown(...)`.
- Key functions/vars:
	- `ensure_model_pulled(OLLAMA_MODEL_NAME)`
	- `get_embedding_model()`
	- `hybrid_search(query, query_embedding, top_k)`
	- `prompt_template(query, context, history)`
	- `st.session_state[...]` for settings/history.
- Example: User asks “Summarize my mark sheet.” → Hybrid search fetches chunks with “grade/CGPA” → prompt assembled → LLM streams answer.

## 5. Document Ingestion Pipeline
### 5.1 File Loading
- Supported: PDFs (via Streamlit uploader). Text extraction: `PyPDF2.PageObject.extract_text()`; fallback OCR with Tesseract (`pytesseract`) if page has images (`ocr.extract_text_from_images`).
- Flow: Save to `uploaded_files/` → read with `PdfReader` → concat per-page text → clean via `utils.clean_text()`.
### 5.2 Chunking (Fixed + Overlap)
- Strategy: Fixed token-ish chunk size with overlap (`chunk_text(text, chunk_size=TEXT_CHUNK_SIZE, overlap=100)`), where chunk size defaults to 300 words.
- Why: Simple, deterministic, good baseline; overlap mitigates boundary loss.
- Example:
	- Original: “Transformers use self-attention to weigh token interactions...” (1200 tokens).
	- Chunks: [0–300], [200–500], [400–700], ... (overlap=100) ensuring context continuity.

### Module Deep-Dive: `ocr.py`, `opensearch.py`, `ingestion.py`, `chat.py`, `pages/1_🤖_Chatbot.py`

- `src/ocr.py`
	- Purpose: Robust PDF text extraction. Uses `PyPDF2.PdfReader` to extract text directly; when a page has no extractable text, iterates `page.images`, decodes image bytes, and runs `pytesseract.image_to_string(image)` to OCR content.
	- Key functions:
		- `extract_text_from_pdf(file_path) -> str`: Loops pages, attempts `.extract_text()`, falls back to `extract_text_from_images(page)`, then normalizes with `utils.clean_text()`.
		- `extract_text_from_images(page: PageObject) -> str`: Converts embedded images to PIL and OCRs them.
	- Why: Many PDFs are scanned; OCR ensures ingestion doesn’t silently skip content.
	- If removed: Scanned PDFs or image-heavy pages would ingest empty text, reducing retrieval quality.

- `src/opensearch.py`
	- Purpose: Connects to OpenSearch and executes hybrid queries.
	- Key functions:
		- `get_opensearch_client() -> OpenSearch`: Configures client with host/port, compression, timeouts, retries.
		- `hybrid_search(query_text: str, query_embedding: List[float], top_k: int)`: Builds OpenSearch `hybrid` query combining `match` on `text` and `knn` on `embedding`, excludes the raw vectors in `_source`, and invokes `search_pipeline="nlp-search-pipeline"` to normalize/merge scores.
	- Why: Centralizes search logic; ensures consistent pipeline usage and response shaping.
	- If removed: Chat can’t perform retrieval; LLM would lack grounding context.

- `src/ingestion.py`
	- Purpose: Index lifecycle and bulk ingestion.
	- Key functions:
		- `load_index_config()`: Loads `src/index_config.json`, sets `embedding.dimension = EMBEDDING_DIMENSION`.
		- `create_index(client)`: Creates `OPENSEARCH_INDEX` if absent with FAISS/HNSW KNN mappings.
		- `bulk_index_documents(documents)`: Converts each chunk to a bulk action with `_id = filename_i`, `_source = {text, embedding, document_name}`, and runs `helpers.bulk`.
		- `delete_documents_by_document_name(name)`: Deletes all chunks belonging to a file.
	- Why: Encapsulates index and ingestion consistency; prevents schema drift.
	- If removed: No index creation; ingestion and deletes would fail.

- `src/chat.py`
	- Purpose: Chat orchestration, integrating embeddings, retrieval, prompt construction, and LLM streaming.
	- Key functions:
		- `ensure_model_pulled(model)`: Ensures Ollama model is available; pulls if missing.
		- `run_llama_streaming(prompt, temperature)`: Calls `ollama.chat(..., stream=True)` and returns a generator of chunks.
		- `prompt_template(query, context, history)`: Assembles the final prompt including retrieved context and conversation history.
		- `generate_response_streaming(query, use_hybrid_search, num_results, temperature, chat_history)`: Encodes query, runs `hybrid_search`, aggregates context, and streams LLM output.
	- Why: Provides deterministic prompt formatting and controlled context inclusion.
	- If removed: Chatbot can’t produce responses.

- `pages/1_🤖_Chatbot.py`
	- Purpose: Chat UI, model/progress indicators, session state, and streaming display.
	- Flow: Initializes logging, OpenSearch client, ensures index, loads embedding + Ollama with progress bar, captures user input via `st.chat_input`, and renders streaming assistant output.
	- Why: User-facing entry for the RAG functionality.
	- If removed: No chat interface.

### Upload Flow: Step-by-Step When You Upload a File
1. User selects one or more PDFs in `pages/2_📄_Upload_Documents.py` (`st.file_uploader`).
2. Files are saved into `uploaded_files/` via `save_uploaded_file()`.
3. Text extraction: Pages read with `PyPDF2.PdfReader`; `page.extract_text()` concatenated. If using OCR path, `ocr.extract_text_from_pdf()` performs fallback OCR.
4. Cleaning: `utils.clean_text()` normalizes whitespace, hyphenation, and newlines.
5. Chunking: `utils.chunk_text(text, chunk_size=TEXT_CHUNK_SIZE, overlap=100)` produces fixed-size overlapping chunks.
6. Embedding: `embeddings.generate_embeddings(chunks)` encodes each chunk into a 768-d vector using `SentenceTransformer` (cached by Streamlit).
7. Indexing: `ingestion.bulk_index_documents([...])` writes chunks into `documents` index with `_id = filename_i`, storing `text`, `embedding`, `document_name`.
8. UI updates: The Upload page displays the file and extracted character count; delete buttons allow removing from filesystem and index via `delete_documents_by_document_name`.

## 6. Embedding Generation
- Model: `sentence-transformers/all-mpnet-base-v2`.
- Dimension: `EMBEDDING_DIMENSION = 768` (enforced into mapping via `ingestion.load_index_config`).
- Input/Output: text chunk → `model.encode(chunk)` → Python list/NumPy array of 768 floats, e.g. `[0.012, -0.041, ...]`.
- Batch: Current implementation encodes per-chunk in a Python list; can be batched for performance.
- Performance: First-load download (~400MB), subsequent cached; encoding throughput depends on CPU/GPU availability.
- Why embeddings: They map semantically similar text to nearby points in vector space, enabling KNN retrieval of meaning-related chunks.

## 7. OpenSearch / Vector DB Layer
### 7.1 Index Creation
- Name: `OPENSEARCH_INDEX = "documents"`.
- Mapping: `src/index_config.json` defines `knn_vector` field `embedding` with FAISS/HNSW, `space_type: l2`.
- `text` is `type: text`, `document_name` is `keyword`.

#### What `GET documents/_mapping` Means
When you run `GET documents/_mapping` in Dev Tools and see:

```
{
	"documents": {
		"mappings": {
			"properties": {
				"document_name": { "type": "keyword" },
				"embedding": {
					"type": "knn_vector",
					"dimension": 768,
					"method": {
						"engine": "faiss",
						"space_type": "l2",
						"name": "hnsw",
						"parameters": {}
					}
				},
				"text": { "type": "text" }
			}
		}
	}
}
```

- **`text: text`**: Full‑text field analyzed by BM25 for keyword search and highlighting. Stores your chunk content.
- **`document_name: keyword`**: Exact‑match field for filtering/aggregations. Holds the original filename for each chunk.
- **`embedding: knn_vector`**: Numeric vector used for nearest‑neighbor queries.
	- **`dimension: 768`**: Matches the embedding size from `all-mpnet-base-v2`.
	- **`method: faiss + hnsw`**: Uses FAISS with HNSW index for fast approximate KNN.
	- **`space_type: l2`**: Distance metric (Euclidean). Lower distance = closer semantic similarity.

Together, these fields enable hybrid search: BM25 over `text` and KNN over `embedding`, then combined by the search pipeline.
### 7.2 Indexing Flow
- Stored metadata: `text` (optionally prefixed if asymmetric), `embedding` (list[float]), `document_name`.
- IDs: `_id` is `"{filename}_{i}"` per chunk; document identity maintained through `document_name`.
### 7.3 Query Flow
- Query embedding: from user query via `get_embedding_model().encode(...)`.
- Hybrid search: OpenSearch `hybrid` combines `match` (text) and `knn` (vector) queries; results normalized/combined via configured pipeline `nlp-search-pipeline` (created once via Dev Tools `PUT /_search/pipeline/...`).
- Top-K: `size=top_k` controls number of hits; embeddings excluded in `_source` for response compactness.
- Example:
	- Query: “What is my CGPA?” → `match` on “CGPA” + KNN on the semantic embedding → returns relevant mark-sheet chunks.

## 8. Retrieval + RAG Logic
- Selection: Iterate hits, collect `['_source']['text']` for top-K.
- Prompt assembly: `prompt_template()` includes context blocks + truncated chat history (last 10 messages) + user query.
- Token budgeting: Controlled implicitly by number/size of chunks (`num_results`) and `TEXT_CHUNK_SIZE`; adjust to fit model context.
- Sample final prompt:
	```
	You are a knowledgeable chatbot assistant.
	Use the following context to answer the question.
	Context:
	Document 0:
	...chunk text...

	Document 1:
	...chunk text...

	Conversation History:
	User: Summarize my mark sheet
	Assistant: ...

	User: What is my CGPA?
	Assistant:
	```

## 9. LLM Interaction
- LLM: Ollama model `OLLAMA_MODEL_NAME` (default `llama3.2:latest`).
- API call: `ollama.chat(model=..., messages=[{role:"user", content: prompt}], stream=True, options={"temperature": temperature})`.
- Temperature: UI slider controls 0.0–1.0.
- Error handling: `ollama.ResponseError` caught; function returns `None` to UI.
- Hallucination reduction: Hybrid search context grounds responses; low temperature; include conversation history for coherence.

### Generation Parameters — Temperature, Top‑P, and Top‑K
These settings affect how the LLM chooses the next token at each step.

- **Temperature:** Rescales token probabilities via softmax smoothing.
	- Lower values (0.1–0.5): more deterministic and concise.
	- Higher values (0.7–1.0): more varied and creative.

- **Top‑P (nucleus sampling):** Limits sampling to the smallest set of tokens whose cumulative probability mass is ≤ top_p.
	- Lower values (0.3–0.6): conservative; avoids long‑tail tokens → fewer hallucinations.
	- Higher values (0.8–1.0): broader candidate set → more diversity.

- **Top‑K (generation):** Restricts sampling to the top K most probable tokens.
	- Lower values (e.g., 20–50): tighter outputs, fewer rare words.
	- Higher values (e.g., 100–200): more variety; can increase noise.

Recommended for RAG (grounded answers):
- **Default:** temperature ≈ 0.4–0.7, top_p ≈ 0.85–0.95, top_k ≈ 40–100.
- If hallucinations appear: lower temperature and/or top_p; reduce top_k.
- If responses feel rigid: increase temperature slightly or top_p.

Example Ollama options (in `chat.py`):
```
options = {"temperature": 0.6, "top_p": 0.9, "top_k": 50}
ollama.chat(model=..., messages=[...], stream=True, options=options)
```

Important distinction with retrieval:
- **Retrieval `top_k` (OpenSearch):** The number of results returned from vector/keyword search (see section 7). This is unrelated to generation top_k. Retrieval `top_k` controls how many context chunks enter the prompt.
- **Generation `top_k`:** The number of top tokens considered when sampling the model’s next token.

## 10. Docker & Infrastructure
### 10.1 OpenSearch via Docker (recommended)
Start OpenSearch locally:
```bash
docker run -d --name opensearch-local \
	-p 9200:9200 -p 9600:9600 \
	-e "discovery.type=single-node" \
	-e "OPENSEARCH_INITIAL_ADMIN_PASSWORD=Admin123!" \
	-e "plugins.security.disabled=true" \
	opensearchproject/opensearch:latest
```
Verify:
```bash
curl http://localhost:9200
```
Create hybrid pipeline (Dev Tools or curl):
```bash
curl -XPUT "http://localhost:9200/_search/pipeline/nlp-search-pipeline" \
	-H 'Content-Type: application/json' -d '{
	"description": "Post processor for hybrid search",
	"phase_results_processors": [
		{"normalization-processor": {
			"normalization": {"technique": "min_max"},
			"combination": {"technique": "arithmetic_mean", "parameters": {"weights": [0.3, 0.7]}}
		}}
	]
}'
```
### 10.2 Dockerfile/Compose
This repo does not include a Dockerfile/Compose for the app itself; Streamlit runs locally. OpenSearch runs in Docker as above.

Alternative explicit commands (OpenSearch + Dashboards 2.11.0):
```bash
docker run -d --name opensearch \
	-p 9200:9200 -p 9600:9600 \
	-e "discovery.type=single-node" \
	-e "DISABLE_SECURITY_PLUGIN=true" \
	opensearchproject/opensearch:2.11.0

docker run -d --name opensearch-dashboards \
	-p 5601:5601 \
	--link opensearch:opensearch \
	-e "OPENSEARCH_HOSTS=http://opensearch:9200" \
	-e "DISABLE_SECURITY_DASHBOARDS_PLUGIN=true" \
	opensearchproject/opensearch-dashboards:2.11.0
```
### 10.3 Commands to Run
```bash
# Python env (ensure Python 3.10–3.12 recommended)
pip install -r requirements.txt

# Run Streamlit
streamlit run Welcome.py

# Optional: run specific pages
streamlit run pages/1_🤖_Chatbot.py
streamlit run pages/2_📄_Upload_Documents.py
```

## API Contract (FastAPI backend for React)
The repository includes `src/server.py` exposing a backend API with endpoints that wrap existing logic:

- `GET /health`
	- Checks OpenSearch availability; returns version and `OLLAMA_MODEL_NAME`.
- `GET /documents`
	- Lists distinct `document_name` values using an aggregation.
- `POST /upload` (multipart)
	- Body: `file` (PDF)
	- Saves file, extracts text, chunks, embeds (768-d), bulk-indexes into `documents`.
	- Response: `{ indexed, errors, chunks }`.
- `DELETE /documents/{documentName}`
	- Deletes all chunks belonging to a specific uploaded file.
- `POST /search`
	- Body: `{ query: string, top_k?: number, use_hybrid?: boolean }`
	- Returns top-K hits with `{ _id, _score, text, document_name }`.
- `POST /chat/stream`
	- Body: `{ query, use_hybrid, num_results, temperature, history }`
	- Streams the LLM response as plain text using the retrieved context.

Run the backend locally:
```bash
pip install fastapi uvicorn
uvicorn src.server:app --host 0.0.0.0 --port 8000
```

## OpenSearch Dashboards Setup
Follow these steps to visualize your `documents` index in OpenSearch Dashboards.

- **Start Dashboards:** Ensure the container is running (see Docker commands above). Default URL: http://localhost:5601
- **Create Data View:**
	- Open Dashboards → Discover → "Create data view".
	- **Name:** `documents`
	- **Index pattern:** `documents`
	- **Timestamp field:** None (leave unset)
	- Save.
- **Discover Documents:**
	- Go to Discover and select the `documents` data view.
	- Add columns `text` and `document_name` for readability.
	- Use search bar to filter, e.g., `CGPA` or `document_name: YourFile.pdf`.
- **Verify Counts:** Compare with Dev Tools `GET documents/_count`.

Screenshot placeholder:

![Dashboards Data View](images/dashboards-data-view.png)

Example frontend calls:
```bash
# Upload
curl -F "file=@/path/to/YourFile.pdf" http://localhost:8000/upload

# Search
curl -X POST http://localhost:8000/search \
	-H 'Content-Type: application/json' \
	-d '{"query":"What is my CGPA?","top_k":5,"use_hybrid":true}'

# Chat stream
curl -X POST http://localhost:8000/chat/stream \
	-H 'Content-Type: application/json' \
	-d '{"query":"Summarize my mark sheet","use_hybrid":true,"num_results":5,"temperature":0.7,"history":[]}'
```

## 11. Environment Variables & Secrets
- `OPENSEARCH_HOST`, `OPENSEARCH_PORT`, `OPENSEARCH_INDEX` in `src/constants.py`.
- `EMBEDDING_MODEL_PATH`, `EMBEDDING_DIMENSION`, `ASSYMETRIC_EMBEDDING`, `OLLAMA_MODEL_NAME` also in `constants.py`.
- Behavior if missing: Changing/removing constants breaks client connections and index creation.

## 12. Error Handling & Edge Cases
- OpenSearch down: Client init or search fails; UI shows spinner error paths; start OpenSearch and retry.
- No results: Hybrid search returns empty hits; prompt built without context; LLM answers generically.
- Empty responses: Streaming generator may be `None` on errors; UI logs and yields empty assistant message.
- Large documents: Chunking with overlap prevents memory blow-up; adjust `TEXT_CHUNK_SIZE`.

## 13. Performance & Scaling Notes
- Bottlenecks: First-time model download; per-chunk encoding; OpenSearch KNN on large corpora.
- Latency: Query embedding + OpenSearch hybrid pipeline + LLM generation.
- Caching: `@st.cache_resource` caches embedding model and Ollama pull checks.
- Scaling: Move Streamlit behind a reverse proxy; separate ingestion worker; run OpenSearch on dedicated hardware; batch embeddings; consider GPU acceleration for embeddings.

## 14. End-to-End Example
1. User uploads `Consolidated_finalMarksheet.pdf` on Upload page.
2. Text extracted (PyPDF2; OCR if needed), cleaned, chunked into ~300-token segments with 100-token overlap.
3. Each chunk encoded → 768-d vectors.
4. Indexed into OpenSearch with `_id = filename_i`, storing `text`, `embedding`, `document_name`.
5. User asks: “What is my CGPA?” in Chatbot.
6. Query embedded; OpenSearch hybrid returns top-5 chunks containing “CGPA” and semantically similar content.
7. Prompt constructed with those chunks + last 10 messages.
8. Ollama `llama3.2:latest` streams answer grounded in context.

### Runtime trace example (from `logs/app.log`)
This is how requests flow across components in practice:
```
2025-12-14 23:44:47,787 - INFO - OpenSearch client initialized.
2025-12-14 23:44:47,808 - INFO - POST http://localhost:9200/documents/_search?search_pipeline=nlp-search-pipeline [status:200 request:0.021s]
2025-12-14 23:44:47,808 - INFO - Hybrid search completed for query 'give the policy number and paid amount ?' with top_k=5.
2025-12-14 23:44:47,808 - INFO - Hybrid search completed.
2025-12-14 23:44:47,808 - INFO - Prompt constructed with context and conversation history.
2025-12-14 23:44:47,808 - INFO - Streaming response from LLaMA model.
2025-12-14 23:44:50,630 - INFO - HTTP Request: POST http://127.0.0.1:11434/api/chat "HTTP/1.1 200 OK"
2025-12-14 23:44:50,947 - INFO - Response generated and displayed.
```
Routing breakdown:
- UI captures the prompt → `chat.generate_response_streaming()`
- Embedding for query computed → `opensearch.hybrid_search()` called
- OpenSearch executes `hybrid` (BM25 match on `text` + KNN on `embedding`) → pipeline normalizes/combines
- Top-K chunks returned → `prompt_template()` builds final prompt
- Ollama streams tokens → UI renders incrementally

## 15. Summary Diagram
- User → Streamlit (UI)
- Streamlit → Backend (`chat.py`, `embeddings.py`, `opensearch.py`)
- Backend → Embeddings (SentenceTransformer)
- Backend → OpenSearch (hybrid search over `documents`)
- OpenSearch → Backend → LLM (Ollama)
- LLM → Streamed Answer → UI

---

## Quick Reference — Dev Tools Queries
Here’s a concise cheat sheet for OpenSearch Dashboards Dev Tools.

- List indices
	- `GET _cat/indices?v`
- Index stats
	- `GET documents/_stats`
- Count docs
	- `GET documents/_count`
- View index mapping (schema)
	- `GET documents/_mapping`
- View index settings
	- `GET documents/_settings`
- Search all docs (hide large vectors)
	- `GET documents/_search
		{
			"query": { "match_all": {} },
			"_source": ["text", "document_name"],
			"size": 10
		}`
- Keyword search
	- `GET documents/_search
		{
			"query": { "match": { "text": "CGPA" } },
			"_source": ["text", "document_name"],
			"size": 5
		}`
 - Hybrid search (BM25 + KNN) in Dev Tools
	- ```POST documents/_search?search_pipeline=nlp-search-pipeline
		{
			"_source": { "exclude": ["embedding"] },
			"query": {
				"hybrid": {
					"queries": [
						{ "match": { "text": { "query": "What is my CGPA?" } } },
						{ "knn": { "embedding": { "vector": [ /* 768 floats */ ], "k": 5 } } }
					]
				}
			},
			"size": 5
		}```
	- Note: Replace `[ /* 768 floats */ ]` with a real 768‑d vector (e.g., from `SentenceTransformer(all-mpnet-base-v2).encode(...).tolist()`). An empty or wrong‑length vector returns 400 `[knn] query vector is empty`.
- Aggregation (list distinct document names)
	- `GET documents/_search
		{
			"size": 0,
			"aggs": { "unique_docs": { "terms": { "field": "document_name", "size": 100 } } }
		}`
- Delete-by-query (remove one uploaded file’s chunks)
	- `POST documents/_delete_by_query
		{
			"query": { "term": { "document_name": "YourFile.pdf" } }
		}`
- View search pipeline config
	- `GET _search/pipeline/nlp-search-pipeline`
- Create search pipeline (normalization + combination)
 ```
   PUT _search/pipeline/nlp-search-pipeline
		{
			"description": "Post processor for hybrid search",
			"phase_results_processors": [
				{
					"normalization-processor": {
						"normalization": { "technique": "min_max" },
						"combination": { "technique": "arithmetic_mean", "parameters": { "weights": [0.3, 0.7] } }
					}
				}
			]
		}
  ```
- Delete index (use with caution)
	- `DELETE documents`
- Recreate index from mapping file (CLI example shown earlier; via app: handled by `create_index`)

Hybrid search example payload (for reference — the app constructs this):
```
POST documents/_search
{
	"_source": { "exclude": ["embedding"] },
	"query": {
		"hybrid": {
			"queries": [
				{ "match": { "text": { "query": "What is my CGPA?" } } },
				{ "knn": { "embedding": { "vector": [/* 768-d vector */], "k": 5 } } }
			]
		}
	},
	"size": 5
}
```

## Worked Example — “Policy number and paid amount” (BM25 + KNN)
Goal: Show how a natural-language query is answered using hybrid retrieval and RAG.

1. User enters in Chat UI: “give the policy number and paid amount ?”.
2. Query preprocessing: If `ASSYMETRIC_EMBEDDING` is False, query is used as-is; otherwise prefixed with `passage:` for asymmetric models.
3. Query embedding: `SentenceTransformer(all-mpnet-base-v2)` encodes the query → a 768-d vector (e.g., `[0.012, -0.041, ...]`).
4. Hybrid search request built by app:
	 ```json
	 {
		 "_source": { "exclude": ["embedding"] },
		 "query": {
			 "hybrid": {
				 "queries": [
					 { "match": { "text": { "query": "give the policy number and paid amount ?" } } },
					 { "knn": { "embedding": { "vector": [/* 768-d */], "k": 5 } } }
				 ]
			 }
		 },
		 "size": 5
	 }
	 ```
5. BM25 contribution: Matches chunks whose `text` contains tokens like “policy”, “number”, “paid”, “amount”.
6. KNN contribution: Retrieves semantically similar chunks even if phrasing differs (e.g., “premium paid”, “policy id”).
7. Pipeline normalization: Scores are normalized (min-max) and combined (weighted mean 0.3 BM25, 0.7 KNN) by `nlp-search-pipeline`.
8. Returned hits: Top-5 chunks with fields `{ text, document_name }` used; embeddings excluded for brevity.
9. Prompt construction:
	 ```
	 You are a knowledgeable chatbot assistant.
	 Use the following context to answer the question.
	 Context:
	 Document 0:
	 ... chunk mentioning policy number and payments ...

	 Document 1:
	 ... chunk mentioning paid amount or premium ...

	 Conversation History:
	 User: give the policy number and paid amount ?

	 Assistant:
	 ```
10. LLM generation: Ollama (`llama3.2:latest`) streams a grounded answer, extracting values from the context chunks (e.g., “Policy Number: X12345; Paid Amount: ₹12,500”).
11. UI displays the streaming response and appends it to `chat_history`.

Observability: The log lines above confirm the chain: OpenSearch search → pipeline → prompt → Ollama chat → streamed response.

## Troubleshooting
- Pipeline error: Create via curl shown above.
- Torch warning about `torch.classes`: benign; can ignore.
- Ollama not serving: run `ollama serve`; confirm `ollama list` shows `llama3.2:latest`.
