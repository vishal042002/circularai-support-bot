# Multi-Tenant FastAPI RAG Service - Implementation Plan

## Overview

Transform the current single-client Streamlit RAG system into a production-ready, multi-tenant FastAPI service with:
- Complete client data isolation via Pinecone namespaces
- Dynamic client configuration via Azure Cosmos DB
- RESTful API with authentication
- Docker deployment ready
- Preserving the excellent existing RAG pipeline

## Current Architecture Summary

**Strengths to Preserve:**
- Clean modular structure: `core/`, `agents/`, `ingestion/`
- Sophisticated RAG pipeline: Hybrid search (70% semantic + 30% BM25)
- Hierarchical document chunking with GPT-4 Vision for screenshots
- LangGraph orchestration with conversation memory
- Azure OpenAI integration

**Key Components:**
- [agents/rag_agent.py](agents/rag_agent.py) - Main LangGraph RAG agent
- [core/search.py](core/search.py) - Hybrid search engine
- [core/vector_store.py](core/vector_store.py) - Pinecone wrapper
- [ingestion/pipeline.py](ingestion/pipeline.py) - 6-step document ingestion
- [config/settings.py](config/settings.py) - Pydantic configuration

**Current Limitations:**
- Single-client only (hardcoded "CircularAI")
- No client/tenant isolation
- No API layer
- Prompts not database-driven

## Target Architecture

```
Client Apps → FastAPI Service → Multi-Tenant RAG Engine
                    ↓                      ↓
              Auth Middleware        Client Config Service
                    ↓                      ↓
              Cosmos DB (configs)    Pinecone (namespaced data)
```

## Implementation Strategy

### Phase 1: FastAPI Foundation (Week 1)
**Goal:** Working FastAPI service with backward compatibility

**New Directory Structure:**
```
api/
├── main.py                    # FastAPI app entry point
├── dependencies.py            # Dependency injection
├── middleware/
│   ├── auth.py               # Client authentication
│   └── error_handler.py      # Global error handling
├── routers/
│   ├── chat.py               # Chat endpoints
│   ├── ingest.py             # Document ingestion
│   ├── admin.py              # Client management
│   └── health.py             # Health checks
└── schemas/
    ├── chat.py               # Request/response models
    ├── ingest.py             # Ingestion models
    └── client.py             # Client config models
```

**Tasks:**
1. Create `api/` directory and core files
2. Implement [api/main.py](api/main.py) with FastAPI app setup
3. Create [api/routers/chat.py](api/routers/chat.py) with POST `/api/v1/chat` endpoint
4. Create [api/schemas/chat.py](api/schemas/chat.py) with Pydantic models
5. Implement [api/routers/health.py](api/routers/health.py) with GET `/health`
6. Create [main.py](main.py) as uvicorn entry point
7. Update [requirements.txt](requirements.txt) with: `fastapi`, `uvicorn[standard]`, `python-multipart`
8. Test: Run FastAPI alongside Streamlit (different ports)

**Validation:**
- `uvicorn api.main:app --reload` starts on port 8000
- `curl http://localhost:8000/health` returns 200
- Streamlit app still works on port 8501

### Phase 2: Cosmos DB Integration (Week 2)
**Goal:** Client configurations loaded from Cosmos DB instead of hardcoded YAML

**New Files:**
```
database/
├── cosmos.py                  # Cosmos DB client & operations
├── models.py                  # Pydantic models for DB documents
└── cache.py                   # In-memory config cache

scripts/
└── setup_cosmos.py            # Initialize Cosmos DB containers
```

**Cosmos DB Schema:**

**Container: clients** (partition key: `/client_id`)
```json
{
  "id": "client_circular_ai",
  "client_id": "circular_ai",
  "name": "CircularAI Platform",
  "api_key_hash": "sha256_hash",
  "status": "active",
  "config": {
    "prompts": {
      "system_prompt": "You are a helpful AI assistant...",
      "greeting": "Hello! I'm here to help...",
      "no_context_response": "I don't have information about that."
    },
    "search_settings": {
      "semantic_weight": 0.7,
      "bm25_weight": 0.3,
      "top_k": 5,
      "similarity_threshold": 0.5
    },
    "llm_settings": {
      "temperature": 0.7,
      "max_tokens": 800
    },
    "features": {
      "image_processing": true,
      "conversation_memory": true
    }
  },
  "pinecone_namespace": "circular_ai"
}
```

**Container: documents** (partition key: `/client_id`)
```json
{
  "id": "doc_circular_ai_user_guide_001",
  "client_id": "circular_ai",
  "filename": "CircularAI_User_Guide.docx",
  "version": "1.0",
  "uploaded_at": "2026-01-01T12:00:00Z",
  "status": "active",
  "ingestion": {
    "total_chunks": 350,
    "images_processed": 15
  }
}
```

**Tasks:**
1. Create [database/cosmos.py](database/cosmos.py) with `CosmosDBClient` class
2. Create [database/models.py](database/models.py) with Pydantic schemas for clients/documents
3. Create [database/cache.py](database/cache.py) for LRU config caching
4. Create [scripts/setup_cosmos.py](scripts/setup_cosmos.py) to initialize containers
5. Update [config/settings.py](config/settings.py) to add `CosmosConfig` class
6. Add to `.env`: `COSMOS_ENDPOINT`, `COSMOS_KEY`, `COSMOS_DATABASE`
7. Update [requirements.txt](requirements.txt): `azure-cosmos`
8. Migrate existing CircularAI config from [config/prompts.yaml](config/prompts.yaml) to Cosmos
9. Test: Load client config from Cosmos successfully

**Key Implementation Details:**
- Use `azure-cosmos` SDK for async operations
- Implement API key hashing with SHA256
- Cache configs in-memory with TTL (5 minutes)
- Graceful fallback to YAML if Cosmos unavailable (dev mode)

### Phase 3: Multi-Tenancy Core (Week 2-3)
**Goal:** Multiple clients can use service with complete data isolation

**New Files:**
```
services/
├── client_service.py          # Client config management
├── rag_service.py             # Multi-tenant RAG orchestration
├── session_service.py         # Session management
└── document_service.py        # Document lifecycle management
```

**Files to Modify:**

**1. [core/vector_store.py](core/vector_store.py)**
```python
class VectorStore:
    def __init__(self, namespace: str = None):
        self.namespace = namespace  # ADD
        # ... existing code ...

    def search(self, query_vector, top_k=5):
        results = self.index.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=True,
            namespace=self.namespace  # ADD
        )
```

**2. [core/search.py](core/search.py)**
```python
class HybridSearch:
    def __init__(self, embedding_service, vector_store, namespace=None):
        self.namespace = namespace  # ADD
        # ... existing code ...

    def _fetch_all_from_pinecone(self):
        # Fetch only from this client's namespace
        results = self.vector_store.index.query(
            vector=dummy_vector,
            top_k=batch_size,
            include_metadata=True,
            namespace=self.namespace  # ADD
        )
```

**3. [agents/rag_agent.py](agents/rag_agent.py)**
```python
class RAGAgent:
    def __init__(self, ..., client_config: Dict = None):
        self.client_config = client_config or {}

        # Load prompts from config instead of YAML
        if client_config:
            self.prompts = client_config.get("prompts", {})
            self.system_prompt = self.prompts["system_prompt"]
```

**4. [ingestion/pipeline.py](ingestion/pipeline.py)**
```python
class IngestionPipeline:
    def ingest_docx(self, docx_path, client_id: str, process_images=True):
        # ... existing code ...

        # Upload to client's namespace
        self.vector_store.namespace = client_id
        self.vector_store.upsert(vectors)
```

**Tasks:**
1. Modify [core/vector_store.py](core/vector_store.py) to add namespace support
2. Modify [core/search.py](core/search.py) to add namespace support
3. Modify [agents/rag_agent.py](agents/rag_agent.py) to inject client config
4. Modify [ingestion/pipeline.py](ingestion/pipeline.py) to accept `client_id` parameter
5. Create [services/rag_service.py](services/rag_service.py) with `MultiTenantRAGService` class
6. Implement per-client agent caching (avoid recreating on every request)
7. Create test client in Cosmos DB: `test_client`
8. Test: Ingest documents to different namespaces, verify isolation

**Key Design Decisions:**
- **Session IDs:** Prefix with `{client_id}_` to ensure isolation (e.g., `circular_ai_session_123`)
- **BM25 Index:** Build per-namespace by fetching only that client's vectors
- **Agent Caching:** Cache agents in-memory keyed by `client_id` (max 100 clients, LRU)
- **SQLite Checkpoints:** Single DB with prefixed thread_ids (simpler Docker volumes)

### Phase 4: Authentication & API Security (Week 3)
**Goal:** API key authentication enforcing client isolation

**Tasks:**
1. Implement [api/middleware/auth.py](api/middleware/auth.py)
   - Extract `X-API-Key` header
   - Hash and lookup in Cosmos DB
   - Attach `client_id` and `client_config` to `request.state`
   - Return 401 for invalid/missing keys
2. Implement [api/middleware/error_handler.py](api/middleware/error_handler.py)
   - Global exception handling
   - Structured error responses
   - Log errors with client context
3. Create [api/routers/admin.py](api/routers/admin.py)
   - POST `/api/v1/admin/clients` - Create client
   - GET `/api/v1/admin/clients/{client_id}` - Get client details
   - PUT `/api/v1/admin/clients/{client_id}/config` - Update config
   - Requires separate admin API key
4. Generate secure API keys (32-byte random, base64-encoded)
5. Test: Invalid key returns 401, valid key grants access to correct namespace

**API Endpoints:**

**Chat:**
```
POST /api/v1/chat
Headers: X-API-Key: <client_api_key>
Body: {"message": "How do I create a report?", "session_id": "user123_session_001"}
Response: {"answer": "...", "sources": [...], "session_id": "...", "client_id": "..."}
```

**Health:**
```
GET /health
Response: {"status": "healthy", "services": {"pinecone": "connected", ...}}
```

### Phase 5: Document Ingestion API (Week 4)
**Goal:** Upload documents via REST API with async processing

**Tasks:**
1. Create [api/routers/ingest.py](api/routers/ingest.py)
2. Implement POST `/api/v1/ingest/document` (multipart file upload)
3. Implement background task processing with FastAPI `BackgroundTasks`
4. Implement GET `/api/v1/ingest/status/{job_id}` for job status tracking
5. Implement GET `/api/v1/ingest/documents` to list client's documents
6. Implement DELETE `/api/v1/ingest/document/{document_id}` to remove documents
7. Save document metadata to Cosmos DB `documents` container
8. Test: Upload DOCX → processing → searchable in chat

**Ingestion Flow:**
1. Client uploads DOCX via POST `/api/v1/ingest/document`
2. Return 202 with `job_id` immediately
3. Background task:
   - Run existing ingestion pipeline
   - Upload to client's Pinecone namespace
   - Save metadata to Cosmos
   - Update job status
4. Client polls GET `/api/v1/ingest/status/{job_id}` for completion

### Phase 6: Docker Deployment (Week 5)
**Goal:** Containerized service ready for local and Azure deployment

**New Files:**
```
docker/
├── Dockerfile
├── docker-compose.yml
└── .dockerignore
```

**Dockerfile Structure:**
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN mkdir -p data/checkpoints data/logs
EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**docker-compose.yml:**
```yaml
services:
  rag-service:
    build: .
    ports:
      - "8000:8000"
    environment:
      - AZURE_OPENAI_ENDPOINT=${AZURE_OPENAI_ENDPOINT}
      - PINECONE_API_KEY=${PINECONE_API_KEY}
      - COSMOS_ENDPOINT=${COSMOS_ENDPOINT}
      # ... other env vars from .env
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    restart: unless-stopped
```

**Tasks:**
1. Create [docker/Dockerfile](docker/Dockerfile)
2. Create [docker/docker-compose.yml](docker/docker-compose.yml)
3. Create [docker/.dockerignore](docker/.dockerignore) (exclude `.env`, `data/`, `.git`)
4. Test: `docker-compose up` builds and runs successfully
5. Test: Access API at `http://localhost:8000/health`
6. Document environment variables required
7. Document volume mount strategy (persist checkpoints, exclude secrets)

**Security Best Practices:**
- Never include `.env` in Docker image
- Use environment variables for all secrets
- Volume mount for persistent data only
- Prepare for Azure Managed Identity (future)

### Phase 7: Migration & Testing (Week 5)
**Goal:** Migrate existing CircularAI data, comprehensive testing

**Migration Tasks:**
1. Create [scripts/migrate_client.py](scripts/migrate_client.py)
   - Fetch all vectors from Pinecone (current default namespace)
   - Re-upload to `circular_ai` namespace
   - Verify vector count matches
   - Delete old vectors (manual confirmation required)
2. Create [scripts/migrate_prompts.py](scripts/migrate_prompts.py)
   - Load from [config/prompts.yaml](config/prompts.yaml)
   - Create client document in Cosmos DB
   - Verify config loads correctly
3. Update documentation (README.md)
4. Create deployment guide

**Testing Tasks:**
1. Create [tests/test_api.py](tests/test_api.py) - FastAPI endpoint tests
2. Create [tests/test_services.py](tests/test_services.py) - Business logic tests
3. Create [tests/test_multi_tenancy.py](tests/test_multi_tenancy.py) - Isolation tests
4. Manual testing:
   - Create 2 test clients with different documents
   - Verify complete data isolation
   - Test session management across clients
   - Test document ingestion for each client

**Validation Checklist:**
- [ ] Multiple clients can chat simultaneously without data leakage
- [ ] Client configs load from Cosmos DB correctly
- [ ] Pinecone namespaces enforce complete isolation
- [ ] API authentication works correctly
- [ ] Document ingestion via API succeeds
- [ ] Docker deployment works locally
- [ ] Health checks validate all dependencies
- [ ] Existing Streamlit app still works (backward compatibility)

## Critical Files Summary

**New Files to Create (15 files):**
1. [api/main.py](api/main.py) - FastAPI app entry point
2. [api/middleware/auth.py](api/middleware/auth.py) - Authentication
3. [api/routers/chat.py](api/routers/chat.py) - Chat endpoints
4. [api/routers/health.py](api/routers/health.py) - Health checks
5. [api/routers/ingest.py](api/routers/ingest.py) - Document ingestion
6. [api/routers/admin.py](api/routers/admin.py) - Client management
7. [api/schemas/chat.py](api/schemas/chat.py) - Pydantic models
8. [database/cosmos.py](database/cosmos.py) - Cosmos DB client
9. [database/models.py](database/models.py) - DB schemas
10. [database/cache.py](database/cache.py) - Config caching
11. [services/rag_service.py](services/rag_service.py) - Multi-tenant orchestration
12. [scripts/setup_cosmos.py](scripts/setup_cosmos.py) - DB initialization
13. [scripts/migrate_client.py](scripts/migrate_client.py) - Data migration
14. [docker/Dockerfile](docker/Dockerfile) - Container definition
15. [docker/docker-compose.yml](docker/docker-compose.yml) - Local deployment

**Existing Files to Modify (5 files):**
1. [core/vector_store.py](core/vector_store.py) - Add namespace parameter
2. [core/search.py](core/search.py) - Add namespace support
3. [agents/rag_agent.py](agents/rag_agent.py) - Inject client config
4. [ingestion/pipeline.py](ingestion/pipeline.py) - Add client_id parameter
5. [config/settings.py](config/settings.py) - Add Cosmos configuration

**Files to Update:**
- [requirements.txt](requirements.txt) - Add: `fastapi`, `uvicorn[standard]`, `azure-cosmos`, `python-multipart`
- `.env` - Add Cosmos DB credentials

## Dependencies to Add

```txt
# API Framework
fastapi==0.109.0
uvicorn[standard]==0.27.0
python-multipart==0.0.6

# Azure Cosmos DB
azure-cosmos==4.5.1

# Testing
pytest==7.4.0
pytest-asyncio==0.21.0
httpx==0.26.0  # For TestClient
```

## Environment Variables Required

```bash
# Existing (keep)
AZURE_OPENAI_ENDPOINT=https://xxx.openai.azure.com/
AZURE_OPENAI_API_KEY=xxx
AZURE_DEPLOYMENT_NAME=gpt-4o
AZURE_EMBEDDING_DEPLOYMENT=text-embedding-3-small
PINECONE_API_KEY=xxx
PINECONE_INDEX_NAME=circularai-support

# New (add)
COSMOS_ENDPOINT=https://xxx.documents.azure.com:443/
COSMOS_KEY=xxx
COSMOS_DATABASE=rag-service-prod
ADMIN_API_KEY=xxx  # For /api/v1/admin endpoints
```

## Design Decisions & Trade-offs

**1. Single Pinecone Index vs. Multiple Indices**
- **Decision:** Single index with namespaces per client
- **Rationale:** Cost-effective, complete isolation, easier management
- **Trade-off:** All clients share index quotas (mitigate with monitoring)

**2. SQLite Checkpointing Strategy**
- **Decision:** Single SQLite DB with prefixed thread_ids (`{client_id}_{session_id}`)
- **Rationale:** Simpler Docker volume management, easier backups
- **Trade-off:** Slight risk of collision (mitigated by prefix)

**3. Agent Caching**
- **Decision:** In-memory LRU cache per client (max 100 clients)
- **Rationale:** Avoid recreating agents on every request, faster responses
- **Trade-off:** Memory usage scales with clients

**4. Async Ingestion**
- **Decision:** FastAPI `BackgroundTasks` for Phase 5, Celery for production scale
- **Rationale:** Document processing is slow (GPT-4 Vision), non-blocking API
- **Trade-off:** Added complexity for job tracking

**5. Cosmos DB Schema**
- **Decision:** Schemaless documents with client_id partition key
- **Rationale:** Flexible client configs, read-heavy pattern optimization
- **Trade-off:** Higher cost vs PostgreSQL (acceptable for enterprise)

## Success Metrics

**Phase 1-2 Complete:**
- FastAPI service responds to `/api/v1/chat` with hardcoded config
- Health endpoint returns 200
- Cosmos DB connected and serving configs

**Phase 3 Complete:**
- 2 test clients with different documents
- Complete data isolation verified (no cross-client results)
- BM25 indexes built per namespace

**Phase 4 Complete:**
- API key authentication working
- Invalid keys rejected with 401
- Admin endpoints create/update clients

**Phase 5-7 Complete:**
- Documents uploaded via API
- Docker deployment working locally
- Migration scripts tested
- All tests passing

## Next Steps After Plan Approval

1. **Phase 1:** Create FastAPI skeleton (2-3 days)
2. **Phase 2:** Cosmos DB integration (2-3 days)
3. **Phase 3:** Multi-tenancy core (3-4 days)
4. **Phase 4:** Authentication (2 days)
5. **Phase 5:** Ingestion API (2-3 days)
6. **Phase 6:** Docker deployment (1-2 days)
7. **Phase 7:** Migration & testing (2-3 days)

**Total estimated time:** 4-5 weeks for complete implementation

## Backward Compatibility

**During Development:**
- Streamlit app ([app.py](app.py)) continues to work alongside FastAPI
- Both use same core modules ([core/](core/), [agents/](agents/), [ingestion/](ingestion/))
- Ports: Streamlit (8501), FastAPI (8000)

**Post-Migration:**
- Streamlit can be kept as admin/demo tool
- Or deprecated in favor of .NET frontend consuming FastAPI
- Core RAG pipeline unchanged, just API layer added

## Risk Mitigation

**Risk 1: Namespace migration fails**
- Mitigation: Test migration script on copy of data first
- Backup: Keep old vectors until verification complete

**Risk 2: BM25 index rebuild slow**
- Mitigation: Cache BM25 indices per namespace, rebuild only on document changes
- Backup: Make BM25 optional, use semantic-only if timeout

**Risk 3: Cosmos DB costs**
- Mitigation: Start with minimal RU/s (400), scale as needed
- Monitor query patterns, optimize partition keys

**Risk 4: Agent caching memory usage**
- Mitigation: LRU cache with max size, evict least-used clients
- Monitor memory, adjust cache size based on instance capacity

This plan preserves your excellent RAG pipeline while adding production-grade multi-tenancy, API access, and enterprise deployment readiness. The phased approach allows testing at each stage and maintains backward compatibility throughout development.
