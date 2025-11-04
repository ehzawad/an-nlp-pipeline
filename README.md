# Dialogue AI Pipeline

Production-ready Bengali dialogue system for National ID (NID) customer support. The project combines intent classification, semantic search, entity extraction, multi-turn forms, and event-sourced session management into a single FastAPI service with rich observability and multi-tenant isolation.

> Tip: The architecture deep dive lives in `docs/SYSTEM_ARCHITECTURE.txt` (ASCII diagram). This README summarises the pieces you will actually touch day-to-day.

---

## Highlights

- **End-to-end pipeline** – 14-stage dialogue flow with hooks, forms, policy routing, summarisation stub, and persistence.
- **Hybrid NLP stack** – Logistic regression classifier over multilingual-e5 embeddings, FAISS semantic search, fractional query detection, and regex + transformer-backed NER.
- **Event sourcing** – Every turn appended to per-tenant JSONL event logs under `event_store/`, plus rotating dialogue logs in `logs/`.
- **Tenant aware** – `TenantContext` lazily wires models, caches, event buses, and session stores per tenant; FastAPI app boots a factory and caches contexts.
- **Extensive test suite** – 65+ unit & integration tests covering forms, policies, infrastructure, hooks, multi-turn pipelines, and tenant isolation.

---

## Repository Tour

| Path | Purpose |
| --- | --- |
| `src/interfaces/api/app.py` | FastAPI application: startup lifecycle, REST + WebSocket endpoints, CORS, tracing, and dialogue pipeline wiring. |
| `src/application/` | Core application services (dialogue pipeline, policies, forms, NLP orchestration, session store, hooks, summarisation). |
| `src/shared/tenant_context.py` | Tenant factory/context with lazy initialisation of models, caches, event store, form registry, etc. |
| `src/infrastructure/` | Cross-cutting utilities: tracing, circuit breaker, event store/bus. |
| `src/domain/events.py` | Immutable domain event definitions and helpers. |
| `pipelines/` | Data preparation and training scripts for classifiers, FAISS indices, and cache utilities. |
| `config/*.json` | Deployment/runtime configuration (classification, semantic search, dialogue tuning, logging). |
| `datasets/` | Raw Bengali corpora (clusters/tags, answers, fractional queries) and processed splits for classification. |
| `models/` | Persisted models: classifier pickle, label encoder, FAISS indices, fractional classifier, and embedding cache. |
| `event_store/` | Append-only JSONL event streams per tenant/session (used for replay & auditing). |
| `tests/` | Unit + integration tests with pytest-asyncio coverage. |
| `docs/SYSTEM_ARCHITECTURE.txt` | Large ASCII architecture write-up (client ➜ gateway ➜ dialogue pipeline). |
| `examples/` | CLI demos, scripted conversations, and proof-of-concept credit-card service form. |

---

## Dialogue Pipeline Overview

`src/application/dialogue_service.py` defines the **EnhancedDialoguePipeline**, chaining 14 explicit stages:

1. Session load / creation (`SessionManager`, `InMemorySessionStore`)
2. Event publishing (`turn.started`)
3. Hook `before_nlp`
4. Pre-processing (summarisation stub + fractional context augmentation)
5. NER extraction (regex + transformer)
6. NLP stage (classification + semantic search)
7. Hook `after_nlp`
8. Hook `before_policy`
9. Policy engine (handler chain for escalation → active forms → form triggers → FAQ → clarification)
10. Hook `after_policy`
11. Response generation
12. Hook `before_response`
13. Session persistence (history + event append)
14. Hook `after_response`

The pipeline emits domain events (`nlp.completed`, form slot updates, etc.) stored under `event_store/<tenant>/<aggregate>.jsonl`. Hooks (`src/application/hooks`) can log, collect metrics, or react to failures at each stage.

---

## NLP Stack & Models

| Component | Location | Notes |
| --- | --- | --- |
| **Embedding model** | `models/embeddings/e5_cache/` | Multilingual E5 large instruct (SentenceTransformer). Loaded lazily per tenant. |
| **Intent classifier** | `models/classification/model.pkl` | Scikit-learn logistic regression (48 clusters) trained via `pipelines/training/train_classifier.py`. Validation metrics stored in `validation_metrics.json` (≈0.91 balanced accuracy). See [Classification Component Guide](docs/CLASSIFICATION_COMPONENT.md) for details. |
| **Semantic search** | `models/semantic_search/` | FAISS `IndexFlatIP` per cluster + merged index; metadata lists question-tag mappings for 3.2k Bengali FAQs. Built with `pipelines/training/build_indices.py`. |
| **Fractional query detector** | `models/context/fractional_classifier.pkl` | Detects context-dependent queries, trained on `datasets/raw/fractional_queries.txt` via `train_fractional_classifier.py`. |
| **NER** | Runtime | `NERExtractor` mixes regex patterns (`entity_patterns.py`) and optional transformer (xlm-roberta) with slot mapping. |
| **Summariser** | Runtime | Placeholder `PassthroughSummarizer` keeps text unchanged; structure ready for real summarisation. |

---

## Domain Data

- **Raw corpora (`datasets/raw/`)**
  - `questions_with_clusters.csv` – 3,200+ Bengali user utterances labelled into 48 support clusters.
  - `questions_with_tags.csv` + `tag_answer.csv` – FAQ tag ↔ answer pairs used by semantic search.
  - `cluster_to_tags_full.json` – mapping of cluster → tag list for policy + forms.
  - `fractional_queries.txt` – Bengali short queries signalling dependency on prior context.
  - `test_english_dataset.csv` – small English demo dataset for cross-language experimentation.
- **Processed splits (`datasets/processed/classification/`)**
  - Stratified train/val/test CSVs produced by `pipelines/data/prepare_datasets.py`, configurable via `config/classification.json`.

---

## Forms & Policy

- Forms live in `src/application/forms/examples/` (NID status, correction request, hotel booking demo, bank account verification).
- `BaseForm`/`BaseSlot` support async validators, regex checks, max retries, and optional on-start/on-complete hooks.
- `FormRunner` encapsulates slot collection, validation, execution, failure handling; `FormInterruptionHandler` sketches how FAQs interrupt active forms.
- Policy chain (`src/application/policy/handlers.py`):
  1. `EscalationHandler` – respects `session_state.escalation_flag`.
  2. `ActiveFormHandler` – continues active forms (with optional interruption detection).
  3. `FormTriggerHandler` – spins up new forms based on top FAQ tags.
  4. `HighConfidenceFAQHandler` – direct answers if classifier confidence ≥ threshold.
  5. `ClarificationFallbackHandler` – clarifies or escalates when uncertain.

---

## Configuration Cheatsheet

| File | Description |
| --- | --- |
| `config/main.json` | Project metadata, path roots, embedding model IDs, logging defaults. |
| `config/classification.json` | Classifier hyperparameters, train/val/test ratios, merged-cluster behaviour. |
| `config/semantic_search.json` | FAISS build options, prompts, artifact paths. |
| `config/dialogue.json` | Feature toggles (forms, summarisation, context augmentation), Redis settings, policy thresholds, form mappings. |
| `config/ner.json` | NER device, regex patterns for structured IDs, Bert slot mapping. |
| `config/cluster_thresholds.json` | Cluster-specific policy thresholds (auto-generated heuristic). |
| `config/logging_config.json` | Detailed logging formatter + rotating file handler. |

`ConfigLoader` (application/config/loader.py) handles overrides, caching, and repo-relative path resolution.

---

## Running the Service

1. **Environment**
   ```bash
   python3.13 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
   > PyTorch + FAISS wheels are listed; expect a sizeable install.

2. **Model Assets**
   - Ensure `models/` contains classifier pickle, label encoder, FAISS indices, fractional classifier, and E5 cache.
   - If missing, retrain/build using scripts in the next section.

3. **Start API**
   ```bash
   uvicorn src.interfaces.api.app:app --reload
   ```
   - Startup registers tenants, initialises shared session stores, and pre-builds dialogue pipelines per tenant.
   - REST endpoints:
     - `GET /` – service metadata
     - `GET /health` – basic readiness ping
     - `POST /api/v2/chat` – main dialogue entrypoint (`query`, optional `session_id`, `tenant_id`)
   - WebSocket: `/ws/chat/{session_id}` for streaming updates (echo stub included).

4. **Observation**
   - Rotating logs: `logs/dialogue_system.log`, `logs/app_YYYY-MM-DD.log`
   - Event streams: `event_store/<tenant>/<session>.jsonl`

---

## Data Prep & Training Pipelines

Located under `pipelines/` (call with project root on `PYTHONPATH`):

| Script | Purpose | Key Inputs/Outputs |
| --- | --- | --- |
| `data/prepare_datasets.py` | Stratified train/val/test split for classifier. | Reads raw cluster CSV → writes `datasets/processed/classification/*.csv`. |
| `training/train_classifier.py` | Train logistic regression intent classifier with optional grid search + calibration. | Consumes processed splits, writes model & label encoder to `models/classification/`, stores metrics. |
| `training/train_fractional_classifier.py` | Train logistic regression classifier to flag fractional queries. | Uses fractional queries + complete samples, emits `models/context/fractional_classifier.pkl` & metadata. |
| `training/build_indices.py` | Compute E5 embeddings and build FAISS indices / metadata for semantic search. | Writes `models/semantic_search/*.index`, `cluster_metadata.json`. |
| `utils/clear_cache.py` | Manage `cache/` directory entries. | Supports `--list` / `--clear`. |

All scripts rely on configuration in `config/*.json` and expect the embedding model to be cached (downloads handled automatically by SentenceTransformers).

---

## Testing

Pytest is configured in `pytest.ini` with coverage thresholds and async defaults.

```bash
pytest                 # run entire suite with coverage
pytest tests/unit      # faster feedback on core units
pytest tests/integration/test_dialogue_flow.py::test_multi_turn_form_completion
```

Tests use extensive fakes/mocks to isolate components (e.g., scripted NLP results, fake tenant contexts). Coverage reports target `src/application|domain|infrastructure|shared`.

---

## Logging, Events & Tracing

- **Logging** – Structured console + rotating file handlers (`config/logging_config.json`). `scripts/add_comprehensive_logging.py` can inject additional instrumentation into legacy files.
- **Event Store** – `EventStore` appends JSONL rows per session inside `event_store/<tenant>/`. Use these for auditing, debugging, or replay.
- **Tracing** – Lightweight context-based spans via `src/infrastructure/tracing.py` (`@trace_async` / `@trace_sync`). Each pipeline stage and NLP operation emits spans with attributes, durations, and error metadata.
- **Circuit Breaker** – `src/infrastructure/circuit_breaker.py` wraps expensive dependencies (e.g., NLP pipeline) with failure thresholds and recovery logic.

---

## Multi-Tenancy Notes

- `TenantContextFactory` registers `TenantConfig` instances and shared session stores.
- Each request obtains a tenant-specific `TenantContext` (with optional correlation ID).
- The context lazily initialises:
  - Event store/bus
  - E5 embeddings + classifier + searcher (`AsyncNLPPipeline` inside `NLPServiceWithCircuitBreaker`)
  - Fractional query augmenter
  - Summariser
  - Hook manager (auto-registers builtin logging + metrics hooks)
  - Form registry (pre-registers example forms)
  - Session manager (using shared store from factory)
- Integration tests (`tests/integration/test_tenant_isolation.py`) assert isolation across tenants.

---

## Demo & Tooling

- `examples/simulate_conversation.py` – Reads `examples/input.txt`, runs queries through NLP service, logs results to `examples/output.txt`.
- `examples/comprehensive_test_inputs.txt` – Canonical manual test plan covering FAQs, forms, NER, interruptions, policy routes, and edge cases.
- `examples/credit_card_service_demo.py` – Illustrates building a new form suite (credit card verification + follow-up forms) atop existing BaseForm implementation.

---

## Next Steps & Ideas

- Swap `PassthroughSummarizer` for real summarisation (OpenAI, HuggingFace, etc.).
- Wire `FormInterruptionHandler` fully (prompt user to continue form vs answer FAQ).
- Persist event store to durable storage (PostgreSQL/EventStoreDB) and add replay tooling.
- Integrate metrics exporter (Prometheus/OpenTelemetry) for spans and hook metadata.
- Package CLI utilities (dataset prep, training) with `typer` for consistent interfaces.

---

## Additional References

- **Architecture** – `docs/SYSTEM_ARCHITECTURE.txt`
- **Classification Component** – `docs/CLASSIFICATION_COMPONENT.md`
- **Log Samples** – `logs/dialogue_system.log`, `logs/app_YYYY-MM-DD.log`
- **Event Samples** – `event_store/default/default/*.jsonl`
- **Validation Metrics** – `models/classification/validation_metrics.json`
- **Config Schema (Pydantic)** – `src/application/config/schemas.py`

---
