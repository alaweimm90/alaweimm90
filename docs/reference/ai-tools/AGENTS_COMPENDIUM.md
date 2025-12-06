# Agents & Orchestration Compendium
_A production-minded handbook covering agents, DAG systems, RAG, orchestration, workflows, integration patterns, and an industry landscape — contextualized for a computational physics + ML research portfolio._

## 0. Objectives
- Define an **adapter-agnostic** agent stack that fits a three-layer governance model (`.metaHub → alawein → organizations`).
- Provide **concrete code** (typed, tested-ready, vendor-neutral) for DAG execution, RAG, and multi-agent coordination.
- Document **integration patterns**, **communication contracts**, **dataflow**, **governance**, and **benchmarks**.
- Summarize **industry patterns** (OpenAI, Anthropic, Google, Microsoft, OSS) without vendor lock-in.

---

## 1. Foundations: Agents & Roles
**Agent**: a program with goals, tools, and policies. Common roles:
- **Planner**: decomposes goals into tasks; emits a plan/DAG.
- **Executor**: runs tasks (LLM calls, tools, HPC jobs).
- **Critic**: evaluates outputs; initiates repair or escalation.
- **Router**: chooses models/tools based on budget/latency/quality priors.
- **Memory**: manages retrieval, state, and long-horizon context.
- **Supervisor**: closes the loop; owns success criteria & guardrails.

**Key constraints for production**:
- **Typed I/O** (JSON Schema/Pydantic); deterministic serialization.
- **Idempotency keys** for retries; **timeouts** + **circuit breakers**.
- **Observability**: traces, spans, metrics (OpenTelemetry-friendly).
- **Policy gates**: budget/safety/PII; OPA-as-code where applicable.

---

## 2. Architectures: DAG vs Event-Driven
- **Deterministic DAG** (Airflow/Dagster/Prefect style): repeatable ETL, eval runs, indexing. Great for batch.
- **Dynamic graphs / FSM** (LangGraph-like): conversational state machines, conditional branches.
- **Event-driven (pub/sub)**: Kafka/NATS/Redis Streams; decoupled agents, streaming updates.
- **Blackboard**: shared state space that agents read/write; supervisor arbitrates conflicts.

**Pattern selection**:
- **Batch**: index builds, offline eval → **DAG**.
- **Interactive**: chat tools, tool-use chains → **FSM/Dynamic graph**.
- **HPC**: job submission, polling, artifact collation → **DAG + event hooks**.

---

## 3. Communication & Dataflow
- **Protocols**: HTTP+JSON, gRPC (typed), NATS/Kafka for async fanout.
- **Schemas**: JSON Schema / Pydantic v2; version all payloads.
- **Contracts**: Request{goal, inputs, policy}, Response{artifacts, metrics, evidence}.
- **State**: short-term in Redis; long-term in Postgres; embeddings in pgvector (default) or Milvus/Weaviate.

**Idempotency**: `request_hash = sha256(sorted_json(request))`.
**Retry**: exponential backoff with jitter; max attempts configurable.

---

## 4. Retrieval-Augmented Generation (RAG)
**Pipeline**: _query → rewrite → retrieve → rerank → synthesize → cite_.
- **Chunking**: fixed (512–1000 tokens) + overlap; or structure-aware (headings).
- **Embeddings**: sentence-level for recall; **cross-encoder** rerank for precision.
- **Stores**: default to **pgvector** (Postgres), scale to Milvus/Weaviate/Pinecone.
- **Agentic RAG**: planner chooses retrievers (docs, code, specs); critic verifies citations.
- **Graph/Topology**: augment with entity/relationship store for multi-hop queries.

**Metrics**: Recall@k, nDCG, MRR (retrieval); citation precision; exact-match / Jaccard (gen).

---

## 5. Benchmarks & Cost/Latency Models
- **Latency**: `T ≈ T_pre + T_retr + T_rerank + T_gen + T_post`. Minimize serialized waits; parallelize retrievers.
- **Cost**: `C ≈ (#calls × price_per_token × tokens) + infra`. Cache embedding & rerank results; batch vectors.
- **Throughput**: `QPS ≈ min(worker_capacity, model_rate_limit)`. Use async I/O and backpressure.

---

## 6. Orchestration Methodologies
- **Policy-first**: budget/time/accuracy thresholds decide routes (router agent).
- **Self-refutation**: propose → critique → repair; archive failures (Hall-of-Failures).
- **Tournaments**: multiple strategies compete; choose Pareto-front or ELO winner.
- **Safety**: sensitive-tool allowlists, redaction pre-processors, refusal paths.

---

## 7. Industry Landscape (patterns, not endorsements)
- **OpenAI**: function/tool calling, JSON mode, Assistants-like session tools; strong structured outputs and parallel tool calls.
- **Anthropic**: tool use + safety guardrails; emphasis on instruction adherence and constitutional constraints.
- **Google**: Vertex AI Agents, grounding with enterprise search; connectors to data sources; model routing.
- **Microsoft**: **Semantic Kernel**, **AutoGen** (multi-agent patterns); Azure OpenAI operational hardening.
- **OSS & startups**: LangChain/LangGraph (graph orchestration), LlamaIndex (RAG graph), Haystack, CrewAI (roles), Guardrails.ai (validation), Langfuse/Phoenix (observability).

Themes: **typed tools, reusable skills, graph-based control flow, eval-first releases, tracing, and budget-aware routing**.

---

## 8. Recommended Stack (contextualized)
- **Storage**: Postgres + pgvector (default), S3/MinIO for blobs, Redis for cache/locks.
- **Orchestration**: Prefect/Dagster (batch), a light **graph FSM** for interactive agents (see `code/mini_dag.py`).
- **Queue**: Redis Streams → Kafka/NATS when needed.
- **Services**: FastAPI microservices; one service per domain tool (DFT submitter, LAMMPS submitter, parser).
- **Observability**: OpenTelemetry → Prometheus/Grafana; app logs in JSONL; traces per turn.
- **Governance**: OPA policies (already present) for repo/project invariants; extend to runtime “tool-use allowlist”.
- **Testing**: golden tests for agents, retrieval eval sets, metamorphic tests for prompt changes.
- **Deployment**: docker-compose dev; K8s or SLURM-integrated runners for HPC workloads.

This aligns with **vendor neutrality**, HPC integration, and cost control.

---

## 9. Integration Patterns (with code)
- **Planner→Executor→Critic loop** with typed messages (see `code/schemas.py` + `code/mini_dag.py`).  
- **RAG Agent** as a tool of the planner (see `code/rag_pipeline.py`).  
- **HPC job tool**: submit/poll/collect pattern (stubbed in `code/mini_dag.py` task example).

---

## 10. Go‑Live Checklists (agent services)
- Typed contracts & versioning present.
- Timeouts, retries, idempotency keys implemented.
- Tool allowlists and redaction filters enabled.
- Traces & metrics exported; dashboards green.
- Golden tests + retrieval eval pass thresholds.
- Disaster switches: degrade to simpler route.
- Budget & latency SLOs enforced by router.

---

## 11. File Index (this pack)
- `docs/AGENTS_COMPENDIUM.md` (this file)
- `code/schemas.py` (typed message contracts)
- `code/mini_dag.py` (deterministic DAG + retries/timeout)
- `code/rag_pipeline.py` (adapter-agnostic RAG engine)
- `code/benchmarks_harness.py` (lightweight eval runner)
- `code/README.md` (how to run & extend)

