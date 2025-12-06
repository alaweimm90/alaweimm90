# Orchestrator Specification

API
- Endpoints:
  - POST `/qapflow/solve` (QAP problem JSON): validates, calls backend adapter, returns solution JSON.
  - POST `/workflow/solve`, `/allocflow/solve` similarly.
  - POST `/mezan/solve` for composite problems routed via C2SC.
  - GET `/health` for mode detection and environment path.
- Settings: `ORCH_MAX_BODY` (bytes cap), `ORCH_REQUEST_TIMEOUT` (seconds), `QAPFLOW_SAFE_MODE` (disable external backend).

Behavior
- Strict schema validation; explicit checks for required keys (e.g., require `fit` or `A/B` for QAPFlow).
- Threaded request timeout for QAPFlow to avoid long blocking responses.

Adapter Integration
- Orchestrator uses `LibriaRouter` and the QAP backend adapter (`Librex.QAP_backend`) to support both A/B and fit/interaction.

Error Handling
- Responds with 400 on invalid JSON or schema violations; 413 on oversized payload; 408 on request timeout.

Observability
- Include backend mode and solve_time in solution metadata; `/health` reports available modes.

