# QAPFlow TODO Board (Initial 50 Tasks)

1) Backend wiring
1. Add support for explicit A/B matrices in QAPFlow problem schema (done in adapter; extend docs).
2. Map fit/interaction reliably to (A,B); document assumptions, add schema examples.
3. Parse parameters for mode, time_limit, seed in backend and pass through to solvers.
4. Wire `fft` mode via solvers.fft_accelerated_solver with normalized outputs.
5. Wire `enhanced` mode via solvers.enhanced_solver with normalized outputs.
6. Wire `nesterov` mode via solvers.nesterov_accelerated_solver with normalized outputs.
7. Wire `instance_adaptive` via solvers.instance_adaptive_optimizer (adaptive_solver_with_params) with params mapping.
8. Validate backend solution against qapflow_solution.json before returning.

2) Modular repo integration
9. Add `solvers/mezan_adapter.py` in modular repo with `solve(A,B,max_time,mode)` uniform signature (external repo task).
10. Provide capabilities.json in modular repo listing supported modes and their defaults.
11. Package modular repo with pyproject for pip install; document `pip install -e .`.
12. Add smoke test callable from MEZAN to verify importability.
13. Align NumPy/Scipy versions and pin in requirements.

3) Methods & pipelines
14. Add FFT-Laplace preconditioning toggle in hybrid mode; ablation in 10 instances.
15. Implement reverse-time homotopy escape (from UNIFIED) as optional stage.
16. Add robust min–max penalty injection; expose `robust_eps` parameter.
17. Parameterize k-Opt swaps; compare to 2-Opt baseline.
18. Formalize size-adaptive parameter curves; serialize to JSON and load dynamically.
19. Add lower bounds/certificates where available; expose in metadata.

4) Benchmarks & validation
20. Add `--bench qaplib` results CSV/JSON aggregator with plots.
21. Compare modes across size regimes; produce a summary table.
22. Robustness tests with noise injected into A/B; report gap deltas.
23. Deterministic seeds per run logged in results.
24. OptiBench integration: checksum + replay script for QAPFlow runs.

5) Orchestrator & ORCHEX
25. Expose `mode` and `time_limit` via orchestrator query params or payload fields.
26. Provide `/qapflow/solve_ab` endpoint accepting A/B directly (optional convenience).
27. Extend LibriaRouter to accept real cost matrices or agent-task costs mapped to A/B.
28. Add end-to-end demo with real QAPLIB instance using HTTP orchestrator.
29. Add health endpoint with mode availability and modular repo path status.

6) Schema & docs
30. Extend problem schema to include `mode`, `time_limit`, `robust` fields (optional, documented).
31. Extend solution schema with `solve_time`, `bound`, and solver telemetry fields.
32. Write QAPFlow usage guide with A/B and fit/interaction examples (added docs/QAPFLOW_INTEGRATION_GUIDE.md).
33. Document mapping from assignment cost to QAP form with caveats.
34. Document each modular mode and its expected strengths.

7) Performance & stability
35. Profile modes under different BLAS backends; document impact.
36. Enforce timeouts and graceful fallback returning best incumbent.
37. Memory guards for large instances; advise on hardware requirements.
38. Thread/process pool toggles to avoid oversubscription.

8) Research consolidation
39. Curate UNIFIED + modular methods into MEZAN research doc; cross-reference code paths.
40. Design ablation toggles and produce controlled experiment report.
41. Draft QAPFlow paper outline (methods, complexity, empirical).

9) QA & CI
42. Add schema validation tests for each mode.
43. Add nightly mini-bench (3 instances) to detect regressions.
44. CI link checker and docs build consistency.

10) Packaging & release
45. Provide editable install guide and wheel packaging for modular repo.
46. Add Makefile target `make qap-bench` to run standard suites.
47. Tag QAPFlow versions as modes are integrated; changelog entries.
48. Provide Dockerfile(s) with pinned dependency stacks.
49. Publish initial technical report and usage blog.
50. Prepare short demo videos and a README quickstart for stakeholders.

Next Steps (Backlog additions)
51. Backend: validate solution dict against `qapflow_solution.json` before returning from `Librex.QAP_backend.solve()`.
52. CLI bench: support comma-separated `--modes` to sweep multiple modes in one run; emit comparative CSV and summary table.
53. Orchestrator: passthrough `mode` and `time_limit` fields from request payload to backend; document in schema and guide.
54. Orchestrator: optionally accept direct A/B payload on `/qapflow/solve` (or create `/qapflow/solve_ab`); update docs.
55. Makefile: add `qap-bench`, `orchestrator-up`, and `orchestrator-test` targets.
56. GitHub workflow: add nightly mini-bench (3 instances) with drift detection and artifacts upload; gate on label.
57. Error UX: add clear messages when `QAP_MODULAR_REPO_PATH` is unset/invalid; suggest fix and show detected default.
58. Docs: extend integration guide with Windows vs WSL notes, path pitfalls, and environment examples.
59. Issue templates: add GitHub issue templates for QAPFlow (bug/feature/bench request) with auto-labels.
60. Roadmap linkage: generate a GitHub issues CSV from this TODO and the 500-task plan when ready to open issues.
