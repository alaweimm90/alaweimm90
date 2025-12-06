# Improvement Backlog (100 Items)

Scope: grounded in current repo (ORCHEX, Libria, docs, CI). Items are actionable and non-destructive unless noted.

Code Quality & Style (12)
1. Add `ruff` config for linting across Python projects.
2. Add `black` formatting config and CI check.
3. Introduce `mypy` with gradual typing in Libria code.
4. Add `pyproject.toml` tools section consolidating dev tooling.
5. Enforce `isort` (or ruff import sorting) in CI.
6. Add pre-commit hooks mirroring CI checks.
7. Adopt consistent logging via `logging` module in Libria.
8. Replace print statements in benchmarks with logger and verbosity flags.
9. Add type hints to baselines (`autofolio.py`, `satzilla.py`).
10. Add docstrings to public classes/functions in `benchmark/` and `libria_meta/`.
11. Normalize NumPy/Pandas imports and dtype usage (float32 policy).
12. Create `coding_guidelines.md` and link from CONTRIBUTING.

Testing (12)
13. Add unit tests for `get_training_arrays` shapes and NaN handling.
14. Add integration tests for SAT11-HAND minimal path through baselines.
15. Add tests for `Librex.Meta.fit/select/update` happy paths with mock solvers.
16. Add tests for AutoFolio pre-solving rule creation and selection fallback.
17. Add tests covering CV parsing with multiple folds and repetitions.
18. Add property-based tests for ARFF parsing with random quoted fields.
19. Add regression test for PAR10 mapping across statuses.
20. Add benchmark smoke test (runs < 5s) gated by env var.
21. Add tests for feature extractor defaults and graph features.
22. Add tests for numeric inference thresholds (50%, 90%).
23. Add tests ensuring category dtype doesn't break pivoting.
24. Add snapshot test for `get_summary()` on sample scenarios.

Performance (10)
25. Parallelize feature normalization with NumPy vector ops.
26. Cache algorithm list and instance list inside loader (invalidate on reload).
27. Add optional memory-map for large ARFF via chunked parsing fallback.
28. Add fast path for `get_performance` using groupby dict cache.
29. Expose `--arrays/--dicts` switch in evaluation scripts.
30. Add `--batch-size` for chunked training-data extraction.
31. Benchmark multiple scenarios in one run and summarize.
32. Add simple progress indicator for large extractions.
33. Use `observed=True` in pivot to silence future warning and speed up.
34. Profile end-to-end selection latency on 1k/5k instances.

Data Handling (10)
35. Implement ARFF relation name capture (optional metadata).
36. Validate required columns (`instance_id`, `algorithm`) per file type.
37. Add schema check for `cv.arff` variants and helpful errors.
38. Support `feature_costs.arff` optional integration in summaries.
39. Provide loader method to list available scenario files.
40. Add CLI to validate a scenario folder and print summary.
41. Add CSV export for features and runtimes (arrays path).
42. Add parquet export option for performance table.
43. Detect and warn about duplicate (instance, algorithm) entries.
44. Add deterministic sort to algorithm names and instance ids.

Architecture (10)
45. Extract evaluation pipeline utilities into `libria_meta/eval_utils.py`.
46. Add interface/protocol for solver wrappers (name, solve, metadata).
47. Introduce config dataclasses for baselines and Librex.Meta.
48. Separate CV split logic into `cv_utils.py` with strategies.
49. Add registry for baselines to simplify CLI selection.
50. Provide a high-level `Scenario` facade object encapsulating loader.
51. Add `libria_meta/cli.py` with subcommands (summary, train, evaluate).
52. Introduce versioning constant for Libria Meta package.
53. Add graceful feature normalization (scaler persistence IO).
54. Make FeatureExtractor pluggable via protocol in Librex.Meta.

Documentation (10)
55. Add docs/README.md (done) and link from root README (done).
56. Create Libria quick start focusing on aslib scenarios.
57. Document ARFF edge cases supported by parser.
58. Write guide on using vectorized `get_training_arrays`.
59. Add benchmarking guide with typical flags (`--arrays`, `--all`).
60. Add architecture overview diagram for Libria Meta.
61. Add ORCHEX↔Libria integration notes in `docs/`.
62. Consolidate legacy root docs into topic pages with summaries.
63. Document repo hygiene rules (venv, caches) in CONTRIBUTING.
64. Add docs for dataset provenance and licenses.

CI/CD (9)
65. Add Python matrix CI for Libria tests (3.9–3.12).
66. Add lint/mypy jobs with annotations.
67. Cache pip dependencies in CI to speed up.
68. Add scheduled job to run benchmarks on small slices.
69. Upload benchmark summaries as job artifacts.
70. Enforce `.gitignore` violations via CI (existing hygiene extended).
71. Add link checker for docs using `lychee`.
72. Publish code coverage to badge (optional, private).
73. Gate slow tests behind marker; run only on nightly.

Security & Reliability (7)
74. Add `safety`/`pip-audit` to check dependency vulnerabilities.
75. Pin dependencies with upper bounds in `requirements.txt`.
76. Add retry wrapper for file IO with clear errors.
77. Validate inputs to selection methods with type checks.
78. Add timeouts to benchmark runs to avoid hangs.
79. Sanitize printed paths in logs.
80. Add resource usage notes (RAM/CPU) in docs.

Developer Experience (10)
81. Make a `make test`, `make lint`, `make bench` at root.
82. Add VS Code settings recommending extensions.
83. Provide sample notebooks reading `get_training_arrays` output.
84. Add example script for training SATzilla/AutoFolio quickly.
85. Add minimal Dockerfile for Libria evaluation only.
86. Add `tox`/`nox` config for local multi-env testing.
87. Add error codes reference for loader exceptions.
88. Add logging formatter and levels via env var.
89. Provide dataset size/time table in docs.
90. Add badges in README referencing CI and tests.

ORCHEX-side (10)
91. Remove lingering committed artifacts (done for venv). Add guard in CI.
92. Ensure ORCHEX unit tests run in CI hygiene job.
93. Add README if missing under `ORCHEX/` with quick start.
94. Document engine/agent architecture in ORCHEX docs.
95. Add example showing Libria selection driving ORCHEX solver choice.
96. Standardize logging configuration between ORCHEX and Libria.
97. Add type hints to key ORCHEX modules (agent, engine).
98. Add `ruff`/`black` to ORCHEX CI as well.
99. Add lightweight perf test for ORCHEX pipelines.
100. Add ORCHEX docs link from root README.

Priorities (suggested first tranche)
- 65, 66, 67: CI lint/mypy/matrix
- 13, 15, 16: Core tests for selection
- 58, 59: Vectorized arrays docs and bench guide
- 28, 33: Loader performance cache and pivot observed flag
- 1, 2, 6: Lint/format/pre-commit

