PYTHONPATH_BASE=src:../turing-features/self-refutation/src:../turing-features/interrogation/src:../turing-features/hall-of-failures/src:../turing-features/meta-learning/src

.PHONY: smoke test-full orchestrator-tests

smoke:
	@echo "Running ATLAS smoke tests (diagnostics + hypothesis generator)…"
	ATLAS_PREFLIGHT=1 PYTHONPATH=$(PYTHONPATH_BASE) pytest tests/test_diagnostics.py tests/test_hypothesis_generator.py -q

test-full:
	@echo "Running full ATLAS test suite…"
	PYTHONPATH=$(PYTHONPATH_BASE) pytest tests -q

orchestrator-tests:
	@echo "Running orchestrator routing tests…"
	PYTHONPATH=../atlas-orchestrator/src pytest ../atlas-orchestrator/tests/test_router.py -q
