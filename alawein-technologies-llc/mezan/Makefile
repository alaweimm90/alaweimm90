.PHONY: help setup lint type test orchestrator bench-html tree

help:
	@echo "Targets:\n  setup (pre-commit)\n  lint (ruff)\n  type (mypy)\n  test (pytest)\n  orchestrator (run HTTP)\n  bench-html (QAPLIB bench to HTML)\n  tree (generate repo trees)"

setup:
	@pre-commit install || true

lint:
	@ruff check Libria/libria-meta || true

type:
	@mypy Libria/libria-meta --config-file Libria/libria-meta/mypy.ini || true

test:
	@pytest -q Libria/libria-meta/tests || true

orchestrator:
	@python -m libria_meta.services.mezan_orchestrator

bench-html:
	@$(MAKE) -C Libria/libria-meta bench-html

tree:
	@python scripts/gen_tree_md.py

