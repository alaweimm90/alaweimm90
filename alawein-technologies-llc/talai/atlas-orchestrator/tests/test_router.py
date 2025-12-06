import sys
import types


class _DummyEncoding:
    def encode(self, text: str):
        return text.encode("utf-8")


def _fake_encoding_for_model(_: str):
    return _DummyEncoding()


sys.modules.setdefault("tiktoken", types.SimpleNamespace(encoding_for_model=_fake_encoding_for_model))

from atlas_orchestrator.core.config import OrchestratorConfig
from atlas_orchestrator.core.router import Router, RoutingStrategy
from atlas_orchestrator.core.task import Task, TaskType


def test_primary_model_leads_fallback_chain():
    config = OrchestratorConfig()
    router = Router(config)
    task = Task(prompt="hello world", task_type=TaskType.CODE)

    chain = router.get_fallback_chain(task, primary_model="claude-opus")

    assert chain[0] == "claude-opus"
    assert chain.count("claude-opus") == 1


def test_primary_inferred_when_not_provided():
    config = OrchestratorConfig()
    router = Router(config, strategy=RoutingStrategy.COST_OPTIMIZED)
    task = Task(prompt="optimize me", task_type=TaskType.CODE)

    primary = router.select_model(task)
    chain = router.get_fallback_chain(task)

    assert chain[0] == primary
