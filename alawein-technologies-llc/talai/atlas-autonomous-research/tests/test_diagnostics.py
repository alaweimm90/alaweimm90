import pytest

from ORCHEX import diagnostics


def test_run_diagnostics_reports_missing(monkeypatch):
    available = {"self_refutation": False, "interrogation": True, "hall_of_failures": True, "atlas_orchestrator": True, "meta_learning": False}

    def fake_find_spec(name: str):
        return object() if available.get(name, True) else None

    monkeypatch.setattr(diagnostics.importlib.util, "find_spec", fake_find_spec)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "abc")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    result = diagnostics.run_diagnostics()
    assert result["healthy"] is False

    critical = {check["module"]: check for check in result["checks"] if check["critical"]}
    assert critical["self_refutation"]["ok"] is False
    assert critical["interrogation"]["ok"] is True

    api = {entry["variable"]: entry for entry in result["api_keys"]}
    assert api["ANTHROPIC_API_KEY"]["ok"] is True
    assert api["OPENAI_API_KEY"]["ok"] is False


def test_summarize_diagnostics_contains_hints(monkeypatch):
    fake_result = {
        "healthy": False,
        "checks": [
            {"name": "Self-Refutation Protocol", "module": "self_refutation", "critical": True, "ok": False, "hint": "Install"},
            {"name": "Meta-Learning Agents", "module": "meta_learning", "critical": False, "ok": False, "hint": "Optional"},
        ],
        "api_keys": [{"variable": "OPENAI_API_KEY", "provider": "OpenAI", "ok": False, "hint": "Set it"}],
    }

    summary = diagnostics.summarize_diagnostics(fake_result)
    assert "Self-Refutation Protocol" in summary
    assert "OPENAI_API_KEY" in summary
    assert "FAIL" in summary
