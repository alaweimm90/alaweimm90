import pytest

from ORCHEX.hypothesis_generator import HypothesisGenerator


@pytest.mark.asyncio
async def test_literature_search_is_cached(monkeypatch):
    generator = HypothesisGenerator()
    calls = {"count": 0}

    def fake_fetch(query: str, limit: int):
        calls["count"] += 1
        return [{"title": "test", "abstract": "example"}]

    async def immediate_to_thread(func, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr(generator, "_fetch_literature", fake_fetch)
    monkeypatch.setattr("ORCHEX.hypothesis_generator.asyncio.to_thread", immediate_to_thread)

    first = await generator._search_literature("quantum", limit=5)
    second = await generator._search_literature("quantum", limit=5)

    assert first == second
    assert calls["count"] == 1
