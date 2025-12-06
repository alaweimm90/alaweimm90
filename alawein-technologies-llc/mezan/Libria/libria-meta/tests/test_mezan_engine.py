import sys, os, json
from pathlib import Path

def test_mezan_engine_composite():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from libria_meta.mezan_engine.orchestrator import MezanOrchestrator

    engine = MezanOrchestrator()
    composite = {
        "assignment": {"A": [[1,0,2],[0,1,2],[2,1,0]], "B": [[1,2,1],[2,1,2],[1,2,1]], "params": {"mode": "hybrid"}},
        "workflow": {"stages": ["designer","critic","validator"], "start": "designer", "end": "validator"},
        "allocation": {"budget": 5, "options": [{"id":"x","cost":2,"expected_return":3},{"id":"y","cost":3,"expected_return":2}]}
    }
    res = engine.solve_composite(composite)
    assert "assignment" in res and "workflow" in res and "allocation" in res
    # include optional flows
    composite2 = {
        "graph": {"nodes": ["a","b","c"], "edges": [{"u":"a","v":"b","weight":1.0},{"u":"b","v":"c","weight":2.0}], "budget": 2},
        "meta": {"solvers": ["hybrid","nesterov"], "contexts": [{},{}], "history": {"hybrid":[0.6,0.7]}},
        "dual": {"budget": 3},
        "evo": {"population": [{"candidate":"x","score":1.0},{"candidate":"y","score":0.9}]}
    }
    res2 = engine.solve_composite(composite2)
    assert "graph" in res2 and "meta" in res2 and "dual" in res2 and "evo" in res2
