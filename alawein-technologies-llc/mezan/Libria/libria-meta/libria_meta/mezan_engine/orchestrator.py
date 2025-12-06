from __future__ import annotations

from typing import Any, Dict, Optional

from .c2sc import C2SCCompiler
from .adapters.libria_adapter import LibriaAdapter


class MezanOrchestrator:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.c2sc = C2SCCompiler()
        self.adapter = LibriaAdapter()

    def solve_composite(self, composite: Dict[str, Any]) -> Dict[str, Any]:
        assignment_pb, workflow_pb, allocation_pb = self.c2sc.compile(composite)
        result: Dict[str, Any] = {"metadata": {}}
        if assignment_pb:
            params = assignment_pb.get('params') or {}
            # Prefer passing A/B directly or fit/interaction mapping as-is
            res = self.adapter.solve_qapflow(assignment_pb, params)
            result['assignment'] = res
        if workflow_pb:
            params = workflow_pb.get('params') or {}
            res = self.adapter.solve_workflow(workflow_pb, params)
            result['workflow'] = res
        if allocation_pb:
            params = allocation_pb.get('params') or {}
            res = self.adapter.solve_allocflow(allocation_pb, params)
            result['allocation'] = res
        # Optional additional flows
        graph_pb = composite.get('graph') or {}
        if graph_pb:
            params = graph_pb.get('params') or {}
            result['graph'] = self.adapter.solve_graphflow(graph_pb, params)
        meta_pb = composite.get('meta') or {}
        if meta_pb:
            params = meta_pb.get('params') or {}
            result['meta'] = self.adapter.solve_metaflow(meta_pb, params)
        dual_pb = composite.get('dual') or {}
        if dual_pb:
            params = dual_pb.get('params') or {}
            result['dual'] = self.adapter.solve_dualflow(dual_pb, params)
        evo_pb = composite.get('evo') or {}
        if evo_pb:
            params = evo_pb.get('params') or {}
            result['evo'] = self.adapter.solve_evoflow(evo_pb, params)
        return result
