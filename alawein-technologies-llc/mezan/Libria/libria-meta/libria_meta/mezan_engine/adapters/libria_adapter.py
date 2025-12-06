from __future__ import annotations

from typing import Any, Dict, Optional

from libria_meta.solvers.workflow import WorkFlow
from libria_meta.solvers.allocflow import AllocFlow
from libria_meta.solvers.graphflow import GraphFlow
from libria_meta.solvers.metaflow import MetaFlow
from libria_meta.solvers.dualflow import DualFlow
from libria_meta.solvers.evoflow import EvoFlow
import Librex.QAP_backend


class LibriaAdapter:
    def __init__(self):
        self._wf = WorkFlow()
        self._alloc = AllocFlow()
        self._graph = GraphFlow()
        self._meta = MetaFlow()
        self._dual = DualFlow()
        self._evo = EvoFlow()

    def solve_qapflow(self, problem: Dict[str, Any], params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return Librex.QAP_backend.solve(problem, parameters=params or {})

    def solve_workflow(self, problem: Dict[str, Any], params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self._wf.solve(problem, parameters=params or {})

    def solve_allocflow(self, problem: Dict[str, Any], params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self._alloc.solve(problem, parameters=params or {})

    def solve_graphflow(self, problem: Dict[str, Any], params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self._graph.solve(problem, parameters=params or {})

    def solve_metaflow(self, problem: Dict[str, Any], params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self._meta.solve(problem, parameters=params or {})

    def solve_dualflow(self, problem: Dict[str, Any], params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self._dual.solve(problem, parameters=params or {})

    def solve_evoflow(self, problem: Dict[str, Any], params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self._evo.solve(problem, parameters=params or {})
