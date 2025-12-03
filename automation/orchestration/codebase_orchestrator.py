#!/usr/bin/env python3
"""
Codebase Orchestrator - Multi-Agent Analysis, Critique, and Validation System

Implements sequential thinking with internal agent orchestration for:
- Deep codebase analysis
- Critical evaluation and issue detection
- Simplification and refactoring recommendations
- Documentation validation and correction
- Cross-validation and consistency checking
- CODEMAP-based structural validation
"""

import asyncio
import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

import yaml


class ThinkingPhase(Enum):
    """Sequential thinking phases for deep analysis."""
    OBSERVE = "observe"      # Gather information
    ANALYZE = "analyze"      # Deep analysis
    CRITIQUE = "critique"    # Find issues
    SIMPLIFY = "simplify"    # Propose simplifications
    VALIDATE = "validate"    # Cross-validate
    SYNTHESIZE = "synthesize"  # Final synthesis


class AgentRole(Enum):
    """Specialized agent roles for codebase orchestration."""
    ANALYZER = "analyzer"
    CRITIC = "critic"
    SIMPLIFIER = "simplifier"
    DOC_VALIDATOR = "doc_validator"
    CROSS_VALIDATOR = "cross_validator"
    CODEMAP_VALIDATOR = "codemap_validator"
    ORCHESTRATOR = "orchestrator"


@dataclass
class Finding:
    """Represents a single finding from an agent."""
    severity: str  # critical, high, medium, low, info
    category: str  # code, docs, structure, consistency, security
    title: str
    description: str
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    suggestion: Optional[str] = None
    agent: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class AgentResult:
    """Result from a single agent execution."""
    agent: AgentRole
    phase: ThinkingPhase
    findings: list[Finding] = field(default_factory=list)
    summary: str = ""
    confidence: float = 0.0
    duration_ms: int = 0
    metadata: dict = field(default_factory=dict)


@dataclass
class ThinkingContext:
    """Shared context across sequential thinking phases."""
    root_path: Path
    codemap: dict = field(default_factory=dict)
    phase_results: dict[ThinkingPhase, AgentResult] = field(default_factory=dict)
    all_findings: list[Finding] = field(default_factory=list)
    file_index: dict[str, dict] = field(default_factory=dict)
    doc_index: dict[str, dict] = field(default_factory=dict)
    consistency_map: dict[str, list] = field(default_factory=dict)


class CodebaseOrchestrator:
    """
    Multi-agent orchestrator for comprehensive codebase analysis.

    Implements sequential thinking pattern:
    1. OBSERVE: Gather codebase information
    2. ANALYZE: Deep analysis of code quality, structure, patterns
    3. CRITIQUE: Critical evaluation, find issues
    4. SIMPLIFY: Identify simplification opportunities
    5. VALIDATE: Cross-validate findings, check consistency
    6. SYNTHESIZE: Generate final report with prioritized actions
    """

    def __init__(self, root_path: str | Path):
        self.root_path = Path(root_path)
        self.context = ThinkingContext(root_path=self.root_path)
        self.agents: dict[AgentRole, Callable] = {
            AgentRole.ANALYZER: self._run_analyzer,
            AgentRole.CRITIC: self._run_critic,
            AgentRole.SIMPLIFIER: self._run_simplifier,
            AgentRole.DOC_VALIDATOR: self._run_doc_validator,
            AgentRole.CROSS_VALIDATOR: self._run_cross_validator,
            AgentRole.CODEMAP_VALIDATOR: self._run_codemap_validator,
        }

    async def orchestrate(self) -> dict[str, Any]:
        """
        Run full orchestration with sequential thinking.

        Returns comprehensive analysis report.
        """
        print("=" * 70)
        print("[BRAIN] CODEBASE ORCHESTRATOR - Sequential Thinking Analysis")
        print("=" * 70)

        start_time = datetime.now()

        # Phase 1: OBSERVE - Gather information
        print(f"\n[1/6] OBSERVE - Gathering codebase information...")
        await self._phase_observe()

        # Phase 2: ANALYZE - Deep analysis
        print(f"\n[2/6] ANALYZE - Deep codebase analysis...")
        await self._phase_analyze()

        # Phase 3: CRITIQUE - Critical evaluation
        print(f"\n[3/6] CRITIQUE - Critical evaluation...")
        await self._phase_critique()

        # Phase 4: SIMPLIFY - Find simplification opportunities
        print(f"\n[4/6] SIMPLIFY - Finding simplification opportunities...")
        await self._phase_simplify()

        # Phase 5: VALIDATE - Cross-validation
        print(f"\n[5/6] VALIDATE - Cross-validating findings...")
        await self._phase_validate()

        # Phase 6: SYNTHESIZE - Final synthesis
        print(f"\n[6/6] SYNTHESIZE - Generating final report...")
        report = await self._phase_synthesize()

        duration = (datetime.now() - start_time).total_seconds()
        report["duration_seconds"] = duration

        print(f"\n{'=' * 70}")
        print(f"[DONE] Analysis complete in {duration:.2f}s")
        print(f"   Findings: {len(self.context.all_findings)}")
        print(f"   Critical: {sum(1 for f in self.context.all_findings if f.severity == 'critical')}")
        print(f"   High: {sum(1 for f in self.context.all_findings if f.severity == 'high')}")
        print(f"{'=' * 70}")

        return report

    # =========================================================================
    # THINKING PHASES
    # =========================================================================

    async def _phase_observe(self) -> None:
        """Phase 1: Gather codebase information."""
        # Load CODEMAP
        codemap_path = self.root_path / "docs" / "CODEMAP.md"
        if codemap_path.exists():
            self.context.codemap = self._parse_codemap(codemap_path)
            print(f"   [+] Loaded CODEMAP")

        # Index files
        self.context.file_index = await self._index_files()
        print(f"   [+] Indexed {len(self.context.file_index)} files")

        # Index documentation
        self.context.doc_index = await self._index_documentation()
        print(f"   [+] Indexed {len(self.context.doc_index)} doc files")

        # Load policies
        policies = await self._load_policies()
        self.context.consistency_map["policies"] = policies
        print(f"   [+] Loaded {len(policies)} policies")

    async def _phase_analyze(self) -> None:
        """Phase 2: Deep analysis with analyzer agent."""
        result = await self.agents[AgentRole.ANALYZER](ThinkingPhase.ANALYZE)
        self.context.phase_results[ThinkingPhase.ANALYZE] = result
        self.context.all_findings.extend(result.findings)
        print(f"   [+] Analysis complete: {len(result.findings)} findings")

    async def _phase_critique(self) -> None:
        """Phase 3: Critical evaluation with critic agent."""
        result = await self.agents[AgentRole.CRITIC](ThinkingPhase.CRITIQUE)
        self.context.phase_results[ThinkingPhase.CRITIQUE] = result
        self.context.all_findings.extend(result.findings)
        print(f"   [+] Critique complete: {len(result.findings)} issues found")

    async def _phase_simplify(self) -> None:
        """Phase 4: Find simplification opportunities."""
        result = await self.agents[AgentRole.SIMPLIFIER](ThinkingPhase.SIMPLIFY)
        self.context.phase_results[ThinkingPhase.SIMPLIFY] = result
        self.context.all_findings.extend(result.findings)
        print(f"   [+] Simplification analysis: {len(result.findings)} opportunities")

    async def _phase_validate(self) -> None:
        """Phase 5: Cross-validation and consistency checking."""
        # Run doc validator
        doc_result = await self.agents[AgentRole.DOC_VALIDATOR](ThinkingPhase.VALIDATE)
        self.context.all_findings.extend(doc_result.findings)
        print(f"   [+] Doc validation: {len(doc_result.findings)} issues")

        # Run cross validator
        cross_result = await self.agents[AgentRole.CROSS_VALIDATOR](ThinkingPhase.VALIDATE)
        self.context.all_findings.extend(cross_result.findings)
        print(f"   [+] Cross-validation: {len(cross_result.findings)} inconsistencies")

        # Run CODEMAP validator
        codemap_result = await self.agents[AgentRole.CODEMAP_VALIDATOR](ThinkingPhase.VALIDATE)
        self.context.all_findings.extend(codemap_result.findings)
        print(f"   [+] CODEMAP validation: {len(codemap_result.findings)} structure issues")

        self.context.phase_results[ThinkingPhase.VALIDATE] = AgentResult(
            agent=AgentRole.CROSS_VALIDATOR,
            phase=ThinkingPhase.VALIDATE,
            findings=doc_result.findings + cross_result.findings + codemap_result.findings,
            summary="Validation complete",
            confidence=0.85
        )

    async def _phase_synthesize(self) -> dict[str, Any]:
        """Phase 6: Generate final synthesis report."""
        # Deduplicate and prioritize findings
        unique_findings = self._deduplicate_findings(self.context.all_findings)
        prioritized = self._prioritize_findings(unique_findings)

        # Generate action plan
        action_plan = self._generate_action_plan(prioritized)

        # Build report
        report = {
            "timestamp": datetime.now().isoformat(),
            "root_path": str(self.root_path),
            "summary": {
                "total_findings": len(unique_findings),
                "by_severity": self._count_by_severity(unique_findings),
                "by_category": self._count_by_category(unique_findings),
            },
            "findings": [self._finding_to_dict(f) for f in prioritized],
            "action_plan": action_plan,
            "phase_summaries": {
                phase.value: {
                    "findings_count": len(result.findings),
                    "confidence": result.confidence,
                    "summary": result.summary
                }
                for phase, result in self.context.phase_results.items()
            },
            "consistency_score": self._calculate_consistency_score(),
            "recommendations": self._generate_recommendations(prioritized)
        }

        return report

    # =========================================================================
    # AGENT IMPLEMENTATIONS
    # =========================================================================

    async def _run_analyzer(self, phase: ThinkingPhase) -> AgentResult:
        """Analyzer agent: Deep codebase analysis."""
        findings = []

        # Analyze code structure
        for file_path, info in self.context.file_index.items():
            # Check file size
            if info.get("lines", 0) > 500:
                findings.append(Finding(
                    severity="medium",
                    category="code",
                    title="Large file detected",
                    description=f"File has {info['lines']} lines. Consider splitting.",
                    file_path=file_path,
                    suggestion="Break into smaller, focused modules",
                    agent="analyzer"
                ))

            # Check for complexity indicators
            if info.get("type") in ["py", "ts", "js"]:
                content = info.get("content", "")

                # Nested callbacks/promises
                if content.count("callback") > 5 or content.count(".then(") > 5:
                    findings.append(Finding(
                        severity="medium",
                        category="code",
                        title="Callback complexity",
                        description="Multiple nested callbacks detected",
                        file_path=file_path,
                        suggestion="Refactor to async/await pattern",
                        agent="analyzer"
                    ))

                # TODO comments
                todo_count = len(re.findall(r'#\s*TODO|//\s*TODO|/\*\s*TODO', content))
                if todo_count > 3:
                    findings.append(Finding(
                        severity="low",
                        category="code",
                        title="Multiple TODOs",
                        description=f"File has {todo_count} TODO comments",
                        file_path=file_path,
                        suggestion="Address or track TODOs in issue tracker",
                        agent="analyzer"
                    ))

        return AgentResult(
            agent=AgentRole.ANALYZER,
            phase=phase,
            findings=findings,
            summary=f"Analyzed {len(self.context.file_index)} files",
            confidence=0.85
        )

    async def _run_critic(self, phase: ThinkingPhase) -> AgentResult:
        """Critic agent: Find issues and problems."""
        findings = []

        # Check for anti-patterns
        for file_path, info in self.context.file_index.items():
            content = info.get("content", "")

            # God objects/files
            if info.get("type") == "py":
                class_count = len(re.findall(r'^class\s+\w+', content, re.MULTILINE))
                if class_count > 5:
                    findings.append(Finding(
                        severity="high",
                        category="code",
                        title="Too many classes in single file",
                        description=f"File contains {class_count} classes - potential god object",
                        file_path=file_path,
                        suggestion="Split into separate modules by responsibility",
                        agent="critic"
                    ))

            # Hardcoded values
            if re.search(r'(?:localhost|127\.0\.0\.1|:3000|:8080)', content):
                findings.append(Finding(
                    severity="medium",
                    category="code",
                    title="Hardcoded values detected",
                    description="Hardcoded hosts/ports found",
                    file_path=file_path,
                    suggestion="Move to configuration/environment variables",
                    agent="critic"
                ))

            # Empty except blocks
            if re.search(r'except.*:\s*pass', content):
                findings.append(Finding(
                    severity="high",
                    category="code",
                    title="Silent exception handling",
                    description="Empty except blocks swallow errors",
                    file_path=file_path,
                    suggestion="Log errors or handle specifically",
                    agent="critic"
                ))

        # Check structural issues
        if not (self.root_path / "README.md").exists():
            findings.append(Finding(
                severity="critical",
                category="docs",
                title="Missing README",
                description="No README.md at root",
                suggestion="Add comprehensive README",
                agent="critic"
            ))

        return AgentResult(
            agent=AgentRole.CRITIC,
            phase=phase,
            findings=findings,
            summary=f"Found {len(findings)} issues",
            confidence=0.80
        )

    async def _run_simplifier(self, phase: ThinkingPhase) -> AgentResult:
        """Simplifier agent: Find simplification opportunities."""
        findings = []

        # Check for redundant files
        seen_names = {}
        for file_path in self.context.file_index:
            name = Path(file_path).stem
            if name in seen_names:
                findings.append(Finding(
                    severity="low",
                    category="structure",
                    title="Potential duplicate file",
                    description=f"Similar filename to {seen_names[name]}",
                    file_path=file_path,
                    suggestion="Review for consolidation",
                    agent="simplifier"
                ))
            seen_names[name] = file_path

        # Check for over-abstraction
        for file_path, info in self.context.file_index.items():
            content = info.get("content", "")

            # Single-use abstractions
            if info.get("type") in ["py", "ts"]:
                # Very short classes
                classes = re.findall(r'class\s+(\w+)[^{]*{[^}]{1,100}}', content)
                for cls in classes:
                    findings.append(Finding(
                        severity="info",
                        category="code",
                        title="Minimal class",
                        description=f"Class {cls} has minimal implementation",
                        file_path=file_path,
                        suggestion="Consider if abstraction is necessary",
                        agent="simplifier"
                    ))

        # Check directory depth
        for file_path in self.context.file_index:
            rel_path = Path(file_path).relative_to(self.root_path)
            if len(rel_path.parts) > 6:
                findings.append(Finding(
                    severity="medium",
                    category="structure",
                    title="Deep nesting",
                    description=f"File is {len(rel_path.parts)} levels deep",
                    file_path=file_path,
                    suggestion="Flatten directory structure",
                    agent="simplifier"
                ))

        return AgentResult(
            agent=AgentRole.SIMPLIFIER,
            phase=phase,
            findings=findings,
            summary=f"Found {len(findings)} simplification opportunities",
            confidence=0.75
        )

    async def _run_doc_validator(self, phase: ThinkingPhase) -> AgentResult:
        """Documentation validator agent."""
        findings = []

        for doc_path, info in self.context.doc_index.items():
            content = info.get("content", "")

            # Check for broken internal links
            links = re.findall(r'\[([^\]]+)\]\(([^)]+)\)', content)
            for text, href in links:
                if href.startswith(("http://", "https://", "#")):
                    continue
                target = (Path(doc_path).parent / href).resolve()
                if not target.exists():
                    findings.append(Finding(
                        severity="medium",
                        category="docs",
                        title="Broken link",
                        description=f"Link to {href} is broken",
                        file_path=doc_path,
                        suggestion=f"Update or remove link to {href}",
                        agent="doc_validator"
                    ))

            # Check for outdated references
            if "TODO" in content or "FIXME" in content:
                findings.append(Finding(
                    severity="low",
                    category="docs",
                    title="Incomplete documentation",
                    description="Contains TODO/FIXME markers",
                    file_path=doc_path,
                    suggestion="Complete documentation",
                    agent="doc_validator"
                ))

            # Check for stale dates
            dates = re.findall(r'202[0-3]', content)
            if dates and '2025' not in content:
                findings.append(Finding(
                    severity="low",
                    category="docs",
                    title="Potentially outdated",
                    description="Contains older date references",
                    file_path=doc_path,
                    suggestion="Review and update if needed",
                    agent="doc_validator"
                ))

        return AgentResult(
            agent=AgentRole.DOC_VALIDATOR,
            phase=phase,
            findings=findings,
            summary=f"Validated {len(self.context.doc_index)} docs",
            confidence=0.80
        )

    async def _run_cross_validator(self, phase: ThinkingPhase) -> AgentResult:
        """Cross-validator agent: Check consistency across codebase."""
        findings = []

        # Check policy consistency
        policies = self.context.consistency_map.get("policies", [])

        # Validate root structure against policy
        root_structure_policy = next(
            (p for p in policies if "root-structure" in str(p.get("path", ""))),
            None
        )

        if root_structure_policy:
            allowed_dirs = root_structure_policy.get("allowed_directories", {})
            actual_dirs = [
                d.name for d in self.root_path.iterdir()
                if d.is_dir() and not d.name.startswith(".")
            ]

            # Check for unexpected directories
            expected = set()
            for category in allowed_dirs.values():
                if isinstance(category, list):
                    expected.update(item.get("name", "") for item in category)

            for dir_name in actual_dirs:
                if dir_name not in expected and dir_name != "node_modules":
                    findings.append(Finding(
                        severity="medium",
                        category="consistency",
                        title="Unexpected directory",
                        description=f"Directory '{dir_name}' not in policy",
                        file_path=str(self.root_path / dir_name),
                        suggestion="Add to policy or remove",
                        agent="cross_validator"
                    ))

        # Check naming conventions
        naming_patterns = {
            "py": r'^[a-z_]+\.py$',
            "ts": r'^[a-z][a-zA-Z0-9-]*\.ts$',
            "yaml": r'^[a-z][a-z0-9-]*\.ya?ml$'
        }

        for file_path in self.context.file_index:
            name = Path(file_path).name
            ext = Path(file_path).suffix[1:] if Path(file_path).suffix else ""

            if ext in naming_patterns:
                if not re.match(naming_patterns[ext], name):
                    findings.append(Finding(
                        severity="low",
                        category="consistency",
                        title="Naming convention violation",
                        description=f"File '{name}' doesn't match convention",
                        file_path=file_path,
                        suggestion="Rename to follow convention",
                        agent="cross_validator"
                    ))

        return AgentResult(
            agent=AgentRole.CROSS_VALIDATOR,
            phase=phase,
            findings=findings,
            summary=f"Found {len(findings)} consistency issues",
            confidence=0.85
        )

    async def _run_codemap_validator(self, phase: ThinkingPhase) -> AgentResult:
        """CODEMAP validator: Check against documented structure."""
        findings = []

        if not self.context.codemap:
            findings.append(Finding(
                severity="high",
                category="docs",
                title="Missing CODEMAP",
                description="No CODEMAP found for validation",
                suggestion="Generate CODEMAP.md",
                agent="codemap_validator"
            ))
            return AgentResult(
                agent=AgentRole.CODEMAP_VALIDATOR,
                phase=phase,
                findings=findings,
                summary="CODEMAP not available",
                confidence=0.0
            )

        # Verify documented directories exist
        documented_dirs = self.context.codemap.get("directories", [])
        for doc_dir in documented_dirs:
            dir_path = self.root_path / doc_dir
            if not dir_path.exists():
                findings.append(Finding(
                    severity="high",
                    category="consistency",
                    title="CODEMAP mismatch",
                    description=f"Documented directory '{doc_dir}' doesn't exist",
                    suggestion="Update CODEMAP or create directory",
                    agent="codemap_validator"
                ))

        # Check for undocumented directories
        actual_dirs = {
            d.name for d in self.root_path.iterdir()
            if d.is_dir() and not d.name.startswith(".") and d.name != "node_modules"
        }
        documented_set = set(documented_dirs)

        undocumented = actual_dirs - documented_set
        for dir_name in undocumented:
            findings.append(Finding(
                severity="medium",
                category="consistency",
                title="Undocumented directory",
                description=f"Directory '{dir_name}' not in CODEMAP",
                file_path=str(self.root_path / dir_name),
                suggestion="Add to CODEMAP or remove",
                agent="codemap_validator"
            ))

        return AgentResult(
            agent=AgentRole.CODEMAP_VALIDATOR,
            phase=phase,
            findings=findings,
            summary=f"Validated against CODEMAP",
            confidence=0.90
        )

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def _parse_codemap(self, path: Path) -> dict:
        """Parse CODEMAP.md and extract structure."""
        content = path.read_text(encoding="utf-8")

        codemap = {"directories": [], "components": []}

        # Extract directory structure from code blocks
        structure_match = re.search(r'```text\n(.*?)\n```', content, re.DOTALL)
        if structure_match:
            structure = structure_match.group(1)
            # Parse tree structure
            for line in structure.split("\n"):
                # Match directory lines like "├── automation/"
                dir_match = re.search(r'[├└]── (\w+)/', line)
                if dir_match:
                    codemap["directories"].append(dir_match.group(1))

        return codemap

    async def _index_files(self) -> dict[str, dict]:
        """Index all code files in repository."""
        index = {}
        extensions = {".py", ".ts", ".js", ".tsx", ".jsx", ".yaml", ".yml", ".json"}

        for path in self.root_path.rglob("*"):
            if path.is_file() and path.suffix in extensions:
                # Skip node_modules, .git, etc.
                if any(part.startswith((".git", "node_modules", ".archive")) for part in path.parts):
                    continue

                try:
                    content = path.read_text(encoding="utf-8", errors="ignore")
                    index[str(path)] = {
                        "type": path.suffix[1:],
                        "lines": content.count("\n") + 1,
                        "size": path.stat().st_size,
                        "content": content[:10000]  # First 10KB for analysis
                    }
                except (OSError, UnicodeError):
                    pass

        return index

    async def _index_documentation(self) -> dict[str, dict]:
        """Index all documentation files."""
        index = {}

        for path in self.root_path.rglob("*.md"):
            if any(part.startswith((".git", "node_modules", ".archive")) for part in path.parts):
                continue

            try:
                content = path.read_text(encoding="utf-8", errors="ignore")
                index[str(path)] = {
                    "content": content,
                    "lines": content.count("\n") + 1
                }
            except (OSError, UnicodeError):
                pass

        return index

    async def _load_policies(self) -> list[dict]:
        """Load governance policies."""
        policies = []
        policy_dir = self.root_path / ".metaHub" / "policies"

        if policy_dir.exists():
            for policy_file in policy_dir.glob("*.yaml"):
                try:
                    content = yaml.safe_load(policy_file.read_text(encoding="utf-8"))
                    content["path"] = str(policy_file)
                    policies.append(content)
                except (yaml.YAMLError, OSError):
                    pass

        return policies

    def _deduplicate_findings(self, findings: list[Finding]) -> list[Finding]:
        """Remove duplicate findings."""
        seen = set()
        unique = []

        for f in findings:
            key = (f.title, f.file_path, f.category)
            if key not in seen:
                seen.add(key)
                unique.append(f)

        return unique

    def _prioritize_findings(self, findings: list[Finding]) -> list[Finding]:
        """Sort findings by priority."""
        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}
        return sorted(findings, key=lambda f: severity_order.get(f.severity, 5))

    def _count_by_severity(self, findings: list[Finding]) -> dict[str, int]:
        """Count findings by severity."""
        counts = {}
        for f in findings:
            counts[f.severity] = counts.get(f.severity, 0) + 1
        return counts

    def _count_by_category(self, findings: list[Finding]) -> dict[str, int]:
        """Count findings by category."""
        counts = {}
        for f in findings:
            counts[f.category] = counts.get(f.category, 0) + 1
        return counts

    def _finding_to_dict(self, f: Finding) -> dict:
        """Convert finding to dictionary."""
        return {
            "severity": f.severity,
            "category": f.category,
            "title": f.title,
            "description": f.description,
            "file_path": f.file_path,
            "line_number": f.line_number,
            "suggestion": f.suggestion,
            "agent": f.agent
        }

    def _generate_action_plan(self, findings: list[Finding]) -> list[dict]:
        """Generate prioritized action plan from findings."""
        actions = []

        # Group by file
        by_file: dict[str, list[Finding]] = {}
        for f in findings:
            key = f.file_path or "general"
            if key not in by_file:
                by_file[key] = []
            by_file[key].append(f)

        # Generate actions
        for file_path, file_findings in by_file.items():
            if any(f.severity in ("critical", "high") for f in file_findings):
                priority = "immediate"
            elif any(f.severity == "medium" for f in file_findings):
                priority = "soon"
            else:
                priority = "backlog"

            actions.append({
                "file": file_path,
                "priority": priority,
                "findings_count": len(file_findings),
                "top_issues": [f.title for f in file_findings[:3]]
            })

        # Sort by priority
        priority_order = {"immediate": 0, "soon": 1, "backlog": 2}
        actions.sort(key=lambda a: priority_order.get(a["priority"], 3))

        return actions

    def _calculate_consistency_score(self) -> float:
        """Calculate overall consistency score (0-100)."""
        if not self.context.all_findings:
            return 100.0

        # Deduct points based on severity
        deductions = {
            "critical": 10,
            "high": 5,
            "medium": 2,
            "low": 0.5,
            "info": 0.1
        }

        total_deduction = sum(
            deductions.get(f.severity, 0)
            for f in self.context.all_findings
            if f.category == "consistency"
        )

        return max(0, 100 - total_deduction)

    def _generate_recommendations(self, findings: list[Finding]) -> list[str]:
        """Generate high-level recommendations."""
        recommendations = []

        # Check for common patterns
        categories = self._count_by_category(findings)
        severities = self._count_by_severity(findings)

        if severities.get("critical", 0) > 0:
            recommendations.append(
                f"[CRITICAL] Address {severities['critical']} critical issues immediately"
            )

        if categories.get("docs", 0) > 5:
            recommendations.append(
                "[DOCS] Documentation needs significant updates"
            )

        if categories.get("consistency", 0) > 10:
            recommendations.append(
                "[SYNC] Run governance sync to fix consistency issues"
            )

        if categories.get("code", 0) > 20:
            recommendations.append(
                "[REFACTOR] Consider dedicated refactoring sprint"
            )

        if not recommendations:
            recommendations.append("[OK] Codebase is in good shape!")

        return recommendations


async def main():
    """Main entry point."""
    import sys

    root_path = sys.argv[1] if len(sys.argv) > 1 else "."

    orchestrator = CodebaseOrchestrator(root_path)
    report = await orchestrator.orchestrate()

    # Save report
    output_path = Path(root_path) / ".atlas" / "reports" / "orchestration-report.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n[SAVED] Report saved to: {output_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS:")
    for rec in report.get("recommendations", []):
        print(f"  {rec}")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
