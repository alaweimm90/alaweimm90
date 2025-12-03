#!/usr/bin/env python3
"""
Claude Reasoning Service - AI-powered code analysis and refactoring suggestions.

This module provides:
- Background Claude analysis for refactoring opportunities
- Code complexity assessment and optimization recommendations
- Integration with parallel workflow executor
- Asynchronous reasoning with caching
- Multi-file analysis and dependency mapping
"""

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
import hashlib
import re
import ast
import subprocess
import threading

from workflow_types import ParallelTask, TaskResult, TaskStatus


class AnalysisType(Enum):
    """Types of code analysis Claude can perform."""
    REFACTORING_OPPORTUNITIES = "refactoring_opportunities"
    COMPLEXITY_ANALYSIS = "complexity_analysis"
    DEPENDENCY_MAPPING = "dependency_mapping"
    SECURITY_VULNERABILITIES = "security_vulnerabilities"
    PERFORMANCE_OPTIMIZATION = "performance_optimization"
    CODE_SMELLS = "code_smells"
    ARCHITECTURE_REVIEW = "architecture_review"


@dataclass
class CodeFile:
    """Representation of a code file for analysis."""
    path: str
    content: str
    language: str
    size_bytes: int
    lines_count: int
    last_modified: datetime
    hash: str


@dataclass
class AnalysisResult:
    """Result of Claude code analysis."""
    analysis_id: str
    analysis_type: AnalysisType
    files_analyzed: List[str]
    findings: List[Dict[str, Any]]
    recommendations: List[str]
    confidence_score: float
    processing_time_ms: int
    timestamp: datetime


@dataclass
class RefactoringOpportunity:
    """A specific refactoring opportunity identified by Claude."""
    file_path: str
    line_range: tuple
    opportunity_type: str
    description: str
    impact: "high" | "medium" | "low"
    effort: "high" | "medium" | "low"
    code_before: str
    code_after: str
    rationale: str


class ClaudeReasoningService:
    """Service for AI-powered code analysis and reasoning."""

    def __init__(self, cache_ttl_minutes: int = 30):
        self.analysis_cache: Dict[str, AnalysisResult] = {}
        self.cache_ttl = timedelta(minutes=cache_ttl_minutes)
        self.analysis_queue = asyncio.Queue()
        self.processing = False
        self.worker_task: Optional[asyncio.Task] = None
        self.supported_languages = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".jsx": "javascript",
            ".tsx": "typescript",
            ".java": "java",
            ".cpp": "cpp",
            ".c": "c",
            ".cs": "csharp",
            ".go": "go",
            ".rs": "rust"
        }

    async def start(self):
        """Start the Claude reasoning service."""
        self.processing = True
        self.worker_task = asyncio.create_task(self._analysis_worker())

    async def stop(self):
        """Stop the Claude reasoning service."""
        self.processing = False
        if self.worker_task:
            self.worker_task.cancel()
            try:
                await self.worker_task
            except asyncio.CancelledError:
                pass

    async def analyze_code(
        self,
        files: List[str],
        analysis_types: List[AnalysisType],
        force_refresh: bool = False
    ) -> List[AnalysisResult]:
        """
        Analyze code files using Claude reasoning.

        Args:
            files: List of file paths to analyze
            analysis_types: Types of analysis to perform
            force_refresh: Skip cache and force fresh analysis

        Returns:
            List of analysis results
        """
        results = []

        for analysis_type in analysis_types:
            # Check cache first
            cache_key = self._generate_cache_key(files, analysis_type)
            if not force_refresh and cache_key in self.analysis_cache:
                cached_result = self.analysis_cache[cache_key]
                if datetime.now() - cached_result.timestamp < self.cache_ttl:
                    results.append(cached_result)
                    continue

            # Queue for analysis
            analysis_task = {
                "id": str(uuid.uuid4()),
                "files": files,
                "analysis_type": analysis_type,
                "cache_key": cache_key
            }

            await self.analysis_queue.put(analysis_task)

        # Wait for all analyses to complete
        await self._wait_for_queue_empty()

        # Collect results from cache
        for analysis_type in analysis_types:
            cache_key = self._generate_cache_key(files, analysis_type)
            if cache_key in self.analysis_cache:
                results.append(self.analysis_cache[cache_key])

        return results

    async def get_refactoring_opportunities(
        self,
        directory: str,
        max_files: int = 50
    ) -> List[RefactoringOpportunity]:
        """
        Get refactoring opportunities for a directory.

        Args:
            directory: Directory to analyze
            max_files: Maximum number of files to analyze

        Returns:
            List of refactoring opportunities
        """
        # Find code files
        code_files = self._find_code_files(directory, max_files)

        if not code_files:
            return []

        # Perform analysis
        analysis_results = await self.analyze_code(
            [f.path for f in code_files],
            [AnalysisType.REFACTORING_OPPORTUNITIES, AnalysisType.COMPLEXITY_ANALYSIS]
        )

        # Extract refactoring opportunities
        opportunities = []
        for result in analysis_results:
            for finding in result.findings:
                if finding.get("type") == "refactoring_opportunity":
                    opportunity = RefactoringOpportunity(
                        file_path=finding["file_path"],
                        line_range=tuple(finding["line_range"]),
                        opportunity_type=finding["opportunity_type"],
                        description=finding["description"],
                        impact=finding["impact"],
                        effort=finding["effort"],
                        code_before=finding.get("code_before", ""),
                        code_after=finding.get("code_after", ""),
                        rationale=finding.get("rationale", "")
                    )
                    opportunities.append(opportunity)

        # Sort by impact and effort
        opportunities.sort(key=lambda x: (
            {"high": 3, "medium": 2, "low": 1}[x.impact],
            {"low": 3, "medium": 2, "high": 1}[x.effort]
        ), reverse=True)

        return opportunities

    async def analyze_dependencies(self, directory: str) -> Dict[str, List[str]]:
        """
        Analyze dependencies between files in a directory.

        Args:
            directory: Directory to analyze

        Returns:
            Dictionary mapping file paths to their dependencies
        """
        code_files = self._find_code_files(directory)

        if not code_files:
            return {}

        # Perform dependency analysis
        analysis_results = await self.analyze_code(
            [f.path for f in code_files],
            [AnalysisType.DEPENDENCY_MAPPING]
        )

        # Extract dependency mapping
        dependencies = {}
        for result in analysis_results:
            for finding in result.findings:
                if finding.get("type") == "dependency":
                    file_path = finding["file_path"]
                    deps = finding["dependencies"]
                    dependencies[file_path] = deps

        return dependencies

    def _generate_cache_key(self, files: List[str], analysis_type: AnalysisType) -> str:
        """Generate cache key for analysis."""
        # Create hash from file paths and modification times
        key_data = {
            "files": sorted(files),
            "analysis_type": analysis_type.value,
            "timestamps": []
        }

        for file_path in files:
            try:
                stat = Path(file_path).stat()
                key_data["timestamps"].append(stat.st_mtime)
            except OSError:
                key_data["timestamps"].append(0)

        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()

    def _find_code_files(self, directory: str, max_files: int = 100) -> List[CodeFile]:
        """Find code files in directory."""
        code_files = []
        directory_path = Path(directory)

        if not directory_path.exists():
            return code_files

        # Walk directory and find supported files
        for file_path in directory_path.rglob("*"):
            if len(code_files) >= max_files:
                break

            if file_path.is_file() and file_path.suffix in self.supported_languages:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()

                    stat = file_path.stat()
                    file_hash = hashlib.md5(content.encode()).hexdigest()

                    code_file = CodeFile(
                        path=str(file_path),
                        content=content,
                        language=self.supported_languages[file_path.suffix],
                        size_bytes=stat.st_size,
                        lines_count=len(content.splitlines()),
                        last_modified=datetime.fromtimestamp(stat.st_mtime),
                        hash=file_hash
                    )

                    code_files.append(code_file)

                except Exception as e:
                    # Skip files that can't be read
                    continue

        return code_files

    async def _analysis_worker(self):
        """Background worker for processing analysis tasks."""
        while self.processing:
            try:
                # Get task from queue
                task = await asyncio.wait_for(self.analysis_queue.get(), timeout=1.0)

                # Perform analysis
                result = await self._perform_analysis(task)

                # Cache result
                self.analysis_cache[task["cache_key"]] = result

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                print(f"Analysis worker error: {e}")

    async def _perform_analysis(self, task: Dict[str, Any]) -> AnalysisResult:
        """Perform actual Claude analysis."""
        start_time = time.time()
        analysis_type = task["analysis_type"]
        files = task["files"]

        # Load file contents
        code_files = []
        for file_path in files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                file_path_obj = Path(file_path)
                language = self.supported_languages.get(file_path_obj.suffix, "unknown")

                code_files.append({
                    "path": file_path,
                    "content": content,
                    "language": language
                })
            except (OSError, UnicodeDecodeError):
                continue

        # Perform analysis based on type
        if analysis_type == AnalysisType.REFACTORING_OPPORTUNITIES:
            findings, recommendations = self._analyze_refactoring_opportunities(code_files)
        elif analysis_type == AnalysisType.COMPLEXITY_ANALYSIS:
            findings, recommendations = self._analyze_complexity(code_files)
        elif analysis_type == AnalysisType.DEPENDENCY_MAPPING:
            findings, recommendations = self._analyze_dependencies(code_files)
        elif analysis_type == AnalysisType.SECURITY_VULNERABILITIES:
            findings, recommendations = self._analyze_security(code_files)
        elif analysis_type == AnalysisType.PERFORMANCE_OPTIMIZATION:
            findings, recommendations = self._analyze_performance(code_files)
        elif analysis_type == AnalysisType.CODE_SMELLS:
            findings, recommendations = self._analyze_code_smells(code_files)
        elif analysis_type == AnalysisType.ARCHITECTURE_REVIEW:
            findings, recommendations = self._analyze_architecture(code_files)
        else:
            findings, recommendations = [], []

        processing_time_ms = int((time.time() - start_time) * 1000)

        return AnalysisResult(
            analysis_id=task["id"],
            analysis_type=analysis_type,
            files_analyzed=files,
            findings=findings,
            recommendations=recommendations,
            confidence_score=0.85,  # Mock confidence score
            processing_time_ms=processing_time_ms,
            timestamp=datetime.now()
        )

    def _analyze_refactoring_opportunities(self, code_files: List[Dict[str, Any]]) -> tuple:
        """Analyze code for refactoring opportunities."""
        findings = []
        recommendations = []

        for file_info in code_files:
            content = file_info["content"]
            file_path = file_info["path"]
            language = file_info["language"]

            lines = content.splitlines()

            # Look for long methods (Python example)
            if language == "python":
                try:
                    tree = ast.parse(content)
                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef):
                            if hasattr(node, 'end_lineno') and node.end_lineno:
                                line_count = node.end_lineno - node.lineno + 1
                                if line_count > 50:
                                    findings.append({
                                        "type": "refactoring_opportunity",
                                        "file_path": file_path,
                                        "line_range": [node.lineno, node.end_lineno],
                                        "opportunity_type": "extract_method",
                                        "description": f"Function '{node.name}' is too long ({line_count} lines)",
                                        "impact": "medium",
                                        "effort": "medium",
                                        "rationale": "Long functions are harder to understand and maintain"
                                    })
                except SyntaxError:
                    pass  # Skip files with syntax errors

            # Look for duplicate code patterns
            duplicate_blocks = self._find_duplicate_blocks(content)
            for block in duplicate_blocks:
                findings.append({
                    "type": "refactoring_opportunity",
                    "file_path": file_path,
                    "line_range": block["line_range"],
                    "opportunity_type": "extract_duplicate_code",
                    "description": "Duplicate code block found",
                    "impact": "medium",
                    "effort": "low",
                    "rationale": "Duplicate code leads to maintenance issues"
                })

            # Look for complex conditional logic
            complex_conditions = self._find_complex_conditions(content, language)
            for condition in complex_conditions:
                findings.append({
                    "type": "refactoring_opportunity",
                    "file_path": file_path,
                    "line_range": condition["line_range"],
                    "opportunity_type": "simplify_condition",
                    "description": "Complex conditional logic detected",
                    "impact": "low",
                    "effort": "low",
                    "rationale": "Complex conditions are hard to understand"
                })

        recommendations = [
            "Consider extracting large functions into smaller, focused methods",
            "Eliminate duplicate code by creating shared utility functions",
            "Simplify complex conditional logic using early returns or strategy pattern",
            "Use descriptive variable and function names",
            "Add proper error handling and logging"
        ]

        return findings, recommendations

    def _analyze_complexity(self, code_files: List[Dict[str, Any]]) -> tuple:
        """Analyze code complexity."""
        findings = []
        recommendations = []

        for file_info in code_files:
            content = file_info["content"]
            file_path = file_info["path"]
            language = file_info["language"]

            # Calculate complexity metrics
            lines = content.splitlines()
            total_lines = len(lines)
            non_empty_lines = len([line for line in lines if line.strip()])

            # Cyclomatic complexity estimation
            complexity_score = self._estimate_complexity(content, language)

            findings.append({
                "type": "complexity_metrics",
                "file_path": file_path,
                "metrics": {
                    "total_lines": total_lines,
                    "non_empty_lines": non_empty_lines,
                    "complexity_score": complexity_score
                }
            })

            if complexity_score > 10:
                recommendations.append(f"High complexity in {file_path} - consider refactoring")

        return findings, recommendations

    def _analyze_dependencies(self, code_files: List[Dict[str, Any]]) -> tuple:
        """Analyze file dependencies."""
        findings = []
        recommendations = []

        for file_info in code_files:
            content = file_info["content"]
            file_path = file_info["path"]
            language = file_info["language"]

            # Extract imports/dependencies
            dependencies = self._extract_dependencies(content, language)

            findings.append({
                "type": "dependency",
                "file_path": file_path,
                "dependencies": dependencies
            })

        recommendations = [
            "Consider reducing circular dependencies",
            "Group related functionality into modules",
            "Use dependency injection for better testability"
        ]

        return findings, recommendations

    def _analyze_security(self, code_files: List[Dict[str, Any]]) -> tuple:
        """Analyze security vulnerabilities."""
        findings = []
        recommendations = []

        security_patterns = {
            "sql_injection": [r"execute\(", r"query\("],
            "hardcoded_secrets": [r"password\s*=\s*['\"][^'\"]+['\"]", r"api_key\s*=\s*['\"][^'\"]+['\"]"],
            "eval_usage": [r"eval\(", r"exec\("],
            "shell_injection": [r"os\.system\(", r"subprocess\.call\("]
        }

        for file_info in code_files:
            content = file_info["content"]
            file_path = file_info["path"]

            lines = content.splitlines()
            for i, line in enumerate(lines, 1):
                for vuln_type, patterns in security_patterns.items():
                    for pattern in patterns:
                        if re.search(pattern, line, re.IGNORECASE):
                            findings.append({
                                "type": "security_vulnerability",
                                "file_path": file_path,
                                "line_number": i,
                                "vulnerability_type": vuln_type,
                                "line_content": line.strip()
                            })

        recommendations = [
            "Use parameterized queries to prevent SQL injection",
            "Store secrets in environment variables or secure vaults",
            "Avoid eval() and exec() with user input",
            "Validate and sanitize all user inputs",
            "Use principle of least privilege for API access"
        ]

        return findings, recommendations

    def _analyze_performance(self, code_files: List[Dict[str, Any]]) -> tuple:
        """Analyze performance optimization opportunities."""
        findings = []
        recommendations = []

        for file_info in code_files:
            content = file_info["content"]
            file_path = file_info["path"]
            language = file_info["language"]

            # Look for performance anti-patterns
            performance_issues = self._find_performance_issues(content, language)

            for issue in performance_issues:
                findings.append({
                    "type": "performance_issue",
                    "file_path": file_path,
                    "line_range": issue["line_range"],
                    "issue_type": issue["type"],
                    "description": issue["description"],
                    "impact": issue["impact"]
                })

        recommendations = [
            "Use caching for frequently accessed data",
            "Optimize database queries with proper indexing",
            "Consider async/await for I/O operations",
            "Profile code to identify bottlenecks",
            "Use efficient data structures and algorithms"
        ]

        return findings, recommendations

    def _analyze_code_smells(self, code_files: List[Dict[str, Any]]) -> tuple:
        """Analyze code smells."""
        findings = []
        recommendations = []

        for file_info in code_files:
            content = file_info["content"]
            file_path = file_info["path"]

            # Look for common code smells
            smells = self._find_code_smells(content)

            for smell in smells:
                findings.append({
                    "type": "code_smell",
                    "file_path": file_path,
                    "line_range": smell["line_range"],
                    "smell_type": smell["type"],
                    "description": smell["description"]
                })

        recommendations = [
            "Follow consistent naming conventions",
            "Keep functions and classes small and focused",
            "Avoid deep nesting and complex logic",
            "Use meaningful variable names",
            "Remove dead code and unused imports"
        ]

        return findings, recommendations

    def _analyze_architecture(self, code_files: List[Dict[str, Any]]) -> tuple:
        """Analyze architectural patterns and issues."""
        findings = []
        recommendations = []

        # Analyze overall project structure
        file_count = len(code_files)
        language_distribution = {}

        for file_info in code_files:
            language = file_info["language"]
            language_distribution[language] = language_distribution.get(language, 0) + 1

        findings.append({
            "type": "architecture_overview",
            "metrics": {
                "total_files": file_count,
                "language_distribution": language_distribution
            }
        })

        if file_count > 100:
            recommendations.append("Large codebase - consider modularization")

        recommendations.extend([
            "Follow SOLID principles for better design",
            "Implement proper separation of concerns",
            "Use design patterns appropriately",
            "Document architectural decisions",
            "Consider microservices for large applications"
        ])

        return findings, recommendations

    def _find_duplicate_blocks(self, content: str) -> List[Dict[str, Any]]:
        """Find duplicate code blocks."""
        # Simplified duplicate detection
        lines = content.splitlines()
        duplicates = []

        # Look for blocks of 5+ lines that appear multiple times
        block_size = 5
        line_blocks = {}

        for i in range(len(lines) - block_size + 1):
            block = "\n".join(lines[i:i + block_size]).strip()
            if len(block) < 50:  # Skip very small blocks
                continue

            block_hash = hashlib.md5(block.encode()).hexdigest()
            if block_hash in line_blocks:
                line_blocks[block_hash].append(i + 1)
            else:
                line_blocks[block_hash] = [i + 1]

        for block_hash, positions in line_blocks.items():
            if len(positions) > 1:
                for pos in positions:
                    duplicates.append({
                        "line_range": [pos, pos + block_size - 1]
                    })

        return duplicates[:5]  # Limit to 5 most significant duplicates

    def _find_complex_conditions(self, content: str, language: str) -> List[Dict[str, Any]]:
        """Find complex conditional statements."""
        complex_conditions = []
        lines = content.splitlines()

        for i, line in enumerate(lines, 1):
            # Count logical operators
            and_count = line.count(" and ") + line.count(" && ")
            or_count = line.count(" or ") + line.count(" || ")

            if and_count + or_count > 3:
                complex_conditions.append({
                    "line_range": [i, i],
                    "complexity_score": and_count + or_count
                })

        return complex_conditions[:10]  # Limit to 10 most complex

    def _estimate_complexity(self, content: str, language: str) -> float:
        """Estimate cyclomatic complexity."""
        # Simplified complexity estimation
        complexity_keywords = {
            "python": ["if", "elif", "for", "while", "except", "with", "and", "or"],
            "javascript": ["if", "else", "for", "while", "catch", "&&", "||"],
            "typescript": ["if", "else", "for", "while", "catch", "&&", "||"]
        }

        keywords = complexity_keywords.get(language, complexity_keywords["python"])
        complexity = 1  # Base complexity

        for keyword in keywords:
            complexity += content.count(keyword)

        return float(complexity)

    def _extract_dependencies(self, content: str, language: str) -> List[str]:
        """Extract dependencies from code."""
        dependencies = []

        if language == "python":
            import_pattern = r"(?:from\s+(\S+)\s+import|import\s+(\S+))"
            matches = re.findall(import_pattern, content)
            for match in matches:
                dep = match[0] if match[0] else match[1]
                dependencies.append(dep)

        elif language in ["javascript", "typescript"]:
            import_patterns = [
                r"import.*from\s+['\"]([^'\"]+)['\"]",
                r"require\(['\"]([^'\"]+)['\"]\)"
            ]
            for pattern in import_patterns:
                matches = re.findall(pattern, content)
                dependencies.extend(matches)

        return list(set(dependencies))

    def _find_performance_issues(self, content: str, language: str) -> List[Dict[str, Any]]:
        """Find performance issues."""
        issues = []
        lines = content.splitlines()

        performance_patterns = {
            "nested_loops": r"for.*for",
            "inefficient_string_concat": r"\+\s*=",
            "sync_io": r"\.read\(|\.write\(",
            "large_data_structures": r"list\(range\("
        }

        for i, line in enumerate(lines, 1):
            for issue_type, pattern in performance_patterns.items():
                if re.search(pattern, line):
                    issues.append({
                        "type": issue_type,
                        "line_range": [i, i],
                        "description": f"Potential performance issue: {issue_type}",
                        "impact": "medium"
                    })

        return issues[:10]

    def _find_code_smells(self, content: str) -> List[Dict[str, Any]]:
        """Find code smells."""
        smells = []
        lines = content.splitlines()

        smell_patterns = {
            "long_line": r".{100,}",  # Lines over 100 characters
            "magic_number": r"\b(?!0|1)\d{2,}\b",  # Numbers >= 10
            "todo_comment": r"# TODO|// TODO",
            "multiple_blank_lines": r"\n\s*\n\s*\n"
        }

        content_with_line_numbers = "\n".join(f"{i+1}:{line}" for i, line in enumerate(lines))

        for smell_type, pattern in smell_patterns.items():
            matches = list(re.finditer(pattern, content_with_line_numbers, re.MULTILINE))
            for match in matches:
                line_num = int(match.group().split(":")[0])
                smells.append({
                    "type": smell_type,
                    "line_range": [line_num, line_num],
                    "description": f"Code smell detected: {smell_type}"
                })

        return smells[:15]

    async def _wait_for_queue_empty(self):
        """Wait for analysis queue to be empty."""
        while not self.analysis_queue.empty():
            await asyncio.sleep(0.1)


# Integration with ParallelWorkflowExecutor
class ClaudeAnalysisHandler:
    """Handler for integrating Claude analysis with parallel workflows."""

    def __init__(self):
        self.claude_service = ClaudeReasoningService()
        self._started = False

    async def ensure_started(self):
        """Ensure the Claude service is started."""
        if not self._started:
            await self.claude_service.start()
            self._started = True

    async def handle_claude_analysis_task(self, task: ParallelTask) -> TaskResult:
        """Handle a Claude analysis task from the parallel executor."""
        await self.ensure_started()

        start_time = time.time()

        try:
            # Extract analysis parameters from task
            stage = task.stage
            context = task.context

            # Get files to analyze from context or stage
            files = stage.get("files", [])
            if not files and "repo_path" in context.inputs:
                # Find code files in repo
                files = [f.path for f in self.claude_service._find_code_files(context.inputs["repo_path"])]

            # Determine analysis types
            analysis_types_str = stage.get("analysis_types", ["refactoring_opportunities"])
            analysis_types = [AnalysisType(t) for t in analysis_types_str]

            # Perform analysis
            results = await self.claude_service.analyze_code(
                files, analysis_types, force_refresh=True
            )

            # Format output
            output = {
                "analysis_results": [
                    {
                        "type": result.analysis_type.value,
                        "files_analyzed": result.files_analyzed,
                        "findings_count": len(result.findings),
                        "recommendations": result.recommendations,
                        "confidence": result.confidence_score,
                        "processing_time_ms": result.processing_time_ms
                    }
                    for result in results
                ],
                "total_findings": sum(len(r.findings) for r in results),
                "total_files_analyzed": len(set(f for r in results for f in r.files_analyzed))
            }

            duration_ms = int((time.time() - start_time) * 1000)

            return TaskResult(
                task_id=task.task_id,
                status=TaskStatus.COMPLETED,
                output=output,
                duration_ms=duration_ms
            )

        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            return TaskResult(
                task_id=task.task_id,
                status=TaskStatus.FAILED,
                error=str(e),
                duration_ms=duration_ms
            )

    async def cleanup(self):
        """Clean up resources."""
        if self._started:
            await self.claude_service.stop()


# Global handler instance
_claude_handler = ClaudeAnalysisHandler()


async def handle_claude_analysis_task(task: ParallelTask) -> TaskResult:
    """Global handler function for Claude analysis tasks."""
    return await _claude_handler.handle_claude_analysis_task(task)
