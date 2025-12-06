"""
Sandbox Executor - Stage 3 of ORCHEX

Safely executes generated experiment code with resource limits.
"""

import subprocess
import json
import time
from typing import Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
import shutil
import tempfile


@dataclass
class ExecutionResult:
    """Result from sandbox execution"""
    experiment_id: str
    success: bool
    exit_code: int
    stdout: str
    stderr: str
    duration: float
    results_data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


class SandboxExecutor:
    """
    Execute experiment code safely with resource limits

    Features:
    - Subprocess isolation
    - Timeout enforcement
    - Memory limits (via ulimit)
    - Result collection
    - Error handling and recovery
    - Cleanup
    """

    def __init__(
        self,
        default_timeout: int = 3600,  # 1 hour
        max_memory_mb: int = 8192,  # 8GB
        cleanup_on_success: bool = False
    ):
        """
        Initialize sandbox executor

        Args:
            default_timeout: Default timeout in seconds
            max_memory_mb: Maximum memory in MB
            cleanup_on_success: Whether to delete files after successful run
        """
        self.default_timeout = default_timeout
        self.max_memory_mb = max_memory_mb
        self.cleanup_on_success = cleanup_on_success

    async def execute(
        self,
        experiment_dir: Path,
        timeout: Optional[int] = None,
        env_vars: Optional[Dict[str, str]] = None
    ) -> ExecutionResult:
        """
        Execute experiment in sandbox

        Args:
            experiment_dir: Directory with experiment code
            timeout: Timeout in seconds (override default)
            env_vars: Additional environment variables

        Returns:
            ExecutionResult with execution outcome
        """
        experiment_id = experiment_dir.name
        timeout = timeout or self.default_timeout

        print(f"\n🔒 Executing experiment in sandbox: {experiment_id}")
        print(f"   Timeout: {timeout}s, Memory limit: {self.max_memory_mb}MB")

        # Prepare environment
        env = self._prepare_environment(env_vars)

        # Install requirements
        install_success = await self._install_requirements(experiment_dir)
        if not install_success:
            return ExecutionResult(
                experiment_id=experiment_id,
                success=False,
                exit_code=-1,
                stdout="",
                stderr="Failed to install requirements",
                duration=0.0,
                error_message="Dependency installation failed"
            )

        # Execute main script
        start_time = time.time()

        try:
            result = subprocess.run(
                ["python", "main.py"],
                cwd=str(experiment_dir),
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout
            )

            duration = time.time() - start_time

            # Collect results
            results_data = self._collect_results(experiment_dir)

            execution_result = ExecutionResult(
                experiment_id=experiment_id,
                success=result.returncode == 0,
                exit_code=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
                duration=duration,
                results_data=results_data
            )

            if result.returncode == 0:
                print(f"   ✓ Execution successful ({duration:.1f}s)")
            else:
                print(f"   ✗ Execution failed (exit code: {result.returncode})")

        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            execution_result = ExecutionResult(
                experiment_id=experiment_id,
                success=False,
                exit_code=-1,
                stdout="",
                stderr=f"Timeout after {timeout}s",
                duration=duration,
                error_message=f"Execution timeout ({timeout}s)"
            )
            print(f"   ⏱️  Timeout after {timeout}s")

        except Exception as e:
            duration = time.time() - start_time
            execution_result = ExecutionResult(
                experiment_id=experiment_id,
                success=False,
                exit_code=-1,
                stdout="",
                stderr=str(e),
                duration=duration,
                error_message=f"Execution error: {e}"
            )
            print(f"   ❌ Error: {e}")

        # Cleanup if requested
        if self.cleanup_on_success and execution_result.success:
            self._cleanup(experiment_dir)

        return execution_result

    async def execute_with_retry(
        self,
        experiment_dir: Path,
        max_retries: int = 3,
        timeout: Optional[int] = None
    ) -> ExecutionResult:
        """
        Execute with automatic retry on transient failures

        Args:
            experiment_dir: Directory with experiment code
            max_retries: Maximum retry attempts
            timeout: Timeout per attempt

        Returns:
            ExecutionResult from final attempt
        """
        last_result = None

        for attempt in range(max_retries):
            if attempt > 0:
                print(f"\n🔄 Retry attempt {attempt + 1}/{max_retries}")

            result = await self.execute(experiment_dir, timeout)

            if result.success:
                return result

            last_result = result

            # Check if error is retryable
            if not self._is_retryable_error(result):
                print(f"   ⚠️  Non-retryable error, stopping")
                break

            # Wait before retry
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                print(f"   ⏳ Waiting {wait_time}s before retry...")
                time.sleep(wait_time)

        return last_result

    def _prepare_environment(
        self,
        env_vars: Optional[Dict[str, str]]
    ) -> Dict[str, str]:
        """Prepare execution environment"""
        import os

        env = os.environ.copy()

        # Add memory limit (Linux only)
        try:
            import resource
            soft, hard = resource.getrlimit(resource.RLIMIT_AS)
            memory_bytes = self.max_memory_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, hard))
        except (ImportError, ValueError):
            # Not on Linux or setrlimit failed
            pass

        # Add custom env vars
        if env_vars:
            env.update(env_vars)

        # Disable GPU by default (can be overridden)
        if "CUDA_VISIBLE_DEVICES" not in env:
            env["CUDA_VISIBLE_DEVICES"] = ""

        return env

    async def _install_requirements(self, experiment_dir: Path) -> bool:
        """Install Python requirements"""
        requirements_file = experiment_dir / "requirements.txt"

        if not requirements_file.exists():
            return True  # No requirements needed

        print(f"   📦 Installing requirements...")

        try:
            result = subprocess.run(
                ["pip", "install", "-q", "-r", str(requirements_file)],
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout for installation
            )

            if result.returncode != 0:
                print(f"   ❌ Installation failed: {result.stderr}")
                return False

            print(f"   ✓ Requirements installed")
            return True

        except subprocess.TimeoutExpired:
            print(f"   ⏱️  Installation timeout")
            return False
        except Exception as e:
            print(f"   ❌ Installation error: {e}")
            return False

    def _collect_results(self, experiment_dir: Path) -> Optional[Dict[str, Any]]:
        """Collect results from experiment directory"""
        results_file = experiment_dir / "results" / "results.json"

        if not results_file.exists():
            # Try alternative locations
            alt_paths = [
                experiment_dir / "results.json",
                experiment_dir / "output" / "results.json",
            ]

            for alt_path in alt_paths:
                if alt_path.exists():
                    results_file = alt_path
                    break
            else:
                return None

        try:
            with open(results_file) as f:
                return json.load(f)
        except Exception as e:
            print(f"   ⚠️  Could not load results: {e}")
            return None

    def _is_retryable_error(self, result: ExecutionResult) -> bool:
        """Check if error is retryable"""
        # Non-retryable: syntax errors, import errors, logic errors
        non_retryable_keywords = [
            "SyntaxError",
            "ImportError",
            "ModuleNotFoundError",
            "NameError",
            "TypeError",
            "AttributeError"
        ]

        stderr_lower = result.stderr.lower()

        for keyword in non_retryable_keywords:
            if keyword.lower() in stderr_lower:
                return False

        # Retryable: timeouts, network errors, resource errors
        retryable_keywords = [
            "timeout",
            "connection",
            "network",
            "temporarily unavailable"
        ]

        for keyword in retryable_keywords:
            if keyword.lower() in stderr_lower:
                return True

        # Default: retry once
        return True

    def _cleanup(self, experiment_dir: Path):
        """Clean up experiment directory"""
        try:
            shutil.rmtree(experiment_dir)
            print(f"   🗑️  Cleaned up: {experiment_dir}")
        except Exception as e:
            print(f"   ⚠️  Cleanup failed: {e}")

    def save_result(self, result: ExecutionResult, output_file: Path):
        """Save execution result to file"""
        result_dict = {
            "experiment_id": result.experiment_id,
            "success": result.success,
            "exit_code": result.exit_code,
            "duration": result.duration,
            "stdout": result.stdout[:1000],  # Truncate long output
            "stderr": result.stderr[:1000],
            "results_data": result.results_data,
            "error_message": result.error_message,
            "timestamp": result.timestamp.isoformat()
        }

        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump(result_dict, f, indent=2)

        print(f"✓ Saved execution result: {output_file}")


class BatchExecutor:
    """Execute multiple experiments in batch"""

    def __init__(self, sandbox_executor: Optional[SandboxExecutor] = None):
        """
        Initialize batch executor

        Args:
            sandbox_executor: SandboxExecutor to use (creates default if None)
        """
        self.sandbox = sandbox_executor or SandboxExecutor()

    async def execute_batch(
        self,
        experiment_dirs: list[Path],
        parallel: bool = False,
        max_workers: int = 4
    ) -> list[ExecutionResult]:
        """
        Execute multiple experiments

        Args:
            experiment_dirs: List of experiment directories
            parallel: Whether to run in parallel
            max_workers: Max parallel workers

        Returns:
            List of ExecutionResults
        """
        print(f"\n🔬 Executing batch: {len(experiment_dirs)} experiments")
        print(f"   Mode: {'parallel' if parallel else 'sequential'}")

        results = []

        if parallel:
            # Parallel execution (simplified - use asyncio.gather in production)
            print(f"   Max workers: {max_workers}")
            for exp_dir in experiment_dirs:
                result = await self.sandbox.execute(exp_dir)
                results.append(result)
        else:
            # Sequential execution
            for i, exp_dir in enumerate(experiment_dirs, 1):
                print(f"\n[{i}/{len(experiment_dirs)}]")
                result = await self.sandbox.execute(exp_dir)
                results.append(result)

        # Summary
        successful = sum(1 for r in results if r.success)
        print(f"\n📊 Batch complete: {successful}/{len(results)} successful")

        return results
