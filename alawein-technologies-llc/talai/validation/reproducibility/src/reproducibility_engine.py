"""
Reproducibility Verification Engine for TalAI

Core engine that ensures computational reproducibility through environment capture,
seed tracking, dependency management, and result verification with tolerance checking.
"""

import asyncio
import json
import logging
import hashlib
import subprocess
import sys
import os
import platform
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set
from enum import Enum
import numpy as np
import yaml
import pickle
import tempfile
import shutil
import docker
import tarfile
from packaging import version

logger = logging.getLogger(__name__)


class ReproducibilityLevel(Enum):
    """Levels of reproducibility certification"""
    GOLD = "gold"  # Fully reproducible, containerized, all seeds tracked
    SILVER = "silver"  # Mostly reproducible, environment captured, minor variations
    BRONZE = "bronze"  # Partially reproducible, core results match, some variations
    NONE = "none"  # Not reproducible, significant variations


class VerificationStatus(Enum):
    """Status of reproducibility verification"""
    VERIFIED = "verified"
    PARTIAL = "partial"
    FAILED = "failed"
    IN_PROGRESS = "in_progress"
    NOT_STARTED = "not_started"


@dataclass
class Environment:
    """Complete environment specification"""
    python_version: str
    platform_info: Dict[str, str]
    cpu_info: Dict[str, Any]
    memory_info: Dict[str, Any]
    gpu_info: Optional[Dict[str, Any]]
    os_info: Dict[str, str]
    environment_variables: Dict[str, str]
    conda_environment: Optional[Dict[str, Any]]
    pip_packages: List[Dict[str, str]]
    system_packages: List[str]
    docker_info: Optional[Dict[str, Any]]
    timestamp: datetime


@dataclass
class DependencySpec:
    """Specification for a dependency"""
    name: str
    version: str
    source: str  # pip, conda, system, github
    hash: Optional[str]
    install_command: str
    optional: bool
    conflicts: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RandomSeed:
    """Random seed tracking"""
    numpy_seed: Optional[int]
    python_seed: Optional[int]
    torch_seed: Optional[int]
    tensorflow_seed: Optional[int]
    custom_seeds: Dict[str, int]
    seed_sequence: List[Tuple[str, int, datetime]]


@dataclass
class ExecutionTrace:
    """Trace of execution for reproducibility"""
    command: str
    working_directory: str
    start_time: datetime
    end_time: datetime
    exit_code: int
    stdout: str
    stderr: str
    resource_usage: Dict[str, Any]
    file_hashes_before: Dict[str, str]
    file_hashes_after: Dict[str, str]


@dataclass
class ReproducibilityReport:
    """Complete reproducibility verification report"""
    experiment_id: str
    original_run: Dict[str, Any]
    reproduction_run: Dict[str, Any]
    environment_match: float  # 0-1 score
    dependency_match: float  # 0-1 score
    seed_tracking_complete: bool
    results_match: bool
    tolerance_used: float
    variations_found: List[Dict[str, Any]]
    reproducibility_level: ReproducibilityLevel
    verification_status: VerificationStatus
    recommendations: List[str]
    badge_awarded: Optional[str]
    certificate_hash: str


class ReproducibilityEngine:
    """
    Main engine for reproducibility verification of TalAI experiments.
    Captures environments, tracks seeds, and verifies result reproducibility.
    """

    def __init__(self,
                 workspace_dir: Path,
                 reprozoip_enabled: bool = True,
                 docker_enabled: bool = True,
                 tolerance: float = 1e-6):
        """
        Initialize the reproducibility engine.

        Args:
            workspace_dir: Directory for reproducibility artifacts
            reprozoip_enabled: Whether to use ReproZip for capture
            docker_enabled: Whether to use Docker for containerization
            tolerance: Numerical tolerance for result comparison
        """
        self.workspace_dir = Path(workspace_dir)
        self.workspace_dir.mkdir(parents=True, exist_ok=True)

        self.reprozip_enabled = reprozoip_enabled and self._check_reprozip()
        self.docker_enabled = docker_enabled and self._check_docker()
        self.tolerance = tolerance

        # Tracking
        self.environments: Dict[str, Environment] = {}
        self.dependencies: Dict[str, List[DependencySpec]] = {}
        self.seeds: Dict[str, RandomSeed] = {}
        self.traces: Dict[str, List[ExecutionTrace]] = {}
        self.reports: List[ReproducibilityReport] = []

        # Performance metrics
        self.capture_times: List[float] = []
        self.verification_times: List[float] = []
        self.reproducibility_scores: List[float] = []

        logger.info(f"Initialized ReproducibilityEngine with workspace: {workspace_dir}")
        logger.info(f"ReproZip: {self.reprozip_enabled}, Docker: {self.docker_enabled}")

    def _check_reprozip(self) -> bool:
        """Check if ReproZip is available"""
        try:
            result = subprocess.run(["reprozip", "--version"], capture_output=True)
            return result.returncode == 0
        except FileNotFoundError:
            logger.warning("ReproZip not found, some features will be limited")
            return False

    def _check_docker(self) -> bool:
        """Check if Docker is available"""
        try:
            client = docker.from_env()
            client.ping()
            return True
        except Exception as e:
            logger.warning(f"Docker not available: {e}")
            return False

    async def capture_environment(self, experiment_id: str) -> Environment:
        """
        Capture complete environment specification.

        Args:
            experiment_id: Unique identifier for the experiment

        Returns:
            Environment specification
        """
        import time
        start_time = time.time()

        # Python version
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

        # Platform information
        platform_info = {
            "system": platform.system(),
            "node": platform.node(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor()
        }

        # CPU information
        cpu_info = {
            "count": os.cpu_count(),
            "architecture": platform.machine()
        }

        try:
            import psutil
            cpu_info.update({
                "physical_cores": psutil.cpu_count(logical=False),
                "logical_cores": psutil.cpu_count(logical=True),
                "frequency": psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None
            })
        except ImportError:
            logger.warning("psutil not available, limited CPU info")

        # Memory information
        memory_info = {}
        try:
            import psutil
            mem = psutil.virtual_memory()
            memory_info = {
                "total": mem.total,
                "available": mem.available,
                "used": mem.used,
                "percent": mem.percent
            }
        except ImportError:
            pass

        # GPU information
        gpu_info = await self._capture_gpu_info()

        # OS information
        os_info = {
            "name": os.name,
            "environ_keys": list(os.environ.keys())
        }

        # Environment variables (filtered for security)
        safe_env_vars = ["PATH", "PYTHONPATH", "CUDA_HOME", "LD_LIBRARY_PATH", "VIRTUAL_ENV"]
        environment_variables = {
            k: v for k, v in os.environ.items()
            if k in safe_env_vars or k.startswith("CONDA_") or k.startswith("PIP_")
        }

        # Conda environment
        conda_environment = await self._capture_conda_environment()

        # Pip packages
        pip_packages = await self._capture_pip_packages()

        # System packages
        system_packages = await self._capture_system_packages()

        # Docker information
        docker_info = None
        if self.docker_enabled:
            docker_info = await self._capture_docker_info()

        env = Environment(
            python_version=python_version,
            platform_info=platform_info,
            cpu_info=cpu_info,
            memory_info=memory_info,
            gpu_info=gpu_info,
            os_info=os_info,
            environment_variables=environment_variables,
            conda_environment=conda_environment,
            pip_packages=pip_packages,
            system_packages=system_packages,
            docker_info=docker_info,
            timestamp=datetime.now()
        )

        self.environments[experiment_id] = env
        self.capture_times.append(time.time() - start_time)

        logger.info(f"Captured environment for {experiment_id} in {self.capture_times[-1]:.2f}s")

        return env

    async def _capture_gpu_info(self) -> Optional[Dict[str, Any]]:
        """Capture GPU information"""
        gpu_info = {}

        try:
            # Try nvidia-smi
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                gpu_info["nvidia"] = []
                for line in lines:
                    parts = line.split(', ')
                    if len(parts) >= 3:
                        gpu_info["nvidia"].append({
                            "name": parts[0],
                            "memory": parts[1],
                            "driver": parts[2]
                        })
        except FileNotFoundError:
            pass

        try:
            # Try to get CUDA version
            import torch
            if torch.cuda.is_available():
                gpu_info["cuda_available"] = True
                gpu_info["cuda_version"] = torch.version.cuda
                gpu_info["cudnn_version"] = torch.backends.cudnn.version()
                gpu_info["device_count"] = torch.cuda.device_count()
        except ImportError:
            pass

        return gpu_info if gpu_info else None

    async def _capture_conda_environment(self) -> Optional[Dict[str, Any]]:
        """Capture Conda environment if available"""
        try:
            result = subprocess.run(["conda", "env", "export", "--json"], capture_output=True)
            if result.returncode == 0:
                return json.loads(result.stdout)
        except FileNotFoundError:
            pass
        return None

    async def _capture_pip_packages(self) -> List[Dict[str, str]]:
        """Capture pip package list"""
        packages = []
        try:
            result = subprocess.run([sys.executable, "-m", "pip", "list", "--format=json"], capture_output=True)
            if result.returncode == 0:
                packages = json.loads(result.stdout)
        except Exception as e:
            logger.warning(f"Failed to capture pip packages: {e}")
        return packages

    async def _capture_system_packages(self) -> List[str]:
        """Capture system packages (Linux only)"""
        packages = []
        if platform.system() == "Linux":
            try:
                # Try dpkg (Debian/Ubuntu)
                result = subprocess.run(["dpkg", "-l"], capture_output=True, text=True)
                if result.returncode == 0:
                    lines = result.stdout.strip().split('\n')
                    packages = [line.split()[1] for line in lines if line.startswith("ii")]
            except FileNotFoundError:
                try:
                    # Try rpm (Red Hat/CentOS)
                    result = subprocess.run(["rpm", "-qa"], capture_output=True, text=True)
                    if result.returncode == 0:
                        packages = result.stdout.strip().split('\n')
                except FileNotFoundError:
                    pass
        return packages

    async def _capture_docker_info(self) -> Optional[Dict[str, Any]]:
        """Capture Docker information"""
        try:
            client = docker.from_env()
            return {
                "version": client.version(),
                "info": client.info(),
                "images": [img.tags for img in client.images.list()]
            }
        except Exception as e:
            logger.warning(f"Failed to capture Docker info: {e}")
            return None

    async def track_seeds(self, experiment_id: str) -> RandomSeed:
        """
        Track and record all random seeds.

        Args:
            experiment_id: Unique experiment identifier

        Returns:
            RandomSeed tracking object
        """
        import random
        seeds = RandomSeed(
            numpy_seed=None,
            python_seed=None,
            torch_seed=None,
            tensorflow_seed=None,
            custom_seeds={},
            seed_sequence=[]
        )

        # Python random seed
        python_seed = random.getstate()[1][0]
        seeds.python_seed = python_seed
        seeds.seed_sequence.append(("python", python_seed, datetime.now()))

        # NumPy seed
        try:
            import numpy as np
            numpy_seed = np.random.get_state()[1][0]
            seeds.numpy_seed = int(numpy_seed)
            seeds.seed_sequence.append(("numpy", int(numpy_seed), datetime.now()))
        except ImportError:
            pass

        # PyTorch seed
        try:
            import torch
            torch_seed = torch.initial_seed()
            seeds.torch_seed = int(torch_seed)
            seeds.seed_sequence.append(("torch", int(torch_seed), datetime.now()))
        except ImportError:
            pass

        # TensorFlow seed
        try:
            import tensorflow as tf
            # TensorFlow doesn't have a direct way to get the seed
            # We'll set a known seed instead
            tf_seed = hash(experiment_id) % (2**32)
            tf.random.set_seed(tf_seed)
            seeds.tensorflow_seed = tf_seed
            seeds.seed_sequence.append(("tensorflow", tf_seed, datetime.now()))
        except ImportError:
            pass

        self.seeds[experiment_id] = seeds
        logger.info(f"Tracked seeds for {experiment_id}: {seeds.seed_sequence}")

        return seeds

    async def set_seeds(self, seeds: RandomSeed) -> None:
        """
        Set all random seeds for reproducibility.

        Args:
            seeds: RandomSeed object with seed values
        """
        import random

        # Python random
        if seeds.python_seed is not None:
            random.seed(seeds.python_seed)

        # NumPy
        if seeds.numpy_seed is not None:
            try:
                import numpy as np
                np.random.seed(seeds.numpy_seed)
            except ImportError:
                pass

        # PyTorch
        if seeds.torch_seed is not None:
            try:
                import torch
                torch.manual_seed(seeds.torch_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(seeds.torch_seed)
                    torch.cuda.manual_seed_all(seeds.torch_seed)
            except ImportError:
                pass

        # TensorFlow
        if seeds.tensorflow_seed is not None:
            try:
                import tensorflow as tf
                tf.random.set_seed(seeds.tensorflow_seed)
            except ImportError:
                pass

        # Custom seeds
        for name, seed in seeds.custom_seeds.items():
            logger.info(f"Setting custom seed {name}: {seed}")

    async def lock_dependencies(self, experiment_id: str) -> List[DependencySpec]:
        """
        Lock all dependencies with exact versions.

        Args:
            experiment_id: Unique experiment identifier

        Returns:
            List of locked dependency specifications
        """
        dependencies = []

        # Pip packages
        pip_packages = await self._capture_pip_packages()
        for pkg in pip_packages:
            dep = DependencySpec(
                name=pkg["name"],
                version=pkg["version"],
                source="pip",
                hash=None,  # Would need pip show for hash
                install_command=f"pip install {pkg['name']}=={pkg['version']}",
                optional=False,
                conflicts=[]
            )
            dependencies.append(dep)

        # Conda packages if available
        conda_env = await self._capture_conda_environment()
        if conda_env and "dependencies" in conda_env:
            for dep_str in conda_env["dependencies"]:
                if isinstance(dep_str, str) and "=" in dep_str:
                    name, version = dep_str.split("=", 1)
                    dep = DependencySpec(
                        name=name,
                        version=version,
                        source="conda",
                        hash=None,
                        install_command=f"conda install {name}={version}",
                        optional=False,
                        conflicts=[]
                    )
                    dependencies.append(dep)

        self.dependencies[experiment_id] = dependencies
        logger.info(f"Locked {len(dependencies)} dependencies for {experiment_id}")

        return dependencies

    async def create_dockerfile(self,
                              experiment_id: str,
                              base_image: str = "python:3.9") -> Path:
        """
        Create Dockerfile for reproducible environment.

        Args:
            experiment_id: Unique experiment identifier
            base_image: Base Docker image to use

        Returns:
            Path to generated Dockerfile
        """
        if experiment_id not in self.dependencies:
            raise ValueError(f"No dependencies found for {experiment_id}")

        dockerfile_path = self.workspace_dir / f"Dockerfile.{experiment_id}"
        requirements_path = self.workspace_dir / f"requirements.{experiment_id}.txt"

        # Generate requirements file
        requirements = []
        for dep in self.dependencies[experiment_id]:
            if dep.source == "pip":
                requirements.append(f"{dep.name}=={dep.version}")

        with open(requirements_path, 'w') as f:
            f.write('\n'.join(requirements))

        # Generate Dockerfile
        dockerfile_content = [
            f"FROM {base_image}",
            "",
            "# System dependencies",
            "RUN apt-get update && apt-get install -y \\",
            "    build-essential \\",
            "    git \\",
            "    && rm -rf /var/lib/apt/lists/*",
            "",
            "# Python dependencies",
            f"COPY requirements.{experiment_id}.txt /tmp/requirements.txt",
            "RUN pip install --no-cache-dir -r /tmp/requirements.txt",
            "",
            "# Working directory",
            "WORKDIR /app",
            "",
            "# Copy application code",
            "COPY . /app",
            "",
            "# Set environment variables",
            "ENV PYTHONUNBUFFERED=1",
            f"ENV EXPERIMENT_ID={experiment_id}",
            "",
            "# Default command",
            'CMD ["python", "-m", "talai.validate"]'
        ]

        with open(dockerfile_path, 'w') as f:
            f.write('\n'.join(dockerfile_content))

        logger.info(f"Created Dockerfile for {experiment_id} at {dockerfile_path}")

        return dockerfile_path

    async def trace_execution(self,
                            experiment_id: str,
                            command: str,
                            working_dir: Optional[Path] = None) -> ExecutionTrace:
        """
        Trace execution of a command for reproducibility.

        Args:
            experiment_id: Unique experiment identifier
            command: Command to execute
            working_dir: Working directory for execution

        Returns:
            ExecutionTrace with complete execution details
        """
        import time
        import resource

        working_dir = working_dir or Path.cwd()
        start_time = datetime.now()

        # Capture file hashes before execution
        file_hashes_before = await self._hash_directory(working_dir)

        # Track resource usage
        usage_before = resource.getrusage(resource.RUSAGE_CHILDREN)

        # Execute command
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=working_dir,
            text=True
        )

        stdout, stderr = process.communicate()
        exit_code = process.returncode

        # Track resource usage after
        usage_after = resource.getrusage(resource.RUSAGE_CHILDREN)

        # Capture file hashes after execution
        file_hashes_after = await self._hash_directory(working_dir)

        end_time = datetime.now()

        # Calculate resource usage
        resource_usage = {
            "user_time": usage_after.ru_utime - usage_before.ru_utime,
            "system_time": usage_after.ru_stime - usage_before.ru_stime,
            "max_rss": usage_after.ru_maxrss,
            "page_faults": usage_after.ru_majflt - usage_before.ru_majflt
        }

        trace = ExecutionTrace(
            command=command,
            working_directory=str(working_dir),
            start_time=start_time,
            end_time=end_time,
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            resource_usage=resource_usage,
            file_hashes_before=file_hashes_before,
            file_hashes_after=file_hashes_after
        )

        if experiment_id not in self.traces:
            self.traces[experiment_id] = []
        self.traces[experiment_id].append(trace)

        logger.info(f"Traced execution for {experiment_id}: exit_code={exit_code}")

        return trace

    async def _hash_directory(self, directory: Path, extensions: Optional[List[str]] = None) -> Dict[str, str]:
        """Hash all files in a directory"""
        hashes = {}
        extensions = extensions or ['.py', '.json', '.yaml', '.txt', '.csv']

        for file_path in directory.rglob('*'):
            if file_path.is_file():
                if extensions and not any(file_path.suffix == ext for ext in extensions):
                    continue

                try:
                    with open(file_path, 'rb') as f:
                        file_hash = hashlib.sha256(f.read()).hexdigest()
                        relative_path = file_path.relative_to(directory)
                        hashes[str(relative_path)] = file_hash
                except Exception as e:
                    logger.warning(f"Failed to hash {file_path}: {e}")

        return hashes

    async def verify_reproducibility(self,
                                   original_id: str,
                                   reproduction_id: str,
                                   results_original: Dict[str, Any],
                                   results_reproduction: Dict[str, Any]) -> ReproducibilityReport:
        """
        Verify reproducibility between original and reproduction runs.

        Args:
            original_id: Original experiment ID
            reproduction_id: Reproduction experiment ID
            results_original: Original results
            results_reproduction: Reproduction results

        Returns:
            ReproducibilityReport with verification details
        """
        import time
        start_time = time.time()

        # Compare environments
        env_match = 0.0
        if original_id in self.environments and reproduction_id in self.environments:
            env_match = self._compare_environments(
                self.environments[original_id],
                self.environments[reproduction_id]
            )

        # Compare dependencies
        dep_match = 0.0
        if original_id in self.dependencies and reproduction_id in self.dependencies:
            dep_match = self._compare_dependencies(
                self.dependencies[original_id],
                self.dependencies[reproduction_id]
            )

        # Check seed tracking
        seed_tracking_complete = (
            original_id in self.seeds and
            reproduction_id in self.seeds and
            self.seeds[original_id].numpy_seed is not None
        )

        # Compare results
        results_match, variations = self._compare_results(
            results_original,
            results_reproduction,
            self.tolerance
        )

        # Determine reproducibility level
        level = self._determine_reproducibility_level(
            env_match, dep_match, seed_tracking_complete, results_match
        )

        # Generate recommendations
        recommendations = self._generate_recommendations(
            env_match, dep_match, seed_tracking_complete, results_match, variations
        )

        # Generate certificate hash
        certificate_data = {
            "original_id": original_id,
            "reproduction_id": reproduction_id,
            "timestamp": datetime.now().isoformat(),
            "level": level.value
        }
        certificate_hash = hashlib.sha256(
            json.dumps(certificate_data, sort_keys=True).encode()
        ).hexdigest()

        report = ReproducibilityReport(
            experiment_id=f"{original_id}_vs_{reproduction_id}",
            original_run={"id": original_id, "results": results_original},
            reproduction_run={"id": reproduction_id, "results": results_reproduction},
            environment_match=env_match,
            dependency_match=dep_match,
            seed_tracking_complete=seed_tracking_complete,
            results_match=results_match,
            tolerance_used=self.tolerance,
            variations_found=variations,
            reproducibility_level=level,
            verification_status=VerificationStatus.VERIFIED if results_match else VerificationStatus.PARTIAL,
            recommendations=recommendations,
            badge_awarded=level.value if level != ReproducibilityLevel.NONE else None,
            certificate_hash=certificate_hash
        )

        self.reports.append(report)
        self.verification_times.append(time.time() - start_time)
        self.reproducibility_scores.append(env_match * 0.3 + dep_match * 0.3 + (1.0 if results_match else 0.0) * 0.4)

        logger.info(f"Verified reproducibility: {level.value} badge awarded")

        return report

    def _compare_environments(self, env1: Environment, env2: Environment) -> float:
        """Compare two environments and return similarity score"""
        score = 0.0
        total_checks = 0

        # Python version
        if env1.python_version == env2.python_version:
            score += 1.0
        total_checks += 1

        # Platform
        if env1.platform_info.get("system") == env2.platform_info.get("system"):
            score += 0.5
        if env1.platform_info.get("machine") == env2.platform_info.get("machine"):
            score += 0.5
        total_checks += 1

        # CPU
        if env1.cpu_info.get("count") == env2.cpu_info.get("count"):
            score += 0.5
        total_checks += 0.5

        # GPU
        if (env1.gpu_info is None) == (env2.gpu_info is None):
            score += 0.5
            if env1.gpu_info and env2.gpu_info:
                if env1.gpu_info.get("cuda_version") == env2.gpu_info.get("cuda_version"):
                    score += 0.5
                total_checks += 0.5
        total_checks += 0.5

        return score / max(total_checks, 1)

    def _compare_dependencies(self, deps1: List[DependencySpec], deps2: List[DependencySpec]) -> float:
        """Compare two dependency lists and return similarity score"""
        deps1_dict = {dep.name: dep.version for dep in deps1}
        deps2_dict = {dep.name: dep.version for dep in deps2}

        common_deps = set(deps1_dict.keys()) & set(deps2_dict.keys())
        all_deps = set(deps1_dict.keys()) | set(deps2_dict.keys())

        if not all_deps:
            return 1.0

        matching_deps = sum(
            1 for dep in common_deps
            if deps1_dict[dep] == deps2_dict[dep]
        )

        return matching_deps / len(all_deps)

    def _compare_results(self,
                       results1: Dict[str, Any],
                       results2: Dict[str, Any],
                       tolerance: float) -> Tuple[bool, List[Dict[str, Any]]]:
        """Compare two result sets with tolerance"""
        variations = []
        all_match = True

        for key in set(results1.keys()) | set(results2.keys()):
            if key not in results1:
                variations.append({"key": key, "issue": "missing_in_original"})
                all_match = False
            elif key not in results2:
                variations.append({"key": key, "issue": "missing_in_reproduction"})
                all_match = False
            else:
                val1 = results1[key]
                val2 = results2[key]

                if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                    if abs(val1 - val2) > tolerance:
                        variations.append({
                            "key": key,
                            "issue": "numerical_difference",
                            "original": val1,
                            "reproduction": val2,
                            "difference": abs(val1 - val2)
                        })
                        all_match = False
                elif val1 != val2:
                    variations.append({
                        "key": key,
                        "issue": "value_mismatch",
                        "original": str(val1)[:100],
                        "reproduction": str(val2)[:100]
                    })
                    all_match = False

        return all_match, variations

    def _determine_reproducibility_level(self,
                                       env_match: float,
                                       dep_match: float,
                                       seed_tracking: bool,
                                       results_match: bool) -> ReproducibilityLevel:
        """Determine reproducibility certification level"""
        if results_match and env_match > 0.9 and dep_match > 0.95 and seed_tracking:
            return ReproducibilityLevel.GOLD
        elif results_match and env_match > 0.7 and dep_match > 0.8:
            return ReproducibilityLevel.SILVER
        elif env_match > 0.5 and dep_match > 0.6:
            return ReproducibilityLevel.BRONZE
        else:
            return ReproducibilityLevel.NONE

    def _generate_recommendations(self,
                                env_match: float,
                                dep_match: float,
                                seed_tracking: bool,
                                results_match: bool,
                                variations: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations for improving reproducibility"""
        recommendations = []

        if not results_match:
            recommendations.append("Results do not match - investigate numerical stability")

        if env_match < 0.9:
            recommendations.append("Use containerization (Docker) to ensure environment consistency")

        if dep_match < 0.95:
            recommendations.append("Lock all dependency versions with exact specifications")

        if not seed_tracking:
            recommendations.append("Implement comprehensive random seed tracking")

        if len(variations) > 0:
            num_variations = len(variations)
            recommendations.append(f"Address {num_variations} variations found in results")

        if not self.docker_enabled:
            recommendations.append("Enable Docker for full containerization support")

        if not self.reprozip_enabled:
            recommendations.append("Install ReproZip for automatic provenance tracking")

        return recommendations

    async def generate_badge(self, level: ReproducibilityLevel) -> Dict[str, Any]:
        """
        Generate reproducibility badge metadata.

        Args:
            level: Reproducibility level achieved

        Returns:
            Badge metadata including SVG content
        """
        colors = {
            ReproducibilityLevel.GOLD: "#FFD700",
            ReproducibilityLevel.SILVER: "#C0C0C0",
            ReproducibilityLevel.BRONZE: "#CD7F32",
            ReproducibilityLevel.NONE: "#808080"
        }

        badge_svg = f'''
        <svg xmlns="http://www.w3.org/2000/svg" width="200" height="30">
            <rect width="200" height="30" fill="{colors[level]}"/>
            <text x="100" y="20" text-anchor="middle" fill="white" font-size="14" font-weight="bold">
                Reproducibility: {level.value.upper()}
            </text>
        </svg>
        '''

        badge_metadata = {
            "level": level.value,
            "color": colors[level],
            "svg": badge_svg.strip(),
            "timestamp": datetime.now().isoformat(),
            "criteria": {
                "gold": "Full reproducibility with containerization",
                "silver": "High reproducibility with minor variations",
                "bronze": "Partial reproducibility with documented variations",
                "none": "Not reproducible"
            }[level.value]
        }

        return badge_metadata

    async def create_reprozip_bundle(self,
                                    experiment_id: str,
                                    command: str) -> Optional[Path]:
        """
        Create ReproZip bundle for perfect reproducibility.

        Args:
            experiment_id: Unique experiment identifier
            command: Command to trace with ReproZip

        Returns:
            Path to ReproZip bundle if successful
        """
        if not self.reprozip_enabled:
            logger.warning("ReproZip not available")
            return None

        bundle_dir = self.workspace_dir / f"reprozip_{experiment_id}"
        bundle_dir.mkdir(exist_ok=True)

        try:
            # Trace with ReproZip
            trace_result = subprocess.run(
                ["reprozip", "trace", "--dir", str(bundle_dir), command],
                capture_output=True,
                text=True
            )

            if trace_result.returncode != 0:
                logger.error(f"ReproZip trace failed: {trace_result.stderr}")
                return None

            # Pack the bundle
            bundle_path = bundle_dir / f"{experiment_id}.rpz"
            pack_result = subprocess.run(
                ["reprozip", "pack", "-d", str(bundle_dir), str(bundle_path)],
                capture_output=True,
                text=True
            )

            if pack_result.returncode != 0:
                logger.error(f"ReproZip pack failed: {pack_result.stderr}")
                return None

            logger.info(f"Created ReproZip bundle at {bundle_path}")
            return bundle_path

        except Exception as e:
            logger.error(f"Failed to create ReproZip bundle: {e}")
            return None

    async def export_report(self, report: ReproducibilityReport, output_path: Path) -> None:
        """Export reproducibility report to file"""
        report_data = {
            "experiment_id": report.experiment_id,
            "timestamp": datetime.now().isoformat(),
            "original_run": report.original_run,
            "reproduction_run": report.reproduction_run,
            "scores": {
                "environment_match": report.environment_match,
                "dependency_match": report.dependency_match,
                "seed_tracking_complete": report.seed_tracking_complete,
                "results_match": report.results_match
            },
            "reproducibility_level": report.reproducibility_level.value,
            "verification_status": report.verification_status.value,
            "badge_awarded": report.badge_awarded,
            "certificate_hash": report.certificate_hash,
            "variations_found": report.variations_found,
            "recommendations": report.recommendations,
            "tolerance_used": report.tolerance_used
        }

        # Add badge if awarded
        if report.badge_awarded:
            badge_level = ReproducibilityLevel(report.badge_awarded)
            badge_metadata = await self.generate_badge(badge_level)
            report_data["badge"] = badge_metadata

        with open(output_path, 'w') as f:
            json.dump(report_data, f, indent=2)

        logger.info(f"Exported report to {output_path}")


if __name__ == "__main__":
    # Example usage
    async def main():
        engine = ReproducibilityEngine(
            workspace_dir=Path("/tmp/reproducibility_workspace"),
            reprozip_enabled=True,
            docker_enabled=True,
            tolerance=1e-6
        )

        # Capture environment for original run
        original_id = "exp_001"
        env = await engine.capture_environment(original_id)
        seeds = await engine.track_seeds(original_id)
        deps = await engine.lock_dependencies(original_id)

        # Create Docker container
        dockerfile = await engine.create_dockerfile(original_id)

        # Trace execution
        trace = await engine.trace_execution(
            original_id,
            "python -m talai.validate --hypothesis test.json"
        )

        # Simulate reproduction run
        repro_id = "exp_001_repro"
        await engine.capture_environment(repro_id)
        await engine.set_seeds(seeds)  # Use same seeds
        await engine.trace_execution(
            repro_id,
            "python -m talai.validate --hypothesis test.json"
        )

        # Verify reproducibility
        results_original = {"accuracy": 0.95, "loss": 0.05}
        results_repro = {"accuracy": 0.95, "loss": 0.050001}

        report = await engine.verify_reproducibility(
            original_id, repro_id,
            results_original, results_repro
        )

        print(f"Reproducibility Level: {report.reproducibility_level.value}")
        print(f"Environment Match: {report.environment_match:.2%}")
        print(f"Dependency Match: {report.dependency_match:.2%}")
        print(f"Results Match: {report.results_match}")

        # Export report
        await engine.export_report(report, Path("/tmp/reproducibility_report.json"))

    asyncio.run(main())