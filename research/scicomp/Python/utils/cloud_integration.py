"""
Cloud integration utilities for SciComp.
This module provides cloud computing integration including:
- AWS, Google Cloud, Azure integration
- Distributed computing with Dask
- Container deployment utilities
- Remote computation management
- Cloud storage interfaces
"""
import numpy as np
import warnings
import os
import json
import subprocess
from typing import Union, List, Tuple, Optional, Dict, Any, Callable
from dataclasses import dataclass
from enum import Enum
import tempfile
import shutil
# Optional cloud and distributed computing libraries
try:
    import dask
    import dask.array as da
    from dask.distributed import Client, as_completed
    from dask import delayed
    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False
    warnings.warn("Dask not available. Distributed computing features limited.")
try:
    import boto3
    from botocore.exceptions import NoCredentialsError, BotoCoreError
    AWS_AVAILABLE = True
except ImportError:
    AWS_AVAILABLE = False
try:
    from google.cloud import storage as gcs
    from google.cloud import compute_v1
    GCP_AVAILABLE = True
except ImportError:
    GCP_AVAILABLE = False
try:
    from azure.storage.blob import BlobServiceClient
    from azure.identity import DefaultAzureCredential
    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False
class CloudProvider(Enum):
    """Supported cloud providers."""
    AWS = "aws"
    GCP = "gcp"
    AZURE = "azure"
    LOCAL = "local"
@dataclass
class ComputeResource:
    """Cloud compute resource specification."""
    provider: CloudProvider
    instance_type: str
    region: str
    cpu_cores: int
    memory_gb: float
    gpu_count: int = 0
    spot_instance: bool = False
@dataclass
class CloudJob:
    """Cloud computation job specification."""
    job_id: str
    function: Callable
    args: Tuple
    kwargs: Dict
    resource: ComputeResource
    status: str = "pending"
    result: Any = None
    error: Optional[str] = None
class CloudStorage:
    """Unified cloud storage interface."""
    def __init__(self, provider: CloudProvider, **credentials):
        """
        Initialize cloud storage client.
        Args:
            provider: Cloud provider
            **credentials: Provider-specific credentials
        """
        self.provider = provider
        self.client = None
        try:
            if provider == CloudProvider.AWS and AWS_AVAILABLE:
                self.client = boto3.client('s3', **credentials)
            elif provider == CloudProvider.GCP and GCP_AVAILABLE:
                self.client = gcs.Client(**credentials)
            elif provider == CloudProvider.AZURE and AZURE_AVAILABLE:
                credential = DefaultAzureCredential()
                account_url = credentials.get('account_url')
                self.client = BlobServiceClient(account_url=account_url, credential=credential)
            else:
                warnings.warn(f"Cloud provider {provider} not available or supported")
        except Exception as e:
            warnings.warn(f"Failed to initialize {provider} client: {e}")
    def upload_array(self, array: np.ndarray, bucket: str, key: str) -> bool:
        """
        Upload numpy array to cloud storage.
        Args:
            array: Numpy array to upload
            bucket: Storage bucket name
            key: Object key/path
        Returns:
            Success status
        """
        try:
            # Serialize array to bytes
            with tempfile.NamedTemporaryFile() as tmp:
                np.save(tmp, array)
                tmp.seek(0)
                data = tmp.read()
            if self.provider == CloudProvider.AWS:
                self.client.put_object(Bucket=bucket, Key=key, Body=data)
            elif self.provider == CloudProvider.GCP:
                blob = self.client.bucket(bucket).blob(key)
                blob.upload_from_string(data)
            elif self.provider == CloudProvider.AZURE:
                blob_client = self.client.get_blob_client(container=bucket, blob=key)
                blob_client.upload_blob(data, overwrite=True)
            return True
        except Exception as e:
            warnings.warn(f"Failed to upload array: {e}")
            return False
    def download_array(self, bucket: str, key: str) -> Optional[np.ndarray]:
        """
        Download numpy array from cloud storage.
        Args:
            bucket: Storage bucket name
            key: Object key/path
        Returns:
            Downloaded array or None if failed
        """
        try:
            if self.provider == CloudProvider.AWS:
                response = self.client.get_object(Bucket=bucket, Key=key)
                data = response['Body'].read()
            elif self.provider == CloudProvider.GCP:
                blob = self.client.bucket(bucket).blob(key)
                data = blob.download_as_bytes()
            elif self.provider == CloudProvider.AZURE:
                blob_client = self.client.get_blob_client(container=bucket, blob=key)
                data = blob_client.download_blob().readall()
            # Deserialize array
            with tempfile.NamedTemporaryFile() as tmp:
                tmp.write(data)
                tmp.seek(0)
                array = np.load(tmp)
            return array
        except Exception as e:
            warnings.warn(f"Failed to download array: {e}")
            return None
    def list_objects(self, bucket: str, prefix: str = "") -> List[str]:
        """
        List objects in cloud storage bucket.
        Args:
            bucket: Storage bucket name
            prefix: Object key prefix filter
        Returns:
            List of object keys
        """
        try:
            if self.provider == CloudProvider.AWS:
                response = self.client.list_objects_v2(Bucket=bucket, Prefix=prefix)
                return [obj['Key'] for obj in response.get('Contents', [])]
            elif self.provider == CloudProvider.GCP:
                blobs = self.client.list_blobs(bucket, prefix=prefix)
                return [blob.name for blob in blobs]
            elif self.provider == CloudProvider.AZURE:
                blob_list = self.client.get_container_client(bucket).list_blobs(name_starts_with=prefix)
                return [blob.name for blob in blob_list]
            return []
        except Exception as e:
            warnings.warn(f"Failed to list objects: {e}")
            return []
class DistributedComputing:
    """Distributed computing using Dask."""
    def __init__(self, scheduler_address: Optional[str] = None):
        """
        Initialize distributed computing client.
        Args:
            scheduler_address: Dask scheduler address (local if None)
        """
        if not DASK_AVAILABLE:
            raise ImportError("Dask required for distributed computing")
        try:
            if scheduler_address:
                self.client = Client(scheduler_address)
            else:
                # Start local cluster
                self.client = Client(processes=True, threads_per_worker=2)
            print(f"Dask dashboard available at: {self.client.dashboard_link}")
        except Exception as e:
            warnings.warn(f"Failed to start Dask client: {e}")
            self.client = None
    def distribute_array_computation(self, arrays: List[np.ndarray],
                                   computation: Callable,
                                   chunk_size: str = "100MB") -> np.ndarray:
        """
        Distribute array computation across workers.
        Args:
            arrays: List of input arrays
            computation: Function to apply to arrays
            chunk_size: Dask array chunk size
        Returns:
            Computed result
        """
        if not self.client:
            raise RuntimeError("Dask client not available")
        # Convert to Dask arrays
        dask_arrays = [da.from_array(arr, chunks=chunk_size) for arr in arrays]
        # Apply computation
        result = computation(*dask_arrays)
        # Compute and return
        return result.compute()
    def parallel_map(self, func: Callable, items: List[Any],
                    batch_size: Optional[int] = None) -> List[Any]:
        """
        Apply function to items in parallel.
        Args:
            func: Function to apply
            items: Items to process
            batch_size: Batch size for processing
        Returns:
            Results list
        """
        if not self.client:
            raise RuntimeError("Dask client not available")
        # Create delayed computations
        delayed_results = [delayed(func)(item) for item in items]
        # Compute in parallel
        if batch_size:
            results = []
            for i in range(0, len(delayed_results), batch_size):
                batch = delayed_results[i:i+batch_size]
                batch_results = dask.compute(*batch)
                results.extend(batch_results)
            return results
        else:
            return dask.compute(*delayed_results)
    def distributed_optimization(self, objective: Callable, bounds: List[Tuple],
                                n_trials: int = 100) -> Dict[str, Any]:
        """
        Distributed optimization using random search.
        Args:
            objective: Objective function to minimize
            bounds: Parameter bounds [(min1, max1), (min2, max2), ...]
            n_trials: Number of optimization trials
        Returns:
            Optimization result
        """
        if not self.client:
            raise RuntimeError("Dask client not available")
        # Generate random parameter combinations
        np.random.seed(42)
        params_list = []
        for _ in range(n_trials):
            params = []
            for low, high in bounds:
                params.append(np.random.uniform(low, high))
            params_list.append(params)
        # Evaluate objective function in parallel
        delayed_results = [delayed(objective)(params) for params in params_list]
        objective_values = dask.compute(*delayed_results)
        # Find best result
        best_idx = np.argmin(objective_values)
        best_params = params_list[best_idx]
        best_value = objective_values[best_idx]
        return {
            'best_params': best_params,
            'best_value': best_value,
            'n_trials': n_trials,
            'all_params': params_list,
            'all_values': list(objective_values)
        }
    def close(self):
        """Close distributed computing client."""
        if self.client:
            self.client.close()
class CloudComputing:
    """Cloud computing resource management."""
    def __init__(self, provider: CloudProvider, **credentials):
        """
        Initialize cloud computing manager.
        Args:
            provider: Cloud provider
            **credentials: Provider-specific credentials
        """
        self.provider = provider
        self.credentials = credentials
        self.active_jobs = {}
        # Initialize cloud clients
        if provider == CloudProvider.AWS and AWS_AVAILABLE:
            self.ec2_client = boto3.client('ec2', **credentials)
        elif provider == CloudProvider.GCP and GCP_AVAILABLE:
            self.compute_client = compute_v1.InstancesClient(**credentials)
    def launch_compute_instance(self, resource: ComputeResource) -> Optional[str]:
        """
        Launch cloud compute instance.
        Args:
            resource: Compute resource specification
        Returns:
            Instance ID if successful
        """
        try:
            if self.provider == CloudProvider.AWS and hasattr(self, 'ec2_client'):
                response = self.ec2_client.run_instances(
                    ImageId='ami-0c55b159cbfafe1d0',  # Example AMI
                    MinCount=1,
                    MaxCount=1,
                    InstanceType=resource.instance_type,
                    SecurityGroupIds=['default'],
                    UserData=self._get_user_data_script()
                )
                return response['Instances'][0]['InstanceId']
            # Add other cloud providers here
            return None
        except Exception as e:
            warnings.warn(f"Failed to launch instance: {e}")
            return None
    def _get_user_data_script(self) -> str:
        """Generate user data script for cloud instance initialization."""
        return """#!/bin/bash
        # Update system
        yum update -y
        # Install Python and scientific libraries
        yum install -y python3 python3-pip
        pip3 install numpy scipy matplotlib scikit-learn dask
        # Install SciComp
        git clone https://github.com/berkeley/scicomp.git
        cd scicomp
        pip3 install -e .
        # Start Dask scheduler
        python3 -m dask.distributed scheduler &
        """
    def submit_job(self, job: CloudJob) -> str:
        """
        Submit computation job to cloud.
        Args:
            job: Job specification
        Returns:
            Job ID
        """
        # In a real implementation, this would submit the job to a cloud compute service
        # For now, simulate job submission
        job.status = "submitted"
        self.active_jobs[job.job_id] = job
        print(f"Job {job.job_id} submitted to {self.provider.value}")
        return job.job_id
    def get_job_status(self, job_id: str) -> str:
        """
        Get status of submitted job.
        Args:
            job_id: Job identifier
        Returns:
            Job status
        """
        if job_id in self.active_jobs:
            return self.active_jobs[job_id].status
        return "not_found"
    def get_job_result(self, job_id: str) -> Any:
        """
        Get result of completed job.
        Args:
            job_id: Job identifier
        Returns:
            Job result
        """
        if job_id in self.active_jobs:
            job = self.active_jobs[job_id]
            if job.status == "completed":
                return job.result
        return None
class ContainerDeployment:
    """Container deployment utilities."""
    def __init__(self):
        """Initialize container deployment manager."""
        self.docker_available = self._check_docker()
    def _check_docker(self) -> bool:
        """Check if Docker is available."""
        try:
            subprocess.run(['docker', '--version'], capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
    def create_dockerfile(self, base_image: str = "python:3.9",
                         requirements: Optional[List[str]] = None) -> str:
        """
        Create Dockerfile for Berkeley SciComp deployment.
        Args:
            base_image: Base Docker image
            requirements: Additional Python requirements
        Returns:
            Dockerfile content
        """
        if requirements is None:
            requirements = [
                "numpy", "scipy", "matplotlib", "scikit-learn",
                "dask", "jupyter", "numba"
            ]
        dockerfile_content = f"""FROM {base_image}
# Set working directory
WORKDIR /app
# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc g++ gfortran \\
    libopenblas-dev liblapack-dev \\
    && rm -rf /var/lib/apt/lists/*
# Install Python dependencies
RUN pip install --no-cache-dir \\
    {' '.join(requirements)}
# Copy SciComp
COPY . /app/
# Install SciComp
RUN pip install -e .
# Set environment variables
ENV PYTHONPATH=/app
ENV BERKELEY_SCICOMP_CONFIG=/app/berkeley_config.json
# Expose ports for Jupyter and Dask
EXPOSE 8888 8787
# Default command
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
"""
        return dockerfile_content
    def build_image(self, tag: str = "berkeley-scicomp:latest",
                   dockerfile_path: str = "Dockerfile") -> bool:
        """
        Build Docker image.
        Args:
            tag: Docker image tag
            dockerfile_path: Path to Dockerfile
        Returns:
            Success status
        """
        if not self.docker_available:
            warnings.warn("Docker not available")
            return False
        try:
            subprocess.run([
                'docker', 'build', '-t', tag, '-f', dockerfile_path, '.'
            ], check=True)
            return True
        except subprocess.CalledProcessError as e:
            warnings.warn(f"Docker build failed: {e}")
            return False
    def run_container(self, tag: str = "berkeley-scicomp:latest",
                     ports: Dict[int, int] = None,
                     volumes: Dict[str, str] = None,
                     environment: Dict[str, str] = None) -> Optional[str]:
        """
        Run Docker container.
        Args:
            tag: Docker image tag
            ports: Port mappings {container_port: host_port}
            volumes: Volume mappings {host_path: container_path}
            environment: Environment variables
        Returns:
            Container ID if successful
        """
        if not self.docker_available:
            warnings.warn("Docker not available")
            return None
        if ports is None:
            ports = {8888: 8888, 8787: 8787}
        cmd = ['docker', 'run', '-d']
        # Add port mappings
        for container_port, host_port in ports.items():
            cmd.extend(['-p', f"{host_port}:{container_port}"])
        # Add volume mappings
        if volumes:
            for host_path, container_path in volumes.items():
                cmd.extend(['-v', f"{host_path}:{container_path}"])
        # Add environment variables
        if environment:
            for key, value in environment.items():
                cmd.extend(['-e', f"{key}={value}"])
        cmd.append(tag)
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            container_id = result.stdout.strip()
            print(f"Container started: {container_id[:12]}")
            return container_id
        except subprocess.CalledProcessError as e:
            warnings.warn(f"Failed to run container: {e}")
            return None
# Convenience functions
def deploy_to_cloud(provider: CloudProvider, computation: Callable,
                   data: np.ndarray, **kwargs) -> Any:
    """
    Deploy computation to cloud with automatic resource management.
    Args:
        provider: Cloud provider
        computation: Function to execute
        data: Input data
        **kwargs: Additional arguments
    Returns:
        Computation result
    """
    if provider == CloudProvider.LOCAL and DASK_AVAILABLE:
        # Use local Dask cluster
        with DistributedComputing() as dc:
            return dc.distribute_array_computation([data], computation)
    else:
        # Cloud deployment would require more setup
        warnings.warn(f"Cloud deployment for {provider} not fully implemented")
        return computation(data)
def create_deployment_package(output_dir: str = "deployment") -> str:
    """
    Create complete deployment package.
    Args:
        output_dir: Output directory for package
    Returns:
        Path to created package
    """
    os.makedirs(output_dir, exist_ok=True)
    # Create container deployment
    container_deploy = ContainerDeployment()
    # Generate Dockerfile
    dockerfile_content = container_deploy.create_dockerfile()
    with open(os.path.join(output_dir, "Dockerfile"), "w") as f:
        f.write(dockerfile_content)
    # Create docker-compose.yml
    compose_content = """version: '3.8'
services:
  berkeley-scicomp:
    build: .
    ports:
      - "8888:8888"
      - "8787:8787"
    volumes:
      - ./data:/app/data
      - ./notebooks:/app/notebooks
    environment:
      - JUPYTER_TOKEN=berkeley
      - DASK_SCHEDULER_ADDRESS=tcp://scheduler:8786
  scheduler:
    image: daskdev/dask:latest
    command: dask-scheduler
    ports:
      - "8786:8786"
      - "8787:8787"
  worker:
    image: daskdev/dask:latest
    command: dask-worker tcp://scheduler:8786
    depends_on:
      - scheduler
"""
    with open(os.path.join(output_dir, "docker-compose.yml"), "w") as f:
        f.write(compose_content)
    # Create deployment README
    readme_content = """# SciComp Deployment
## Docker Deployment
1. Build the image:
   ```bash
   docker build -t berkeley-scicomp .
   ```
2. Run with docker-compose:
   ```bash
   docker-compose up
   ```
3. Access Jupyter Lab at: http://localhost:8888
   Token: berkeley
4. Access Dask Dashboard at: http://localhost:8787
## Cloud Deployment
See cloud-specific deployment guides in the docs/ directory.
"""
    with open(os.path.join(output_dir, "README.md"), "w") as f:
        f.write(readme_content)
    print(f"Deployment package created in: {output_dir}")
    return output_dir