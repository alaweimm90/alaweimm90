# cloud_integration
**Module:** `Python/utils/cloud_integration.py`
## Overview
Cloud integration utilities for SciComp.
This module provides cloud computing integration including:
- AWS, Google Cloud, Azure integration
- Distributed computing with Dask
- Container deployment utilities
- Remote computation management
- Cloud storage interfaces
## Functions
### `deploy_to_cloud(provider, computation, data)`
Deploy computation to cloud with automatic resource management.
Args:
provider: Cloud provider
computation: Function to execute
data: Input data
**kwargs: Additional arguments
Returns:
Computation result
**Source:** [Line 601](Python/utils/cloud_integration.py#L601)
### `create_deployment_package(output_dir)`
Create complete deployment package.
Args:
output_dir: Output directory for package
Returns:
Path to created package
**Source:** [Line 625](Python/utils/cloud_integration.py#L625)
## Classes
### `CloudProvider`
Supported cloud providers.
**Class Source:** [Line 56](Python/utils/cloud_integration.py#L56)
### `ComputeResource`
Cloud compute resource specification.
**Class Source:** [Line 65](Python/utils/cloud_integration.py#L65)
### `CloudJob`
Cloud computation job specification.
**Class Source:** [Line 77](Python/utils/cloud_integration.py#L77)
### `CloudStorage`
Unified cloud storage interface.
#### Methods
##### `__init__(self, provider)`
Initialize cloud storage client.
Args:
provider: Cloud provider
**credentials: Provider-specific credentials
**Source:** [Line 92](Python/utils/cloud_integration.py#L92)
##### `upload_array(self, array, bucket, key)`
Upload numpy array to cloud storage.
Args:
array: Numpy array to upload
bucket: Storage bucket name
key: Object key/path
Returns:
Success status
**Source:** [Line 117](Python/utils/cloud_integration.py#L117)
##### `download_array(self, bucket, key)`
Download numpy array from cloud storage.
Args:
bucket: Storage bucket name
key: Object key/path
Returns:
Downloaded array or None if failed
**Source:** [Line 150](Python/utils/cloud_integration.py#L150)
##### `list_objects(self, bucket, prefix)`
List objects in cloud storage bucket.
Args:
bucket: Storage bucket name
prefix: Object key prefix filter
Returns:
List of object keys
**Source:** [Line 183](Python/utils/cloud_integration.py#L183)
**Class Source:** [Line 89](Python/utils/cloud_integration.py#L89)
### `DistributedComputing`
Distributed computing using Dask.
#### Methods
##### `__init__(self, scheduler_address)`
Initialize distributed computing client.
Args:
scheduler_address: Dask scheduler address (local if None)
**Source:** [Line 214](Python/utils/cloud_integration.py#L214)
##### `distribute_array_computation(self, arrays, computation, chunk_size)`
Distribute array computation across workers.
Args:
arrays: List of input arrays
computation: Function to apply to arrays
chunk_size: Dask array chunk size
Returns:
Computed result
**Source:** [Line 236](Python/utils/cloud_integration.py#L236)
##### `parallel_map(self, func, items, batch_size)`
Apply function to items in parallel.
Args:
func: Function to apply
items: Items to process
batch_size: Batch size for processing
Returns:
Results list
**Source:** [Line 262](Python/utils/cloud_integration.py#L262)
##### `distributed_optimization(self, objective, bounds, n_trials)`
Distributed optimization using random search.
Args:
objective: Objective function to minimize
bounds: Parameter bounds [(min1, max1), (min2, max2), ...]
n_trials: Number of optimization trials
Returns:
Optimization result
**Source:** [Line 292](Python/utils/cloud_integration.py#L292)
##### `close(self)`
Close distributed computing client.
**Source:** [Line 334](Python/utils/cloud_integration.py#L334)
**Class Source:** [Line 211](Python/utils/cloud_integration.py#L211)
### `CloudComputing`
Cloud computing resource management.
#### Methods
##### `__init__(self, provider)`
Initialize cloud computing manager.
Args:
provider: Cloud provider
**credentials: Provider-specific credentials
**Source:** [Line 343](Python/utils/cloud_integration.py#L343)
##### `launch_compute_instance(self, resource)`
Launch cloud compute instance.
Args:
resource: Compute resource specification
Returns:
Instance ID if successful
**Source:** [Line 361](Python/utils/cloud_integration.py#L361)
##### `_get_user_data_script(self)`
Generate user data script for cloud instance initialization.
**Source:** [Line 390](Python/utils/cloud_integration.py#L390)
##### `submit_job(self, job)`
Submit computation job to cloud.
Args:
job: Job specification
Returns:
Job ID
**Source:** [Line 409](Python/utils/cloud_integration.py#L409)
##### `get_job_status(self, job_id)`
Get status of submitted job.
Args:
job_id: Job identifier
Returns:
Job status
**Source:** [Line 427](Python/utils/cloud_integration.py#L427)
##### `get_job_result(self, job_id)`
Get result of completed job.
Args:
job_id: Job identifier
Returns:
Job result
**Source:** [Line 441](Python/utils/cloud_integration.py#L441)
**Class Source:** [Line 340](Python/utils/cloud_integration.py#L340)
### `ContainerDeployment`
Container deployment utilities.
#### Methods
##### `__init__(self)`
Initialize container deployment manager.
**Source:** [Line 461](Python/utils/cloud_integration.py#L461)
##### `_check_docker(self)`
Check if Docker is available.
**Source:** [Line 465](Python/utils/cloud_integration.py#L465)
##### `create_dockerfile(self, base_image, requirements)`
Create Dockerfile for Berkeley SciComp deployment.
Args:
base_image: Base Docker image
requirements: Additional Python requirements
Returns:
Dockerfile content
**Source:** [Line 473](Python/utils/cloud_integration.py#L473)
##### `build_image(self, tag, dockerfile_path)`
Build Docker image.
Args:
tag: Docker image tag
dockerfile_path: Path to Dockerfile
Returns:
Success status
**Source:** [Line 524](Python/utils/cloud_integration.py#L524)
##### `run_container(self, tag, ports, volumes, environment)`
Run Docker container.
Args:
tag: Docker image tag
ports: Port mappings {container_port: host_port}
volumes: Volume mappings {host_path: container_path}
environment: Environment variables
Returns:
Container ID if successful
**Source:** [Line 549](Python/utils/cloud_integration.py#L549)
**Class Source:** [Line 458](Python/utils/cloud_integration.py#L458)