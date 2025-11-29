# SuperTool GCP GKE Module Variables

# General Configuration
variable "environment" {
  description = "Environment name (dev, staging, production)"
  type        = string

  validation {
    condition     = can(regex("^(dev|staging|production)$", var.environment))
    error_message = "Environment must be dev, staging, or production."
  }
}

variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "region" {
  description = "GCP region"
  type        = string
  default     = "us-central1"
}

variable "zone" {
  description = "GCP zone (used if not regional cluster)"
  type        = string
  default     = "us-central1-a"
}

variable "regional_cluster" {
  description = "Create a regional cluster (true) or zonal (false)"
  type        = bool
  default     = true
}

variable "common_labels" {
  description = "Common labels to apply to all resources"
  type        = map(string)
  default     = {}
}

# Networking
variable "subnet_cidr" {
  description = "Subnet CIDR range"
  type        = string
  default     = "10.0.0.0/24"
}

variable "pods_cidr" {
  description = "Pods secondary CIDR range"
  type        = string
  default     = "10.1.0.0/16"
}

variable "services_cidr" {
  description = "Services secondary CIDR range"
  type        = string
  default     = "10.2.0.0/16"
}

variable "enable_private_nodes" {
  description = "Enable private nodes"
  type        = bool
  default     = true
}

variable "enable_private_endpoint" {
  description = "Enable private endpoint"
  type        = bool
  default     = false
}

variable "master_ipv4_cidr_block" {
  description = "Master IPv4 CIDR block"
  type        = string
  default     = "172.16.0.0/28"
}

variable "master_authorized_networks" {
  description = "Master authorized networks"
  type = list(object({
    cidr_block   = string
    display_name = string
  }))
  default = []
}

# Artifact Registry
variable "enable_artifact_registry" {
  description = "Create Artifact Registry repository"
  type        = bool
  default     = true
}

# GKE Cluster Configuration
variable "release_channel" {
  description = "GKE release channel (RAPID, REGULAR, STABLE, UNSPECIFIED)"
  type        = string
  default     = "REGULAR"

  validation {
    condition     = contains(["RAPID", "REGULAR", "STABLE", "UNSPECIFIED"], var.release_channel)
    error_message = "Release channel must be RAPID, REGULAR, STABLE, or UNSPECIFIED."
  }
}

variable "maintenance_start_time" {
  description = "Maintenance window start time (HH:MM format)"
  type        = string
  default     = "03:00"
}

variable "enable_network_policy" {
  description = "Enable network policy"
  type        = bool
  default     = true
}

variable "enable_binary_authorization" {
  description = "Enable binary authorization"
  type        = bool
  default     = false
}

variable "enable_cloud_logging" {
  description = "Enable Cloud Logging"
  type        = bool
  default     = true
}

variable "enable_cloud_monitoring" {
  description = "Enable Cloud Monitoring"
  type        = bool
  default     = true
}

# Cluster Autoscaling
variable "enable_cluster_autoscaling" {
  description = "Enable cluster autoscaling"
  type        = bool
  default     = true
}

variable "cluster_autoscaling_cpu_min" {
  description = "Minimum CPU cores for cluster autoscaling"
  type        = number
  default     = 1
}

variable "cluster_autoscaling_cpu_max" {
  description = "Maximum CPU cores for cluster autoscaling"
  type        = number
  default     = 100
}

variable "cluster_autoscaling_memory_min" {
  description = "Minimum memory GB for cluster autoscaling"
  type        = number
  default     = 1
}

variable "cluster_autoscaling_memory_max" {
  description = "Maximum memory GB for cluster autoscaling"
  type        = number
  default     = 1000
}

variable "cluster_autoscaling_profile" {
  description = "Cluster autoscaling profile (BALANCED or OPTIMIZE_UTILIZATION)"
  type        = string
  default     = "BALANCED"

  validation {
    condition     = contains(["BALANCED", "OPTIMIZE_UTILIZATION"], var.cluster_autoscaling_profile)
    error_message = "Profile must be BALANCED or OPTIMIZE_UTILIZATION."
  }
}

# Node Pool Configuration
variable "machine_type" {
  description = "Machine type for nodes"
  type        = string
  default     = "e2-standard-4"
}

variable "disk_size_gb" {
  description = "Disk size in GB"
  type        = number
  default     = 100

  validation {
    condition     = var.disk_size_gb >= 10 && var.disk_size_gb <= 65536
    error_message = "Disk size must be between 10 and 65536 GB."
  }
}

variable "disk_type" {
  description = "Disk type (pd-standard, pd-balanced, pd-ssd)"
  type        = string
  default     = "pd-standard"

  validation {
    condition     = contains(["pd-standard", "pd-balanced", "pd-ssd"], var.disk_type)
    error_message = "Disk type must be pd-standard, pd-balanced, or pd-ssd."
  }
}

variable "node_count" {
  description = "Initial node count (for zonal cluster)"
  type        = number
  default     = 3

  validation {
    condition     = var.node_count >= 1 && var.node_count <= 1000
    error_message = "Node count must be between 1 and 1000."
  }
}

variable "node_count_per_zone" {
  description = "Node count per zone (for regional cluster)"
  type        = number
  default     = 1

  validation {
    condition     = var.node_count_per_zone >= 1
    error_message = "Node count per zone must be at least 1."
  }
}

variable "autoscaling_min_nodes" {
  description = "Minimum nodes for autoscaling"
  type        = number
  default     = 1

  validation {
    condition     = var.autoscaling_min_nodes >= 0
    error_message = "Minimum nodes must be at least 0."
  }
}

variable "autoscaling_max_nodes" {
  description = "Maximum nodes for autoscaling"
  type        = number
  default     = 10

  validation {
    condition     = var.autoscaling_max_nodes >= var.autoscaling_min_nodes
    error_message = "Maximum nodes must be >= minimum nodes."
  }
}

variable "node_tags" {
  description = "Network tags for nodes"
  type        = list(string)
  default     = ["gke-node"]
}

variable "use_preemptible_nodes" {
  description = "Use preemptible nodes (for dev only)"
  type        = bool
  default     = false
}

variable "use_spot_nodes" {
  description = "Use spot nodes (for dev only)"
  type        = bool
  default     = false
}

variable "max_surge" {
  description = "Maximum surge during node pool upgrade"
  type        = number
  default     = 1
}

variable "max_unavailable" {
  description = "Maximum unavailable during node pool upgrade"
  type        = number
  default     = 0
}

# Application Configuration
variable "container_image" {
  description = "Container image repository"
  type        = string
  default     = "gcr.io/project-id/supertool"
}

variable "container_tag" {
  description = "Container image tag"
  type        = string
  default     = "latest"
}

variable "app_version" {
  description = "Application version"
  type        = string
  default     = "1.0.0"
}

variable "container_port" {
  description = "Container port"
  type        = number
  default     = 8080

  validation {
    condition     = var.container_port > 0 && var.container_port < 65536
    error_message = "Container port must be between 1 and 65535."
  }
}

variable "replica_count" {
  description = "Number of pod replicas"
  type        = number
  default     = 3

  validation {
    condition     = var.replica_count >= 1
    error_message = "Replica count must be at least 1."
  }
}

variable "log_level" {
  description = "Application log level"
  type        = string
  default     = "info"

  validation {
    condition     = contains(["debug", "info", "warn", "error"], var.log_level)
    error_message = "Log level must be debug, info, warn, or error."
  }
}

# Container Resources
variable "container_cpu_request" {
  description = "CPU request (e.g., 100m, 1)"
  type        = string
  default     = "250m"
}

variable "container_memory_request" {
  description = "Memory request (e.g., 128Mi, 1Gi)"
  type        = string
  default     = "512Mi"
}

variable "container_cpu_limit" {
  description = "CPU limit (e.g., 100m, 1)"
  type        = string
  default     = "1000m"
}

variable "container_memory_limit" {
  description = "Memory limit (e.g., 128Mi, 1Gi)"
  type        = string
  default     = "2Gi"
}

# Service Configuration
variable "service_type" {
  description = "Kubernetes service type (ClusterIP, LoadBalancer, NodePort)"
  type        = string
  default     = "LoadBalancer"

  validation {
    condition     = contains(["ClusterIP", "LoadBalancer", "NodePort"], var.service_type)
    error_message = "Service type must be ClusterIP, LoadBalancer, or NodePort."
  }
}

variable "load_balancer_type" {
  description = "Load balancer type (External or Internal)"
  type        = string
  default     = "External"

  validation {
    condition     = contains(["External", "Internal"], var.load_balancer_type)
    error_message = "Load balancer type must be External or Internal."
  }
}

# Horizontal Pod Autoscaling
variable "enable_hpa" {
  description = "Enable Horizontal Pod Autoscaler"
  type        = bool
  default     = true
}

variable "hpa_min_replicas" {
  description = "HPA minimum replicas"
  type        = number
  default     = 3

  validation {
    condition     = var.hpa_min_replicas >= 1
    error_message = "HPA minimum replicas must be at least 1."
  }
}

variable "hpa_max_replicas" {
  description = "HPA maximum replicas"
  type        = number
  default     = 20

  validation {
    condition     = var.hpa_max_replicas >= var.hpa_min_replicas
    error_message = "HPA maximum must be >= minimum."
  }
}

variable "hpa_cpu_target" {
  description = "HPA CPU utilization target percentage"
  type        = number
  default     = 70

  validation {
    condition     = var.hpa_cpu_target > 0 && var.hpa_cpu_target <= 100
    error_message = "HPA CPU target must be between 1 and 100."
  }
}

variable "hpa_memory_target" {
  description = "HPA memory utilization target percentage"
  type        = number
  default     = 80

  validation {
    condition     = var.hpa_memory_target > 0 && var.hpa_memory_target <= 100
    error_message = "HPA memory target must be between 1 and 100."
  }
}

# Pod Disruption Budget
variable "pdb_min_available" {
  description = "Minimum available pods during disruptions"
  type        = number
  default     = 2

  validation {
    condition     = var.pdb_min_available >= 1
    error_message = "PDB minimum available must be at least 1."
  }
}
