# SuperTool Azure AKS Module Variables

# General Configuration
variable "environment" {
  description = "Environment name (dev, staging, production)"
  type        = string

  validation {
    condition     = can(regex("^(dev|staging|production)$", var.environment))
    error_message = "Environment must be dev, staging, or production."
  }
}

variable "azure_region" {
  description = "Azure region for resources"
  type        = string
  default     = "eastus"
}

variable "common_tags" {
  description = "Common tags to apply to all resources"
  type        = map(string)
  default     = {}
}

# Container Registry
variable "create_acr" {
  description = "Create Azure Container Registry"
  type        = bool
  default     = true
}

variable "acr_sku" {
  description = "ACR SKU (Basic, Standard, Premium)"
  type        = string
  default     = "Standard"

  validation {
    condition     = contains(["Basic", "Standard", "Premium"], var.acr_sku)
    error_message = "ACR SKU must be Basic, Standard, or Premium."
  }
}

# Networking
variable "vnet_address_space" {
  description = "Virtual network address space"
  type        = string
  default     = "10.1.0.0/16"
}

variable "aks_subnet_address_prefix" {
  description = "AKS subnet address prefix"
  type        = string
  default     = "10.1.1.0/24"
}

variable "dns_service_ip" {
  description = "DNS service IP address"
  type        = string
  default     = "10.2.0.10"
}

variable "service_cidr" {
  description = "Kubernetes service CIDR"
  type        = string
  default     = "10.2.0.0/16"
}

# AKS Cluster
variable "kubernetes_version" {
  description = "Kubernetes version"
  type        = string
  default     = "1.28"
}

variable "node_count" {
  description = "Initial number of nodes"
  type        = number
  default     = 3

  validation {
    condition     = var.node_count >= 1 && var.node_count <= 100
    error_message = "Node count must be between 1 and 100."
  }
}

variable "node_vm_size" {
  description = "VM size for nodes"
  type        = string
  default     = "Standard_D2s_v3"
}

variable "node_os_disk_size" {
  description = "OS disk size in GB"
  type        = number
  default     = 100

  validation {
    condition     = var.node_os_disk_size >= 30 && var.node_os_disk_size <= 2048
    error_message = "OS disk size must be between 30 and 2048 GB."
  }
}

variable "enable_autoscaling" {
  description = "Enable cluster autoscaling"
  type        = bool
  default     = true
}

variable "min_node_count" {
  description = "Minimum number of nodes for autoscaling"
  type        = number
  default     = 2

  validation {
    condition     = var.min_node_count >= 1
    error_message = "Minimum node count must be at least 1."
  }
}

variable "max_node_count" {
  description = "Maximum number of nodes for autoscaling"
  type        = number
  default     = 10

  validation {
    condition     = var.max_node_count >= var.min_node_count
    error_message = "Maximum node count must be greater than or equal to minimum."
  }
}

variable "automatic_channel_upgrade" {
  description = "Automatic channel upgrade (patch, stable, rapid, node-image, none)"
  type        = string
  default     = "stable"

  validation {
    condition     = contains(["patch", "stable", "rapid", "node-image", "none"], var.automatic_channel_upgrade)
    error_message = "Must be patch, stable, rapid, node-image, or none."
  }
}

# Azure AD Integration
variable "admin_group_object_ids" {
  description = "Azure AD admin group object IDs"
  type        = list(string)
  default     = []
}

# Monitoring
variable "enable_container_insights" {
  description = "Enable Azure Container Insights"
  type        = bool
  default     = true
}

variable "log_retention_days" {
  description = "Log retention in days"
  type        = number
  default     = 30

  validation {
    condition     = contains([30, 60, 90, 120, 180, 270, 365, 550, 730], var.log_retention_days)
    error_message = "Log retention must be 30, 60, 90, 120, 180, 270, 365, 550, or 730 days."
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

# Application Configuration
variable "container_image" {
  description = "Container image repository"
  type        = string
  default     = "ghcr.io/username/supertool"
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

# Network Policy
variable "enable_network_policy" {
  description = "Enable network policies"
  type        = bool
  default     = true
}
