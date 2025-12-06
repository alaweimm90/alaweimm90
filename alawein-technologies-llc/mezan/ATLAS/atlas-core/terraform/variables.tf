variable "environment" {
  description = "Environment name (development, staging, production)"
  type        = string
  validation {
    condition     = contains(["development", "staging", "production"], var.environment)
    error_message = "Environment must be development, staging, or production."
  }
}

variable "aws_region" {
  description = "AWS region to deploy resources"
  type        = string
  default     = "us-west-2"
}

variable "cost_center" {
  description = "Cost center for billing"
  type        = string
  default     = "engineering"
}

# VPC Configuration
variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "private_subnet_cidrs" {
  description = "CIDR blocks for private subnets"
  type        = list(string)
  default     = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
}

variable "public_subnet_cidrs" {
  description = "CIDR blocks for public subnets"
  type        = list(string)
  default     = ["10.0.101.0/24", "10.0.102.0/24", "10.0.103.0/24"]
}

# EKS Configuration
variable "kubernetes_version" {
  description = "Kubernetes version for EKS cluster"
  type        = string
  default     = "1.28"
}

variable "cluster_endpoint_public_access_cidrs" {
  description = "List of CIDR blocks that can access the public API server endpoint"
  type        = list(string)
  default     = ["0.0.0.0/0"]
}

variable "node_group_min_size" {
  description = "Minimum number of nodes in the general node group"
  type        = number
  default     = 2
}

variable "node_group_max_size" {
  description = "Maximum number of nodes in the general node group"
  type        = number
  default     = 10
}

variable "node_group_desired_size" {
  description = "Desired number of nodes in the general node group"
  type        = number
  default     = 3
}

variable "node_instance_types" {
  description = "Instance types for general node group"
  type        = list(string)
  default     = ["t3.large", "t3a.large"]
}

variable "compute_instance_types" {
  description = "Instance types for compute-intensive node group"
  type        = list(string)
  default     = ["c5.2xlarge", "c5a.2xlarge"]
}

# RDS Configuration
variable "db_instance_class" {
  description = "Instance class for RDS PostgreSQL"
  type        = string
  default     = "db.t3.medium"
}

variable "db_allocated_storage" {
  description = "Allocated storage for RDS in GB"
  type        = number
  default     = 20
}

variable "db_max_allocated_storage" {
  description = "Maximum allocated storage for RDS autoscaling in GB"
  type        = number
  default     = 100
}

# ElastiCache Configuration
variable "redis_node_type" {
  description = "Node type for ElastiCache Redis"
  type        = string
  default     = "cache.t3.medium"
}

# Application Configuration
variable "mezan_atlas_config" {
  description = "Configuration for MEZAN ATLAS application"
  type = object({
    anthropic_api_key = string
    openai_api_key    = string
    redis_password    = string
    db_password       = string
    jwt_secret        = string
    session_secret    = string
  })
  sensitive = true
}

# Feature Flags
variable "enable_monitoring" {
  description = "Enable monitoring stack (Prometheus, Grafana)"
  type        = bool
  default     = true
}

variable "enable_logging" {
  description = "Enable centralized logging (ELK/EFK stack)"
  type        = bool
  default     = true
}

variable "enable_service_mesh" {
  description = "Enable service mesh (Istio)"
  type        = bool
  default     = false
}

variable "enable_backup" {
  description = "Enable automated backups"
  type        = bool
  default     = true
}

# Tags
variable "additional_tags" {
  description = "Additional tags to apply to all resources"
  type        = map(string)
  default     = {}
}