variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "environment" {
  description = "Environment name"
  type        = string
  validation {
    condition     = contains(["development", "staging", "production"], var.environment)
    error_message = "Environment must be development, staging, or production."
  }
}

variable "project_name" {
  description = "Project name"
  type        = string
  default     = "crazyideas"
}

variable "owner" {
  description = "Owner of the infrastructure"
  type        = string
  default     = "DevOps Team"
}

variable "cost_center" {
  description = "Cost center for billing"
  type        = string
  default     = "Engineering"
}

# VPC Configuration
variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "public_subnets" {
  description = "CIDR blocks for public subnets"
  type        = list(string)
  default     = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
}

variable "private_subnets" {
  description = "CIDR blocks for private subnets"
  type        = list(string)
  default     = ["10.0.10.0/24", "10.0.11.0/24", "10.0.12.0/24"]
}

variable "database_subnets" {
  description = "CIDR blocks for database subnets"
  type        = list(string)
  default     = ["10.0.20.0/24", "10.0.21.0/24", "10.0.22.0/24"]
}

variable "enable_nat_gateway" {
  description = "Enable NAT Gateway for private subnets"
  type        = bool
  default     = true
}

variable "enable_vpn_gateway" {
  description = "Enable VPN Gateway"
  type        = bool
  default     = false
}

# Database Configuration
variable "database_name" {
  description = "Name of the database"
  type        = string
  default     = "crazyideas"
}

variable "database_username" {
  description = "Master username for database"
  type        = string
  default     = "postgres"
  sensitive   = true
}

variable "database_password" {
  description = "Master password for database"
  type        = string
  sensitive   = true
}

variable "rds_instance_class" {
  description = "RDS instance class"
  type        = string
  default     = "db.t3.medium"
}

variable "rds_allocated_storage" {
  description = "Allocated storage for RDS in GB"
  type        = number
  default     = 100
}

variable "backup_retention_period" {
  description = "Backup retention period in days"
  type        = number
  default     = 30
}

# Redis Configuration
variable "redis_node_type" {
  description = "ElastiCache Redis node type"
  type        = string
  default     = "cache.t3.micro"
}

variable "redis_num_nodes" {
  description = "Number of Redis cache nodes"
  type        = number
  default     = 1
}

# ECS Configuration
variable "container_cpu" {
  description = "CPU units for container (1024 = 1 vCPU)"
  type        = number
  default     = 256
}

variable "container_memory" {
  description = "Memory for container in MB"
  type        = number
  default     = 512
}

variable "desired_count" {
  description = "Desired number of tasks"
  type        = number
  default     = 2
}

variable "min_capacity" {
  description = "Minimum number of tasks for auto-scaling"
  type        = number
  default     = 1
}

variable "max_capacity" {
  description = "Maximum number of tasks for auto-scaling"
  type        = number
  default     = 10
}

variable "api_container_cpu" {
  description = "CPU units for API container"
  type        = number
  default     = 512
}

variable "api_container_memory" {
  description = "Memory for API container in MB"
  type        = number
  default     = 1024
}

variable "api_desired_count" {
  description = "Desired number of API tasks"
  type        = number
  default     = 3
}

variable "api_min_capacity" {
  description = "Minimum number of API tasks"
  type        = number
  default     = 2
}

variable "api_max_capacity" {
  description = "Maximum number of API tasks"
  type        = number
  default     = 20
}

# Container Images
variable "ecr_repository_url" {
  description = "ECR repository URL"
  type        = string
}

variable "image_tag" {
  description = "Docker image tag"
  type        = string
  default     = "latest"
}

# Domain Configuration
variable "domain_names" {
  description = "List of domain names"
  type        = list(string)
  default     = ["crazyideas.com"]
}

variable "acm_certificate_arn" {
  description = "ACM certificate ARN for SSL"
  type        = string
}

# Security Configuration
variable "jwt_secret" {
  description = "JWT secret key"
  type        = string
  sensitive   = true
}

variable "cors_origins" {
  description = "Allowed CORS origins"
  type        = list(string)
  default = [
    "https://ghost-researcher.crazyideas.com",
    "https://scientific-tinder.crazyideas.com",
    "https://chaos-engine.crazyideas.com"
  ]
}

variable "waf_rate_limit" {
  description = "WAF rate limit per 5 minutes"
  type        = number
  default     = 2000
}

# Monitoring Configuration
variable "alert_email" {
  description = "Email address for alerts"
  type        = string
}

# Backup Configuration
variable "backup_retention_days" {
  description = "Backup retention in days"
  type        = number
  default     = 30
}

# Tags
variable "tags" {
  description = "Additional tags for resources"
  type        = map(string)
  default     = {}
}