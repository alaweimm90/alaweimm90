# REPZ Platform - Terraform Variables
# Environment-specific and configurable variables

# Environment configuration
variable "environment" {
  description = "Environment name (development, staging, production)"
  type        = string
  validation {
    condition     = contains(["development", "staging", "production"], var.environment)
    error_message = "Environment must be development, staging, or production."
  }
}

variable "aws_region" {
  description = "AWS region for resources"
  type        = string
  default     = "us-east-1"
}

# Network configuration
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

variable "allowed_cidr_blocks" {
  description = "CIDR blocks allowed to access resources"
  type        = list(string)
  default     = ["0.0.0.0/0"]  # Restrict in production
}

# Database configuration
variable "db_allocated_storage" {
  description = "Allocated storage for RDS instance (GB)"
  type        = number
  default     = 20
}

variable "db_max_allocated_storage" {
  description = "Maximum allocated storage for RDS instance (GB)"
  type        = number
  default     = 100
}

variable "db_username" {
  description = "Database master username"
  type        = string
  default     = "repzadmin"
  sensitive   = true
}

variable "db_password" {
  description = "Database master password"
  type        = string
  sensitive   = true
}

# Redis configuration
variable "redis_node_type" {
  description = "ElastiCache Redis node type"
  type        = string
  default     = "cache.t3.micro"
}

variable "redis_auth_token" {
  description = "Redis authentication token"
  type        = string
  sensitive   = true
}

# Application configuration
variable "app_image" {
  description = "Docker image for the application"
  type        = string
  default     = "repz/platform:latest"
}

# SSL certificate
variable "ssl_certificate_arn" {
  description = "ARN of SSL certificate for HTTPS"
  type        = string
  default     = ""
}

# Domain configuration
variable "create_dns_records" {
  description = "Whether to create Route53 DNS records"
  type        = bool
  default     = false
}

variable "domain_name" {
  description = "Domain name for the application"
  type        = string
  default     = "repzcoach.com"
}

# External service configuration
variable "supabase_url" {
  description = "Supabase project URL"
  type        = string
  sensitive   = true
}

variable "supabase_anon_key" {
  description = "Supabase anonymous key"
  type        = string
  sensitive   = true
}

variable "stripe_public_key" {
  description = "Stripe publishable key"
  type        = string
  sensitive   = true
}

variable "stripe_secret_key" {
  description = "Stripe secret key"
  type        = string
  sensitive   = true
}

variable "jwt_secret" {
  description = "JWT signing secret"
  type        = string
  sensitive   = true
}

# Monitoring configuration
variable "sns_alert_topic_arn" {
  description = "SNS topic ARN for alerts"
  type        = string
  default     = ""
}

# Feature flags
variable "enable_cdn" {
  description = "Enable CloudFront CDN"
  type        = bool
  default     = true
}

variable "enable_waf" {
  description = "Enable AWS WAF"
  type        = bool
  default     = true
}

variable "enable_backup" {
  description = "Enable automated backups"
  type        = bool
  default     = true
}

variable "enable_monitoring" {
  description = "Enable comprehensive monitoring"
  type        = bool
  default     = true
}

# Cost optimization
variable "enable_spot_instances" {
  description = "Use spot instances for non-critical workloads"
  type        = bool
  default     = false
}

variable "enable_auto_scaling" {
  description = "Enable auto scaling for ECS services"
  type        = bool
  default     = true
}

# Security configuration
variable "enable_encryption" {
  description = "Enable encryption at rest for all services"
  type        = bool
  default     = true
}

variable "enable_vpc_flow_logs" {
  description = "Enable VPC flow logs"
  type        = bool
  default     = true
}

variable "enable_config_rules" {
  description = "Enable AWS Config compliance rules"
  type        = bool
  default     = true
}

# Compliance and governance
variable "data_retention_days" {
  description = "Data retention period in days"
  type        = number
  default     = 2555  # 7 years for healthcare compliance
}

variable "audit_log_retention_days" {
  description = "Audit log retention period in days"
  type        = number
  default     = 2555  # 7 years for compliance
}

variable "enable_hipaa_compliance" {
  description = "Enable HIPAA compliance features"
  type        = bool
  default     = true
}

# Multi-region configuration
variable "enable_multi_region" {
  description = "Enable multi-region deployment"
  type        = bool
  default     = false
}

variable "backup_region" {
  description = "Secondary region for backups and disaster recovery"
  type        = string
  default     = "us-west-2"
}

# Performance optimization
variable "enable_caching" {
  description = "Enable Redis caching layer"
  type        = bool
  default     = true
}

variable "enable_cdn_caching" {
  description = "Enable CloudFront caching"
  type        = bool
  default     = true
}

variable "cdn_cache_ttl" {
  description = "Default TTL for CDN cache in seconds"
  type        = number
  default     = 86400  # 24 hours
}

# Development and testing
variable "enable_debug_mode" {
  description = "Enable debug mode (development only)"
  type        = bool
  default     = false
}

variable "enable_test_data" {
  description = "Create test data (non-production only)"
  type        = bool
  default     = false
}

# Resource tagging
variable "additional_tags" {
  description = "Additional tags to apply to all resources"
  type        = map(string)
  default     = {}
}

variable "cost_center" {
  description = "Cost center for billing allocation"
  type        = string
  default     = "REPZ-Platform"
}

variable "owner_email" {
  description = "Email of the resource owner"
  type        = string
  default     = "devops@repzcoach.com"
}