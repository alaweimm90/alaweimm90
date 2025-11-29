# SuperTool VPC Networking Module Variables

# General Configuration
variable "environment" {
  description = "Environment name (dev, staging, production)"
  type        = string

  validation {
    condition     = can(regex("^(dev|staging|production)$", var.environment))
    error_message = "Environment must be dev, staging, or production."
  }
}

variable "vpc_name" {
  description = "Name of the VPC"
  type        = string
  default     = "supertool"
}

variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "common_tags" {
  description = "Common tags to apply to all resources"
  type        = map(string)
  default     = {}
}

# VPC Configuration
variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.0.0.0/16"

  validation {
    condition     = can(cidrhost(var.vpc_cidr, 0))
    error_message = "VPC CIDR must be a valid IPv4 CIDR block."
  }
}

variable "enable_dns_hostnames" {
  description = "Enable DNS hostnames in the VPC"
  type        = bool
  default     = true
}

variable "enable_dns_support" {
  description = "Enable DNS support in the VPC"
  type        = bool
  default     = true
}

# Availability Zones
variable "azs" {
  description = "List of availability zones"
  type        = list(string)
  default     = ["us-east-1a", "us-east-1b", "us-east-1c"]

  validation {
    condition     = length(var.azs) >= 2
    error_message = "At least 2 availability zones are required for high availability."
  }
}

# Subnets
variable "public_subnets" {
  description = "List of public subnet CIDR blocks"
  type        = list(string)
  default     = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]

  validation {
    condition     = length(var.public_subnets) >= 2
    error_message = "At least 2 public subnets are required for high availability."
  }
}

variable "private_subnets" {
  description = "List of private subnet CIDR blocks"
  type        = list(string)
  default     = ["10.0.11.0/24", "10.0.12.0/24", "10.0.13.0/24"]

  validation {
    condition     = length(var.private_subnets) >= 2
    error_message = "At least 2 private subnets are required for high availability."
  }
}

variable "database_subnets" {
  description = "List of database subnet CIDR blocks"
  type        = list(string)
  default     = []
}

# NAT Gateway
variable "enable_nat_gateway" {
  description = "Enable NAT Gateway for private subnets"
  type        = bool
  default     = true
}

variable "single_nat_gateway" {
  description = "Use a single NAT Gateway for all private subnets (cost optimization)"
  type        = bool
  default     = false
}

# Internet Gateway
variable "create_internet_gateway" {
  description = "Create an Internet Gateway"
  type        = bool
  default     = true
}

# VPN Gateway
variable "enable_vpn_gateway" {
  description = "Enable VPN Gateway"
  type        = bool
  default     = false
}

# VPC Flow Logs
variable "enable_flow_logs" {
  description = "Enable VPC Flow Logs"
  type        = bool
  default     = true
}

variable "flow_logs_traffic_type" {
  description = "Traffic type to log (ACCEPT, REJECT, ALL)"
  type        = string
  default     = "ALL"

  validation {
    condition     = contains(["ACCEPT", "REJECT", "ALL"], var.flow_logs_traffic_type)
    error_message = "Flow logs traffic type must be ACCEPT, REJECT, or ALL."
  }
}

variable "flow_logs_retention_days" {
  description = "Flow logs retention in days"
  type        = number
  default     = 30

  validation {
    condition     = contains([0, 1, 3, 5, 7, 14, 30, 60, 90, 120, 150, 180, 365, 400, 545, 731, 1827, 3653], var.flow_logs_retention_days)
    error_message = "Flow logs retention must be a valid CloudWatch Logs retention period."
  }
}

# VPC Endpoints
variable "enable_s3_endpoint" {
  description = "Enable S3 VPC Endpoint"
  type        = bool
  default     = true
}

variable "enable_ecr_endpoints" {
  description = "Enable ECR VPC Endpoints (API and DKR)"
  type        = bool
  default     = true
}

# Default Resources Management
variable "manage_default_network_acl" {
  description = "Manage default Network ACL"
  type        = bool
  default     = false
}

variable "manage_default_security_group" {
  description = "Manage default Security Group"
  type        = bool
  default     = false
}
