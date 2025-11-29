# SuperTool AWS ECS Module Variables

# General
variable "environment" {
  description = "Environment name (dev, staging, production)"
  type        = string

  validation {
    condition     = can(regex("^(dev|staging|production)$", var.environment))
    error_message = "Environment must be dev, staging, or production."
  }
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

# Container Configuration
variable "container_image" {
  description = "Docker image for SuperTool"
  type        = string
  default     = "supertool"
}

variable "container_tag" {
  description = "Docker image tag"
  type        = string
  default     = "latest"
}

variable "container_port" {
  description = "Port exposed by the container"
  type        = number
  default     = 8080
}

# ECS Task Configuration
variable "task_cpu" {
  description = "CPU units for the task (256, 512, 1024, 2048, 4096)"
  type        = number
  default     = 256

  validation {
    condition     = contains([256, 512, 1024, 2048, 4096], var.task_cpu)
    error_message = "Task CPU must be 256, 512, 1024, 2048, or 4096."
  }
}

variable "task_memory" {
  description = "Memory for the task in MB"
  type        = number
  default     = 512

  validation {
    condition     = var.task_memory >= 512 && var.task_memory <= 30720
    error_message = "Task memory must be between 512 and 30720 MB."
  }
}

variable "desired_count" {
  description = "Number of task instances to run"
  type        = number
  default     = 2

  validation {
    condition     = var.desired_count >= 1
    error_message = "Desired count must be at least 1."
  }
}

# Networking
variable "vpc_id" {
  description = "VPC ID for ECS deployment"
  type        = string
}

variable "private_subnet_ids" {
  description = "Private subnet IDs for ECS tasks"
  type        = list(string)

  validation {
    condition     = length(var.private_subnet_ids) >= 2
    error_message = "At least 2 private subnets required for high availability."
  }
}

variable "public_subnet_ids" {
  description = "Public subnet IDs for ALB"
  type        = list(string)

  validation {
    condition     = length(var.public_subnet_ids) >= 2
    error_message = "At least 2 public subnets required for high availability."
  }
}

variable "allowed_cidr_blocks" {
  description = "CIDR blocks allowed to access the ALB"
  type        = list(string)
  default     = ["0.0.0.0/0"]
}

# Load Balancer
variable "alb_internal" {
  description = "Whether the ALB is internal"
  type        = bool
  default     = false
}

variable "alb_listener_port" {
  description = "Port for ALB listener"
  type        = number
  default     = 80
}

variable "alb_listener_protocol" {
  description = "Protocol for ALB listener (HTTP or HTTPS)"
  type        = string
  default     = "HTTP"

  validation {
    condition     = contains(["HTTP", "HTTPS"], var.alb_listener_protocol)
    error_message = "ALB listener protocol must be HTTP or HTTPS."
  }
}

variable "acm_certificate_arn" {
  description = "ARN of ACM certificate for HTTPS (required if protocol is HTTPS)"
  type        = string
  default     = ""
}

# Auto Scaling
variable "enable_autoscaling" {
  description = "Enable auto scaling for ECS service"
  type        = bool
  default     = true
}

variable "autoscaling_min_capacity" {
  description = "Minimum number of tasks"
  type        = number
  default     = 2
}

variable "autoscaling_max_capacity" {
  description = "Maximum number of tasks"
  type        = number
  default     = 10
}

variable "autoscaling_cpu_target" {
  description = "Target CPU utilization for autoscaling (%)"
  type        = number
  default     = 70

  validation {
    condition     = var.autoscaling_cpu_target > 0 && var.autoscaling_cpu_target <= 100
    error_message = "CPU target must be between 1 and 100."
  }
}

variable "autoscaling_memory_target" {
  description = "Target memory utilization for autoscaling (%)"
  type        = number
  default     = 80

  validation {
    condition     = var.autoscaling_memory_target > 0 && var.autoscaling_memory_target <= 100
    error_message = "Memory target must be between 1 and 100."
  }
}

# Logging
variable "log_retention_days" {
  description = "CloudWatch log retention in days"
  type        = number
  default     = 7

  validation {
    condition = contains([
      1, 3, 5, 7, 14, 30, 60, 90, 120, 150, 180, 365, 400, 545, 731, 1827, 3653
    ], var.log_retention_days)
    error_message = "Log retention must be a valid CloudWatch retention period."
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

# Monitoring
variable "enable_container_insights" {
  description = "Enable Container Insights for ECS cluster"
  type        = bool
  default     = true
}

# Security
variable "enable_exec_command" {
  description = "Enable ECS Exec for debugging"
  type        = bool
  default     = false
}

variable "enable_deletion_protection" {
  description = "Enable deletion protection for ALB"
  type        = bool
  default     = false
}
