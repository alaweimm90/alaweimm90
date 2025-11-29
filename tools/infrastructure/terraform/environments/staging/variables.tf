# SuperTool Staging Environment Variables

variable "aws_region" {
  description = "AWS region for staging deployment"
  type        = string
  default     = "us-east-1"
}

variable "container_image" {
  description = "Docker image for SuperTool"
  type        = string
  default     = "ghcr.io/username/supertool"
}

variable "container_tag" {
  description = "Docker image tag (version)"
  type        = string
  default     = "staging-latest"
}

variable "allowed_cidr_blocks" {
  description = "CIDR blocks allowed to access the ALB"
  type        = list(string)
  default     = ["0.0.0.0/0"]  # Restrict this for better security
}

variable "acm_certificate_arn" {
  description = "ARN of ACM certificate for HTTPS"
  type        = string
  default     = ""  # Provide via tfvars or environment variable
}

variable "sns_topic_arn" {
  description = "SNS topic ARN for alarms"
  type        = string
  default     = ""
}
