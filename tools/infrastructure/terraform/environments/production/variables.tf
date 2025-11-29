# SuperTool Production Environment Variables

variable "aws_region" {
  description = "AWS region for production deployment"
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
  default     = "latest"
}

variable "vpc_id" {
  description = "VPC ID for deployment"
  type        = string
}

variable "private_subnet_ids" {
  description = "Private subnet IDs for ECS tasks"
  type        = list(string)
}

variable "public_subnet_ids" {
  description = "Public subnet IDs for ALB"
  type        = list(string)
}

variable "allowed_cidr_blocks" {
  description = "CIDR blocks allowed to access the ALB"
  type        = list(string)
  default     = ["0.0.0.0/0"]  # Restrict this in production!
}

variable "acm_certificate_arn" {
  description = "ARN of ACM certificate for HTTPS"
  type        = string
}

variable "sns_topic_arn" {
  description = "SNS topic ARN for alarms"
  type        = string
  default     = ""
}

variable "create_dns_record" {
  description = "Create Route53 DNS record"
  type        = bool
  default     = false
}

variable "route53_zone_id" {
  description = "Route53 hosted zone ID"
  type        = string
  default     = ""
}

variable "domain_name" {
  description = "Domain name for SuperTool"
  type        = string
  default     = ""
}
