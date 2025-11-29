# SuperTool Dev Environment Variables

variable "aws_region" {
  description = "AWS region for dev deployment"
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
  default     = "dev-latest"
}

variable "allowed_cidr_blocks" {
  description = "CIDR blocks allowed to access the ALB"
  type        = list(string)
  default     = ["0.0.0.0/0"]  # Open for dev - restrict for security in real scenarios
}
