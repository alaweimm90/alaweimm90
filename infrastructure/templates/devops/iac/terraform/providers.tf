# {{PROJECT_NAME}} Providers

terraform {
  required_version = ">= 1.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  # Backend configuration (uncomment for remote state)
  # backend "s3" {
  #   bucket         = "{{PROJECT_NAME}}-terraform-state"
  #   key            = "state/terraform.tfstate"
  #   region         = "{{AWS_REGION}}"
  #   encrypt        = true
  #   dynamodb_table = "{{PROJECT_NAME}}-terraform-locks"
  # }
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project     = var.project_name
      Environment = var.environment
      ManagedBy   = "terraform"
    }
  }
}
