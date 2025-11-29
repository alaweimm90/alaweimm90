# SuperTool Dev Environment
# Complete infrastructure for development deployment

terraform {
  required_version = ">= 1.5.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  backend "s3" {
    bucket         = "supertool-terraform-state"
    key            = "dev/terraform.tfstate"
    region         = "us-east-1"
    encrypt        = true
    dynamodb_table = "supertool-terraform-locks"
  }
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Environment = "dev"
      Project     = "SuperTool"
      ManagedBy   = "Terraform"
      Owner       = "Development Team"
    }
  }
}

# Data sources
data "aws_availability_zones" "available" {
  state = "available"
}

# VPC Module
module "vpc" {
  source = "../../modules/networking/vpc"

  environment = "dev"
  vpc_name    = "supertool"
  aws_region  = var.aws_region

  vpc_cidr        = "10.2.0.0/16"
  azs             = slice(data.aws_availability_zones.available.names, 0, 2) # 2 AZs for dev
  public_subnets  = ["10.2.1.0/24", "10.2.2.0/24"]
  private_subnets = ["10.2.11.0/24", "10.2.12.0/24"]

  # Cost optimization for dev
  enable_nat_gateway = true
  single_nat_gateway = true # Single NAT for cost savings

  # Minimal monitoring for dev
  enable_flow_logs        = false # Disabled for cost savings
  flow_logs_traffic_type  = "ALL"
  flow_logs_retention_days = 7

  # VPC Endpoints
  enable_s3_endpoint  = true
  enable_ecr_endpoints = true

  common_tags = {
    Application = "SuperTool"
    Tier        = "Infrastructure"
  }
}

# SuperTool ECS Deployment
module "supertool_ecs" {
  source = "../../modules/aws/ecs"

  environment = "dev"
  aws_region  = var.aws_region

  # Container Configuration
  container_image = var.container_image
  container_tag   = var.container_tag
  container_port  = 8080

  # Task Configuration (minimal for dev)
  task_cpu      = 256  # 0.25 vCPU
  task_memory   = 512  # 512 MB
  desired_count = 1    # 1 task for dev

  # Networking
  vpc_id             = module.vpc.vpc_id
  private_subnet_ids = module.vpc.private_subnet_ids
  public_subnet_ids  = module.vpc.public_subnet_ids
  allowed_cidr_blocks = var.allowed_cidr_blocks

  # Load Balancer
  alb_internal        = false
  alb_listener_port   = 80 # HTTP only for dev
  alb_listener_protocol = "HTTP"
  acm_certificate_arn = "" # No HTTPS in dev

  # Auto Scaling
  enable_autoscaling       = false # Disabled for dev
  autoscaling_min_capacity = 1
  autoscaling_max_capacity = 2
  autoscaling_cpu_target   = 80
  autoscaling_memory_target = 90

  # Monitoring & Logging
  enable_container_insights = false # Disabled for cost savings
  log_retention_days        = 3 # Minimal retention for dev
  log_level                 = "debug" # Very verbose logging for dev

  # Security
  enable_exec_command        = true  # Enable for debugging
  enable_deletion_protection = false # Disabled for easier testing

  common_tags = {
    Application = "SuperTool"
    Tier        = "Application"
  }

  depends_on = [module.vpc]
}
