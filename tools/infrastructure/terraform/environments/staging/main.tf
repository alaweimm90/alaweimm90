# SuperTool Staging Environment
# Complete infrastructure for staging deployment

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
    key            = "staging/terraform.tfstate"
    region         = "us-east-1"
    encrypt        = true
    dynamodb_table = "supertool-terraform-locks"
  }
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Environment = "staging"
      Project     = "SuperTool"
      ManagedBy   = "Terraform"
      Owner       = "DevOps Team"
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

  environment = "staging"
  vpc_name    = "supertool"
  aws_region  = var.aws_region

  vpc_cidr        = "10.1.0.0/16"
  azs             = slice(data.aws_availability_zones.available.names, 0, 2) # 2 AZs for staging
  public_subnets  = ["10.1.1.0/24", "10.1.2.0/24"]
  private_subnets = ["10.1.11.0/24", "10.1.12.0/24"]

  # Cost optimization for staging
  enable_nat_gateway = true
  single_nat_gateway = true # Single NAT for cost savings

  # Monitoring
  enable_flow_logs        = true
  flow_logs_traffic_type  = "ALL"
  flow_logs_retention_days = 7 # Shorter retention for staging

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

  environment = "staging"
  aws_region  = var.aws_region

  # Container Configuration
  container_image = var.container_image
  container_tag   = var.container_tag
  container_port  = 8080

  # Task Configuration (smaller for staging)
  task_cpu      = 512   # 0.5 vCPU
  task_memory   = 1024  # 1 GB
  desired_count = 2     # 2 tasks for staging

  # Networking
  vpc_id             = module.vpc.vpc_id
  private_subnet_ids = module.vpc.private_subnet_ids
  public_subnet_ids  = module.vpc.public_subnet_ids
  allowed_cidr_blocks = var.allowed_cidr_blocks

  # Load Balancer
  alb_internal        = false
  alb_listener_port   = 443
  alb_listener_protocol = "HTTPS"
  acm_certificate_arn = var.acm_certificate_arn

  # Auto Scaling
  enable_autoscaling       = true
  autoscaling_min_capacity = 2
  autoscaling_max_capacity = 6
  autoscaling_cpu_target   = 75
  autoscaling_memory_target = 85

  # Monitoring & Logging
  enable_container_insights = true
  log_retention_days        = 7 # Shorter retention for staging
  log_level                 = "debug" # More verbose logging for staging

  # Security
  enable_exec_command        = true  # Enable for debugging in staging
  enable_deletion_protection = false # Disabled for easier testing

  common_tags = {
    Application = "SuperTool"
    Tier        = "Application"
  }

  depends_on = [module.vpc]
}

# CloudWatch Alarms
resource "aws_cloudwatch_metric_alarm" "ecs_cpu_high" {
  alarm_name          = "supertool-staging-cpu-high"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "2"
  metric_name         = "CPUUtilization"
  namespace           = "AWS/ECS"
  period              = "300"
  statistic           = "Average"
  threshold           = "80"
  alarm_description   = "This metric monitors ECS CPU utilization"
  alarm_actions       = var.sns_topic_arn != "" ? [var.sns_topic_arn] : []

  dimensions = {
    ClusterName = module.supertool_ecs.cluster_name
    ServiceName = module.supertool_ecs.service_name
  }
}

resource "aws_cloudwatch_metric_alarm" "ecs_memory_high" {
  alarm_name          = "supertool-staging-memory-high"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "2"
  metric_name         = "MemoryUtilization"
  namespace           = "AWS/ECS"
  period              = "300"
  statistic           = "Average"
  threshold           = "85"
  alarm_description   = "This metric monitors ECS memory utilization"
  alarm_actions       = var.sns_topic_arn != "" ? [var.sns_topic_arn] : []

  dimensions = {
    ClusterName = module.supertool_ecs.cluster_name
    ServiceName = module.supertool_ecs.service_name
  }
}
