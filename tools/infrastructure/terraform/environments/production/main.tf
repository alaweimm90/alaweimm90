# SuperTool Production Environment
# Complete infrastructure for production deployment

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
    key            = "production/terraform.tfstate"
    region         = "us-east-1"
    encrypt        = true
    dynamodb_table = "supertool-terraform-locks"
  }
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Environment = "production"
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

# VPC Module (to be created)
# module "vpc" {
#   source = "../../modules/networking/vpc"
#
#   environment = "production"
#   vpc_cidr    = "10.0.0.0/16"
#   azs         = slice(data.aws_availability_zones.available.names, 0, 3)
#
#   private_subnets = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
#   public_subnets  = ["10.0.101.0/24", "10.0.102.0/24", "10.0.103.0/24"]
#
#   enable_nat_gateway = true
#   enable_vpn_gateway = false
#
#   tags = {
#     Terraform   = "true"
#     Environment = "production"
#   }
# }

# SuperTool ECS Deployment
module "supertool_ecs" {
  source = "../../modules/aws/ecs"

  environment = "production"
  aws_region  = var.aws_region

  # Container Configuration
  container_image = var.container_image
  container_tag   = var.container_tag
  container_port  = 8080

  # Task Configuration
  task_cpu      = 1024  # 1 vCPU
  task_memory   = 2048  # 2 GB
  desired_count = 3     # High availability

  # Networking
  vpc_id             = var.vpc_id
  private_subnet_ids = var.private_subnet_ids
  public_subnet_ids  = var.public_subnet_ids
  allowed_cidr_blocks = var.allowed_cidr_blocks

  # Load Balancer
  alb_internal        = false
  alb_listener_port   = 443
  alb_listener_protocol = "HTTPS"
  acm_certificate_arn = var.acm_certificate_arn

  # Auto Scaling
  enable_autoscaling       = true
  autoscaling_min_capacity = 3
  autoscaling_max_capacity = 20
  autoscaling_cpu_target   = 70
  autoscaling_memory_target = 80

  # Monitoring & Logging
  enable_container_insights = true
  log_retention_days        = 30
  log_level                 = "info"

  # Security
  enable_exec_command        = false  # Disabled in production
  enable_deletion_protection = true   # Protect ALB

  common_tags = {
    Application = "SuperTool"
    Tier        = "Application"
    Backup      = "Daily"
  }
}

# CloudWatch Alarms
resource "aws_cloudwatch_metric_alarm" "ecs_cpu_high" {
  alarm_name          = "supertool-production-cpu-high"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "2"
  metric_name         = "CPUUtilization"
  namespace           = "AWS/ECS"
  period              = "300"
  statistic           = "Average"
  threshold           = "85"
  alarm_description   = "This metric monitors ECS CPU utilization"
  alarm_actions       = [var.sns_topic_arn]

  dimensions = {
    ClusterName = module.supertool_ecs.cluster_name
    ServiceName = module.supertool_ecs.service_name
  }
}

resource "aws_cloudwatch_metric_alarm" "ecs_memory_high" {
  alarm_name          = "supertool-production-memory-high"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "2"
  metric_name         = "MemoryUtilization"
  namespace           = "AWS/ECS"
  period              = "300"
  statistic           = "Average"
  threshold           = "90"
  alarm_description   = "This metric monitors ECS memory utilization"
  alarm_actions       = [var.sns_topic_arn]

  dimensions = {
    ClusterName = module.supertool_ecs.cluster_name
    ServiceName = module.supertool_ecs.service_name
  }
}

# Route53 DNS (optional)
# resource "aws_route53_record" "supertool" {
#   count   = var.create_dns_record ? 1 : 0
#   zone_id = var.route53_zone_id
#   name    = var.domain_name
#   type    = "A"
#
#   alias {
#     name                   = module.supertool_ecs.alb_dns_name
#     zone_id                = module.supertool_ecs.alb_zone_id
#     evaluate_target_health = true
#   }
# }
