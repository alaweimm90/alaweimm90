terraform {
  required_version = ">= 1.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.5"
    }
    null = {
      source  = "hashicorp/null"
      version = "~> 3.2"
    }
  }

  backend "s3" {
    bucket         = "crazyideas-terraform-state"
    key            = "infrastructure/terraform.tfstate"
    region         = "us-east-1"
    encrypt        = true
    dynamodb_table = "terraform-state-lock"
  }
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project     = "CrazyIdeas"
      Environment = var.environment
      ManagedBy   = "Terraform"
      Owner       = var.owner
      CostCenter  = var.cost_center
    }
  }
}

# Data sources
data "aws_availability_zones" "available" {
  state = "available"
}

data "aws_caller_identity" "current" {}

# VPC Module
module "vpc" {
  source = "./modules/vpc"

  environment         = var.environment
  vpc_cidr           = var.vpc_cidr
  availability_zones = data.aws_availability_zones.available.names
  public_subnets     = var.public_subnets
  private_subnets    = var.private_subnets
  database_subnets   = var.database_subnets
  enable_nat_gateway = var.enable_nat_gateway
  enable_vpn_gateway = var.enable_vpn_gateway
}

# Security Groups Module
module "security_groups" {
  source = "./modules/security-groups"

  vpc_id      = module.vpc.vpc_id
  environment = var.environment
}

# ECS Cluster Module
module "ecs_cluster" {
  source = "./modules/ecs"

  environment                = var.environment
  cluster_name              = "${var.project_name}-${var.environment}"
  vpc_id                    = module.vpc.vpc_id
  private_subnet_ids        = module.vpc.private_subnet_ids
  public_subnet_ids         = module.vpc.public_subnet_ids
  alb_security_group_id     = module.security_groups.alb_security_group_id
  ecs_tasks_security_group_id = module.security_groups.ecs_tasks_security_group_id
}

# RDS Module (PostgreSQL)
module "rds" {
  source = "./modules/rds"

  environment            = var.environment
  database_name         = var.database_name
  database_username     = var.database_username
  database_password     = var.database_password
  instance_class        = var.rds_instance_class
  allocated_storage     = var.rds_allocated_storage
  vpc_id               = module.vpc.vpc_id
  database_subnet_ids   = module.vpc.database_subnet_ids
  security_group_id     = module.security_groups.database_security_group_id
  backup_retention_period = var.backup_retention_period
  multi_az             = var.environment == "production" ? true : false
}

# ElastiCache Module (Redis)
module "elasticache" {
  source = "./modules/elasticache"

  environment          = var.environment
  cluster_id          = "${var.project_name}-${var.environment}-redis"
  node_type           = var.redis_node_type
  num_cache_nodes     = var.redis_num_nodes
  vpc_id              = module.vpc.vpc_id
  subnet_ids          = module.vpc.private_subnet_ids
  security_group_id   = module.security_groups.redis_security_group_id
}

# S3 Buckets Module
module "s3" {
  source = "./modules/s3"

  environment     = var.environment
  project_name    = var.project_name
  aws_account_id  = data.aws_caller_identity.current.account_id
}

# CloudFront CDN Module
module "cloudfront" {
  source = "./modules/cloudfront"

  environment          = var.environment
  project_name        = var.project_name
  origin_domain_name  = module.ecs_cluster.alb_dns_name
  s3_bucket_domain    = module.s3.static_assets_bucket_domain
  acm_certificate_arn = var.acm_certificate_arn
  domain_names        = var.domain_names
}

# Application Deployments
module "ghost_researcher" {
  source = "./modules/ecs-service"

  environment               = var.environment
  service_name             = "ghost-researcher"
  cluster_id               = module.ecs_cluster.cluster_id
  task_role_arn           = module.ecs_cluster.task_role_arn
  execution_role_arn      = module.ecs_cluster.execution_role_arn
  vpc_id                  = module.vpc.vpc_id
  private_subnet_ids      = module.vpc.private_subnet_ids
  security_group_id       = module.security_groups.ecs_tasks_security_group_id
  alb_target_group_arn    = module.ecs_cluster.ghost_researcher_target_group_arn

  container_image         = "${var.ecr_repository_url}/ghost-researcher:${var.image_tag}"
  container_port         = 3000
  container_cpu          = var.container_cpu
  container_memory       = var.container_memory
  desired_count          = var.desired_count
  min_capacity           = var.min_capacity
  max_capacity           = var.max_capacity

  environment_variables = {
    NODE_ENV               = var.environment
    DATABASE_URL          = module.rds.connection_string
    REDIS_URL             = module.elasticache.redis_endpoint
    NEXT_PUBLIC_API_URL   = "https://api.${var.domain_names[0]}"
  }
}

module "scientific_tinder" {
  source = "./modules/ecs-service"

  environment               = var.environment
  service_name             = "scientific-tinder"
  cluster_id               = module.ecs_cluster.cluster_id
  task_role_arn           = module.ecs_cluster.task_role_arn
  execution_role_arn      = module.ecs_cluster.execution_role_arn
  vpc_id                  = module.vpc.vpc_id
  private_subnet_ids      = module.vpc.private_subnet_ids
  security_group_id       = module.security_groups.ecs_tasks_security_group_id
  alb_target_group_arn    = module.ecs_cluster.scientific_tinder_target_group_arn

  container_image         = "${var.ecr_repository_url}/scientific-tinder:${var.image_tag}"
  container_port         = 3001
  container_cpu          = var.container_cpu
  container_memory       = var.container_memory
  desired_count          = var.desired_count
  min_capacity           = var.min_capacity
  max_capacity           = var.max_capacity

  environment_variables = {
    NODE_ENV               = var.environment
    DATABASE_URL          = module.rds.connection_string
    REDIS_URL             = module.elasticache.redis_endpoint
    NEXT_PUBLIC_API_URL   = "https://api.${var.domain_names[0]}"
  }
}

module "chaos_engine" {
  source = "./modules/ecs-service"

  environment               = var.environment
  service_name             = "chaos-engine"
  cluster_id               = module.ecs_cluster.cluster_id
  task_role_arn           = module.ecs_cluster.task_role_arn
  execution_role_arn      = module.ecs_cluster.execution_role_arn
  vpc_id                  = module.vpc.vpc_id
  private_subnet_ids      = module.vpc.private_subnet_ids
  security_group_id       = module.security_groups.ecs_tasks_security_group_id
  alb_target_group_arn    = module.ecs_cluster.chaos_engine_target_group_arn

  container_image         = "${var.ecr_repository_url}/chaos-engine:${var.image_tag}"
  container_port         = 3002
  container_cpu          = var.container_cpu
  container_memory       = var.container_memory
  desired_count          = var.desired_count
  min_capacity           = var.min_capacity
  max_capacity           = var.max_capacity

  environment_variables = {
    NODE_ENV               = var.environment
    DATABASE_URL          = module.rds.connection_string
    REDIS_URL             = module.elasticache.redis_endpoint
    NEXT_PUBLIC_API_URL   = "https://api.${var.domain_names[0]}"
  }
}

module "api_backend" {
  source = "./modules/ecs-service"

  environment               = var.environment
  service_name             = "api-backend"
  cluster_id               = module.ecs_cluster.cluster_id
  task_role_arn           = module.ecs_cluster.task_role_arn
  execution_role_arn      = module.ecs_cluster.execution_role_arn
  vpc_id                  = module.vpc.vpc_id
  private_subnet_ids      = module.vpc.private_subnet_ids
  security_group_id       = module.security_groups.ecs_tasks_security_group_id
  alb_target_group_arn    = module.ecs_cluster.api_target_group_arn

  container_image         = "${var.ecr_repository_url}/api-backend:${var.image_tag}"
  container_port         = 8000
  container_cpu          = var.api_container_cpu
  container_memory       = var.api_container_memory
  desired_count          = var.api_desired_count
  min_capacity           = var.api_min_capacity
  max_capacity           = var.api_max_capacity

  environment_variables = {
    NODE_ENV               = var.environment
    DATABASE_URL          = module.rds.connection_string
    REDIS_URL             = module.elasticache.redis_endpoint
    JWT_SECRET            = var.jwt_secret
    CORS_ORIGINS          = join(",", var.cors_origins)
  }
}

# WAF Module
module "waf" {
  source = "./modules/waf"

  environment     = var.environment
  project_name    = var.project_name
  alb_arn        = module.ecs_cluster.alb_arn
  rate_limit     = var.waf_rate_limit
}

# Monitoring Module
module "monitoring" {
  source = "./modules/monitoring"

  environment          = var.environment
  project_name        = var.project_name
  cluster_name        = module.ecs_cluster.cluster_name
  alb_arn_suffix      = module.ecs_cluster.alb_arn_suffix
  target_group_arns   = module.ecs_cluster.target_group_arns
  rds_instance_id     = module.rds.instance_id
  redis_cluster_id    = module.elasticache.cluster_id
  sns_email_endpoint  = var.alert_email
}

# Backup Module
module "backup" {
  source = "./modules/backup"

  environment            = var.environment
  project_name          = var.project_name
  rds_instance_arn      = module.rds.instance_arn
  s3_bucket_arns        = module.s3.bucket_arns
  backup_retention_days = var.backup_retention_days
}

# Outputs
output "vpc_id" {
  description = "VPC ID"
  value       = module.vpc.vpc_id
}

output "alb_dns_name" {
  description = "ALB DNS name"
  value       = module.ecs_cluster.alb_dns_name
}

output "cloudfront_distribution_id" {
  description = "CloudFront distribution ID"
  value       = module.cloudfront.distribution_id
}

output "cloudfront_domain_name" {
  description = "CloudFront domain name"
  value       = module.cloudfront.domain_name
}

output "database_endpoint" {
  description = "RDS instance endpoint"
  value       = module.rds.endpoint
  sensitive   = true
}

output "redis_endpoint" {
  description = "Redis cluster endpoint"
  value       = module.elasticache.redis_endpoint
  sensitive   = true
}

output "s3_buckets" {
  description = "S3 bucket names"
  value       = module.s3.bucket_names
}

output "ecr_repository_urls" {
  description = "ECR repository URLs"
  value = {
    ghost_researcher = "${var.ecr_repository_url}/ghost-researcher"
    scientific_tinder = "${var.ecr_repository_url}/scientific-tinder"
    chaos_engine = "${var.ecr_repository_url}/chaos-engine"
    api_backend = "${var.ecr_repository_url}/api-backend"
  }
}