# REPZ Platform - Terraform Infrastructure
# Multi-environment AWS infrastructure for the REPZ Coach platform

terraform {
  required_version = ">= 1.0"
  
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.1"
    }
  }
  
  backend "s3" {
    # Configure in terraform.tfvars or via CLI
    # bucket = "repz-terraform-state"
    # key    = "platform/terraform.tfstate"
    # region = "us-east-1"
  }
}

# Provider configuration
provider "aws" {
  region = var.aws_region
  
  default_tags {
    tags = {
      Project     = "REPZ-Platform"
      Environment = var.environment
      ManagedBy   = "Terraform"
      Owner       = "REPZ-DevOps"
    }
  }
}

# Local values for common configurations
locals {
  name_prefix = "repz-${var.environment}"
  
  common_tags = {
    Project     = "REPZ-Platform"
    Environment = var.environment
    ManagedBy   = "Terraform"
  }
  
  # Environment-specific configurations
  env_config = {
    development = {
      instance_type     = "t3.small"
      min_capacity      = 1
      max_capacity      = 2
      desired_capacity  = 1
      db_instance_class = "db.t3.micro"
      multi_az          = false
    }
    staging = {
      instance_type     = "t3.medium"
      min_capacity      = 1
      max_capacity      = 3
      desired_capacity  = 2
      db_instance_class = "db.t3.small"
      multi_az          = false
    }
    production = {
      instance_type     = "t3.large"
      min_capacity      = 2
      max_capacity      = 10
      desired_capacity  = 3
      db_instance_class = "db.r6g.large"
      multi_az          = true
    }
  }
  
  current_config = local.env_config[var.environment]
}

# Data sources
data "aws_availability_zones" "available" {
  state = "available"
}

data "aws_caller_identity" "current" {}

data "aws_region" "current" {}

# VPC Module
module "vpc" {
  source = "./modules/vpc"
  
  name_prefix = local.name_prefix
  environment = var.environment
  
  vpc_cidr             = var.vpc_cidr
  availability_zones   = data.aws_availability_zones.available.names
  private_subnet_cidrs = var.private_subnet_cidrs
  public_subnet_cidrs  = var.public_subnet_cidrs
  
  enable_nat_gateway   = true
  enable_vpn_gateway   = false
  enable_dns_hostnames = true
  enable_dns_support   = true
  
  tags = local.common_tags
}

# Security Groups Module
module "security_groups" {
  source = "./modules/security"
  
  name_prefix = local.name_prefix
  environment = var.environment
  vpc_id      = module.vpc.vpc_id
  
  allowed_cidr_blocks = var.allowed_cidr_blocks
  
  tags = local.common_tags
}

# RDS Database Module
module "database" {
  source = "./modules/database"
  
  name_prefix = local.name_prefix
  environment = var.environment
  
  # Network configuration
  subnet_ids         = module.vpc.private_subnet_ids
  security_group_ids = [module.security_groups.database_security_group_id]
  
  # Database configuration
  engine               = "postgres"
  engine_version       = "15.4"
  instance_class       = local.current_config.db_instance_class
  allocated_storage    = var.db_allocated_storage
  max_allocated_storage = var.db_max_allocated_storage
  
  db_name  = "repzplatform"
  username = var.db_username
  password = var.db_password
  
  # High availability
  multi_az               = local.current_config.multi_az
  backup_retention_period = var.environment == "production" ? 30 : 7
  backup_window          = "03:00-04:00"
  maintenance_window     = "sun:04:00-sun:05:00"
  
  # Performance and monitoring
  performance_insights_enabled = var.environment == "production"
  monitoring_interval         = var.environment == "production" ? 60 : 0
  
  # Security
  deletion_protection = var.environment == "production"
  encrypt_at_rest    = true
  
  tags = local.common_tags
}

# ElastiCache Redis Module
module "redis" {
  source = "./modules/cache"
  
  name_prefix = local.name_prefix
  environment = var.environment
  
  # Network configuration
  subnet_ids         = module.vpc.private_subnet_ids
  security_group_ids = [module.security_groups.cache_security_group_id]
  
  # Redis configuration
  node_type               = var.redis_node_type
  num_cache_nodes         = var.environment == "production" ? 2 : 1
  engine_version          = "7.0"
  parameter_group_name    = "default.redis7"
  
  # Security
  at_rest_encryption_enabled = true
  transit_encryption_enabled = true
  auth_token                 = var.redis_auth_token
  
  tags = local.common_tags
}

# Application Load Balancer Module
module "load_balancer" {
  source = "./modules/load_balancer"
  
  name_prefix = local.name_prefix
  environment = var.environment
  
  # Network configuration
  vpc_id            = module.vpc.vpc_id
  public_subnet_ids = module.vpc.public_subnet_ids
  security_group_ids = [module.security_groups.alb_security_group_id]
  
  # SSL certificate
  certificate_arn = var.ssl_certificate_arn
  
  # Health check configuration
  health_check_path     = "/health"
  health_check_matcher  = "200"
  
  tags = local.common_tags
}

# ECS Cluster Module
module "ecs" {
  source = "./modules/ecs"
  
  name_prefix = local.name_prefix
  environment = var.environment
  
  # Network configuration
  vpc_id           = module.vpc.vpc_id
  private_subnet_ids = module.vpc.private_subnet_ids
  public_subnet_ids  = module.vpc.public_subnet_ids
  
  # Load balancer configuration
  target_group_arn = module.load_balancer.target_group_arn
  security_group_ids = [
    module.security_groups.ecs_security_group_id,
    module.security_groups.app_security_group_id
  ]
  
  # Application configuration
  app_image = var.app_image
  app_port  = 3000
  
  # Scaling configuration
  min_capacity     = local.current_config.min_capacity
  max_capacity     = local.current_config.max_capacity
  desired_capacity = local.current_config.desired_capacity
  
  # Database connection
  database_url = module.database.connection_string
  redis_url    = module.redis.connection_string
  
  # Environment variables
  environment_variables = {
    NODE_ENV                    = var.environment
    VITE_SUPABASE_URL          = var.supabase_url
    VITE_SUPABASE_ANON_KEY     = var.supabase_anon_key
    VITE_STRIPE_PUBLIC_KEY     = var.stripe_public_key
    DATABASE_URL               = module.database.connection_string
    REDIS_URL                  = module.redis.connection_string
    JWT_SECRET                 = var.jwt_secret
    STRIPE_SECRET_KEY          = var.stripe_secret_key
  }
  
  tags = local.common_tags
}

# S3 Buckets Module
module "storage" {
  source = "./modules/storage"
  
  name_prefix = local.name_prefix
  environment = var.environment
  
  # Create buckets for different purposes
  buckets = {
    app-assets = {
      versioning = true
      lifecycle  = true
      cors       = true
    }
    user-uploads = {
      versioning = true
      lifecycle  = true
      cors       = true
    }
    backups = {
      versioning = true
      lifecycle  = true
      cors       = false
    }
    logs = {
      versioning = false
      lifecycle  = true
      cors       = false
    }
  }
  
  tags = local.common_tags
}

# CloudWatch Monitoring Module
module "monitoring" {
  source = "./modules/monitoring"
  
  name_prefix = local.name_prefix
  environment = var.environment
  
  # ECS cluster monitoring
  ecs_cluster_name = module.ecs.cluster_name
  ecs_service_name = module.ecs.service_name
  
  # Database monitoring
  db_instance_id = module.database.instance_id
  
  # Load balancer monitoring
  alb_arn_suffix = module.load_balancer.arn_suffix
  target_group_arn_suffix = module.load_balancer.target_group_arn_suffix
  
  # Alert notification
  sns_topic_arn = var.sns_alert_topic_arn
  
  tags = local.common_tags
}

# IAM Roles and Policies Module
module "iam" {
  source = "./modules/iam"
  
  name_prefix = local.name_prefix
  environment = var.environment
  
  # S3 bucket ARNs for policy attachment
  s3_bucket_arns = module.storage.bucket_arns
  
  tags = local.common_tags
}

# Route53 DNS Module (optional)
module "dns" {
  source = "./modules/dns"
  count  = var.create_dns_records ? 1 : 0
  
  name_prefix = local.name_prefix
  environment = var.environment
  
  # Domain configuration
  domain_name    = var.domain_name
  subdomain      = var.environment == "production" ? "app" : var.environment
  
  # Load balancer configuration
  alb_dns_name = module.load_balancer.dns_name
  alb_zone_id  = module.load_balancer.zone_id
  
  tags = local.common_tags
}