terraform {
  required_version = ">= 1.5.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.23"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.11"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.5"
    }
  }

  backend "s3" {
    bucket         = "mezan-atlas-terraform-state"
    key            = "infrastructure/terraform.tfstate"
    region         = "us-west-2"
    encrypt        = true
    dynamodb_table = "terraform-state-lock"
  }
}

# Configure providers
provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project     = "MEZAN-ATLAS"
      Environment = var.environment
      ManagedBy   = "Terraform"
      Owner       = "meshal@berkeley.edu"
      CostCenter  = var.cost_center
    }
  }
}

provider "kubernetes" {
  host                   = module.eks.cluster_endpoint
  cluster_ca_certificate = base64decode(module.eks.cluster_certificate_authority_data)
  token                  = data.aws_eks_cluster_auth.cluster.token
}

provider "helm" {
  kubernetes {
    host                   = module.eks.cluster_endpoint
    cluster_ca_certificate = base64decode(module.eks.cluster_certificate_authority_data)
    token                  = data.aws_eks_cluster_auth.cluster.token
  }
}

# Data sources
data "aws_eks_cluster_auth" "cluster" {
  name = module.eks.cluster_name
}

data "aws_availability_zones" "available" {
  state = "available"
}

data "aws_caller_identity" "current" {}

# Random resources for unique naming
resource "random_string" "suffix" {
  length  = 8
  special = false
  upper   = false
}

# VPC Module
module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "~> 5.0"

  name = "mezan-atlas-${var.environment}-vpc"
  cidr = var.vpc_cidr

  azs             = data.aws_availability_zones.available.names
  private_subnets = var.private_subnet_cidrs
  public_subnets  = var.public_subnet_cidrs

  enable_nat_gateway   = true
  single_nat_gateway   = var.environment == "development"
  enable_dns_hostnames = true
  enable_dns_support   = true

  enable_flow_log                      = true
  create_flow_log_cloudwatch_iam_role  = true
  create_flow_log_cloudwatch_log_group = true

  public_subnet_tags = {
    "kubernetes.io/role/elb"                        = 1
    "kubernetes.io/cluster/${local.cluster_name}"   = "shared"
  }

  private_subnet_tags = {
    "kubernetes.io/role/internal-elb"               = 1
    "kubernetes.io/cluster/${local.cluster_name}"   = "shared"
  }

  tags = {
    "kubernetes.io/cluster/${local.cluster_name}" = "shared"
  }
}

# EKS Module
module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "~> 19.0"

  cluster_name    = local.cluster_name
  cluster_version = var.kubernetes_version

  cluster_endpoint_private_access = true
  cluster_endpoint_public_access  = true
  cluster_endpoint_public_access_cidrs = var.cluster_endpoint_public_access_cidrs

  cluster_addons = {
    coredns = {
      most_recent = true
    }
    kube-proxy = {
      most_recent = true
    }
    vpc-cni = {
      most_recent = true
    }
    aws-ebs-csi-driver = {
      most_recent = true
    }
  }

  vpc_id                   = module.vpc.vpc_id
  subnet_ids               = module.vpc.private_subnets
  control_plane_subnet_ids = module.vpc.private_subnets

  # Enable IRSA
  enable_irsa = true

  # Cluster security group
  cluster_security_group_additional_rules = {
    egress_nodes_ephemeral_ports_tcp = {
      description                = "To node 1025-65535"
      protocol                   = "tcp"
      from_port                  = 1025
      to_port                    = 65535
      type                       = "egress"
      source_node_security_group = true
    }
  }

  # Node security group
  node_security_group_additional_rules = {
    ingress_self_all = {
      description = "Node to node all ports/protocols"
      protocol    = "-1"
      from_port   = 0
      to_port     = 0
      type        = "ingress"
      self        = true
    }
    ingress_cluster_all = {
      description                   = "Cluster to node all ports/protocols"
      protocol                      = "-1"
      from_port                     = 0
      to_port                       = 0
      type                          = "ingress"
      source_cluster_security_group = true
    }
  }

  # EKS Managed Node Groups
  eks_managed_node_groups = {
    general = {
      name            = "general-${var.environment}"
      use_name_prefix = true

      subnet_ids = module.vpc.private_subnets

      min_size     = var.node_group_min_size
      max_size     = var.node_group_max_size
      desired_size = var.node_group_desired_size

      instance_types = var.node_instance_types
      capacity_type  = var.environment == "production" ? "ON_DEMAND" : "SPOT"

      disk_size = 100
      disk_type = "gp3"

      enable_monitoring = true

      labels = {
        Environment = var.environment
        NodeGroup   = "general"
      }

      taints = []

      tags = {
        "k8s.io/cluster-autoscaler/enabled"             = "true"
        "k8s.io/cluster-autoscaler/${local.cluster_name}" = "owned"
      }
    }

    compute = {
      name            = "compute-${var.environment}"
      use_name_prefix = true

      subnet_ids = module.vpc.private_subnets

      min_size     = var.environment == "production" ? 2 : 0
      max_size     = var.environment == "production" ? 10 : 5
      desired_size = var.environment == "production" ? 3 : 1

      instance_types = var.compute_instance_types
      capacity_type  = "SPOT"

      disk_size = 200
      disk_type = "gp3"

      labels = {
        Environment = var.environment
        NodeGroup   = "compute"
        Workload    = "compute-intensive"
      }

      taints = [
        {
          key    = "compute"
          value  = "true"
          effect = "NO_SCHEDULE"
        }
      ]

      tags = {
        "k8s.io/cluster-autoscaler/enabled"             = "true"
        "k8s.io/cluster-autoscaler/${local.cluster_name}" = "owned"
      }
    }
  }

  # aws-auth configmap
  manage_aws_auth_configmap = true

  aws_auth_roles = [
    {
      rolearn  = "arn:aws:iam::${data.aws_caller_identity.current.account_id}:role/DevOps"
      username = "devops"
      groups   = ["system:masters"]
    },
  ]

  aws_auth_users = [
    {
      userarn  = "arn:aws:iam::${data.aws_caller_identity.current.account_id}:user/meshal"
      username = "meshal"
      groups   = ["system:masters"]
    },
  ]

  tags = {
    Environment = var.environment
    GithubRepo  = "AlaweinOS/MEZAN"
  }
}

# RDS PostgreSQL for persistent storage
module "rds" {
  source  = "terraform-aws-modules/rds/aws"
  version = "~> 6.0"

  identifier = "mezan-atlas-${var.environment}-db"

  engine               = "postgres"
  engine_version       = "15.4"
  family               = "postgres15"
  major_engine_version = "15"
  instance_class       = var.db_instance_class

  allocated_storage     = var.db_allocated_storage
  max_allocated_storage = var.db_max_allocated_storage

  db_name  = "mezan_atlas"
  username = "mezan_admin"
  port     = 5432

  multi_az               = var.environment == "production"
  create_db_subnet_group = true
  subnet_ids             = module.vpc.private_subnets
  vpc_security_group_ids = [module.security_group_rds.security_group_id]

  maintenance_window              = "Mon:00:00-Mon:03:00"
  backup_window                   = "03:00-06:00"
  enabled_cloudwatch_logs_exports = ["postgresql"]
  create_cloudwatch_log_group     = true

  backup_retention_period = var.environment == "production" ? 30 : 7
  skip_final_snapshot     = var.environment != "production"
  deletion_protection     = var.environment == "production"

  performance_insights_enabled          = var.environment == "production"
  performance_insights_retention_period = 7
  create_monitoring_role                = true
  monitoring_interval                   = 60
  monitoring_role_name                  = "mezan-atlas-rds-monitoring-${var.environment}"
  monitoring_role_use_name_prefix       = true

  parameters = [
    {
      name  = "autovacuum"
      value = 1
    },
    {
      name  = "client_encoding"
      value = "utf8"
    },
    {
      name  = "shared_preload_libraries"
      value = "pg_stat_statements,pgaudit"
    }
  ]

  tags = {
    Environment = var.environment
  }
}

# ElastiCache Redis Cluster
module "elasticache" {
  source  = "terraform-aws-modules/elasticache/aws"
  version = "~> 1.0"

  cluster_id               = "mezan-atlas-${var.environment}"
  create_cluster           = true
  create_replication_group = var.environment == "production"

  engine          = "redis"
  engine_version  = "7.0"
  node_type       = var.redis_node_type
  num_cache_nodes = var.environment == "production" ? 3 : 1

  subnet_ids         = module.vpc.private_subnets
  security_group_ids = [module.security_group_redis.security_group_id]

  # Replication group settings for production
  replication_group_id       = var.environment == "production" ? "mezan-atlas-${var.environment}" : null
  automatic_failover_enabled = var.environment == "production"
  multi_az_enabled           = var.environment == "production"

  snapshot_retention_limit = var.environment == "production" ? 5 : 1
  snapshot_window          = "03:00-05:00"

  apply_immediately = var.environment != "production"

  log_delivery_configuration = [
    {
      destination      = aws_cloudwatch_log_group.redis.name
      destination_type = "cloudwatch-logs"
      log_format       = "json"
      log_type         = "slow-log"
    }
  ]

  tags = {
    Environment = var.environment
  }
}

# S3 Bucket for backups and artifacts
module "s3_bucket" {
  source  = "terraform-aws-modules/s3-bucket/aws"
  version = "~> 3.0"

  bucket = "mezan-atlas-${var.environment}-${random_string.suffix.result}"

  force_destroy = var.environment != "production"

  # Bucket policies
  attach_policy                         = true
  attach_deny_insecure_transport_policy = true
  attach_require_latest_tls_policy      = true

  # S3 bucket-level Public Access Block
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true

  # Encryption
  server_side_encryption_configuration = {
    rule = {
      apply_server_side_encryption_by_default = {
        sse_algorithm = "AES256"
      }
    }
  }

  # Versioning
  versioning = {
    enabled = true
  }

  # Lifecycle rules
  lifecycle_rule = [
    {
      id      = "transition-old-backups"
      enabled = true

      transition = [
        {
          days          = 30
          storage_class = "STANDARD_IA"
        },
        {
          days          = 90
          storage_class = "GLACIER"
        }
      ]

      expiration = {
        days = var.environment == "production" ? 365 : 90
      }
    }
  ]

  tags = {
    Environment = var.environment
    Purpose     = "Backups and Artifacts"
  }
}

# IAM Role for Service Account (IRSA)
module "mezan_atlas_irsa" {
  source  = "terraform-aws-modules/iam/aws//modules/iam-role-for-service-accounts-eks"
  version = "~> 5.0"

  role_name = "mezan-atlas-${var.environment}-irsa"

  attach_external_secrets_policy = true

  oidc_providers = {
    main = {
      provider_arn               = module.eks.oidc_provider_arn
      namespace_service_accounts = ["mezan-atlas:mezan-atlas-sa"]
    }
  }

  tags = {
    Environment = var.environment
  }
}

# Security Groups
module "security_group_rds" {
  source  = "terraform-aws-modules/security-group/aws"
  version = "~> 5.0"

  name        = "mezan-atlas-rds-${var.environment}"
  description = "Security group for RDS PostgreSQL"
  vpc_id      = module.vpc.vpc_id

  ingress_with_source_security_group_id = [
    {
      from_port                = 5432
      to_port                  = 5432
      protocol                 = "tcp"
      description              = "PostgreSQL from EKS nodes"
      source_security_group_id = module.eks.node_security_group_id
    }
  ]

  tags = {
    Environment = var.environment
  }
}

module "security_group_redis" {
  source  = "terraform-aws-modules/security-group/aws"
  version = "~> 5.0"

  name        = "mezan-atlas-redis-${var.environment}"
  description = "Security group for ElastiCache Redis"
  vpc_id      = module.vpc.vpc_id

  ingress_with_source_security_group_id = [
    {
      from_port                = 6379
      to_port                  = 6379
      protocol                 = "tcp"
      description              = "Redis from EKS nodes"
      source_security_group_id = module.eks.node_security_group_id
    }
  ]

  tags = {
    Environment = var.environment
  }
}

# CloudWatch Log Groups
resource "aws_cloudwatch_log_group" "redis" {
  name              = "/aws/elasticache/mezan-atlas-${var.environment}"
  retention_in_days = var.environment == "production" ? 30 : 7

  tags = {
    Environment = var.environment
    Service     = "ElastiCache"
  }
}

# Locals
locals {
  cluster_name = "mezan-atlas-${var.environment}-${random_string.suffix.result}"
}