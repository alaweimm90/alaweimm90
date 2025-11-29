# SuperTool VPC Networking Module
# Cloud-agnostic VPC configuration for AWS

terraform {
  required_version = ">= 1.5.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

# VPC
resource "aws_vpc" "main" {
  cidr_block           = var.vpc_cidr
  enable_dns_hostnames = var.enable_dns_hostnames
  enable_dns_support   = var.enable_dns_support

  tags = merge(
    var.common_tags,
    {
      Name        = "${var.environment}-${var.vpc_name}"
      Environment = var.environment
    }
  )
}

# Internet Gateway
resource "aws_internet_gateway" "main" {
  count = var.create_internet_gateway ? 1 : 0

  vpc_id = aws_vpc.main.id

  tags = merge(
    var.common_tags,
    {
      Name        = "${var.environment}-${var.vpc_name}-igw"
      Environment = var.environment
    }
  )
}

# Elastic IPs for NAT Gateways
resource "aws_eip" "nat" {
  count = var.enable_nat_gateway ? (var.single_nat_gateway ? 1 : length(var.azs)) : 0

  domain = "vpc"

  tags = merge(
    var.common_tags,
    {
      Name        = "${var.environment}-${var.vpc_name}-nat-${count.index + 1}"
      Environment = var.environment
    }
  )

  depends_on = [aws_internet_gateway.main]
}

# Public Subnets
resource "aws_subnet" "public" {
  count = length(var.public_subnets)

  vpc_id                  = aws_vpc.main.id
  cidr_block              = var.public_subnets[count.index]
  availability_zone       = var.azs[count.index]
  map_public_ip_on_launch = true

  tags = merge(
    var.common_tags,
    {
      Name        = "${var.environment}-${var.vpc_name}-public-${var.azs[count.index]}"
      Environment = var.environment
      Type        = "public"
      Tier        = "public"
    }
  )
}

# Private Subnets
resource "aws_subnet" "private" {
  count = length(var.private_subnets)

  vpc_id            = aws_vpc.main.id
  cidr_block        = var.private_subnets[count.index]
  availability_zone = var.azs[count.index]

  tags = merge(
    var.common_tags,
    {
      Name        = "${var.environment}-${var.vpc_name}-private-${var.azs[count.index]}"
      Environment = var.environment
      Type        = "private"
      Tier        = "private"
    }
  )
}

# Database Subnets
resource "aws_subnet" "database" {
  count = length(var.database_subnets)

  vpc_id            = aws_vpc.main.id
  cidr_block        = var.database_subnets[count.index]
  availability_zone = var.azs[count.index]

  tags = merge(
    var.common_tags,
    {
      Name        = "${var.environment}-${var.vpc_name}-database-${var.azs[count.index]}"
      Environment = var.environment
      Type        = "database"
      Tier        = "data"
    }
  )
}

# NAT Gateways
resource "aws_nat_gateway" "main" {
  count = var.enable_nat_gateway ? (var.single_nat_gateway ? 1 : length(var.azs)) : 0

  allocation_id = aws_eip.nat[count.index].id
  subnet_id     = aws_subnet.public[count.index].id

  tags = merge(
    var.common_tags,
    {
      Name        = "${var.environment}-${var.vpc_name}-nat-${var.azs[count.index]}"
      Environment = var.environment
    }
  )

  depends_on = [aws_internet_gateway.main]
}

# Public Route Table
resource "aws_route_table" "public" {
  count = length(var.public_subnets) > 0 ? 1 : 0

  vpc_id = aws_vpc.main.id

  tags = merge(
    var.common_tags,
    {
      Name        = "${var.environment}-${var.vpc_name}-public"
      Environment = var.environment
      Type        = "public"
    }
  )
}

# Public Routes
resource "aws_route" "public_internet_gateway" {
  count = var.create_internet_gateway && length(var.public_subnets) > 0 ? 1 : 0

  route_table_id         = aws_route_table.public[0].id
  destination_cidr_block = "0.0.0.0/0"
  gateway_id             = aws_internet_gateway.main[0].id
}

# Public Route Table Associations
resource "aws_route_table_association" "public" {
  count = length(var.public_subnets)

  subnet_id      = aws_subnet.public[count.index].id
  route_table_id = aws_route_table.public[0].id
}

# Private Route Tables
resource "aws_route_table" "private" {
  count = var.single_nat_gateway ? 1 : length(var.private_subnets)

  vpc_id = aws_vpc.main.id

  tags = merge(
    var.common_tags,
    {
      Name        = "${var.environment}-${var.vpc_name}-private-${var.single_nat_gateway ? "shared" : var.azs[count.index]}"
      Environment = var.environment
      Type        = "private"
    }
  )
}

# Private Routes to NAT Gateway
resource "aws_route" "private_nat_gateway" {
  count = var.enable_nat_gateway ? (var.single_nat_gateway ? 1 : length(var.private_subnets)) : 0

  route_table_id         = aws_route_table.private[count.index].id
  destination_cidr_block = "0.0.0.0/0"
  nat_gateway_id         = aws_nat_gateway.main[var.single_nat_gateway ? 0 : count.index].id
}

# Private Route Table Associations
resource "aws_route_table_association" "private" {
  count = length(var.private_subnets)

  subnet_id      = aws_subnet.private[count.index].id
  route_table_id = aws_route_table.private[var.single_nat_gateway ? 0 : count.index].id
}

# Database Route Tables
resource "aws_route_table" "database" {
  count = length(var.database_subnets) > 0 ? 1 : 0

  vpc_id = aws_vpc.main.id

  tags = merge(
    var.common_tags,
    {
      Name        = "${var.environment}-${var.vpc_name}-database"
      Environment = var.environment
      Type        = "database"
    }
  )
}

# Database Route Table Associations
resource "aws_route_table_association" "database" {
  count = length(var.database_subnets)

  subnet_id      = aws_subnet.database[count.index].id
  route_table_id = aws_route_table.database[0].id
}

# VPC Flow Logs
resource "aws_flow_log" "main" {
  count = var.enable_flow_logs ? 1 : 0

  iam_role_arn    = aws_iam_role.flow_logs[0].arn
  log_destination = aws_cloudwatch_log_group.flow_logs[0].arn
  traffic_type    = var.flow_logs_traffic_type
  vpc_id          = aws_vpc.main.id

  tags = merge(
    var.common_tags,
    {
      Name        = "${var.environment}-${var.vpc_name}-flow-logs"
      Environment = var.environment
    }
  )
}

# CloudWatch Log Group for Flow Logs
resource "aws_cloudwatch_log_group" "flow_logs" {
  count = var.enable_flow_logs ? 1 : 0

  name              = "/aws/vpc/${var.environment}-${var.vpc_name}"
  retention_in_days = var.flow_logs_retention_days

  tags = merge(
    var.common_tags,
    {
      Name        = "${var.environment}-${var.vpc_name}-flow-logs"
      Environment = var.environment
    }
  )
}

# IAM Role for Flow Logs
resource "aws_iam_role" "flow_logs" {
  count = var.enable_flow_logs ? 1 : 0

  name = "${var.environment}-${var.vpc_name}-flow-logs"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "vpc-flow-logs.amazonaws.com"
        }
      }
    ]
  })

  tags = var.common_tags
}

# IAM Policy for Flow Logs
resource "aws_iam_role_policy" "flow_logs" {
  count = var.enable_flow_logs ? 1 : 0

  name = "${var.environment}-${var.vpc_name}-flow-logs"
  role = aws_iam_role.flow_logs[0].id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents",
          "logs:DescribeLogGroups",
          "logs:DescribeLogStreams"
        ]
        Effect   = "Allow"
        Resource = "*"
      }
    ]
  })
}

# VPN Gateway
resource "aws_vpn_gateway" "main" {
  count = var.enable_vpn_gateway ? 1 : 0

  vpc_id = aws_vpc.main.id

  tags = merge(
    var.common_tags,
    {
      Name        = "${var.environment}-${var.vpc_name}-vpn-gw"
      Environment = var.environment
    }
  )
}

# VPC Endpoints for AWS Services (S3)
resource "aws_vpc_endpoint" "s3" {
  count = var.enable_s3_endpoint ? 1 : 0

  vpc_id       = aws_vpc.main.id
  service_name = "com.amazonaws.${var.aws_region}.s3"

  tags = merge(
    var.common_tags,
    {
      Name        = "${var.environment}-${var.vpc_name}-s3-endpoint"
      Environment = var.environment
    }
  )
}

# S3 Endpoint Route Table Associations
resource "aws_vpc_endpoint_route_table_association" "s3_private" {
  count = var.enable_s3_endpoint ? length(aws_route_table.private) : 0

  route_table_id  = aws_route_table.private[count.index].id
  vpc_endpoint_id = aws_vpc_endpoint.s3[0].id
}

# VPC Endpoint for ECR API
resource "aws_vpc_endpoint" "ecr_api" {
  count = var.enable_ecr_endpoints ? 1 : 0

  vpc_id              = aws_vpc.main.id
  service_name        = "com.amazonaws.${var.aws_region}.ecr.api"
  vpc_endpoint_type   = "Interface"
  subnet_ids          = aws_subnet.private[*].id
  security_group_ids  = [aws_security_group.vpc_endpoints[0].id]
  private_dns_enabled = true

  tags = merge(
    var.common_tags,
    {
      Name        = "${var.environment}-${var.vpc_name}-ecr-api-endpoint"
      Environment = var.environment
    }
  )
}

# VPC Endpoint for ECR DKR
resource "aws_vpc_endpoint" "ecr_dkr" {
  count = var.enable_ecr_endpoints ? 1 : 0

  vpc_id              = aws_vpc.main.id
  service_name        = "com.amazonaws.${var.aws_region}.ecr.dkr"
  vpc_endpoint_type   = "Interface"
  subnet_ids          = aws_subnet.private[*].id
  security_group_ids  = [aws_security_group.vpc_endpoints[0].id]
  private_dns_enabled = true

  tags = merge(
    var.common_tags,
    {
      Name        = "${var.environment}-${var.vpc_name}-ecr-dkr-endpoint"
      Environment = var.environment
    }
  )
}

# Security Group for VPC Endpoints
resource "aws_security_group" "vpc_endpoints" {
  count = var.enable_ecr_endpoints ? 1 : 0

  name        = "${var.environment}-${var.vpc_name}-vpc-endpoints"
  description = "Security group for VPC endpoints"
  vpc_id      = aws_vpc.main.id

  ingress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = [var.vpc_cidr]
    description = "HTTPS from VPC"
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
    description = "Allow all outbound"
  }

  tags = merge(
    var.common_tags,
    {
      Name        = "${var.environment}-${var.vpc_name}-vpc-endpoints-sg"
      Environment = var.environment
    }
  )
}

# Default Network ACL
resource "aws_default_network_acl" "main" {
  count = var.manage_default_network_acl ? 1 : 0

  default_network_acl_id = aws_vpc.main.default_network_acl_id

  ingress {
    protocol   = -1
    rule_no    = 100
    action     = "allow"
    cidr_block = "0.0.0.0/0"
    from_port  = 0
    to_port    = 0
  }

  egress {
    protocol   = -1
    rule_no    = 100
    action     = "allow"
    cidr_block = "0.0.0.0/0"
    from_port  = 0
    to_port    = 0
  }

  tags = merge(
    var.common_tags,
    {
      Name        = "${var.environment}-${var.vpc_name}-default-nacl"
      Environment = var.environment
    }
  )
}

# Default Security Group
resource "aws_default_security_group" "main" {
  count = var.manage_default_security_group ? 1 : 0

  vpc_id = aws_vpc.main.id

  tags = merge(
    var.common_tags,
    {
      Name        = "${var.environment}-${var.vpc_name}-default-sg"
      Environment = var.environment
    }
  )
}
