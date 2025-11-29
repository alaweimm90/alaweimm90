# SuperTool VPC Networking Module Outputs

# VPC
output "vpc_id" {
  description = "VPC ID"
  value       = aws_vpc.main.id
}

output "vpc_arn" {
  description = "VPC ARN"
  value       = aws_vpc.main.arn
}

output "vpc_cidr" {
  description = "VPC CIDR block"
  value       = aws_vpc.main.cidr_block
}

# Subnets
output "public_subnet_ids" {
  description = "List of public subnet IDs"
  value       = aws_subnet.public[*].id
}

output "private_subnet_ids" {
  description = "List of private subnet IDs"
  value       = aws_subnet.private[*].id
}

output "database_subnet_ids" {
  description = "List of database subnet IDs"
  value       = aws_subnet.database[*].id
}

output "public_subnet_cidrs" {
  description = "List of public subnet CIDR blocks"
  value       = aws_subnet.public[*].cidr_block
}

output "private_subnet_cidrs" {
  description = "List of private subnet CIDR blocks"
  value       = aws_subnet.private[*].cidr_block
}

output "database_subnet_cidrs" {
  description = "List of database subnet CIDR blocks"
  value       = aws_subnet.database[*].cidr_block
}

# Availability Zones
output "availability_zones" {
  description = "List of availability zones"
  value       = var.azs
}

# Internet Gateway
output "internet_gateway_id" {
  description = "Internet Gateway ID"
  value       = var.create_internet_gateway ? aws_internet_gateway.main[0].id : null
}

# NAT Gateways
output "nat_gateway_ids" {
  description = "List of NAT Gateway IDs"
  value       = aws_nat_gateway.main[*].id
}

output "nat_gateway_public_ips" {
  description = "List of NAT Gateway public IPs"
  value       = aws_eip.nat[*].public_ip
}

# Route Tables
output "public_route_table_ids" {
  description = "List of public route table IDs"
  value       = aws_route_table.public[*].id
}

output "private_route_table_ids" {
  description = "List of private route table IDs"
  value       = aws_route_table.private[*].id
}

output "database_route_table_ids" {
  description = "List of database route table IDs"
  value       = aws_route_table.database[*].id
}

# VPN Gateway
output "vpn_gateway_id" {
  description = "VPN Gateway ID"
  value       = var.enable_vpn_gateway ? aws_vpn_gateway.main[0].id : null
}

# VPC Flow Logs
output "flow_logs_id" {
  description = "VPC Flow Logs ID"
  value       = var.enable_flow_logs ? aws_flow_log.main[0].id : null
}

output "flow_logs_log_group_name" {
  description = "CloudWatch Log Group name for Flow Logs"
  value       = var.enable_flow_logs ? aws_cloudwatch_log_group.flow_logs[0].name : null
}

output "flow_logs_log_group_arn" {
  description = "CloudWatch Log Group ARN for Flow Logs"
  value       = var.enable_flow_logs ? aws_cloudwatch_log_group.flow_logs[0].arn : null
}

# VPC Endpoints
output "s3_endpoint_id" {
  description = "S3 VPC Endpoint ID"
  value       = var.enable_s3_endpoint ? aws_vpc_endpoint.s3[0].id : null
}

output "ecr_api_endpoint_id" {
  description = "ECR API VPC Endpoint ID"
  value       = var.enable_ecr_endpoints ? aws_vpc_endpoint.ecr_api[0].id : null
}

output "ecr_dkr_endpoint_id" {
  description = "ECR DKR VPC Endpoint ID"
  value       = var.enable_ecr_endpoints ? aws_vpc_endpoint.ecr_dkr[0].id : null
}

output "vpc_endpoints_security_group_id" {
  description = "Security Group ID for VPC Endpoints"
  value       = var.enable_ecr_endpoints ? aws_security_group.vpc_endpoints[0].id : null
}

# Network Summary
output "network_summary" {
  description = "Summary of network configuration"
  value = {
    vpc_id              = aws_vpc.main.id
    vpc_cidr            = aws_vpc.main.cidr_block
    availability_zones  = var.azs
    public_subnets      = length(aws_subnet.public)
    private_subnets     = length(aws_subnet.private)
    database_subnets    = length(aws_subnet.database)
    nat_gateways        = length(aws_nat_gateway.main)
    flow_logs_enabled   = var.enable_flow_logs
    s3_endpoint_enabled = var.enable_s3_endpoint
    ecr_endpoints_enabled = var.enable_ecr_endpoints
  }
}
