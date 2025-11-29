# SuperTool Dev Environment Outputs

# VPC Outputs
output "vpc_id" {
  description = "VPC ID"
  value       = module.vpc.vpc_id
}

output "vpc_cidr" {
  description = "VPC CIDR block"
  value       = module.vpc.vpc_cidr
}

output "public_subnet_ids" {
  description = "Public subnet IDs"
  value       = module.vpc.public_subnet_ids
}

output "private_subnet_ids" {
  description = "Private subnet IDs"
  value       = module.vpc.private_subnet_ids
}

# ECS Outputs
output "supertool_url" {
  description = "URL to access SuperTool"
  value       = module.supertool_ecs.endpoint_url
}

output "alb_dns_name" {
  description = "ALB DNS name"
  value       = module.supertool_ecs.alb_dns_name
}

output "cluster_name" {
  description = "ECS cluster name"
  value       = module.supertool_ecs.cluster_name
}

output "service_name" {
  description = "ECS service name"
  value       = module.supertool_ecs.service_name
}

output "cloudwatch_log_group" {
  description = "CloudWatch log group name"
  value       = module.supertool_ecs.cloudwatch_log_group_name
}

output "security_group_id" {
  description = "ECS tasks security group ID"
  value       = module.supertool_ecs.ecs_tasks_security_group_id
}
