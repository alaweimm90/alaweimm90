# SuperTool Production Environment Outputs

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
