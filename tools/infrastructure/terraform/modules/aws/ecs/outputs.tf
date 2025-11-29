# SuperTool AWS ECS Module Outputs

# Cluster
output "cluster_id" {
  description = "ID of the ECS cluster"
  value       = aws_ecs_cluster.supertool.id
}

output "cluster_arn" {
  description = "ARN of the ECS cluster"
  value       = aws_ecs_cluster.supertool.arn
}

output "cluster_name" {
  description = "Name of the ECS cluster"
  value       = aws_ecs_cluster.supertool.name
}

# Service
output "service_id" {
  description = "ID of the ECS service"
  value       = aws_ecs_service.supertool.id
}

output "service_name" {
  description = "Name of the ECS service"
  value       = aws_ecs_service.supertool.name
}

# Task Definition
output "task_definition_arn" {
  description = "ARN of the task definition"
  value       = aws_ecs_task_definition.supertool.arn
}

output "task_definition_family" {
  description = "Family of the task definition"
  value       = aws_ecs_task_definition.supertool.family
}

output "task_definition_revision" {
  description = "Revision of the task definition"
  value       = aws_ecs_task_definition.supertool.revision
}

# Load Balancer
output "alb_id" {
  description = "ID of the Application Load Balancer"
  value       = aws_lb.supertool.id
}

output "alb_arn" {
  description = "ARN of the Application Load Balancer"
  value       = aws_lb.supertool.arn
}

output "alb_dns_name" {
  description = "DNS name of the Application Load Balancer"
  value       = aws_lb.supertool.dns_name
}

output "alb_zone_id" {
  description = "Zone ID of the Application Load Balancer"
  value       = aws_lb.supertool.zone_id
}

output "target_group_arn" {
  description = "ARN of the target group"
  value       = aws_lb_target_group.supertool.arn
}

# Security Groups
output "alb_security_group_id" {
  description = "ID of the ALB security group"
  value       = aws_security_group.alb.id
}

output "ecs_tasks_security_group_id" {
  description = "ID of the ECS tasks security group"
  value       = aws_security_group.ecs_tasks.id
}

# IAM Roles
output "ecs_execution_role_arn" {
  description = "ARN of the ECS execution role"
  value       = aws_iam_role.ecs_execution_role.arn
}

output "ecs_task_role_arn" {
  description = "ARN of the ECS task role"
  value       = aws_iam_role.ecs_task_role.arn
}

# CloudWatch
output "cloudwatch_log_group_name" {
  description = "Name of the CloudWatch log group"
  value       = aws_cloudwatch_log_group.supertool.name
}

output "cloudwatch_log_group_arn" {
  description = "ARN of the CloudWatch log group"
  value       = aws_cloudwatch_log_group.supertool.arn
}

# Auto Scaling
output "autoscaling_target_id" {
  description = "ID of the autoscaling target"
  value       = var.enable_autoscaling ? aws_appautoscaling_target.ecs_target[0].id : null
}

# Connection Details
output "endpoint_url" {
  description = "URL to access SuperTool"
  value       = "${var.alb_listener_protocol == "HTTPS" ? "https" : "http"}://${aws_lb.supertool.dns_name}"
}

output "health_check_url" {
  description = "Health check endpoint URL"
  value       = "${var.alb_listener_protocol == "HTTPS" ? "https" : "http"}://${aws_lb.supertool.dns_name}/health"
}
