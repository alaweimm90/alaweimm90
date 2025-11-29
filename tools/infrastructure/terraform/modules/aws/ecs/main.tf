# SuperTool AWS ECS Terraform Module
# Deploys SuperTool to AWS ECS Fargate

terraform {
  required_version = ">= 1.5.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

# ECS Cluster
resource "aws_ecs_cluster" "supertool" {
  name = "${var.environment}-supertool-cluster"

  setting {
    name  = "containerInsights"
    value = var.enable_container_insights ? "enabled" : "disabled"
  }

  tags = merge(
    var.common_tags,
    {
      Name        = "${var.environment}-supertool-cluster"
      Environment = var.environment
      ManagedBy   = "Terraform"
    }
  )
}

# CloudWatch Log Group
resource "aws_cloudwatch_log_group" "supertool" {
  name              = "/ecs/${var.environment}/supertool"
  retention_in_days = var.log_retention_days

  tags = merge(
    var.common_tags,
    {
      Name        = "${var.environment}-supertool-logs"
      Environment = var.environment
    }
  )
}

# ECS Task Definition
resource "aws_ecs_task_definition" "supertool" {
  family                   = "${var.environment}-supertool"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = var.task_cpu
  memory                   = var.task_memory
  execution_role_arn       = aws_iam_role.ecs_execution_role.arn
  task_role_arn            = aws_iam_role.ecs_task_role.arn

  container_definitions = jsonencode([
    {
      name  = "supertool"
      image = "${var.container_image}:${var.container_tag}"

      portMappings = [
        {
          containerPort = var.container_port
          protocol      = "tcp"
        }
      ]

      environment = [
        {
          name  = "NODE_ENV"
          value = var.environment
        },
        {
          name  = "LOG_LEVEL"
          value = var.log_level
        }
      ]

      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.supertool.name
          "awslogs-region"        = var.aws_region
          "awslogs-stream-prefix" = "ecs"
        }
      }

      healthCheck = {
        command     = ["CMD-SHELL", "node -e \"console.log('healthy')\" || exit 1"]
        interval    = 30
        timeout     = 5
        retries     = 3
        startPeriod = 60
      }

      essential = true

      readonlyRootFilesystem = true

      linuxParameters = {
        capabilities = {
          drop = ["ALL"]
        }
      }
    }
  ])

  tags = merge(
    var.common_tags,
    {
      Name        = "${var.environment}-supertool-task"
      Environment = var.environment
    }
  )
}

# ECS Service
resource "aws_ecs_service" "supertool" {
  name            = "${var.environment}-supertool"
  cluster         = aws_ecs_cluster.supertool.id
  task_definition = aws_ecs_task_definition.supertool.arn
  desired_count   = var.desired_count
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = var.private_subnet_ids
    security_groups  = [aws_security_group.ecs_tasks.id]
    assign_public_ip = false
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.supertool.arn
    container_name   = "supertool"
    container_port   = var.container_port
  }

  deployment_configuration {
    maximum_percent         = 200
    minimum_healthy_percent = 100
  }

  deployment_circuit_breaker {
    enable   = true
    rollback = true
  }

  enable_execute_command = var.enable_exec_command

  tags = merge(
    var.common_tags,
    {
      Name        = "${var.environment}-supertool-service"
      Environment = var.environment
    }
  )

  depends_on = [aws_lb_listener.supertool]
}

# Auto Scaling Target
resource "aws_appautoscaling_target" "ecs_target" {
  count = var.enable_autoscaling ? 1 : 0

  max_capacity       = var.autoscaling_max_capacity
  min_capacity       = var.autoscaling_min_capacity
  resource_id        = "service/${aws_ecs_cluster.supertool.name}/${aws_ecs_service.supertool.name}"
  scalable_dimension = "ecs:service:DesiredCount"
  service_namespace  = "ecs"
}

# Auto Scaling Policy - CPU
resource "aws_appautoscaling_policy" "ecs_policy_cpu" {
  count = var.enable_autoscaling ? 1 : 0

  name               = "${var.environment}-supertool-cpu-autoscaling"
  policy_type        = "TargetTrackingScaling"
  resource_id        = aws_appautoscaling_target.ecs_target[0].resource_id
  scalable_dimension = aws_appautoscaling_target.ecs_target[0].scalable_dimension
  service_namespace  = aws_appautoscaling_target.ecs_target[0].service_namespace

  target_tracking_scaling_policy_configuration {
    predefined_metric_specification {
      predefined_metric_type = "ECSServiceAverageCPUUtilization"
    }
    target_value = var.autoscaling_cpu_target
  }
}

# Auto Scaling Policy - Memory
resource "aws_appautoscaling_policy" "ecs_policy_memory" {
  count = var.enable_autoscaling ? 1 : 0

  name               = "${var.environment}-supertool-memory-autoscaling"
  policy_type        = "TargetTrackingScaling"
  resource_id        = aws_appautoscaling_target.ecs_target[0].resource_id
  scalable_dimension = aws_appautoscaling_target.ecs_target[0].scalable_dimension
  service_namespace  = aws_appautoscaling_target.ecs_target[0].service_namespace

  target_tracking_scaling_policy_configuration {
    predefined_metric_specification {
      predefined_metric_type = "ECSServiceAverageMemoryUtilization"
    }
    target_value = var.autoscaling_memory_target
  }
}

# Application Load Balancer
resource "aws_lb" "supertool" {
  name               = "${var.environment}-supertool-alb"
  internal           = var.alb_internal
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb.id]
  subnets            = var.public_subnet_ids

  enable_deletion_protection = var.enable_deletion_protection
  enable_http2               = true
  enable_cross_zone_load_balancing = true

  drop_invalid_header_fields = true

  tags = merge(
    var.common_tags,
    {
      Name        = "${var.environment}-supertool-alb"
      Environment = var.environment
    }
  )
}

# ALB Target Group
resource "aws_lb_target_group" "supertool" {
  name        = "${var.environment}-supertool-tg"
  port        = var.container_port
  protocol    = "HTTP"
  vpc_id      = var.vpc_id
  target_type = "ip"

  health_check {
    enabled             = true
    healthy_threshold   = 2
    interval            = 30
    matcher             = "200"
    path                = "/health"
    port                = "traffic-port"
    protocol            = "HTTP"
    timeout             = 5
    unhealthy_threshold = 3
  }

  deregistration_delay = 30

  tags = merge(
    var.common_tags,
    {
      Name        = "${var.environment}-supertool-tg"
      Environment = var.environment
    }
  )
}

# ALB Listener
resource "aws_lb_listener" "supertool" {
  load_balancer_arn = aws_lb.supertool.arn
  port              = var.alb_listener_port
  protocol          = var.alb_listener_protocol

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.supertool.arn
  }

  dynamic "certificate_arn" {
    for_each = var.alb_listener_protocol == "HTTPS" ? [1] : []
    content {
      certificate_arn = var.acm_certificate_arn
    }
  }
}

# Security Group - ALB
resource "aws_security_group" "alb" {
  name        = "${var.environment}-supertool-alb-sg"
  description = "Security group for SuperTool ALB"
  vpc_id      = var.vpc_id

  ingress {
    description = "HTTP from anywhere"
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = var.allowed_cidr_blocks
  }

  ingress {
    description = "HTTPS from anywhere"
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = var.allowed_cidr_blocks
  }

  egress {
    description = "All outbound traffic"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = merge(
    var.common_tags,
    {
      Name        = "${var.environment}-supertool-alb-sg"
      Environment = var.environment
    }
  )
}

# Security Group - ECS Tasks
resource "aws_security_group" "ecs_tasks" {
  name        = "${var.environment}-supertool-ecs-tasks-sg"
  description = "Security group for SuperTool ECS tasks"
  vpc_id      = var.vpc_id

  ingress {
    description     = "Traffic from ALB"
    from_port       = var.container_port
    to_port         = var.container_port
    protocol        = "tcp"
    security_groups = [aws_security_group.alb.id]
  }

  egress {
    description = "All outbound traffic"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = merge(
    var.common_tags,
    {
      Name        = "${var.environment}-supertool-ecs-tasks-sg"
      Environment = var.environment
    }
  )
}

# IAM Role - ECS Execution Role
resource "aws_iam_role" "ecs_execution_role" {
  name = "${var.environment}-supertool-ecs-execution-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ecs-tasks.amazonaws.com"
        }
      }
    ]
  })

  tags = merge(
    var.common_tags,
    {
      Name        = "${var.environment}-supertool-ecs-execution-role"
      Environment = var.environment
    }
  )
}

resource "aws_iam_role_policy_attachment" "ecs_execution_role_policy" {
  role       = aws_iam_role.ecs_execution_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

# IAM Role - ECS Task Role
resource "aws_iam_role" "ecs_task_role" {
  name = "${var.environment}-supertool-ecs-task-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ecs-tasks.amazonaws.com"
        }
      }
    ]
  })

  tags = merge(
    var.common_tags,
    {
      Name        = "${var.environment}-supertool-ecs-task-role"
      Environment = var.environment
    }
  )
}

# Optional: ECS Exec permissions
resource "aws_iam_role_policy" "ecs_exec_policy" {
  count = var.enable_exec_command ? 1 : 0

  name = "${var.environment}-supertool-ecs-exec-policy"
  role = aws_iam_role.ecs_task_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "ssmmessages:CreateControlChannel",
          "ssmmessages:CreateDataChannel",
          "ssmmessages:OpenControlChannel",
          "ssmmessages:OpenDataChannel"
        ]
        Resource = "*"
      }
    ]
  })
}
