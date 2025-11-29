# SuperTool Azure AKS Deployment Module
# Complete AKS cluster with SuperTool deployment

terraform {
  required_version = ">= 1.5.0"

  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.0"
    }
  }
}

# Resource Group
resource "azurerm_resource_group" "supertool" {
  name     = "${var.environment}-supertool-rg"
  location = var.azure_region

  tags = merge(
    var.common_tags,
    {
      Environment = var.environment
      ManagedBy   = "Terraform"
    }
  )
}

# Azure Container Registry (ACR)
resource "azurerm_container_registry" "supertool" {
  count = var.create_acr ? 1 : 0

  name                = replace("${var.environment}supertoolacr", "-", "")
  resource_group_name = azurerm_resource_group.supertool.name
  location            = azurerm_resource_group.supertool.location
  sku                 = var.acr_sku
  admin_enabled       = false

  identity {
    type = "SystemAssigned"
  }

  tags = var.common_tags
}

# Virtual Network
resource "azurerm_virtual_network" "supertool" {
  name                = "${var.environment}-supertool-vnet"
  resource_group_name = azurerm_resource_group.supertool.name
  location            = azurerm_resource_group.supertool.location
  address_space       = [var.vnet_address_space]

  tags = var.common_tags
}

# AKS Subnet
resource "azurerm_subnet" "aks" {
  name                 = "${var.environment}-aks-subnet"
  resource_group_name  = azurerm_resource_group.supertool.name
  virtual_network_name = azurerm_virtual_network.supertool.name
  address_prefixes     = [var.aks_subnet_address_prefix]
}

# Log Analytics Workspace for Container Insights
resource "azurerm_log_analytics_workspace" "supertool" {
  count = var.enable_container_insights ? 1 : 0

  name                = "${var.environment}-supertool-logs"
  resource_group_name = azurerm_resource_group.supertool.name
  location            = azurerm_resource_group.supertool.location
  sku                 = "PerGB2018"
  retention_in_days   = var.log_retention_days

  tags = var.common_tags
}

# AKS Cluster
resource "azurerm_kubernetes_cluster" "supertool" {
  name                = "${var.environment}-supertool-aks"
  resource_group_name = azurerm_resource_group.supertool.name
  location            = azurerm_resource_group.supertool.location
  dns_prefix          = "${var.environment}-supertool"
  kubernetes_version  = var.kubernetes_version

  # Default node pool
  default_node_pool {
    name                = "default"
    node_count          = var.node_count
    vm_size             = var.node_vm_size
    os_disk_size_gb     = var.node_os_disk_size
    vnet_subnet_id      = azurerm_subnet.aks.id
    enable_auto_scaling = var.enable_autoscaling
    min_count           = var.enable_autoscaling ? var.min_node_count : null
    max_count           = var.enable_autoscaling ? var.max_node_count : null
    max_pods            = 110

    upgrade_settings {
      max_surge = "33%"
    }

    tags = var.common_tags
  }

  # Identity
  identity {
    type = "SystemAssigned"
  }

  # Network Profile
  network_profile {
    network_plugin     = "azure"
    network_policy     = "azure"
    dns_service_ip     = var.dns_service_ip
    service_cidr       = var.service_cidr
    load_balancer_sku  = "standard"
  }

  # Azure AD Integration
  azure_active_directory_role_based_access_control {
    managed                = true
    azure_rbac_enabled     = true
    admin_group_object_ids = var.admin_group_object_ids
  }

  # Monitoring
  dynamic "oms_agent" {
    for_each = var.enable_container_insights ? [1] : []
    content {
      log_analytics_workspace_id = azurerm_log_analytics_workspace.supertool[0].id
    }
  }

  # Security
  key_vault_secrets_provider {
    secret_rotation_enabled = true
  }

  # Auto-upgrade
  automatic_channel_upgrade = var.automatic_channel_upgrade

  # Maintenance Window
  maintenance_window {
    allowed {
      day   = "Sunday"
      hours = [2, 3, 4]
    }
  }

  tags = var.common_tags

  lifecycle {
    ignore_changes = [
      default_node_pool[0].node_count
    ]
  }
}

# ACR Role Assignment (Allow AKS to pull from ACR)
resource "azurerm_role_assignment" "aks_acr_pull" {
  count = var.create_acr ? 1 : 0

  principal_id                     = azurerm_kubernetes_cluster.supertool.kubelet_identity[0].object_id
  role_definition_name             = "AcrPull"
  scope                            = azurerm_container_registry.supertool[0].id
  skip_service_principal_aad_check = true
}

# Kubernetes Provider Configuration
provider "kubernetes" {
  host                   = azurerm_kubernetes_cluster.supertool.kube_config[0].host
  client_certificate     = base64decode(azurerm_kubernetes_cluster.supertool.kube_config[0].client_certificate)
  client_key             = base64decode(azurerm_kubernetes_cluster.supertool.kube_config[0].client_key)
  cluster_ca_certificate = base64decode(azurerm_kubernetes_cluster.supertool.kube_config[0].cluster_ca_certificate)
}

# SuperTool Namespace
resource "kubernetes_namespace" "supertool" {
  metadata {
    name = "supertool"

    labels = {
      name        = "supertool"
      environment = var.environment
    }
  }

  depends_on = [azurerm_kubernetes_cluster.supertool]
}

# SuperTool Deployment
resource "kubernetes_deployment" "supertool" {
  metadata {
    name      = "supertool"
    namespace = kubernetes_namespace.supertool.metadata[0].name

    labels = {
      app         = "supertool"
      version     = var.app_version
      environment = var.environment
    }
  }

  spec {
    replicas = var.replica_count

    selector {
      match_labels = {
        app = "supertool"
      }
    }

    template {
      metadata {
        labels = {
          app         = "supertool"
          version     = var.app_version
          environment = var.environment
        }
      }

      spec {
        container {
          name  = "supertool"
          image = "${var.container_image}:${var.container_tag}"

          port {
            container_port = var.container_port
            name           = "http"
          }

          resources {
            requests = {
              cpu    = var.container_cpu_request
              memory = var.container_memory_request
            }
            limits = {
              cpu    = var.container_cpu_limit
              memory = var.container_memory_limit
            }
          }

          liveness_probe {
            http_get {
              path = "/health"
              port = var.container_port
            }
            initial_delay_seconds = 30
            period_seconds        = 10
            timeout_seconds       = 5
            failure_threshold     = 3
          }

          readiness_probe {
            http_get {
              path = "/ready"
              port = var.container_port
            }
            initial_delay_seconds = 10
            period_seconds        = 5
            timeout_seconds       = 3
            failure_threshold     = 3
          }

          env {
            name  = "NODE_ENV"
            value = var.environment == "production" ? "production" : "development"
          }

          env {
            name  = "LOG_LEVEL"
            value = var.log_level
          }

          security_context {
            read_only_root_filesystem = true
            run_as_non_root           = true
            run_as_user               = 1001
            allow_privilege_escalation = false

            capabilities {
              drop = ["ALL"]
            }
          }
        }

        security_context {
          fs_group = 1001
        }
      }
    }

    strategy {
      type = "RollingUpdate"

      rolling_update {
        max_surge       = "25%"
        max_unavailable = "25%"
      }
    }
  }

  depends_on = [
    kubernetes_namespace.supertool,
    azurerm_role_assignment.aks_acr_pull
  ]
}

# SuperTool Service
resource "kubernetes_service" "supertool" {
  metadata {
    name      = "supertool"
    namespace = kubernetes_namespace.supertool.metadata[0].name

    labels = {
      app = "supertool"
    }
  }

  spec {
    type = var.service_type

    selector = {
      app = "supertool"
    }

    port {
      name        = "http"
      port        = 80
      target_port = var.container_port
      protocol    = "TCP"
    }

    session_affinity = "ClientIP"
  }

  depends_on = [kubernetes_deployment.supertool]
}

# Horizontal Pod Autoscaler
resource "kubernetes_horizontal_pod_autoscaler_v2" "supertool" {
  count = var.enable_hpa ? 1 : 0

  metadata {
    name      = "supertool-hpa"
    namespace = kubernetes_namespace.supertool.metadata[0].name
  }

  spec {
    scale_target_ref {
      api_version = "apps/v1"
      kind        = "Deployment"
      name        = kubernetes_deployment.supertool.metadata[0].name
    }

    min_replicas = var.hpa_min_replicas
    max_replicas = var.hpa_max_replicas

    metric {
      type = "Resource"
      resource {
        name = "cpu"
        target {
          type                = "Utilization"
          average_utilization = var.hpa_cpu_target
        }
      }
    }

    metric {
      type = "Resource"
      resource {
        name = "memory"
        target {
          type                = "Utilization"
          average_utilization = var.hpa_memory_target
        }
      }
    }

    behavior {
      scale_down {
        stabilization_window_seconds = 300
        policy {
          type           = "Percent"
          value          = 50
          period_seconds = 60
        }
      }

      scale_up {
        stabilization_window_seconds = 0
        policy {
          type           = "Percent"
          value          = 100
          period_seconds = 60
        }
      }
    }
  }

  depends_on = [kubernetes_deployment.supertool]
}

# Pod Disruption Budget
resource "kubernetes_pod_disruption_budget_v1" "supertool" {
  metadata {
    name      = "supertool-pdb"
    namespace = kubernetes_namespace.supertool.metadata[0].name
  }

  spec {
    min_available = var.pdb_min_available

    selector {
      match_labels = {
        app = "supertool"
      }
    }
  }

  depends_on = [kubernetes_deployment.supertool]
}

# Network Policy
resource "kubernetes_network_policy" "supertool" {
  count = var.enable_network_policy ? 1 : 0

  metadata {
    name      = "supertool-netpol"
    namespace = kubernetes_namespace.supertool.metadata[0].name
  }

  spec {
    pod_selector {
      match_labels = {
        app = "supertool"
      }
    }

    policy_types = ["Ingress", "Egress"]

    ingress {
      from {
        namespace_selector {
          match_labels = {
            name = "ingress-nginx"
          }
        }
      }

      ports {
        protocol = "TCP"
        port     = var.container_port
      }
    }

    egress {
      to {
        namespace_selector {}
      }

      ports {
        protocol = "TCP"
        port     = "53"
      }
      ports {
        protocol = "UDP"
        port     = "53"
      }
    }

    egress {
      to {
        ip_block {
          cidr = "0.0.0.0/0"
        }
      }

      ports {
        protocol = "TCP"
        port     = "443"
      }
    }
  }

  depends_on = [kubernetes_deployment.supertool]
}
