# SuperTool Azure AKS Module Outputs

# AKS Cluster
output "cluster_name" {
  description = "AKS cluster name"
  value       = azurerm_kubernetes_cluster.supertool.name
}

output "cluster_id" {
  description = "AKS cluster ID"
  value       = azurerm_kubernetes_cluster.supertool.id
}

output "cluster_fqdn" {
  description = "AKS cluster FQDN"
  value       = azurerm_kubernetes_cluster.supertool.fqdn
}

output "kube_config" {
  description = "Kubernetes configuration"
  value       = azurerm_kubernetes_cluster.supertool.kube_config_raw
  sensitive   = true
}

output "cluster_endpoint" {
  description = "Kubernetes API server endpoint"
  value       = azurerm_kubernetes_cluster.supertool.kube_config[0].host
}

# Node Pool
output "node_resource_group" {
  description = "Node resource group name"
  value       = azurerm_kubernetes_cluster.supertool.node_resource_group
}

output "kubelet_identity" {
  description = "Kubelet managed identity"
  value = {
    client_id   = azurerm_kubernetes_cluster.supertool.kubelet_identity[0].client_id
    object_id   = azurerm_kubernetes_cluster.supertool.kubelet_identity[0].object_id
    user_assigned_identity_id = azurerm_kubernetes_cluster.supertool.kubelet_identity[0].user_assigned_identity_id
  }
}

# Container Registry
output "acr_id" {
  description = "Azure Container Registry ID"
  value       = var.create_acr ? azurerm_container_registry.supertool[0].id : null
}

output "acr_login_server" {
  description = "ACR login server URL"
  value       = var.create_acr ? azurerm_container_registry.supertool[0].login_server : null
}

output "acr_name" {
  description = "ACR name"
  value       = var.create_acr ? azurerm_container_registry.supertool[0].name : null
}

# Networking
output "resource_group_name" {
  description = "Resource group name"
  value       = azurerm_resource_group.supertool.name
}

output "vnet_id" {
  description = "Virtual network ID"
  value       = azurerm_virtual_network.supertool.id
}

output "vnet_name" {
  description = "Virtual network name"
  value       = azurerm_virtual_network.supertool.name
}

output "aks_subnet_id" {
  description = "AKS subnet ID"
  value       = azurerm_subnet.aks.id
}

# Monitoring
output "log_analytics_workspace_id" {
  description = "Log Analytics workspace ID"
  value       = var.enable_container_insights ? azurerm_log_analytics_workspace.supertool[0].id : null
}

output "log_analytics_workspace_name" {
  description = "Log Analytics workspace name"
  value       = var.enable_container_insights ? azurerm_log_analytics_workspace.supertool[0].name : null
}

# Kubernetes Resources
output "namespace" {
  description = "SuperTool Kubernetes namespace"
  value       = kubernetes_namespace.supertool.metadata[0].name
}

output "service_name" {
  description = "SuperTool Kubernetes service name"
  value       = kubernetes_service.supertool.metadata[0].name
}

output "service_endpoint" {
  description = "SuperTool service endpoint"
  value       = var.service_type == "LoadBalancer" ? (
    length(kubernetes_service.supertool.status[0].load_balancer[0].ingress) > 0 ?
    kubernetes_service.supertool.status[0].load_balancer[0].ingress[0].ip : "pending"
  ) : null
}

output "deployment_name" {
  description = "SuperTool deployment name"
  value       = kubernetes_deployment.supertool.metadata[0].name
}

# Connection Commands
output "get_credentials_command" {
  description = "Command to get AKS credentials"
  value       = "az aks get-credentials --resource-group ${azurerm_resource_group.supertool.name} --name ${azurerm_kubernetes_cluster.supertool.name}"
}

output "kubectl_config" {
  description = "kubectl configuration context"
  value       = azurerm_kubernetes_cluster.supertool.name
}
