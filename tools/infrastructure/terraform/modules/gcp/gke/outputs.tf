# SuperTool GCP GKE Module Outputs

# GKE Cluster
output "cluster_name" {
  description = "GKE cluster name"
  value       = google_container_cluster.supertool.name
}

output "cluster_id" {
  description = "GKE cluster ID"
  value       = google_container_cluster.supertool.id
}

output "cluster_endpoint" {
  description = "GKE cluster endpoint"
  value       = google_container_cluster.supertool.endpoint
}

output "cluster_ca_certificate" {
  description = "Cluster CA certificate"
  value       = google_container_cluster.supertool.master_auth[0].cluster_ca_certificate
  sensitive   = true
}

output "cluster_location" {
  description = "GKE cluster location"
  value       = google_container_cluster.supertool.location
}

# Networking
output "network_name" {
  description = "VPC network name"
  value       = google_compute_network.supertool.name
}

output "network_id" {
  description = "VPC network ID"
  value       = google_compute_network.supertool.id
}

output "subnet_name" {
  description = "Subnet name"
  value       = google_compute_subnetwork.gke.name
}

output "subnet_id" {
  description = "Subnet ID"
  value       = google_compute_subnetwork.gke.id
}

output "subnet_cidr" {
  description = "Subnet CIDR range"
  value       = google_compute_subnetwork.gke.ip_cidr_range
}

output "pods_cidr" {
  description = "Pods secondary CIDR range"
  value       = google_compute_subnetwork.gke.secondary_ip_range[0].ip_cidr_range
}

output "services_cidr" {
  description = "Services secondary CIDR range"
  value       = google_compute_subnetwork.gke.secondary_ip_range[1].ip_cidr_range
}

# Artifact Registry
output "artifact_registry_id" {
  description = "Artifact Registry repository ID"
  value       = var.enable_artifact_registry ? google_artifact_registry_repository.supertool[0].id : null
}

output "artifact_registry_name" {
  description = "Artifact Registry repository name"
  value       = var.enable_artifact_registry ? google_artifact_registry_repository.supertool[0].name : null
}

output "artifact_registry_url" {
  description = "Artifact Registry repository URL"
  value = var.enable_artifact_registry ? (
    "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.supertool[0].repository_id}"
  ) : null
}

# Service Accounts
output "node_service_account_email" {
  description = "Node pool service account email"
  value       = google_service_account.gke_nodes.email
}

output "app_service_account_email" {
  description = "Application service account email"
  value       = google_service_account.supertool_app.email
}

# Node Pool
output "node_pool_name" {
  description = "Primary node pool name"
  value       = google_container_node_pool.primary.name
}

output "node_pool_id" {
  description = "Primary node pool ID"
  value       = google_container_node_pool.primary.id
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
  description = "Command to get GKE credentials"
  value       = "gcloud container clusters get-credentials ${google_container_cluster.supertool.name} --region ${var.region} --project ${var.project_id}"
}

output "kubectl_context" {
  description = "kubectl context name"
  value       = "gke_${var.project_id}_${google_container_cluster.supertool.location}_${google_container_cluster.supertool.name}"
}

# Monitoring
output "monitoring_dashboard_url" {
  description = "GKE monitoring dashboard URL"
  value       = "https://console.cloud.google.com/kubernetes/clusters/details/${google_container_cluster.supertool.location}/${google_container_cluster.supertool.name}/observability?project=${var.project_id}"
}

output "workloads_url" {
  description = "GKE workloads URL"
  value       = "https://console.cloud.google.com/kubernetes/workload?project=${var.project_id}"
}
