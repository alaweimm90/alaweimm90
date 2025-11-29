# SuperTool GCP GKE Deployment Module
# Complete GKE cluster with SuperTool deployment

terraform {
  required_version = ">= 1.5.0"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.0"
    }
  }
}

# VPC Network
resource "google_compute_network" "supertool" {
  name                    = "${var.environment}-supertool-network"
  auto_create_subnetworks = false
  project                 = var.project_id
}

# Subnet for GKE
resource "google_compute_subnetwork" "gke" {
  name          = "${var.environment}-gke-subnet"
  ip_cidr_range = var.subnet_cidr
  region        = var.region
  network       = google_compute_network.supertool.id
  project       = var.project_id

  secondary_ip_range {
    range_name    = "pods"
    ip_cidr_range = var.pods_cidr
  }

  secondary_ip_range {
    range_name    = "services"
    ip_cidr_range = var.services_cidr
  }

  private_ip_google_access = true
}

# Cloud NAT for private nodes
resource "google_compute_router" "router" {
  name    = "${var.environment}-supertool-router"
  region  = var.region
  network = google_compute_network.supertool.id
  project = var.project_id
}

resource "google_compute_router_nat" "nat" {
  name                               = "${var.environment}-supertool-nat"
  router                             = google_compute_router.router.name
  region                             = var.region
  nat_ip_allocate_option            = "AUTO_ONLY"
  source_subnetwork_ip_ranges_to_nat = "ALL_SUBNETWORKS_ALL_IP_RANGES"
  project                            = var.project_id

  log_config {
    enable = true
    filter = "ERRORS_ONLY"
  }
}

# Service Account for GKE nodes
resource "google_service_account" "gke_nodes" {
  account_id   = "${var.environment}-gke-nodes"
  display_name = "GKE Nodes Service Account"
  project      = var.project_id
}

# IAM bindings for node service account
resource "google_project_iam_member" "gke_nodes_log_writer" {
  project = var.project_id
  role    = "roles/logging.logWriter"
  member  = "serviceAccount:${google_service_account.gke_nodes.email}"
}

resource "google_project_iam_member" "gke_nodes_metric_writer" {
  project = var.project_id
  role    = "roles/monitoring.metricWriter"
  member  = "serviceAccount:${google_service_account.gke_nodes.email}"
}

resource "google_project_iam_member" "gke_nodes_monitoring_viewer" {
  project = var.project_id
  role    = "roles/monitoring.viewer"
  member  = "serviceAccount:${google_service_account.gke_nodes.email}"
}

resource "google_project_iam_member" "gke_nodes_resource_metadata_writer" {
  project = var.project_id
  role    = "roles/stackdriver.resourceMetadata.writer"
  member  = "serviceAccount:${google_service_account.gke_nodes.email}"
}

resource "google_project_iam_member" "gke_nodes_artifact_registry_reader" {
  count   = var.enable_artifact_registry ? 1 : 0
  project = var.project_id
  role    = "roles/artifactregistry.reader"
  member  = "serviceAccount:${google_service_account.gke_nodes.email}"
}

# Artifact Registry Repository
resource "google_artifact_registry_repository" "supertool" {
  count = var.enable_artifact_registry ? 1 : 0

  location      = var.region
  repository_id = "${var.environment}-supertool"
  description   = "SuperTool container images"
  format        = "DOCKER"
  project       = var.project_id

  labels = var.common_labels
}

# GKE Cluster
resource "google_container_cluster" "supertool" {
  name     = "${var.environment}-supertool-gke"
  location = var.regional_cluster ? var.region : var.zone
  project  = var.project_id

  # We can't create a cluster with no node pool, so we create the smallest possible default
  # node pool and immediately delete it.
  remove_default_node_pool = true
  initial_node_count       = 1

  # Networking
  network    = google_compute_network.supertool.id
  subnetwork = google_compute_subnetwork.gke.id

  networking_mode = "VPC_NATIVE"
  ip_allocation_policy {
    cluster_secondary_range_name  = "pods"
    services_secondary_range_name = "services"
  }

  # Private cluster configuration
  private_cluster_config {
    enable_private_nodes    = var.enable_private_nodes
    enable_private_endpoint = var.enable_private_endpoint
    master_ipv4_cidr_block  = var.master_ipv4_cidr_block
  }

  # Master authorized networks
  dynamic "master_authorized_networks_config" {
    for_each = length(var.master_authorized_networks) > 0 ? [1] : []
    content {
      dynamic "cidr_blocks" {
        for_each = var.master_authorized_networks
        content {
          cidr_block   = cidr_blocks.value.cidr_block
          display_name = cidr_blocks.value.display_name
        }
      }
    }
  }

  # Workload Identity
  workload_identity_config {
    workload_pool = "${var.project_id}.svc.id.goog"
  }

  # Addons
  addons_config {
    http_load_balancing {
      disabled = false
    }

    horizontal_pod_autoscaling {
      disabled = false
    }

    network_policy_config {
      disabled = !var.enable_network_policy
    }

    gcp_filestore_csi_driver_config {
      enabled = false
    }

    gcs_fuse_csi_driver_config {
      enabled = false
    }
  }

  # Network Policy
  network_policy {
    enabled  = var.enable_network_policy
    provider = var.enable_network_policy ? "PROVIDER_UNSPECIFIED" : null
  }

  # Binary Authorization
  dynamic "binary_authorization" {
    for_each = var.enable_binary_authorization ? [1] : []
    content {
      evaluation_mode = "PROJECT_SINGLETON_POLICY_ENFORCE"
    }
  }

  # Logging and Monitoring
  logging_service    = var.enable_cloud_logging ? "logging.googleapis.com/kubernetes" : "none"
  monitoring_service = var.enable_cloud_monitoring ? "monitoring.googleapis.com/kubernetes" : "none"

  # Maintenance window
  maintenance_policy {
    daily_maintenance_window {
      start_time = var.maintenance_start_time
    }
  }

  # Release channel
  release_channel {
    channel = var.release_channel
  }

  # Security
  enable_shielded_nodes = true

  # Resource labels
  resource_labels = var.common_labels

  # Cluster autoscaling
  dynamic "cluster_autoscaling" {
    for_each = var.enable_cluster_autoscaling ? [1] : []
    content {
      enabled = true
      resource_limits {
        resource_type = "cpu"
        minimum       = var.cluster_autoscaling_cpu_min
        maximum       = var.cluster_autoscaling_cpu_max
      }
      resource_limits {
        resource_type = "memory"
        minimum       = var.cluster_autoscaling_memory_min
        maximum       = var.cluster_autoscaling_memory_max
      }
      autoscaling_profile = var.cluster_autoscaling_profile
    }
  }

  depends_on = [
    google_project_iam_member.gke_nodes_log_writer,
    google_project_iam_member.gke_nodes_metric_writer,
    google_project_iam_member.gke_nodes_monitoring_viewer,
    google_project_iam_member.gke_nodes_resource_metadata_writer,
  ]
}

# GKE Node Pool
resource "google_container_node_pool" "primary" {
  name       = "${var.environment}-primary-pool"
  location   = var.regional_cluster ? var.region : var.zone
  cluster    = google_container_cluster.supertool.name
  project    = var.project_id
  node_count = var.regional_cluster ? var.node_count_per_zone : var.node_count

  # Autoscaling
  autoscaling {
    min_node_count = var.autoscaling_min_nodes
    max_node_count = var.autoscaling_max_nodes
  }

  # Node configuration
  node_config {
    machine_type = var.machine_type
    disk_size_gb = var.disk_size_gb
    disk_type    = var.disk_type

    # Google recommends custom service accounts with minimal permissions
    service_account = google_service_account.gke_nodes.email
    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform"
    ]

    # Metadata
    metadata = {
      disable-legacy-endpoints = "true"
    }

    # Labels
    labels = merge(
      var.common_labels,
      {
        environment = var.environment
      }
    )

    # Tags
    tags = var.node_tags

    # Shielded instance config
    shielded_instance_config {
      enable_secure_boot          = true
      enable_integrity_monitoring = true
    }

    # Workload metadata config
    workload_metadata_config {
      mode = "GKE_METADATA"
    }

    # Image type
    image_type = "COS_CONTAINERD"

    # Preemptible nodes for dev
    preemptible  = var.environment == "dev" ? var.use_preemptible_nodes : false
    spot         = var.environment == "dev" ? var.use_spot_nodes : false
  }

  # Management
  management {
    auto_repair  = true
    auto_upgrade = true
  }

  # Upgrade settings
  upgrade_settings {
    max_surge       = var.max_surge
    max_unavailable = var.max_unavailable
    strategy        = "SURGE"
  }
}

# Kubernetes Provider Configuration
data "google_client_config" "default" {}

provider "kubernetes" {
  host  = "https://${google_container_cluster.supertool.endpoint}"
  token = data.google_client_config.default.access_token
  cluster_ca_certificate = base64decode(
    google_container_cluster.supertool.master_auth[0].cluster_ca_certificate,
  )
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

  depends_on = [google_container_node_pool.primary]
}

# Workload Identity binding
resource "google_service_account" "supertool_app" {
  account_id   = "${var.environment}-supertool-app"
  display_name = "SuperTool Application Service Account"
  project      = var.project_id
}

resource "google_service_account_iam_member" "supertool_workload_identity" {
  service_account_id = google_service_account.supertool_app.name
  role               = "roles/iam.workloadIdentityUser"
  member             = "serviceAccount:${var.project_id}.svc.id.goog[supertool/supertool]"
}

# Kubernetes Service Account
resource "kubernetes_service_account" "supertool" {
  metadata {
    name      = "supertool"
    namespace = kubernetes_namespace.supertool.metadata[0].name

    annotations = {
      "iam.gke.io/gcp-service-account" = google_service_account.supertool_app.email
    }
  }

  depends_on = [kubernetes_namespace.supertool]
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
        service_account_name = kubernetes_service_account.supertool.metadata[0].name

        container {
          name  = "supertool"
          image = "${var.container_image}:${var.container_tag}"

          port {
            container_port = var.container_port
            name           = "http"
            protocol       = "TCP"
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

          env {
            name  = "GCP_PROJECT"
            value = var.project_id
          }

          security_context {
            read_only_root_filesystem  = true
            run_as_non_root            = true
            run_as_user                = 1001
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
    kubernetes_service_account.supertool,
    google_service_account_iam_member.supertool_workload_identity
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

    annotations = var.service_type == "LoadBalancer" ? {
      "cloud.google.com/neg"                   = "{\"ingress\": true}"
      "cloud.google.com/backend-config"        = kubernetes_service.supertool.metadata[0].name
      "networking.gke.io/load-balancer-type"   = var.load_balancer_type
    } : {}
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
