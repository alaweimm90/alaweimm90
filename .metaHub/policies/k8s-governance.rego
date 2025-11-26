# Kubernetes Governance Policy
# Enforces basic labels and security best practices in K8s manifests

package k8s_governance

default deny := []
default warn := []

is_k8s := input.kind

# Require owner and environment labels on all metadata
deny[msg] {
  is_k8s
  not input.metadata.labels.owner
  msg := sprintf("%s/%s missing label 'owner'", [input.kind, input.metadata.name])
}

deny[msg] {
  is_k8s
  not input.metadata.labels.env
  msg := sprintf("%s/%s missing label 'env'", [input.kind, input.metadata.name])
}

# Containers should not run privileged
deny[msg] {
  is_k8s
  c := input.spec.template.spec.containers[_]
  c.securityContext.privileged
  msg := sprintf("Container '%s' in %s/%s runs privileged", [c.name, input.kind, input.metadata.name])
}

# Recommend resource requests/limits
warn[msg] {
  is_k8s
  c := input.spec.template.spec.containers[_]
  not c.resources.requests
  msg := sprintf("Container '%s' missing resource requests", [c.name])
}

warn[msg] {
  is_k8s
  c := input.spec.template.spec.containers[_]
  not c.resources.limits
  msg := sprintf("Container '%s' missing resource limits", [c.name])
}

