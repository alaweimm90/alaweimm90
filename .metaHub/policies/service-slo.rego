# Service SLO Policy
# Ensures each service in service-catalog.json declares SLOs

package service_slo

default deny := []
default warn := []

catalog := input

# Deny if catalog missing required fields
deny[msg] {
  not catalog.services
  msg := "service-catalog.json must contain 'services' array"
}

# Warn for each service missing SLO block
warn[msg] {
  svc := catalog.services[_]
  not svc.slo
  msg := sprintf("Service '%s' missing SLO configuration", [svc.name])
}

# Warn when SLO exists but missing key targets
warn[msg] {
  svc := catalog.services[_]
  svc.slo
  not svc.slo.availability
  msg := sprintf("Service '%s' SLO missing 'availability' target", [svc.name])
}

warn[msg] {
  svc := catalog.services[_]
  svc.slo
  not svc.slo.latency
  msg := sprintf("Service '%s' SLO missing 'latency' target", [svc.name])
}

