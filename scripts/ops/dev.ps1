Param([string]$Registry='127.0.0.1:5000')

function Build-All {
  docker compose --env-file .env build
}
function Up-Core {
  docker compose up -d simcore repz benchbarrier attributa mag-logic custom-exporters infra
}
function Up-Obs {
  docker compose --env-file .env up -d traefik status prometheus grafana loki promtail node-exporter cadvisor
}
function Down-All {
  docker compose down --remove-orphans
}
function Push-Latest {
  foreach ($img in @('simcore','repz','benchbarrier','mag-logic','attributa','custom-exporters','infra')) {
    docker tag $img`:latest $Registry/$img`:latest
    docker push $Registry/$img`:latest
  }
}
function Status {
  docker ps --format "table {{.Names}}`t{{.Status}}`t{{.Ports}}"
}
Export-ModuleMember -Function *
