Param(
  [string[]]$Images = @('simcore','repz','benchbarrier','mag-logic','attributa','custom-exporters','infra'),
  [string]$Registry = '127.0.0.1:5000'
)

function Get-GitSha {
  try { (git rev-parse --short HEAD).Trim() } catch { "latest" }
}

$sha = Get-GitSha
Write-Output "Using tag: $sha"

foreach ($img in $Images) {
  $fullLatest = "$Registry/$img`:latest"
  $fullSha = "$Registry/$img`:$sha"
  docker pull $fullLatest | Out-Null
  docker tag $fullLatest $fullSha
  docker push $fullSha
}

Write-Output "Pushed immutable tags for images"
