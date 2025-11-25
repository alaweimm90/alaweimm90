$ErrorActionPreference = 'Stop'

param(
  [string]$Owner = 'alaweimm90',
  [string]$Visibility = 'public'
)

if (-not $env:GITHUB_TOKEN) {
  throw 'Set GITHUB_TOKEN env var with repo scope before running.'
}

$headers = @{ Authorization = "Bearer $($env:GITHUB_TOKEN)"; 'User-Agent' = 'metaHub-repo-creator' }
$apiBase = 'https://api.github.com'

$catalogPath = Join-Path $PSScriptRoot '..' '.metaHub' 'service-catalog.json'
$reposRoot = Join-Path $PSScriptRoot '..' 'repos'
$catalog = Get-Content $catalogPath | ConvertFrom-Json

foreach ($svc in $catalog.services) {
  $name = $svc.name
  Write-Host "Creating remote repo: $Owner/$name"
  $body = @{ name = $name; private = ($Visibility -ne 'public') } | ConvertTo-Json
  Invoke-RestMethod -Method Post -Uri "$apiBase/user/repos" -Headers $headers -Body $body -ContentType 'application/json' | Out-Null

  $localPath = Join-Path $reposRoot $name
  if (Test-Path $localPath) {
    if (-not (Test-Path (Join-Path $localPath '.git'))) { git -C $localPath init | Out-Null }
    git -C $localPath remote remove origin 2>$null
    git -C $localPath remote add origin "https://github.com/$Owner/$name.git"
    Write-Host "Repo created. Add commit and push when ready: git -C '$localPath' add .; git -C '$localPath' commit -m 'init'; git -C '$localPath' branch -M main; git -C '$localPath' push -u origin main"
  } else {
    Write-Warning "Local path not found: $localPath"
  }
}

Write-Host "Done. Verify on https://github.com/$Owner?tab=repositories"
