$ErrorActionPreference = 'Stop'

$catalogPath = Join-Path $PSScriptRoot '..' '.metaHub' 'service-catalog.json'
$templatePath = Join-Path $PSScriptRoot '..' '.metaHub' 'templates' 'ai-starter'
$reposRoot = Join-Path $PSScriptRoot '..' 'repos'

Write-Host "Reading catalog: $catalogPath"
$catalog = Get-Content $catalogPath | ConvertFrom-Json
$hasGit = $null -ne (Get-Command git -ErrorAction SilentlyContinue)

if (-not (Test-Path $templatePath)) {
  throw "Template path not found: $templatePath"
}

New-Item -ItemType Directory -Force -Path $reposRoot | Out-Null

foreach ($svc in $catalog.services) {
  $name = $svc.name
  $dest = Join-Path $reposRoot $name
  Write-Host "Creating repo: $name at $dest (template: $templatePath)"
  Copy-Item -Recurse -Force $templatePath $dest

  # Customize package.json name
  $pkgPath = Join-Path $dest 'package.json'
  if (Test-Path $pkgPath) {
    $pkg = Get-Content $pkgPath | ConvertFrom-Json
    $pkg.name = $name
    $pkg | ConvertTo-Json -Depth 10 | Set-Content $pkgPath -Encoding UTF8
  }

  # Add Dockerfile for services
  $dockerfile = @(
    "# syntax=docker/dockerfile:1",
    "FROM node:20.10-alpine@sha256:0d3b1e18b0a0cbecbd",
    "WORKDIR /app",
    "COPY package*.json ./",
    "RUN npm ci --only=production",
    "COPY --chown=node:node . .",
    "USER node",
    "EXPOSE 3000",
    "HEALTHCHECK --interval=30s --timeout=5s CMD node -e \"require('http').request({host:'localhost',port:3000,path:'/'},r=>r.statusCode===200?process.exit(0):process.exit(1)).on('error',()=>process.exit(1)).end()\"",
    "CMD [\"node\", \"dist/index.js\"]"
  )
  Set-Content -Path (Join-Path $dest 'Dockerfile') -Value ($dockerfile -join "`n") -Encoding UTF8

  # Initialize git repo (no commit per policy)
  if ($hasGit -and -not (Test-Path (Join-Path $dest '.git'))) {
    git -C $dest init | Out-Null
  }
}

Write-Host "Done. Repos created under: $reposRoot"
