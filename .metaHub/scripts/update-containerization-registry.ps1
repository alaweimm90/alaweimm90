# ============================================================================
# Update Project Registry - Containerization Status & Health Scores
# ============================================================================
# Updates the projects-registry.json with containerized status and health scores
# for the 5 priority projects that were successfully containerized

param(
    [Parameter(Mandatory=$false)]
    [string]$RegistryPath = ".metaHub\projects-registry.json",

    [Parameter(Mandatory=$false)]
    [switch]$WhatIf
)

# Project IDs for the 5 containerized projects
$containerizedProjects = @{
    "proj-001" = @{
        name = "SimCore"
        org = "AlaweinOS"
        healthScoreIncrement = 1
    }
    "proj-005" = @{
        name = "repz"
        org = "alaweimm90-business"
        healthScoreIncrement = 1
    }
    "proj-018" = @{
        name = "benchbarrier"
        org = "alaweimm90-business"
        healthScoreIncrement = 1
    }
    "proj-034" = @{
        name = "mag-logic"
        org = "alaweimm90-science"
        healthScoreIncrement = 1
    }
    "proj-040" = @{
        name = "Attributa"
        org = "AlaweinOS"
        healthScoreIncrement = 1
    }
}

Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host "Project Registry Update - Containerization Phase 2" -ForegroundColor Cyan
Write-Host "============================================================================`n" -ForegroundColor Cyan

# Check if registry exists
if (-not (Test-Path $RegistryPath)) {
    Write-Host "ERROR: Registry file not found at: $RegistryPath" -ForegroundColor Red
    exit 1
}

# Read the registry
Write-Host "Reading registry from: $RegistryPath" -ForegroundColor Yellow
$registry = Get-Content $RegistryPath -Raw | ConvertFrom-Json

if (-not $registry) {
    Write-Host "ERROR: Failed to parse registry JSON" -ForegroundColor Red
    exit 1
}

Write-Host "Registry loaded successfully. Total projects: $($registry.projects.Count)`n" -ForegroundColor Green

# Track changes
$updatedCount = 0
$alreadyContainerized = 0
$notFound = @()
$changes = @()

# Update each project
foreach ($projectId in $containerizedProjects.Keys) {
    $projectInfo = $containerizedProjects[$projectId]

    Write-Host "Processing: $projectId ($($projectInfo.name))" -ForegroundColor Cyan

    # Find the project in registry
    $project = $registry.projects | Where-Object { $_.id -eq $projectId }

    if (-not $project) {
        Write-Host "  WARNING: Project not found in registry!" -ForegroundColor Yellow
        $notFound += $projectId
        continue
    }

    # Check current state
    $wasContainerized = $project.containerized -eq $true
    $oldHealthScore = $project.healthScore

    if ($wasContainerized) {
        Write-Host "  Already containerized: true" -ForegroundColor Yellow
        $alreadyContainerized++
    } else {
        Write-Host "  Containerized: false -> true" -ForegroundColor Green
    }

    # Update containerization status
    if (-not $WhatIf -and -not $wasContainerized) {
        $project.containerized = $true
    }

    # Update health score (cap at 10)
    $newHealthScore = [Math]::Min($oldHealthScore + $projectInfo.healthScoreIncrement, 10)

    if ($newHealthScore -ne $oldHealthScore) {
        Write-Host "  Health Score: $oldHealthScore -> $newHealthScore" -ForegroundColor Green

        if (-not $WhatIf) {
            $project.healthScore = $newHealthScore
        }
    } else {
        Write-Host "  Health Score: $oldHealthScore (already at max)" -ForegroundColor Yellow
    }

    # Record change
    $changes += @{
        projectId = $projectId
        name = $projectInfo.name
        org = $projectInfo.org
        wasContainerized = $wasContainerized
        oldHealthScore = $oldHealthScore
        newHealthScore = $newHealthScore
    }

    if (-not $wasContainerized) {
        $updatedCount++
    }

    Write-Host ""
}

# Calculate updated metrics
$totalProjects = $registry.projects.Count
$containerizedCount = ($registry.projects | Where-Object { $_.containerized -eq $true }).Count
$containerizationRate = [Math]::Round(($containerizedCount / $totalProjects) * 100, 1)

Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host "Update Summary" -ForegroundColor Cyan
Write-Host "============================================================================`n" -ForegroundColor Cyan

Write-Host "Projects updated: $updatedCount" -ForegroundColor Green
Write-Host "Already containerized: $alreadyContainerized" -ForegroundColor Yellow
Write-Host "Not found in registry: $($notFound.Count)" -ForegroundColor $(if ($notFound.Count -gt 0) { "Red" } else { "Green" })

if ($notFound.Count -gt 0) {
    Write-Host "  Missing projects: $($notFound -join ', ')" -ForegroundColor Red
}

Write-Host "`nRegistry Metrics:" -ForegroundColor Cyan
Write-Host "  Total projects: $totalProjects" -ForegroundColor White
Write-Host "  Containerized: $containerizedCount" -ForegroundColor White
Write-Host "  Containerization rate: $containerizationRate%" -ForegroundColor White

# Update registry metadata
if (-not $WhatIf) {
    $registry.lastUpdated = (Get-Date -Format "yyyy-MM-ddTHH:mm:ssZ")
    $registry.containerizationRate = $containerizationRate
}

# Save updated registry
if ($WhatIf) {
    Write-Host "`n[WhatIf] Skipping registry save. Run without -WhatIf to apply changes." -ForegroundColor Yellow
} else {
    Write-Host "`nSaving updated registry..." -ForegroundColor Yellow

    # Create backup
    $backupPath = "$RegistryPath.backup-$(Get-Date -Format 'yyyyMMdd-HHmmss')"
    Copy-Item $RegistryPath $backupPath
    Write-Host "  Backup created: $backupPath" -ForegroundColor Green

    # Save updated registry
    $registry | ConvertTo-Json -Depth 100 | Set-Content $RegistryPath
    Write-Host "  Registry saved successfully" -ForegroundColor Green
}

# Output change log
Write-Host "`n============================================================================" -ForegroundColor Cyan
Write-Host "Change Log" -ForegroundColor Cyan
Write-Host "============================================================================`n" -ForegroundColor Cyan

$changes | Format-Table -Property `
    @{Label="Project ID"; Expression={$_.projectId}}, `
    @{Label="Name"; Expression={$_.name}}, `
    @{Label="Organization"; Expression={$_.org}}, `
    @{Label="Containerized"; Expression={if ($_.wasContainerized) {"Yes"} else {"No -> Yes"}}}, `
    @{Label="Health Score"; Expression={"$($_.oldHealthScore) -> $($_.newHealthScore)"}} `
    -AutoSize

Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host "Update Complete!" -ForegroundColor Green
Write-Host "============================================================================`n" -ForegroundColor Cyan

# Return summary object
return @{
    success = $true
    updatedCount = $updatedCount
    alreadyContainerized = $alreadyContainerized
    notFound = $notFound
    containerizationRate = $containerizationRate
    changes = $changes
}
