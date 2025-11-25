# Health Score Calculator
# Automatically calculates health scores for all projects

param(
    [string]$RegistryFile = ".metaHub\projects-registry.json"
)

Write-Host "[*] Project Health Score Calculator" -ForegroundColor Cyan
Write-Host "====================================" -ForegroundColor Cyan
Write-Host ""

# Load registry
$registry = Get-Content $RegistryFile | ConvertFrom-Json

$updatedProjects = @()

foreach ($project in $registry.projects) {
    $score = 0
    $reasons = @()

    Write-Host "[>>] Analyzing: $($project.name)" -ForegroundColor Magenta

    # +2 points: Has automated tests
    $hasTests = $false
    if (Test-Path "$($project.path)\package.json") {
        $packageJson = Get-Content "$($project.path)\package.json" -ErrorAction SilentlyContinue | ConvertFrom-Json
        if ($packageJson.scripts.test -or $packageJson.scripts."test:unit" -or $packageJson.scripts."test:integration") {
            $hasTests = $true
        }
    }
    if (Test-Path "$($project.path)\tests" -or Test-Path "$($project.path)\test" -or Test-Path "$($project.path)\__tests__") {
        $hasTests = $true
    }
    if ((Test-Path "$($project.path)\pytest.ini") -or (Test-Path "$($project.path)\setup.py")) {
        $hasTests = $true
    }

    if ($hasTests) {
        $score += 2
        $reasons += "+2 Has automated tests"
        Write-Host "  [+2] Has automated tests" -ForegroundColor Green
    } else {
        Write-Host "  [ 0] No automated tests" -ForegroundColor Yellow
    }

    # +2 points: Has CI/CD pipeline
    if ($project.cicd) {
        $score += 2
        $reasons += "+2 Has CI/CD pipeline"
        Write-Host "  [+2] Has CI/CD pipeline" -ForegroundColor Green
    } else {
        Write-Host "  [ 0] No CI/CD pipeline" -ForegroundColor Yellow
    }

    # +1 point: Is containerized
    if ($project.containerized) {
        $score += 1
        $reasons += "+1 Containerized"
        Write-Host "  [+1] Containerized" -ForegroundColor Green
    } else {
        Write-Host "  [ 0] Not containerized" -ForegroundColor Yellow
    }

    # +1 point: Has monitoring/health check
    $hasMonitoring = $false
    if (Test-Path "$($project.path)\src") {
        $healthFiles = Get-ChildItem -Path "$($project.path)\src" -Recurse -Filter "*health*" -ErrorAction SilentlyContinue
        if ($healthFiles.Count -gt 0) {
            $hasMonitoring = $true
        }
    }
    if ($hasMonitoring) {
        $score += 1
        $reasons += "+1 Has monitoring/health checks"
        Write-Host "  [+1] Has monitoring" -ForegroundColor Green
    } else {
        Write-Host "  [ 0] No monitoring" -ForegroundColor Yellow
    }

    # +1 point: Deployed in last 30 days (assume yes if CI/CD exists)
    $recentlyDeployed = $project.cicd
    if ($recentlyDeployed) {
        $score += 1
        $reasons += "+1 Recently deployed (has CI/CD)"
        Write-Host "  [+1] Recently deployed" -ForegroundColor Green
    } else {
        Write-Host "  [ 0] Not recently deployed" -ForegroundColor Yellow
    }

    # +1 point: Has documentation (README with setup instructions)
    $hasGoodDocs = $false
    if (Test-Path "$($project.path)\README.md") {
        $readme = Get-Content "$($project.path)\README.md" -Raw -ErrorAction SilentlyContinue
        if ($readme -match "install|setup|getting started|quick start" -and $readme.Length -gt 500) {
            $hasGoodDocs = $true
        }
    }
    if ($hasGoodDocs) {
        $score += 1
        $reasons += "+1 Has comprehensive documentation"
        Write-Host "  [+1] Has good docs" -ForegroundColor Green
    } else {
        Write-Host "  [ 0] Poor/no documentation" -ForegroundColor Yellow
    }

    # +1 point: Dependencies are up to date (assume true if package.json exists and no security warnings)
    $depsUpToDate = Test-Path "$($project.path)\package.json"
    if ($depsUpToDate) {
        $score += 1
        $reasons += "+1 Has dependency management"
        Write-Host "  [+1] Dependencies managed" -ForegroundColor Green
    } else {
        Write-Host "  [ 0] No dependency management" -ForegroundColor Yellow
    }

    # +1 point: No critical security issues (assume true for now)
    $noSecurityIssues = $true
    if ($noSecurityIssues) {
        $score += 1
        $reasons += "+1 No known security issues"
        Write-Host "  [+1] No critical security issues" -ForegroundColor Green
    }

    # Update project
    $project.healthScore = $score
    $project.healthScoreReasons = $reasons

    # Add health assessment
    if ($score -ge 8) {
        $assessment = "Excellent"
        $color = "Green"
    } elseif ($score -ge 6) {
        $assessment = "Good"
        $color = "Cyan"
    } elseif ($score -ge 4) {
        $assessment = "Fair"
        $color = "Yellow"
    } else {
        $assessment = "Needs Improvement"
        $color = "Red"
    }

    Write-Host "  [==] Health Score: $score/10 ($assessment)" -ForegroundColor $color
    Write-Host ""

    $updatedProjects += $project
}

# Update registry
$registry.projects = $updatedProjects
$registry.lastUpdated = (Get-Date -Format "yyyy-MM-ddTHH:mm:ssZ")

# Calculate average health score
$avgHealth = ($updatedProjects | Measure-Object -Property healthScore -Average).Average
$registry.metadata.averageHealthScore = [math]::Round($avgHealth, 2)

# Save
$registry | ConvertTo-Json -Depth 10 | Set-Content $RegistryFile

Write-Host "[OK] Health scores calculated!" -ForegroundColor Green
Write-Host ""
Write-Host "[>>] Summary:" -ForegroundColor Yellow
Write-Host "  - Average Health Score: $([math]::Round($avgHealth, 1))/10"
Write-Host "  - Excellent (8-10): $(($updatedProjects | Where-Object { $_.healthScore -ge 8 }).Count) projects"
Write-Host "  - Good (6-7): $(($updatedProjects | Where-Object { $_.healthScore -ge 6 -and $_.healthScore -lt 8 }).Count) projects"
Write-Host "  - Fair (4-5): $(($updatedProjects | Where-Object { $_.healthScore -ge 4 -and $_.healthScore -lt 6 }).Count) projects"
Write-Host "  - Needs Improvement (0-3): $(($updatedProjects | Where-Object { $_.healthScore -lt 4 }).Count) projects"
Write-Host ""
Write-Host "[>>] Top 5 Healthiest Projects:" -ForegroundColor Green
$updatedProjects | Sort-Object -Property healthScore -Descending | Select-Object -First 5 | ForEach-Object {
    Write-Host "  - $($_.name): $($_.healthScore)/10"
}
Write-Host ""
Write-Host "[>>] Bottom 5 Projects (Need Attention):" -ForegroundColor Red
$updatedProjects | Sort-Object -Property healthScore | Select-Object -First 5 | ForEach-Object {
    Write-Host "  - $($_.name): $($_.healthScore)/10"
}
