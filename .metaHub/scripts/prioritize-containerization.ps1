# Containerization Prioritization Script
# Identifies top projects to containerize based on multiple criteria

param(
    [string]$RegistryFile = ".metaHub\projects-registry.json"
)

Write-Host "[*] Containerization Prioritization Analysis" -ForegroundColor Cyan
Write-Host "===========================================" -ForegroundColor Cyan
Write-Host ""

# Load registry
$registry = Get-Content $RegistryFile | ConvertFrom-Json

# Organization priority weights
$orgPriorityWeights = @{
    "high" = 3
    "medium" = 2
    "low" = 1
}

# Category priorities
$categoryWeights = @{
    "backend-service" = 3
    "frontend-app" = 2.5
    "library" = 2
    "tool" = 1.5
    "infrastructure" = 2
    "unknown" = 1
}

$candidates = @()

Write-Host "[>>] Analyzing non-containerized projects..." -ForegroundColor Magenta
Write-Host ""

foreach ($project in $registry.projects) {
    # Only consider projects that are NOT containerized
    if ($project.containerized -eq $false) {

        # Get organization priority
        $org = $registry.organizations.PSObject.Properties | Where-Object { $_.Value.id -eq $project.organizationId } | Select-Object -First 1
        $orgPriority = if ($org) { $org.Value.priority } else { "low" }
        $orgWeight = $orgPriorityWeights[$orgPriority]

        # Get category weight
        $categoryWeight = $categoryWeights[$project.category]

        # Calculate priority score
        # Formula: (healthScore * 10) + (orgWeight * 10) + (categoryWeight * 10) + (CI/CD bonus)
        $priorityScore = ($project.healthScore * 10) + ($orgWeight * 10) + ($categoryWeight * 10)

        # Bonus: +15 if already has CI/CD (easy to containerize)
        if ($project.cicd) {
            $priorityScore += 15
        }

        # Determine complexity tier
        $tier = "Tier 3 (Complex)"
        $estimatedEffort = "2-3 days"

        if ($project.cicd -and $project.healthScore -ge 6) {
            $tier = "Tier 1 (Easy)"
            $estimatedEffort = "2-4 hours"
        } elseif ($project.cicd -or $project.healthScore -ge 5) {
            $tier = "Tier 2 (Medium)"
            $estimatedEffort = "1 day"
        }

        # Build reasoning
        $reasons = @()
        $reasons += "Health Score: $($project.healthScore)/10"
        $reasons += "Org Priority: $orgPriority"
        $reasons += "Category: $($project.category)"
        if ($project.cicd) { $reasons += "Has CI/CD (easy win!)" }
        $reasons += "Complexity: $tier"

        $candidate = [PSCustomObject]@{
            id = $project.id
            name = $project.name
            organizationId = $project.organizationId
            organizationName = if ($org) { $org.Value.name } else { "unknown" }
            organizationPriority = $orgPriority
            category = $project.category
            healthScore = $project.healthScore
            hasCICD = $project.cicd
            priorityScore = $priorityScore
            tier = $tier
            estimatedEffort = $estimatedEffort
            reasons = $reasons -join " | "
            path = $project.path
        }

        $candidates += $candidate
    }
}

# Sort by priority score (descending)
$topCandidates = $candidates | Sort-Object -Property priorityScore -Descending | Select-Object -First 10

Write-Host "[OK] Analysis complete!" -ForegroundColor Green
Write-Host ""
Write-Host "[>>] Non-containerized projects: $($candidates.Count)" -ForegroundColor Yellow
Write-Host ""

# Display top 10
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "TOP 10 CONTAINERIZATION PRIORITIES" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$rank = 1
foreach ($candidate in $topCandidates) {
    $color = switch ($candidate.tier) {
        "Tier 1 (Easy)" { "Green" }
        "Tier 2 (Medium)" { "Yellow" }
        "Tier 3 (Complex)" { "Red" }
    }

    Write-Host "[$rank] $($candidate.name)" -ForegroundColor Cyan
    Write-Host "    Organization: $($candidate.organizationName) ($($candidate.organizationPriority) priority)" -ForegroundColor White
    Write-Host "    Priority Score: $($candidate.priorityScore)" -ForegroundColor White
    Write-Host "    $($candidate.tier) - Est. $($candidate.estimatedEffort)" -ForegroundColor $color
    Write-Host "    Path: $($candidate.path)" -ForegroundColor DarkGray
    Write-Host "    Reasoning: $($candidate.reasons)" -ForegroundColor DarkGray
    Write-Host ""

    $rank++
}

# Export top 5 recommendations
Write-Host "========================================" -ForegroundColor Green
Write-Host "TOP 5 RECOMMENDATIONS (for Week 2-4)" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""

$top5 = $topCandidates | Select-Object -First 5

$recommendations = @{
    generatedAt = (Get-Date -Format "yyyy-MM-ddTHH:mm:ssZ")
    summary = @{
        totalNonContainerized = $candidates.Count
        analyzedProjects = $candidates.Count
        top5Selected = $true
    }
    recommendations = $top5 | ForEach-Object {
        @{
            rank = $top5.IndexOf($_) + 1
            projectId = $_.id
            projectName = $_.name
            organizationId = $_.organizationId
            organizationName = $_.organizationName
            priorityScore = $_.priorityScore
            tier = $_.tier
            estimatedEffort = $_.estimatedEffort
            reasons = $_.reasons
            path = $_.path
            nextSteps = @(
                "Review project structure and dependencies",
                "Create Dockerfile using .metaHub/templates/containers/ template",
                "Add .dockerignore file",
                "Test local build: docker build -t $($_.name) .",
                "Test local run and verify functionality",
                "Update projects-registry.json: containerized = true",
                "Commit Dockerfile and update documentation"
            )
        }
    }
}

# Save recommendations
$outputFile = ".metaHub\reports\containerization-priorities.json"
$recommendations | ConvertTo-Json -Depth 10 | Set-Content $outputFile

Write-Host "[OK] Recommendations saved to: $outputFile" -ForegroundColor Green
Write-Host ""

# Display summary
Write-Host "[>>] Action Plan Summary:" -ForegroundColor Yellow
Write-Host ""
Write-Host "  Week 2-3: Containerize Top 5" -ForegroundColor Cyan
foreach ($rec in $top5) {
    Write-Host "    - $($rec.name) ($($rec.tier))" -ForegroundColor White
}
Write-Host ""
Write-Host "  Expected Outcomes:" -ForegroundColor Cyan
Write-Host "    - 5 additional projects containerized"
Write-Host "    - Containerization rate: $(($registry.metadata.containerizationRate * 100))% -> $([math]::Round((($registry.projects | Where-Object { $_.containerized }).Count + 5) / $registry.projects.Count * 100, 1))%"
Write-Host ""
Write-Host "  Next Command:" -ForegroundColor Yellow
Write-Host "    cat $outputFile | jq '.recommendations[0]'" -ForegroundColor White
Write-Host ""
