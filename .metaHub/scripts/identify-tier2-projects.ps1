# ============================================================================
# Identify Tier 2 Containerization Candidates
# ============================================================================
# Analyzes remaining non-containerized projects and identifies the next 5
# priority targets based on health scores, complexity, and organizational needs

param(
    [Parameter(Mandatory=$false)]
    [string]$RegistryPath = ".metaHub\projects-registry.json",

    [Parameter(Mandatory=$false)]
    [int]$TopN = 5,

    [Parameter(Mandatory=$false)]
    [switch]$IncludeTier3
)

Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host "Tier 2 Project Identification" -ForegroundColor Cyan
Write-Host "============================================================================`n" -ForegroundColor Cyan

# Read registry
if (-not (Test-Path $RegistryPath)) {
    Write-Host "ERROR: Registry not found at $RegistryPath" -ForegroundColor Red
    exit 1
}

$registry = Get-Content $RegistryPath -Raw | ConvertFrom-Json
Write-Host "Registry loaded: $($registry.projects.Count) total projects`n" -ForegroundColor Green

# Filter non-containerized projects
$nonContainerized = $registry.projects | Where-Object { $_.containerized -ne $true }
Write-Host "Non-containerized projects: $($nonContainerized.Count)`n" -ForegroundColor Yellow

# Exclude Tier 1 projects that are already in progress
$tier1InProgress = @("proj-001", "proj-005", "proj-018", "proj-034", "proj-040")
$candidates = $nonContainerized | Where-Object { $_.id -notin $tier1InProgress }

Write-Host "Available candidates: $($candidates.Count)`n" -ForegroundColor Yellow

# Complexity assessment function
function Get-ComplexityScore {
    param($project)

    $complexity = 0

    # Tech stack complexity
    if ($project.techStack -contains "Python") { $complexity += 2 }
    if ($project.techStack -contains "Go") { $complexity += 1 }
    if ($project.techStack -contains "Rust") { $complexity += 2 }
    if ($project.techStack -contains "C#") { $complexity += 2 }
    if ($project.techStack -contains "Java") { $complexity += 3 }
    if ($project.techStack -contains "TypeScript") { $complexity += 1 }

    # Project characteristics
    if ($project.hasDatabase) { $complexity += 2 }
    if ($project.hasMultipleServices) { $complexity += 3 }
    if ($project.requiresGPU) { $complexity += 4 }
    if ($project.hasExternalDependencies) { $complexity += 2 }

    # Organization complexity
    if ($project.organization -eq "alaweimm90-personal") { $complexity += 1 }
    if ($project.organization -eq "alaweimm90-community") { $complexity += 1 }

    return $complexity
}

# Calculate priority scores
$scoredCandidates = $candidates | ForEach-Object {
    $project = $_

    # Organizational priority
    $orgPriority = switch ($project.organization) {
        "AlaweinOS" { 1.0 }
        "alaweimm90-business" { 0.9 }
        "alaweimm90-science" { 0.8 }
        "alaweimm90-personal" { 0.6 }
        "alaweimm90-community" { 0.5 }
        default { 0.5 }
    }

    # Complexity assessment
    $complexity = Get-ComplexityScore $project

    # CI/CD bonus
    $cicdBonus = if ($project.hasCICD) { 0.3 } else { 0 }

    # Documentation bonus
    $docsBonus = if ($project.hasReadme) { 0.2 } else { 0 }

    # Calculate final score
    $priorityScore = ($project.healthScore * 0.4) +
                     ($orgPriority * 0.3) +
                     ($cicdBonus) +
                     ($docsBonus) -
                     ($complexity * 0.05)

    # Estimate effort (in hours)
    $effortEstimate = if ($complexity -le 5) {
        "Medium (4-8h)"
        $tier = 2
    } elseif ($complexity -le 10) {
        "High (8-16h)"
        $tier = 3
    } else {
        "Very High (16-24h)"
        $tier = 3
    }

    [PSCustomObject]@{
        Id = $project.id
        Name = $project.name
        Organization = $project.organization
        HealthScore = $project.healthScore
        TechStack = ($project.techStack -join ", ")
        Complexity = $complexity
        Tier = $tier
        EffortEstimate = $effortEstimate
        PriorityScore = [Math]::Round($priorityScore, 2)
        HasCICD = $project.hasCICD
        HasDocs = $project.hasReadme
        OrgPriority = $orgPriority
    }
}

# Filter by tier if specified
if (-not $IncludeTier3) {
    $scoredCandidates = $scoredCandidates | Where-Object { $_.Tier -eq 2 }
    Write-Host "Filtering to Tier 2 projects only (medium complexity)`n" -ForegroundColor Yellow
}

# Sort by priority score
$topCandidates = $scoredCandidates |
    Sort-Object -Property PriorityScore -Descending |
    Select-Object -First $TopN

Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host "Top $TopN Tier 2 Candidates" -ForegroundColor Cyan
Write-Host "============================================================================`n" -ForegroundColor Cyan

# Display results
$topCandidates | Format-Table -Property `
    @{Label="Rank"; Expression={$topCandidates.IndexOf($_) + 1}}, `
    @{Label="Project"; Expression={$_.Name}}, `
    @{Label="Organization"; Expression={$_.Organization}}, `
    @{Label="Health"; Expression={$_.HealthScore}}, `
    @{Label="Tier"; Expression={$_.Tier}}, `
    @{Label="Effort"; Expression={$_.EffortEstimate}}, `
    @{Label="Priority"; Expression={$_.PriorityScore}} `
    -AutoSize

Write-Host "`n============================================================================" -ForegroundColor Cyan
Write-Host "Detailed Analysis" -ForegroundColor Cyan
Write-Host "============================================================================`n" -ForegroundColor Cyan

foreach ($candidate in $topCandidates) {
    $rank = $topCandidates.IndexOf($candidate) + 1

    Write-Host "[$rank] $($candidate.Name)" -ForegroundColor Green
    Write-Host "    Organization: $($candidate.Organization)" -ForegroundColor White
    Write-Host "    Health Score: $($candidate.HealthScore)/10" -ForegroundColor White
    Write-Host "    Tech Stack: $($candidate.TechStack)" -ForegroundColor White
    Write-Host "    Complexity: $($candidate.Complexity) (Tier $($candidate.Tier))" -ForegroundColor White
    Write-Host "    Effort Estimate: $($candidate.EffortEstimate)" -ForegroundColor White
    Write-Host "    Priority Score: $($candidate.PriorityScore)" -ForegroundColor White
    Write-Host "    Has CI/CD: $(if ($candidate.HasCICD) {'Yes'} else {'No'})" -ForegroundColor $(if ($candidate.HasCICD) {'Green'} else {'Yellow'})
    Write-Host "    Has Docs: $(if ($candidate.HasDocs) {'Yes'} else {'No'})" -ForegroundColor $(if ($candidate.HasDocs) {'Green'} else {'Yellow'})
    Write-Host ""
}

# Summary statistics
Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host "Summary Statistics" -ForegroundColor Cyan
Write-Host "============================================================================`n" -ForegroundColor Cyan

$avgHealth = [Math]::Round(($topCandidates | Measure-Object -Property HealthScore -Average).Average, 2)
$avgComplexity = [Math]::Round(($topCandidates | Measure-Object -Property Complexity -Average).Average, 2)
$avgPriority = [Math]::Round(($topCandidates | Measure-Object -Property PriorityScore -Average).Average, 2)

Write-Host "Average Health Score: $avgHealth/10" -ForegroundColor White
Write-Host "Average Complexity: $avgComplexity" -ForegroundColor White
Write-Host "Average Priority Score: $avgPriority" -ForegroundColor White

# Effort breakdown
$mediumEffort = ($topCandidates | Where-Object { $_.Tier -eq 2 }).Count
$highEffort = ($topCandidates | Where-Object { $_.Tier -eq 3 }).Count

Write-Host "`nEffort Breakdown:" -ForegroundColor Cyan
Write-Host "  Tier 2 (Medium): $mediumEffort projects" -ForegroundColor White
Write-Host "  Tier 3 (High): $highEffort projects" -ForegroundColor White

# Total effort estimate
$totalEffortMin = ($mediumEffort * 4) + ($highEffort * 8)
$totalEffortMax = ($mediumEffort * 8) + ($highEffort * 16)

Write-Host "`nTotal Estimated Effort: $totalEffortMin-$totalEffortMax hours" -ForegroundColor Yellow

# Organization distribution
Write-Host "`nOrganization Distribution:" -ForegroundColor Cyan
$topCandidates | Group-Object -Property Organization | ForEach-Object {
    Write-Host "  $($_.Name): $($_.Count) projects" -ForegroundColor White
}

# Technology distribution
Write-Host "`nTechnology Distribution:" -ForegroundColor Cyan
$techStacks = $topCandidates | ForEach-Object { $_.TechStack -split ", " } | Group-Object
$techStacks | Sort-Object Count -Descending | ForEach-Object {
    Write-Host "  $($_.Name): $($_.Count) projects" -ForegroundColor White
}

# Export to JSON
$outputPath = ".metaHub/docs/containerization/TIER2_ASSESSMENT.json"
$exportData = @{
    timestamp = (Get-Date -Format "yyyy-MM-ddTHH:mm:ssZ")
    totalCandidates = $candidates.Count
    tier2Candidates = ($scoredCandidates | Where-Object { $_.Tier -eq 2 }).Count
    tier3Candidates = ($scoredCandidates | Where-Object { $_.Tier -eq 3 }).Count
    topN = $TopN
    projects = $topCandidates
    summary = @{
        avgHealthScore = $avgHealth
        avgComplexity = $avgComplexity
        avgPriorityScore = $avgPriority
        totalEffortHours = "$totalEffortMin-$totalEffortMax"
    }
}

$exportData | ConvertTo-Json -Depth 10 | Set-Content $outputPath
Write-Host "`n✅ Assessment exported to: $outputPath" -ForegroundColor Green

Write-Host "`n============================================================================" -ForegroundColor Cyan
Write-Host "Tier 2 Identification Complete" -ForegroundColor Green
Write-Host "============================================================================`n" -ForegroundColor Cyan

return $topCandidates
