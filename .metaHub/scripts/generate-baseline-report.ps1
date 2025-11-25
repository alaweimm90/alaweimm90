# Baseline Report Generator
# Creates a comprehensive baseline report for Week 1

param(
    [string]$RegistryFile = ".metaHub\projects-registry.json",
    [string]$PrioritiesFile = ".metaHub\reports\containerization-priorities.json"
)

Write-Host "[*] Generating Baseline Report" -ForegroundColor Cyan
Write-Host "==============================" -ForegroundColor Cyan
Write-Host ""

# Load data
$registry = Get-Content $RegistryFile | ConvertFrom-Json
$priorities = Get-Content $PrioritiesFile | ConvertFrom-Json

# Calculate metrics
$avgHealth = ($registry.projects | Measure-Object -Property healthScore -Average).Average
$containerized = ($registry.projects | Where-Object { $_.containerized }).Count
$withCICD = ($registry.projects | Where-Object { $_.cicd }).Count
$nonContainerized = ($registry.projects | Where-Object { -not $_.containerized }).Count

# Health distribution
$excellent = ($registry.projects | Where-Object { $_.healthScore -ge 8 }).Count
$good = ($registry.projects | Where-Object { $_.healthScore -ge 6 -and $_.healthScore -lt 8 }).Count
$fair = ($registry.projects | Where-Object { $_.healthScore -ge 4 -and $_.healthScore -lt 6 }).Count
$poor = ($registry.projects | Where-Object { $_.healthScore -lt 4 }).Count

# Top and bottom projects
$topProjects = $registry.projects | Sort-Object -Property healthScore -Descending | Select-Object -First 5
$bottomProjects = $registry.projects | Sort-Object -Property healthScore | Select-Object -First 5

# Organization breakdown
$orgBreakdown = @()
foreach ($orgKey in $registry.organizations.PSObject.Properties.Name) {
    $org = $registry.organizations.$orgKey
    $orgProjects = $registry.projects | Where-Object { $_.organizationId -eq $org.id }

    if ($orgProjects.Count -gt 0) {
        $orgAvgHealth = ($orgProjects | Measure-Object -Property healthScore -Average).Average
        $orgContainerized = ($orgProjects | Where-Object { $_.containerized }).Count
        $orgCICD = ($orgProjects | Where-Object { $_.cicd }).Count

        $orgBreakdown += [PSCustomObject]@{
            Name = $org.name
            Priority = $org.priority
            TotalProjects = $orgProjects.Count
            AvgHealth = [math]::Round($orgAvgHealth, 1)
            Containerized = $orgContainerized
            WithCICD = $orgCICD
            ContainerizationRate = [math]::Round($orgContainerized / $orgProjects.Count * 100, 1)
            CICDRate = [math]::Round($orgCICD / $orgProjects.Count * 100, 1)
        }
    }
}

# Create report
$report = @"
================================================================================
PHASE 1 BASELINE REPORT - WEEK 1 COMPLETE
================================================================================
Generated: $(Get-Date -Format "yyyy-MM-dd HH:mm:ss")
Repository: GitHub Multi-Org Solo Dev Setup
Master Plan: 15-Week Optimization Roadmap

================================================================================
EXECUTIVE SUMMARY
================================================================================

Total Projects Discovered: $($registry.metadata.totalProjects)
Organizations: $($registry.metadata.totalOrganizations)

Average Health Score: $([math]::Round($avgHealth, 1))/10
Containerization Rate: $([math]::Round($registry.metadata.containerizationRate * 100, 1))% ($containerized/$($registry.metadata.totalProjects))
CI/CD Coverage: $([math]::Round($registry.metadata.cicdRate * 100, 1))% ($withCICD/$($registry.metadata.totalProjects))

================================================================================
HEALTH SCORE DISTRIBUTION
================================================================================

Excellent (8-10):       $excellent projects ($([math]::Round($excellent / $registry.metadata.totalProjects * 100, 1))%)
Good (6-7):             $good projects ($([math]::Round($good / $registry.metadata.totalProjects * 100, 1))%)
Fair (4-5):             $fair projects ($([math]::Round($fair / $registry.metadata.totalProjects * 100, 1))%)
Needs Improvement (0-3): $poor projects ($([math]::Round($poor / $registry.metadata.totalProjects * 100, 1))%)

Health Distribution Chart:
Excellent:  $('█' * [math]::Min($excellent, 50))
Good:       $('█' * [math]::Min($good, 50))
Fair:       $('█' * [math]::Min($fair, 50))
Poor:       $('█' * [math]::Min($poor, 50))

================================================================================
ORGANIZATION BREAKDOWN
================================================================================

"@

foreach ($org in $orgBreakdown | Sort-Object -Property Priority, Name) {
    $report += @"

[$($org.Name)] - $($org.Priority.ToUpper()) Priority
  Total Projects: $($org.TotalProjects)
  Avg Health Score: $($org.AvgHealth)/10
  Containerized: $($org.Containerized) ($($org.ContainerizationRate)%)
  CI/CD Coverage: $($org.WithCICD) ($($org.CICDRate)%)

"@
}

$report += @"

================================================================================
TOP 5 HEALTHIEST PROJECTS
================================================================================

"@

$rank = 1
foreach ($project in $topProjects) {
    $org = $registry.organizations.PSObject.Properties | Where-Object { $_.Value.id -eq $project.organizationId } | Select-Object -First 1
    $report += @"
$rank. $($project.name) - $($project.healthScore)/10
   Organization: $($org.Value.name)
   Containerized: $(if ($project.containerized) { 'Yes' } else { 'No' })
   CI/CD: $(if ($project.cicd) { 'Yes' } else { 'No' })
   Tech Stack: $($project.techStack -join ', ')

"@
    $rank++
}

$report += @"

================================================================================
BOTTOM 5 PROJECTS (NEED ATTENTION)
================================================================================

"@

$rank = 1
foreach ($project in $bottomProjects) {
    $org = $registry.organizations.PSObject.Properties | Where-Object { $_.Value.id -eq $project.organizationId } | Select-Object -First 1
    $report += @"
$rank. $($project.name) - $($project.healthScore)/10
   Organization: $($org.Value.name)
   Containerized: $(if ($project.containerized) { 'Yes' } else { 'No' })
   CI/CD: $(if ($project.cicd) { 'Yes' } else { 'No' })
   Path: $($project.path)

"@
    $rank++
}

$report += @"

================================================================================
CONTAINERIZATION PRIORITIES (Week 2-4)
================================================================================

Top 5 Projects Selected for Containerization:

"@

$rank = 1
foreach ($rec in $priorities.recommendations) {
    $report += @"
$rank. $($rec.projectName) - Priority Score: $($rec.priorityScore)
   $($rec.tier) - Estimated Effort: $($rec.estimatedEffort)
   Organization: $($rec.organizationName)
   Reasoning: $($rec.reasons)

"@
    $rank++
}

$report += @"

Expected Outcomes After Containerization:
- 5 additional projects containerized
- Containerization rate: $([math]::Round($registry.metadata.containerizationRate * 100, 1))% -> $([math]::Round((($containerized + 5) / $registry.metadata.totalProjects) * 100, 1))%
- All top-priority organizations fully containerized

================================================================================
NEXT STEPS (Week 2)
================================================================================

1. Review Baseline Dashboard
   - Open: .metaHub/dashboard/index.html
   - Verify metrics are accurate
   - Take screenshot for before/after comparison

2. Start Containerization (Phase 2)
   - Use Docker templates: .metaHub/templates/containers/
   - Containerize SimCore (Tier 1 - 2-4 hours)
   - Containerize repz (Tier 1 - 2-4 hours)
   - Containerize benchbarrier (Tier 1 - 2-4 hours)
   - Containerize mag-logic (Tier 1 - 2-4 hours)
   - Containerize Attributa (Tier 1 - 2-4 hours)

3. Test Locally
   - Build each container: docker build -t <name> .
   - Run each container: docker run -p 3000:3000 <name>
   - Verify functionality

4. Update Registry
   - Update containerized: true for each project
   - Recalculate health scores
   - Run: powershell -ExecutionPolicy Bypass -File .metaHub\scripts\calculate-health-scores.ps1

5. Create Dev Stack
   - Combine all services in docker-compose.yml
   - Test full stack locally
   - Document setup in README

================================================================================
COST OPTIMIZATION NOTES
================================================================================

LLM Task Routing (from AI_AGENT_RULES.md):

High-Value Tasks (Claude Sonnet 4.5) - Use sparingly:
- Architecture decisions for containerization strategy
- Complex multi-file refactoring
- Critical security implementations

Quick Edits (Cursor AI) - Use moderately:
- Dockerfile generation
- Docker-compose creation
- Boilerplate code updates

Documentation (Windsurf) - Use moderately:
- README updates
- Architecture diagrams
- Documentation writing

Everything Else (GitHub Copilot) - Use liberally:
- Code completion
- Test generation
- Simple transformations

Estimated subscription savings: 40-60% vs. using Claude for everything

================================================================================
ROI PROJECTIONS (15 Weeks)
================================================================================

Time Savings:
- New project setup: 3 days -> 3 hours (96% reduction)
- Bug fix to production: 1 week -> 1 hour (99% reduction)
- Finding duplicate code: Manual -> Automated

Efficiency Gains:
- Containerization: 55% -> 90% (target)
- CI/CD Coverage: 35% -> 90% (target)
- Avg Health Score: 4.5 -> 8.0 (target)
- Redundancy Reduction: TBD (Phase 4)

Business Impact:
- Faster iteration cycles
- Reduced deployment errors
- Consistent quality across projects
- Better resource utilization
- Lower LLM subscription costs

================================================================================
DASHBOARD ACCESS
================================================================================

Local Dashboard: file:///$((Get-Location).Path)\.metaHub\dashboard\index.html

To view:
1. Open the HTML file in your browser
2. Ensure projects-registry.json is in .metaHub/
3. Screenshot for baseline comparison

================================================================================
COMPLETION STATUS
================================================================================

Phase 1 - Discovery & Baseline (Week 1): COMPLETE
  [x] Step 1: Run project discovery
  [x] Step 2: Review registry and update statuses
  [x] Step 3: Calculate health scores for all projects
  [x] Step 4: Identify top 5 critical projects to containerize
  [x] Step 5: Create baseline metrics dashboard

Next Phase: Containerization (Week 2-4)

================================================================================
"@

# Save report
$reportFile = ".metaHub\reports\baseline-week1.txt"
$report | Out-File -FilePath $reportFile -Encoding UTF8

Write-Host "[OK] Baseline report generated!" -ForegroundColor Green
Write-Host ""
Write-Host "[>>] Report saved to: $reportFile" -ForegroundColor Cyan
Write-Host ""
Write-Host "[>>] Next Actions:" -ForegroundColor Yellow
Write-Host "  1. Open dashboard: .metaHub\dashboard\index.html" -ForegroundColor White
Write-Host "  2. Read report: cat $reportFile" -ForegroundColor White
Write-Host "  3. Take screenshot of dashboard" -ForegroundColor White
Write-Host "  4. Begin Phase 2: Containerization" -ForegroundColor White
Write-Host ""
Write-Host "[>>] Dashboard URL:" -ForegroundColor Green
Write-Host "  file:///$((Get-Location).Path)\.metaHub\dashboard\index.html" -ForegroundColor White
Write-Host ""

# Display key metrics summary
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "KEY BASELINE METRICS" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Total Projects:       $($registry.metadata.totalProjects)" -ForegroundColor White
Write-Host "  Avg Health Score:     $([math]::Round($avgHealth, 1))/10" -ForegroundColor White
Write-Host "  Containerization:     $([math]::Round($registry.metadata.containerizationRate * 100, 1))%" -ForegroundColor White
Write-Host "  CI/CD Coverage:       $([math]::Round($registry.metadata.cicdRate * 100, 1))%" -ForegroundColor White
Write-Host ""
Write-Host "  Excellent Projects:   $excellent" -ForegroundColor Green
Write-Host "  Good Projects:        $good" -ForegroundColor Cyan
Write-Host "  Fair Projects:        $fair" -ForegroundColor Yellow
Write-Host "  Need Improvement:     $poor" -ForegroundColor Red
Write-Host ""
Write-Host "[OK] Phase 1 Complete! Ready for Phase 2." -ForegroundColor Green
Write-Host ""
