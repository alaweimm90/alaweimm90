# ATLAS Comprehensive Demo Script (PowerShell)
# This script demonstrates the complete ATLAS workflow

param(
    [switch]$SkipSetup,
    [switch]$Verbose
)

# Configuration
$DemoDir = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$RepoDir = Join-Path $DemoDir "test-repos"
$LogDir = Join-Path $DemoDir "logs"
$DashboardDir = Join-Path $DemoDir "dashboards"
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$LogFile = Join-Path $LogDir "demo_$Timestamp.log"

# Ensure log directory exists
if (!(Test-Path $LogDir)) { New-Item -ItemType Directory -Path $LogDir | Out-Null }

# Logging function
function Write-Log {
    param([string]$Message, [string]$Level = "INFO")
    $LogMessage = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') - [$Level] $Message"
    Write-Host $LogMessage
    Add-Content -Path $LogFile -Value $LogMessage
}

function Write-Header {
    param([string]$Title)
    $Separator = "=" * 80
    Write-Host "$Separator" -ForegroundColor Blue
    Write-Host "  $Title" -ForegroundColor Blue
    Write-Host "$Separator" -ForegroundColor Blue
    Write-Log "Starting: $Title"
}

function Write-Success {
    param([string]$Message)
    Write-Host "✓ $Message" -ForegroundColor Green
    Write-Log "SUCCESS: $Message"
}

function Write-Error {
    param([string]$Message)
    Write-Host "✗ $Message" -ForegroundColor Red
    Write-Log "ERROR: $Message"
}

function Write-Warning {
    param([string]$Message)
    Write-Host "⚠ $Message" -ForegroundColor Yellow
    Write-Log "WARNING: $Message"
}

# Setup function
function Initialize-Demo {
    Write-Header "Setting up ATLAS Demo Environment"

    # Check if ATLAS CLI is available
    try {
        $atlasVersion = & atlas --version 2>$null
        Write-Success "ATLAS CLI found: $atlasVersion"
    } catch {
        Write-Error "ATLAS CLI not found. Please ensure ATLAS is properly installed."
        exit 1
    }

    # Check test repositories
    $repos = @("messy-python", "complex-js", "spaghetti-ts")
    foreach ($repo in $repos) {
        $repoPath = Join-Path $RepoDir $repo
        if (!(Test-Path $repoPath)) {
            Write-Error "Test repository not found: $repoPath"
            exit 1
        }
    }
    Write-Success "All test repositories verified"
}

# Run analysis on a repository
function Analyze-Repository {
    param([string]$RepoName)

    $repoPath = Join-Path $RepoDir $RepoName
    Write-Header "Analyzing $RepoName Repository"

    # Full repository analysis
    $analysisFile = Join-Path $LogDir "${RepoName}_analysis_$Timestamp.json"
    try {
        & atlas analyze repo $repoPath --format json | Out-File -FilePath $analysisFile -Encoding UTF8
        Write-Success "Repository analysis completed for $RepoName"
    } catch {
        Write-Error "Repository analysis failed for $RepoName"
        return
    }

    # Complexity analysis
    $complexityFile = Join-Path $LogDir "${RepoName}_complexity_$Timestamp.txt"
    try {
        & atlas analyze complexity $repoPath | Out-File -FilePath $complexityFile -Encoding UTF8
        Write-Success "Complexity analysis completed for $RepoName"
    } catch {
        Write-Warning "Complexity analysis failed for $RepoName"
    }

    # Chaos analysis
    $chaosFile = Join-Path $LogDir "${RepoName}_chaos_$Timestamp.txt"
    try {
        & atlas analyze chaos $repoPath --detailed | Out-File -FilePath $chaosFile -Encoding UTF8
        Write-Success "Chaos analysis completed for $RepoName"
    } catch {
        Write-Warning "Chaos analysis failed for $RepoName"
    }

    # Quick scan
    $scanFile = Join-Path $LogDir "${RepoName}_scan_$Timestamp.txt"
    try {
        & atlas analyze scan $repoPath | Out-File -FilePath $scanFile -Encoding UTF8
        Write-Success "Quick scan completed for $RepoName"
    } catch {
        Write-Warning "Quick scan failed for $RepoName"
    }
}

# Generate performance benchmarks
function Invoke-PerformanceBenchmark {
    Write-Header "Running Performance Benchmarks"

    $benchmarkFile = Join-Path $LogDir "benchmark_$Timestamp.txt"

    $repos = @("messy-python", "complex-js", "spaghetti-ts")
    foreach ($repo in $repos) {
        $repoPath = Join-Path $RepoDir $repo
        $startTime = Get-Date

        try {
            & atlas analyze repo $repoPath --format json | Out-Null
            $endTime = Get-Date
            $duration = ($endTime - $startTime).TotalSeconds
            $benchmarkLine = "$repo`: $($duration.ToString('F3'))s"
            Add-Content -Path $benchmarkFile -Value $benchmarkLine
            Write-Success "Benchmarked $repo`: $($duration.ToString('F3'))s"
        } catch {
            Write-Error "Benchmark failed for $repo"
        }
    }
}

# Generate dashboard data
function New-DashboardData {
    Write-Header "Generating Dashboard Data"

    $dashboardFile = Join-Path $DashboardDir "metrics_$Timestamp.json"

    # Collect analysis results
    $repos = @("messy-python", "complex-js", "spaghetti-ts")
    $repoData = @{}

    foreach ($repo in $repos) {
        $analysisFile = Join-Path $LogDir "${repo}_analysis_$Timestamp.json"
        if (Test-Path $analysisFile) {
            try {
                $content = Get-Content $analysisFile -Raw
                $repoData[$repo] = $content | ConvertFrom-Json
            } catch {
                $repoData[$repo] = @{}
            }
        } else {
            $repoData[$repo] = @{}
        }
    }

    # Collect benchmarks
    $benchmarkFile = Join-Path $LogDir "benchmark_$Timestamp.txt"
    $benchmarks = @{}
    if (Test-Path $benchmarkFile) {
        $benchmarkLines = Get-Content $benchmarkFile
        foreach ($line in $benchmarkLines) {
            if ($line -match "^(.+?):\s*(.+)$") {
                $benchmarks[$matches[1]] = $matches[2]
            }
        }
    }

    $dashboardData = @{
        timestamp = [int](Get-Date -UFormat %s)
        repositories = $repoData
        benchmarks = $benchmarks
    }

    $dashboardData | ConvertTo-Json -Depth 10 | Out-File -FilePath $dashboardFile -Encoding UTF8
    Write-Success "Dashboard data generated"
}

# Main demo workflow
function Start-AtlasDemo {
    Write-Header "ATLAS Comprehensive Demo"
    Write-Log "Starting ATLAS demo workflow"

    # Setup
    if (!$SkipSetup) {
        Initialize-Demo
    }

    # Analyze each test repository
    $repos = @("messy-python", "complex-js", "spaghetti-ts")
    foreach ($repo in $repos) {
        Analyze-Repository -RepoName $repo
    }

    # Performance benchmarking
    Invoke-PerformanceBenchmark

    # Generate dashboard
    New-DashboardData

    # Final summary
    Write-Header "Demo Complete"
    Write-Success "ATLAS demo workflow completed successfully"
    Write-Host ""
    Write-Host "Results:" -ForegroundColor Blue
    Write-Host "  Logs: $LogDir\"
    Write-Host "  Dashboard: $(Join-Path $DashboardDir "dashboard_$Timestamp.html")"
    Write-Host "  Metrics: $(Join-Path $DashboardDir "metrics_$Timestamp.json")"
    Write-Host ""
    Write-Host "Note: HTML dashboard generation requires manual creation or additional tooling." -ForegroundColor Yellow

    Write-Log "Demo workflow completed"
}

# Run main function
Start-AtlasDemo