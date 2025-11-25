# Docker Health Monitor
# Monitors health status of all services and provides alerts

param(
    [int]$IntervalSeconds = 30,
    [switch]$ContinuousMode,
    [switch]$AlertOnUnhealthy,
    [string]$LogPath = ".metaHub/logs/health-monitor.log"
)

$ErrorActionPreference = "Stop"

function Write-Log {
    param([string]$Message, [string]$Level = "INFO")
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $logMessage = "[$timestamp] [$Level] $Message"
    Write-Host $logMessage

    if ($LogPath) {
        $logDir = Split-Path $LogPath -Parent
        if (-not (Test-Path $logDir)) {
            New-Item -ItemType Directory -Path $logDir -Force | Out-Null
        }
        Add-Content -Path $LogPath -Value $logMessage
    }
}

function Get-ContainerHealth {
    $containers = docker compose ps --format json | ConvertFrom-Json

    $healthStatus = @{
        healthy = @()
        unhealthy = @()
        starting = @()
        stopped = @()
        total = 0
    }

    foreach ($container in $containers) {
        $healthStatus.total++

        $inspect = docker inspect $container.Name --format '{{.State.Health.Status}}' 2>$null

        if (-not $inspect -or $inspect -eq "<no value>") {
            # No health check defined, check if running
            $state = docker inspect $container.Name --format '{{.State.Status}}' 2>$null
            if ($state -eq "running") {
                $inspect = "running (no healthcheck)"
            } else {
                $inspect = "stopped"
            }
        }

        $containerInfo = @{
            Name = $container.Name
            Service = $container.Service
            Status = $inspect
            Uptime = $container.Status
        }

        switch -Regex ($inspect) {
            "healthy|running" { $healthStatus.healthy += $containerInfo }
            "unhealthy" { $healthStatus.unhealthy += $containerInfo }
            "starting" { $healthStatus.starting += $containerInfo }
            default { $healthStatus.stopped += $containerInfo }
        }
    }

    return $healthStatus
}

function Show-HealthReport {
    param($HealthStatus)

    Write-Host "`n╔══════════════════════════════════════════════════════════╗"
    Write-Host "║          DOCKER HEALTH MONITOR REPORT                   ║"
    Write-Host "╚══════════════════════════════════════════════════════════╝`n"

    Write-Host "Timestamp: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')`n"

    # Overall Status
    $healthyCount = $HealthStatus.healthy.Count
    $totalCount = $HealthStatus.total
    $healthPercentage = if ($totalCount -gt 0) { [math]::Round(($healthyCount / $totalCount) * 100, 1) } else { 0 }

    Write-Host "Overall Status: " -NoNewline
    if ($healthPercentage -eq 100) {
        Write-Host "✅ ALL HEALTHY" -ForegroundColor Green
    } elseif ($healthPercentage -ge 80) {
        Write-Host "⚠️  MOSTLY HEALTHY" -ForegroundColor Yellow
    } else {
        Write-Host "❌ UNHEALTHY" -ForegroundColor Red
    }

    Write-Host "Health Score: $healthyCount/$totalCount ($healthPercentage%)`n"

    # Healthy Services
    if ($HealthStatus.healthy.Count -gt 0) {
        Write-Host "✅ Healthy Services ($($HealthStatus.healthy.Count)):" -ForegroundColor Green
        foreach ($container in $HealthStatus.healthy) {
            Write-Host "   • $($container.Service) - $($container.Status) - $($container.Uptime)" -ForegroundColor Green
        }
        Write-Host ""
    }

    # Starting Services
    if ($HealthStatus.starting.Count -gt 0) {
        Write-Host "⏳ Starting Services ($($HealthStatus.starting.Count)):" -ForegroundColor Yellow
        foreach ($container in $HealthStatus.starting) {
            Write-Host "   • $($container.Service) - $($container.Uptime)" -ForegroundColor Yellow
        }
        Write-Host ""
    }

    # Unhealthy Services
    if ($HealthStatus.unhealthy.Count -gt 0) {
        Write-Host "❌ Unhealthy Services ($($HealthStatus.unhealthy.Count)):" -ForegroundColor Red
        foreach ($container in $HealthStatus.unhealthy) {
            Write-Host "   • $($container.Service) - $($container.Status)" -ForegroundColor Red

            # Show recent logs
            Write-Host "   Recent logs:" -ForegroundColor Gray
            $logs = docker logs $container.Name --tail 5 2>&1
            $logs -split "`n" | ForEach-Object {
                Write-Host "     $_" -ForegroundColor Gray
            }
        }
        Write-Host ""
    }

    # Stopped Services
    if ($HealthStatus.stopped.Count -gt 0) {
        Write-Host "⏹️  Stopped Services ($($HealthStatus.stopped.Count)):" -ForegroundColor Red
        foreach ($container in $HealthStatus.stopped) {
            Write-Host "   • $($container.Service)" -ForegroundColor Red
        }
        Write-Host ""
    }

    # Resource Usage
    Write-Host "📊 Resource Usage:" -ForegroundColor Cyan
    $stats = docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}" | Out-String
    Write-Host $stats -ForegroundColor Cyan
}

function Send-Alert {
    param([string]$Message)

    Write-Log "ALERT: $Message" -Level "ALERT"

    # Could integrate with notification services here:
    # - Slack webhook
    # - Email
    # - Discord webhook
    # - PagerDuty

    Write-Host "`n🚨 ALERT: $Message" -ForegroundColor Red -BackgroundColor Yellow
}

# Main execution
Write-Log "Starting Docker Health Monitor"

if ($ContinuousMode) {
    Write-Host "Running in continuous mode (Ctrl+C to stop)"
    Write-Host "Monitoring interval: $IntervalSeconds seconds`n"

    while ($true) {
        try {
            $health = Get-ContainerHealth
            Show-HealthReport -HealthStatus $health

            # Check for alerts
            if ($AlertOnUnhealthy -and $health.unhealthy.Count -gt 0) {
                foreach ($container in $health.unhealthy) {
                    Send-Alert "Container $($container.Service) is unhealthy!"
                }
            }

            if ($AlertOnUnhealthy -and $health.stopped.Count -gt 0) {
                foreach ($container in $health.stopped) {
                    Send-Alert "Container $($container.Service) has stopped!"
                }
            }

            Write-Host "`nNext check in $IntervalSeconds seconds...`n" -ForegroundColor Gray
            Start-Sleep -Seconds $IntervalSeconds

        } catch {
            Write-Log "Error during health check: $_" -Level "ERROR"
            Start-Sleep -Seconds $IntervalSeconds
        }
    }
} else {
    # Single run
    $health = Get-ContainerHealth
    Show-HealthReport -HealthStatus $health

    Write-Log "Health check complete: $($health.healthy.Count)/$($health.total) healthy"

    # Exit code based on health
    if ($health.unhealthy.Count -gt 0 -or $health.stopped.Count -gt 0) {
        exit 1
    } else {
        exit 0
    }
}
