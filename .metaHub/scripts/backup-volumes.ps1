# Docker Volume Backup Script
# Backs up all Docker volumes to compressed archives

param(
    [string]$BackupPath = ".metaHub/backups/volumes",
    [string[]]$Volumes = @("simcore-data", "repz-data", "benchbarrier-data", "mag-logic-data", "attributa-data"),
    [switch]$Compress = $true,
    [switch]$StopContainers = $false
)

$ErrorActionPreference = "Stop"

function Write-Status {
    param([string]$Message, [string]$Type = "INFO")
    $icons = @{
        "INFO" = "ℹ️"
        "SUCCESS" = "✅"
        "ERROR" = "❌"
        "WARNING" = "⚠️"
    }
    Write-Host "$($icons[$Type]) $Message"
}

# Create backup directory
if (-not (Test-Path $BackupPath)) {
    New-Item -ItemType Directory -Path $BackupPath -Force | Out-Null
    Write-Status "Created backup directory: $BackupPath" "SUCCESS"
}

$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$backupDir = Join-Path $BackupPath $timestamp
New-Item -ItemType Directory -Path $backupDir -Force | Out-Null

Write-Host "`n╔══════════════════════════════════════════════════════════╗"
Write-Host "║          DOCKER VOLUME BACKUP                            ║"
Write-Host "╚══════════════════════════════════════════════════════════╝`n"

Write-Status "Backup started at: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Status "Backup location: $backupDir`n"

# Stop containers if requested
if ($StopContainers) {
    Write-Status "Stopping containers..." "WARNING"
    docker compose stop
    Start-Sleep -Seconds 5
}

$backupResults = @()

foreach ($volume in $Volumes) {
    Write-Host "`n📦 Backing up volume: $volume"

    try {
        # Check if volume exists
        $volumeExists = docker volume inspect $volume 2>$null
        if (-not $volumeExists) {
            Write-Status "Volume $volume does not exist, skipping" "WARNING"
            continue
        }

        # Get volume size
        $volumeInfo = docker volume inspect $volume | ConvertFrom-Json
        $volumePath = $volumeInfo[0].Mountpoint

        # Create backup using alpine container
        $backupFile = Join-Path $backupDir "$volume.tar"
        Write-Status "Creating archive..."

        $result = docker run --rm `
            -v "${volume}:/data" `
            -v "${backupDir}:/backup" `
            alpine tar cf "/backup/$volume.tar" -C /data . 2>&1

        if ($LASTEXITCODE -eq 0) {
            $archiveSize = (Get-Item $backupFile).Length / 1MB
            Write-Status "Archive created: $([math]::Round($archiveSize, 2)) MB" "SUCCESS"

            # Compress if requested
            if ($Compress) {
                Write-Status "Compressing archive..."
                $compressedFile = "$backupFile.gz"

                docker run --rm `
                    -v "${backupDir}:/backup" `
                    alpine sh -c "gzip /backup/$volume.tar"

                if ($LASTEXITCODE -eq 0) {
                    $compressedSize = (Get-Item $compressedFile).Length / 1MB
                    $compressionRatio = [math]::Round((1 - ($compressedSize / $archiveSize)) * 100, 1)
                    Write-Status "Compressed to: $([math]::Round($compressedSize, 2)) MB (saved $compressionRatio%)" "SUCCESS"

                    $backupResults += @{
                        Volume = $volume
                        Status = "Success"
                        Size = "$([math]::Round($compressedSize, 2)) MB"
                        File = "$volume.tar.gz"
                    }
                } else {
                    Write-Status "Compression failed" "ERROR"
                }
            } else {
                $backupResults += @{
                    Volume = $volume
                    Status = "Success"
                    Size = "$([math]::Round($archiveSize, 2)) MB"
                    File = "$volume.tar"
                }
            }
        } else {
            Write-Status "Backup failed: $result" "ERROR"
            $backupResults += @{
                Volume = $volume
                Status = "Failed"
                Size = "N/A"
                File = "N/A"
            }
        }

    } catch {
        Write-Status "Error backing up $volume : $_" "ERROR"
        $backupResults += @{
            Volume = $volume
            Status = "Error"
            Size = "N/A"
            File = "N/A"
        }
    }
}

# Restart containers if they were stopped
if ($StopContainers) {
    Write-Status "`nRestarting containers..." "INFO"
    docker compose start
}

# Create backup manifest
$manifest = @{
    Timestamp = $timestamp
    Date = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Volumes = $backupResults
    TotalSize = (Get-ChildItem $backupDir -File | Measure-Object -Property Length -Sum).Sum / 1MB
}

$manifestPath = Join-Path $backupDir "manifest.json"
$manifest | ConvertTo-Json -Depth 5 | Set-Content $manifestPath

Write-Host "`n╔══════════════════════════════════════════════════════════╗"
Write-Host "║          BACKUP SUMMARY                                  ║"
Write-Host "╚══════════════════════════════════════════════════════════╝`n"

Write-Status "Backup completed at: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Status "Backup location: $backupDir"
Write-Status "Total backup size: $([math]::Round($manifest.TotalSize, 2)) MB`n"

Write-Host "Volume Backups:"
foreach ($result in $backupResults) {
    $statusIcon = if ($result.Status -eq "Success") { "✅" } else { "❌" }
    Write-Host "  $statusIcon $($result.Volume) - $($result.Size) - $($result.File)"
}

Write-Host "`n📄 Manifest saved to: $manifestPath"

# Cleanup old backups (keep last 7)
Write-Host "`n🧹 Cleaning up old backups..."
$allBackups = Get-ChildItem $BackupPath -Directory | Sort-Object Name -Descending
if ($allBackups.Count -gt 7) {
    $toDelete = $allBackups | Select-Object -Skip 7
    foreach ($old in $toDelete) {
        Write-Status "Removing old backup: $($old.Name)"
        Remove-Item $old.FullName -Recurse -Force
    }
    Write-Status "Kept 7 most recent backups" "SUCCESS"
}

Write-Host "`n✅ Backup complete!`n"

# Return success/failure
$failedCount = ($backupResults | Where-Object { $_.Status -ne "Success" }).Count
exit $failedCount
