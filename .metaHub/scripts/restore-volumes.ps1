# Docker Volume Restore Script
# Restores Docker volumes from backup archives

param(
    [Parameter(Mandatory=$true)]
    [string]$BackupTimestamp,
    [string]$BackupPath = ".metaHub/backups/volumes",
    [string[]]$Volumes = @(),  # Empty = restore all
    [switch]$Force = $false,
    [switch]$StopContainers = $true
)

$ErrorActionPreference = "Stop"

function Write-Status {
    param([string]$Message, [string]$Type = "INFO")
    $icons = @{
        "INFO" = "ℹ️"
        "SUCCESS" = "✅"
        "ERROR" = "❌"
        "WARNING" = "⚠️"
        "QUESTION" = "❓"
    }
    Write-Host "$($icons[$Type]) $Message"
}

Write-Host "`n╔══════════════════════════════════════════════════════════╗"
Write-Host "║          DOCKER VOLUME RESTORE                           ║"
Write-Host "╚══════════════════════════════════════════════════════════╝`n"

# Validate backup exists
$backupDir = Join-Path $BackupPath $BackupTimestamp
if (-not (Test-Path $backupDir)) {
    Write-Status "Backup directory not found: $backupDir" "ERROR"
    Write-Status "Available backups:" "INFO"
    Get-ChildItem $BackupPath -Directory | ForEach-Object {
        Write-Host "  • $($_.Name)"
    }
    exit 1
}

# Load manifest
$manifestPath = Join-Path $backupDir "manifest.json"
if (-not (Test-Path $manifestPath)) {
    Write-Status "Manifest not found in backup" "ERROR"
    exit 1
}

$manifest = Get-Content $manifestPath | ConvertFrom-Json
Write-Status "Backup date: $($manifest.Date)"
Write-Status "Backup location: $backupDir`n"

# Show what will be restored
Write-Host "📦 Available volumes in backup:"
$availableVolumes = @()
foreach ($volumeBackup in $manifest.Volumes) {
    if ($volumeBackup.Status -eq "Success") {
        $availableVolumes += $volumeBackup.Volume
        Write-Host "  • $($volumeBackup.Volume) - $($volumeBackup.Size)"
    }
}

# Determine which volumes to restore
$volumesToRestore = if ($Volumes.Count -eq 0) {
    $availableVolumes
} else {
    $Volumes | Where-Object { $availableVolumes -contains $_ }
}

if ($volumesToRestore.Count -eq 0) {
    Write-Status "No volumes to restore" "ERROR"
    exit 1
}

Write-Host "`n⚠️  WARNING: This will OVERWRITE existing volume data!`n"
Write-Host "Volumes to be restored:"
foreach ($vol in $volumesToRestore) {
    Write-Host "  • $vol" -ForegroundColor Yellow
}

if (-not $Force) {
    Write-Host "`nType 'yes' to continue or 'no' to cancel: " -NoNewline -ForegroundColor Yellow
    $confirmation = Read-Host
    if ($confirmation -ne "yes") {
        Write-Status "Restore cancelled by user" "WARNING"
        exit 0
    }
}

# Stop containers if requested
if ($StopContainers) {
    Write-Status "`nStopping containers..." "WARNING"
    docker compose stop
    Start-Sleep -Seconds 5
}

$restoreResults = @()

foreach ($volume in $volumesToRestore) {
    Write-Host "`n📥 Restoring volume: $volume"

    try {
        # Find backup file (check for compressed and uncompressed)
        $backupFile = Join-Path $backupDir "$volume.tar.gz"
        $isCompressed = $true

        if (-not (Test-Path $backupFile)) {
            $backupFile = Join-Path $backupDir "$volume.tar"
            $isCompressed = $false
        }

        if (-not (Test-Path $backupFile)) {
            Write-Status "Backup file not found for $volume" "ERROR"
            $restoreResults += @{
                Volume = $volume
                Status = "Failed"
                Reason = "Backup file not found"
            }
            continue
        }

        # Check if volume exists, create if not
        $volumeExists = docker volume inspect $volume 2>$null
        if (-not $volumeExists) {
            Write-Status "Creating volume $volume..."
            docker volume create $volume | Out-Null
        } else {
            Write-Status "Volume exists, will overwrite data" "WARNING"
        }

        # Restore based on file type
        if ($isCompressed) {
            Write-Status "Extracting compressed backup..."
            $result = docker run --rm `
                -v "${volume}:/data" `
                -v "${backupDir}:/backup" `
                alpine sh -c "cd /data && rm -rf * && tar xzf /backup/$volume.tar.gz" 2>&1
        } else {
            Write-Status "Extracting backup..."
            $result = docker run --rm `
                -v "${volume}:/data" `
                -v "${backupDir}:/backup" `
                alpine sh -c "cd /data && rm -rf * && tar xf /backup/$volume.tar" 2>&1
        }

        if ($LASTEXITCODE -eq 0) {
            Write-Status "Volume restored successfully" "SUCCESS"
            $restoreResults += @{
                Volume = $volume
                Status = "Success"
                Reason = "Restored from $BackupTimestamp"
            }
        } else {
            Write-Status "Restore failed: $result" "ERROR"
            $restoreResults += @{
                Volume = $volume
                Status = "Failed"
                Reason = $result
            }
        }

    } catch {
        Write-Status "Error restoring $volume : $_" "ERROR"
        $restoreResults += @{
            Volume = $volume
            Status = "Error"
            Reason = $_.Exception.Message
        }
    }
}

# Restart containers if they were stopped
if ($StopContainers) {
    Write-Status "`nRestarting containers..." "INFO"
    docker compose start
    Start-Sleep -Seconds 10
}

# Summary
Write-Host "`n╔══════════════════════════════════════════════════════════╗"
Write-Host "║          RESTORE SUMMARY                                 ║"
Write-Host "╚══════════════════════════════════════════════════════════╝`n"

Write-Status "Restore completed at: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Status "Backup source: $BackupTimestamp`n"

Write-Host "Restore Results:"
foreach ($result in $restoreResults) {
    $statusIcon = if ($result.Status -eq "Success") { "✅" } else { "❌" }
    Write-Host "  $statusIcon $($result.Volume) - $($result.Status)"
    if ($result.Status -ne "Success") {
        Write-Host "     Reason: $($result.Reason)" -ForegroundColor Gray
    }
}

$successCount = ($restoreResults | Where-Object { $_.Status -eq "Success" }).Count
Write-Host "`n✅ Successfully restored: $successCount/$($restoreResults.Count) volumes"

# Return failure count
$failedCount = ($restoreResults | Where-Object { $_.Status -ne "Success" }).Count
exit $failedCount
