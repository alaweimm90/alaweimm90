Param(
  [string[]]$Hosts = @("simcore.local","repz.local","benchbarrier.local","attributa.local","infra.local","metrics.local","traefik.local","status.local"),
  [string]$IP = "127.0.0.1"
)

$path = "$Env:WINDIR\System32\drivers\etc\hosts"
try {
  $existing = Get-Content $path -ErrorAction Stop
  foreach ($h in $Hosts) {
    if ($existing -notmatch "\b$h\b") {
      Add-Content -Path $path -Value "$IP`t$h"
    }
  }
  Write-Output "Hosts updated"
} catch {
  Write-Output "Failed to update hosts. Try running as Administrator. Error: $($_.Exception.Message)"
}
