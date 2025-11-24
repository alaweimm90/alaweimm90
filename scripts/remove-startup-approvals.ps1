[CmdletBinding()]
param()

function Remove-JsonBlockComments([string]$t){ [Regex]::Replace($t, "/\*.*?\*/", '', 'Singleline') }

function Get-IdeTaskPaths {
  $base = $env:APPDATA
  $targets = @()
  foreach ($name in @('Code','Trae','Windsurf','Cursor')) {
    $dir = Join-Path $base $name
    $user = Join-Path $dir 'User'
    $tasks = Join-Path $user 'tasks.json'
    if (Test-Path $tasks) { $targets += $tasks }
  }
  $targets | Sort-Object -Unique
}

function Remove-StartupApproveTask([string]$tasksPath){
  $raw = Get-Content -LiteralPath $tasksPath -Raw
  $jsonText = Remove-JsonBlockComments $raw
  try { $obj = ConvertFrom-Json -InputObject $jsonText -ErrorAction Stop } catch { return }
  if (-not $obj.tasks) { return }
  $label = 'Auto-Approve: Approve Next 50 (User Auto)'
  $out = New-Object System.Collections.Generic.List[object]
  foreach ($t in $obj.tasks) { if ($t.label -ne $label) { [void]$out.Add($t) } }
  $obj.tasks = $out.ToArray()
  ($obj | ConvertTo-Json -Depth 20) | Set-Content -LiteralPath $tasksPath -Encoding UTF8
}

$targets = Get-IdeTaskPaths
foreach ($p in $targets) {
  try { Write-Host "Removing startup approval task in: $p" -ForegroundColor Cyan; Remove-StartupApproveTask -tasksPath $p; Write-Host "✅ Updated: $p" -ForegroundColor Green } catch { Write-Error $_ }
}
