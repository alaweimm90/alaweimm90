<#
Adds user-level tasks to auto-approve commands on folder open.

- Primary: Approve Next N (default 50)
- Optional: Wrap + Report (runs approve then generates wrapper report)

Targets VS Code family IDEs under %APPDATA%: Code, Trae, Windsurf, Cursor.
#>
[CmdletBinding()]
param(
  [int]$ApproveNext = 50,
  [switch]$IncludeWrapReport,
  [switch]$NoRunOnOpen
)

function Remove-JsonBlockComments([string]$t){ [Regex]::Replace($t, "/\*.*?\*/", '', 'Singleline') }

function Get-IdeTaskPaths {
  $base = $env:APPDATA
  $targets = @()
  foreach ($name in @('Code','Trae','Windsurf','Cursor')) {
    $dir = Join-Path $base $name
    $user = Join-Path $dir 'User'
    $tasks = Join-Path $user 'tasks.json'
    if (-not (Test-Path $user)) { New-Item -ItemType Directory -Path $user -Force | Out-Null }
    if (-not (Test-Path $tasks)) { Set-Content -LiteralPath $tasks -Value '{\n  "version": "2.0.0",\n  "tasks": []\n}' -Encoding UTF8 }
    $targets += $tasks
  }
  $targets | Sort-Object -Unique
}

function Ensure-StartupApproveTask([string]$tasksPath){
  $raw = Get-Content -LiteralPath $tasksPath -Raw
  $jsonText = Remove-JsonBlockComments $raw
  try { $obj = ConvertFrom-Json -InputObject $jsonText -ErrorAction Stop } catch { $obj = @{ version = '2.0.0'; tasks = @() } }
  if (-not $obj.version) { $obj | Add-Member -NotePropertyName version -NotePropertyValue '2.0.0' -Force }
  if (-not $obj.tasks) { $obj | Add-Member -NotePropertyName tasks -NotePropertyValue @() -Force }
  $label = "Auto-Approve: Approve Next $ApproveNext (User Auto)"
  $existing = @()
  foreach ($t in $obj.tasks) { if ($t.label -eq $label) { $existing += $t } }
  if ($existing.Count -eq 0) {
    $task = [ordered]@{
      label = $label
      type  = 'shell'
      command = 'pwsh'
      args = @('.\\.tools\\ide-auto-approve.ps1','-ApproveNext',"$ApproveNext")
      options = @{ cwd = '${workspaceFolder}' }
      runOptions = $null
      presentation = @{ reveal = 'never' }
      problemMatcher = @()
    }
    if (-not $NoRunOnOpen) { $task.runOptions = @{ runOn = 'folderOpen' } }
    $obj.tasks = @($obj.tasks + $task)
  }
  if ($IncludeWrapReport) {
    $label2 = "Auto-Approve: Wrap + Report (User Auto)"
    $exists2 = $false
    foreach ($t in $obj.tasks) { if ($t.label -eq $label2) { $exists2 = $true; break } }
    if (-not $exists2) {
      $cmd = "pwsh ./.tools/ide-auto-approve.ps1 -ApproveNext $ApproveNext -VerboseLog; pwsh ./.tools/ide-wrapper-report.ps1"
      $task2 = [ordered]@{
        label = $label2
        type  = 'shell'
        command = 'pwsh'
        args = @('-NoProfile','-Command', $cmd)
        options = @{ cwd = '${workspaceFolder}' }
        runOptions = $null
        presentation = @{ reveal = 'never' }
        problemMatcher = @()
      }
      if (-not $NoRunOnOpen) { $task2.runOptions = @{ runOn = 'folderOpen' } }
      $obj.tasks = @($obj.tasks + $task2)
    }
  }
  ($obj | ConvertTo-Json -Depth 20) | Set-Content -LiteralPath $tasksPath -Encoding UTF8
}

$targets = Get-IdeTaskPaths
foreach ($p in $targets) {
  try { Write-Host "Setting startup approval task in: $p" -ForegroundColor Cyan; Ensure-StartupApproveTask -tasksPath $p; Write-Host "✅ Updated: $p" -ForegroundColor Green } catch { Write-Error $_ }
}
