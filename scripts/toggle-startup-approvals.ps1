<#
Toggle or set startup auto-approve tasks across VS Code-like IDEs (Code, Trae, Windsurf, Cursor).
- Adjust the ApproveNext count
- Enable/Disable/Toggle runOn: folderOpen
- Include/omit Wrap + Report task
#>
[CmdletBinding()]
param(
  [int]$ApproveNext = 100,
  [ValidateSet('Enable','Disable','Toggle')]
  [string]$RunOnOpen = 'Toggle',
  [switch]$IncludeWrapReport
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
    if (-not (Test-Path $tasks)) { Set-Content -LiteralPath $tasks -Value '{"version": "2.0.0", "tasks": []}' -Encoding UTF8 }
    $targets += $tasks
  }
  $targets | Sort-Object -Unique
}

function Ensure-TasksObject([string]$tasksPath){
  $raw = Get-Content -LiteralPath $tasksPath -Raw
  $jsonText = Remove-JsonBlockComments $raw
  try { $obj = ConvertFrom-Json -InputObject $jsonText -ErrorAction Stop } catch { $obj = @{ version = '2.0.0'; tasks = @() } }
  if (-not $obj.version) { $obj | Add-Member -NotePropertyName version -NotePropertyValue '2.0.0' -Force }
  if (-not $obj.tasks) { $obj | Add-Member -NotePropertyName tasks -NotePropertyValue @() -Force }
  return $obj
}

function Update-TaskRunOn {
  param([ref]$task, [string]$Mode)
  $t = $task.Value
  $cur = if ($t.runOptions) { $t.runOptions.runOn } else { $null }
  switch ($Mode) {
    'Enable' { if (-not $t.runOptions) { $t.runOptions = @{ runOn = 'folderOpen' } } else { $t.runOptions.runOn = 'folderOpen' } }
    'Disable' { if ($t.runOptions) { $t.runOptions.PSObject.Properties.Remove('runOn') | Out-Null } }
    'Toggle' {
      if ($cur -eq 'folderOpen') { if ($t.runOptions) { $t.runOptions.PSObject.Properties.Remove('runOn') | Out-Null } }
      else { if (-not $t.runOptions) { $t.runOptions = @{ runOn = 'folderOpen' } } else { $t.runOptions.runOn = 'folderOpen' } }
    }
  }
}

$targets = Get-IdeTaskPaths
foreach ($p in $targets) {
  try {
    Write-Host "Toggling startup approvals in: $p" -ForegroundColor Cyan
    $obj = Ensure-TasksObject $p
    $labelMain = "Auto-Approve: Approve Next $ApproveNext (User Auto)"
    $labelWrap = "Auto-Approve: Wrap + Report (User Auto)"

    # Try to find any existing Auto-Approve tasks regardless of number to retarget
    $main = $null; $wrap = $null
    foreach ($t in $obj.tasks) {
      if ($t.label -match '^Auto-Approve: Approve Next .* \(User Auto\)$') { $main = $t }
      if ($t.label -eq $labelWrap) { $wrap = $t }
    }
    # Create missing via add script
    if (-not $main) {
      if ($RunOnOpen -eq 'Disable') { pwsh -NoProfile -File (Join-Path $PSScriptRoot 'add-startup-approvals.ps1') -ApproveNext $ApproveNext -IncludeWrapReport:$IncludeWrapReport -NoRunOnOpen }
      else { pwsh -NoProfile -File (Join-Path $PSScriptRoot 'add-startup-approvals.ps1') -ApproveNext $ApproveNext -IncludeWrapReport:$IncludeWrapReport }
      # Reload
      $obj = Ensure-TasksObject $p
      foreach ($t in $obj.tasks) { if ($t.label -match '^Auto-Approve: Approve Next .* \(User Auto\)$') { $main = $t } }
      if ($IncludeWrapReport -and -not $wrap) { foreach ($t in $obj.tasks) { if ($t.label -eq $labelWrap) { $wrap = $t } } }
    } else {
      # Update label and args for ApproveNext
      $main.label = $labelMain
      $main.args = @('.\\.tools\\ide-auto-approve.ps1','-ApproveNext',"$ApproveNext")
      Update-TaskRunOn ([ref]$main) -Mode $RunOnOpen
      if ($IncludeWrapReport) {
        if (-not $wrap) {
          $cmd = "pwsh ./.tools/ide-auto-approve.ps1 -ApproveNext $ApproveNext -VerboseLog; pwsh ./.tools/ide-wrapper-report.ps1"
          $wrap = [ordered]@{
            label = $labelWrap; type='shell'; command='pwsh'; args=@('-NoProfile','-Command',$cmd); options=@{cwd='${workspaceFolder}'}; runOptions=$null; presentation=@{reveal='never'}; problemMatcher=@() }
          Update-TaskRunOn ([ref]$wrap) -Mode $RunOnOpen
          $obj.tasks = @($obj.tasks + $wrap)
        } else {
          Update-TaskRunOn ([ref]$wrap) -Mode $RunOnOpen
        }
      } else {
        # If wrap exists but IncludeWrapReport not set, leave as-is (non-destructive)
      }
    }
    ($obj | ConvertTo-Json -Depth 30) | Set-Content -LiteralPath $p -Encoding UTF8
    Write-Host "✅ Updated: $p" -ForegroundColor Green
  } catch { Write-Error $_ }
}
