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

$rows = New-Object System.Collections.Generic.List[object]
foreach ($p in Get-IdeTaskPaths) {
  try {
    $raw = Get-Content -LiteralPath $p -Raw
    $jsonText = Remove-JsonBlockComments $raw
    $obj = ConvertFrom-Json -InputObject $jsonText -ErrorAction Stop
    $ide = (Split-Path (Split-Path $p -Parent) -Parent | Split-Path -Leaf)
    foreach ($t in ($obj.tasks | Where-Object { $_.label -like 'Auto-Approve:*' })) {
      $approve = $null
      # Try parse from args first
      if ($t.args -and $t.args.Count -ge 3) {
        for ($i=0; $i -lt $t.args.Count; $i++) { if ($t.args[$i] -eq '-ApproveNext' -and $i+1 -lt $t.args.Count) { $approve = $t.args[$i+1]; break } }
      }
      if (-not $approve -and $t.label -match 'Approve Next (\d+)') { $approve = $Matches[1] }
      $runOn = $null
      if ($t.runOptions) { $runOn = $t.runOptions.runOn }
      $rows.Add([pscustomobject]@{
        IDE = $ide
        TasksJson = $p
        Label = $t.label
        ApproveNext = $approve
        RunOnOpen = if ($runOn) { $runOn } else { '-' }
      }) | Out-Null
    }
  } catch {
    Write-Warning "Failed to parse: $p -> $($_.Exception.Message)"
  }
}

if ($rows.Count -eq 0) { Write-Host "No Auto-Approve tasks found." -ForegroundColor Yellow; exit 0 }
$rows | Sort-Object IDE, Label | Format-Table -AutoSize
