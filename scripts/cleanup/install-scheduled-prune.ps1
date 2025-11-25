Param(
  [string]$ScriptPath = "c:\Users\mesha\Desktop\GitHub\scripts\cleanup\prune-logs.ps1",
  [int]$DaysToKeep = 14,
  [string]$TaskName = "RepoLogPrune",
  [string]$Time = "02:15"
)

$arg = "-NoProfile -ExecutionPolicy Bypass -File `"$ScriptPath`" -DaysToKeep $DaysToKeep"
$action = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument $arg
$trigger = New-ScheduledTaskTrigger -Daily -At ([DateTime]::Parse($Time))
Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Description "Prune repo logs older than $DaysToKeep days" -Force
Write-Output "Scheduled task '$TaskName' created"
