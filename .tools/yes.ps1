param([string]$Text="yes",[int]$Count=0,[int]$DelayMs=0)
if($Count -le 0){
  while($true){
    Write-Output $Text
    if($DelayMs -gt 0){ Start-Sleep -Milliseconds $DelayMs }
  }
} else {
  for($i=0; $i -lt $Count; $i++){
    Write-Output $Text
    if($DelayMs -gt 0){ Start-Sleep -Milliseconds $DelayMs }
  }
}
