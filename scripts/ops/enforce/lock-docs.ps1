Param([string[]]$Paths = @('docs','.metaHub/docs'))
foreach ($p in $Paths) {
  if (Test-Path $p) { attrib +R $p /S }
}
Write-Output "Docs set to read-only"
