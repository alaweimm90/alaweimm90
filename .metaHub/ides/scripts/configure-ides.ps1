param(  
    [switch]$All,  
    [switch]$NotepadPlus,  
    [switch]$TeXstudio,  
    [switch]$SublimeText  
)  
  
Write-Host "IDE Configuration Script Started"  
  
if ($All -or $NotepadPlus) {  
    Write-Host "Configuring Notepad++..."  
    $configPath = "$env:APPDATA\Notepad++\config.xml"  
    if (Test-Path $configPath) {  
        $content = Get-Content $configPath -Raw  
        $content = $content -replace '<GUIConfig name="SaveAllConfirm">yes</GUIConfig>', '<GUIConfig name="SaveAllConfirm">no</GUIConfig>'        $content = $content -replace 'confirmReplaceInAllOpenDocs="yes"', 'confirmReplaceInAllOpenDocs="no"'        $content | Set-Content $configPath -Encoding UTF8        Write-Host "Notepad++ configured successfully"    }}if ($All -or $TeXstudio) {    Write-Host "Configuring TeXstudio..."    $configPath = "$env:APPDATA\\texstudio\\texstudio.ini"    if (Test-Path $configPath) {        $content = Get-Content $configPath -Raw        $content = $content -replace 'Files\\\\Autosave=0', 'Files\\\\Autosave=30'        $content = $content -replace 'Editor\\\\RemoveTrailingWsOnSave=false', 'Editor\\\\RemoveTrailingWsOnSave=true'        $content | Set-Content $configPath -Encoding UTF8        Write-Host "TeXstudio configured successfully"    }}Write-Host "Configuration completed"
