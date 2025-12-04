# PowerShell completion for Meta-Orchestration CLI tools
# Add to $PROFILE: . tools/cli/completions/powershell-completion.ps1

# Meta CLI completion
Register-ArgumentCompleter -CommandName 'meta' -ScriptBlock {
    param($wordToComplete, $commandAst, $cursorPosition)
    
    $commands = @{
        '' = @('sync', 'health', 'test', 'dashboard', 'start', 'complete', 'context', 'metrics', 'help')
        'sync' = @('--deep', '--force', '--dry-run')
        'health' = @('--json', '--verbose')
        'test' = @('--coverage', '--watch')
        'dashboard' = @('--live', '--json')
        'start' = @('--type', '--agent', '--priority')
        '--type' = @('feature', 'bugfix', 'refactor', 'docs', 'test', 'infra')
        '--agent' = @('claude', 'copilot', 'kilo', 'cursor')
    }
    
    $elements = $commandAst.CommandElements
    $lastWord = if ($elements.Count -gt 1) { $elements[-1].ToString() } else { '' }
    $prevWord = if ($elements.Count -gt 2) { $elements[-2].ToString() } else { '' }
    
    $suggestions = if ($commands.ContainsKey($prevWord)) { $commands[$prevWord] }
                   elseif ($commands.ContainsKey($lastWord)) { $commands[$lastWord] }
                   else { $commands[''] }
    
    $suggestions | Where-Object { $_ -like "$wordToComplete*" } | ForEach-Object {
        [System.Management.Automation.CompletionResult]::new($_, $_, 'ParameterValue', $_)
    }
}

# Governance CLI completion
Register-ArgumentCompleter -CommandName 'governance' -ScriptBlock {
    param($wordToComplete, $commandAst, $cursorPosition)
    
    $commands = @{
        '' = @('enforce', 'checkpoint', 'catalog', 'audit', 'sync', 'health', 'status', 'help')
        'enforce' = @('--file', '--level', '--auto-fix')
        '--level' = @('error', 'warning', 'info')
        'checkpoint' = @('--repo', '--all')
        'catalog' = @('--type', '--status')
        '--type' = @('template', 'workflow', 'policy')
        'audit' = @('--scope', '--report')
        '--scope' = @('security', 'compliance', 'performance')
    }
    
    $elements = $commandAst.CommandElements
    $lastWord = if ($elements.Count -gt 1) { $elements[-1].ToString() } else { '' }
    $prevWord = if ($elements.Count -gt 2) { $elements[-2].ToString() } else { '' }
    
    $suggestions = if ($commands.ContainsKey($prevWord)) { $commands[$prevWord] }
                   elseif ($commands.ContainsKey($lastWord)) { $commands[$lastWord] }
                   else { $commands[''] }
    
    $suggestions | Where-Object { $_ -like "$wordToComplete*" } | ForEach-Object {
        [System.Management.Automation.CompletionResult]::new($_, $_, 'ParameterValue', $_)
    }
}

# Atlas CLI completion
Register-ArgumentCompleter -CommandName 'atlas' -ScriptBlock {
    param($wordToComplete, $commandAst, $cursorPosition)
    
    $commands = @{
        '' = @('analyze', 'scan', 'report', 'config', 'agents', 'status', 'help')
        'analyze' = @('--repo', '--path', '--format')
        '--format' = @('json', 'yaml', 'markdown', 'html')
        'scan' = @('--type', '--depth')
        '--type' = @('full', 'quick', 'security', 'compliance')
        'agents' = @('list', 'status', 'configure')
    }
    
    $elements = $commandAst.CommandElements
    $lastWord = if ($elements.Count -gt 1) { $elements[-1].ToString() } else { '' }
    $prevWord = if ($elements.Count -gt 2) { $elements[-2].ToString() } else { '' }
    
    $suggestions = if ($commands.ContainsKey($prevWord)) { $commands[$prevWord] }
                   elseif ($commands.ContainsKey($lastWord)) { $commands[$lastWord] }
                   else { $commands[''] }
    
    $suggestions | Where-Object { $_ -like "$wordToComplete*" } | ForEach-Object {
        [System.Management.Automation.CompletionResult]::new($_, $_, 'ParameterValue', $_)
    }
}

# DevOps CLI completion
Register-ArgumentCompleter -CommandName 'devops' -ScriptBlock {
    param($wordToComplete, $commandAst, $cursorPosition)
    
    $commands = @{
        '' = @('sync', 'analyze', 'health', 'ci', 'cd', 'deploy', 'rollback', 'help')
        'ci' = @('run', 'status', 'logs')
        'cd' = @('deploy', 'promote', 'rollback')
        'deploy' = @('--env', '--version', '--dry-run')
        '--env' = @('dev', 'staging', 'prod')
    }
    
    $elements = $commandAst.CommandElements
    $lastWord = if ($elements.Count -gt 1) { $elements[-1].ToString() } else { '' }
    $prevWord = if ($elements.Count -gt 2) { $elements[-2].ToString() } else { '' }
    
    $suggestions = if ($commands.ContainsKey($prevWord)) { $commands[$prevWord] }
                   elseif ($commands.ContainsKey($lastWord)) { $commands[$lastWord] }
                   else { $commands[''] }
    
    $suggestions | Where-Object { $_ -like "$wordToComplete*" } | ForEach-Object {
        [System.Management.Automation.CompletionResult]::new($_, $_, 'ParameterValue', $_)
    }
}

Write-Host "✅ CLI completions loaded for: meta, governance, atlas, devops" -ForegroundColor Green

