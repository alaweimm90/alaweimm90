#!/usr/bin/env bash
# Bash completion for Meta-Orchestration CLI tools
# Source this file: source tools/cli/completions/bash-completion.sh

# Meta CLI completion
_meta_completion() {
    local cur prev opts
    COMPREPLY=()
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"
    
    case "${prev}" in
        meta)
            opts="sync health test dashboard start complete context metrics help"
            ;;
        sync)
            opts="--deep --force --dry-run"
            ;;
        health)
            opts="--json --verbose"
            ;;
        test)
            opts="--coverage --watch"
            ;;
        dashboard)
            opts="--live --json"
            ;;
        start)
            opts="--type --agent --priority"
            ;;
        --type)
            opts="feature bugfix refactor docs test infra"
            ;;
        --agent)
            opts="claude copilot kilo cursor"
            ;;
        *)
            return 0
            ;;
    esac
    
    COMPREPLY=( $(compgen -W "${opts}" -- ${cur}) )
    return 0
}
complete -F _meta_completion meta

# Governance CLI completion
_governance_completion() {
    local cur prev opts
    COMPREPLY=()
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"
    
    case "${prev}" in
        governance)
            opts="enforce checkpoint catalog audit sync health status help"
            ;;
        enforce)
            opts="--file --level --auto-fix"
            ;;
        --level)
            opts="error warning info"
            ;;
        checkpoint)
            opts="--repo --all"
            ;;
        catalog)
            opts="--type --status"
            ;;
        --type)
            opts="template workflow policy"
            ;;
        audit)
            opts="--scope --report"
            ;;
        --scope)
            opts="security compliance performance"
            ;;
        *)
            return 0
            ;;
    esac
    
    COMPREPLY=( $(compgen -W "${opts}" -- ${cur}) )
    return 0
}
complete -F _governance_completion governance

# Atlas CLI completion
_atlas_completion() {
    local cur prev opts
    COMPREPLY=()
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"
    
    case "${prev}" in
        atlas)
            opts="analyze scan report config agents status help"
            ;;
        analyze)
            opts="--repo --path --format"
            ;;
        --format)
            opts="json yaml markdown html"
            ;;
        scan)
            opts="--type --depth"
            ;;
        --type)
            opts="full quick security compliance"
            ;;
        agents)
            opts="list status configure"
            ;;
        *)
            return 0
            ;;
    esac
    
    COMPREPLY=( $(compgen -W "${opts}" -- ${cur}) )
    return 0
}
complete -F _atlas_completion atlas

# DevOps CLI completion
_devops_completion() {
    local cur prev opts
    COMPREPLY=()
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"
    
    case "${prev}" in
        devops)
            opts="sync analyze health ci cd deploy rollback help"
            ;;
        ci)
            opts="run status logs"
            ;;
        cd)
            opts="deploy promote rollback"
            ;;
        deploy)
            opts="--env --version --dry-run"
            ;;
        --env)
            opts="dev staging prod"
            ;;
        *)
            return 0
            ;;
    esac
    
    COMPREPLY=( $(compgen -W "${opts}" -- ${cur}) )
    return 0
}
complete -F _devops_completion devops

echo "✅ CLI completions loaded for: meta, governance, atlas, devops"

