#!/bin/bash
# SpinCirc Repository Management Shortcuts
# Source this file in your ~/.bashrc: source /mnt/c/Users/mesha/Documents/GitHub/SpinCirc/.spincirc_shortcuts.sh

# Define the SpinCirc repository path
SPINCIRC_PATH="/mnt/c/Users/mesha/Documents/GitHub/SpinCirc"

# BEFORE CLAUDE SESSION: Navigate to repo and get latest changes
alias repo-start='cd "$SPINCIRC_PATH" && echo "🚀 Starting SpinCirc session..." && git pull origin main && echo "📊 Repository status:" && git status'

# DURING CLAUDE SESSION: Quick status check
alias repo-check='cd "$SPINCIRC_PATH" && echo "📊 Current repository status:" && git status && echo "📝 Recent changes:" && git diff --name-only'

# AFTER CLAUDE SESSION: Review changes in detail
alias repo-review='cd "$SPINCIRC_PATH" && echo "🔍 Detailed change review:" && git status && echo -e "\n📋 Files changed:" && git diff --stat && echo -e "\n💡 Use repo-stage to add files, then repo-commit"'

# Stage specific files (interactive)
alias repo-stage='cd "$SPINCIRC_PATH" && echo "📁 Staging files for commit..." && git add'

# Stage all changes
alias repo-stage-all='cd "$SPINCIRC_PATH" && echo "📁 Staging ALL changes..." && git add . && echo "✅ All changes staged. Use repo-commit next."'

# Interactive commit function
repo-commit() {
    cd "$SPINCIRC_PATH"
    if [ -z "$1" ]; then
        echo "❌ Usage: repo-commit \"Your commit message\""
        echo "💡 Example: repo-commit \"Add new magnetization solver\""
        return 1
    fi
    echo "💾 Committing changes..."
    git commit -m "$1"
    echo "✅ Changes committed! Use repo-push to upload to GitHub."
}

# Push to GitHub
alias repo-push='cd "$SPINCIRC_PATH" && echo "🚀 Pushing to GitHub..." && git push origin main && echo "✅ Successfully pushed to GitHub!"'

# COMPLETE WORKFLOW: Add all, commit, and push in one command
repo-save() {
    cd "$SPINCIRC_PATH"
    if [ -z "$1" ]; then
        echo "❌ Usage: repo-save \"Your commit message\""
        echo "💡 Example: repo-save \"Enhanced transport solver with GPU support\""
        return 1
    fi
    echo "💾 Complete save: staging, committing, and pushing..."
    git add .
    git commit -m "$1"
    git push origin main
    echo "✅ All changes saved and pushed to GitHub!"
}

# Emergency: Show what changed since last commit
alias repo-diff='cd "$SPINCIRC_PATH" && echo "🔍 Changes since last commit:" && git diff'

# Show recent commit history
alias repo-log='cd "$SPINCIRC_PATH" && echo "📜 Recent commit history:" && git log --oneline -10'

# Quick navigation to SpinCirc directory
alias spincirc='cd "$SPINCIRC_PATH" && pwd && ls -la'

# Help function
repo-help() {
    echo "🎯 SpinCirc Repository Shortcuts:"
    echo ""
    echo "🚀 BEFORE CLAUDE SESSION:"
    echo "   repo-start     - Navigate to repo, pull latest changes, check status"
    echo ""
    echo "⚡ DURING CLAUDE SESSION:"
    echo "   repo-check     - Quick status and recent changes summary"
    echo "   repo-diff      - Show detailed changes since last commit"
    echo ""
    echo "💾 AFTER CLAUDE SESSION:"
    echo "   repo-review    - Review all changes in detail"
    echo "   repo-stage     - Stage specific files (repo-stage file1.m file2.py)"
    echo "   repo-stage-all - Stage all changes"
    echo "   repo-commit    - Commit with message (repo-commit \"message\")"
    echo "   repo-push      - Push to GitHub"
    echo ""
    echo "🎯 ONE-COMMAND WORKFLOWS:"
    echo "   repo-save      - Stage all, commit, and push (repo-save \"message\")"
    echo ""
    echo "📋 UTILITIES:"
    echo "   repo-log       - Show recent commit history"
    echo "   spincirc       - Navigate to SpinCirc directory"
    echo "   repo-help      - Show this help"
    echo ""
    echo "💡 TYPICAL WORKFLOW:"
    echo "   1. repo-start           # Before Claude"
    echo "   2. [Use Claude Code]    # During session"
    echo "   3. repo-review          # After Claude"
    echo "   4. repo-save \"message\"  # Save everything"
}

# Auto-complete for common commit messages
_repo_commit_completions() {
    local common_messages=(
        "Add new device model"
        "Enhance transport solver"
        "Fix numerical stability issue"
        "Update documentation"
        "Add validation example"
        "Optimize performance"
        "Refactor code structure"
        "Add unit tests"
        "Fix bug in magnetization dynamics"
        "Update material database"
    )
    
    COMPREPLY=($(compgen -W "${common_messages[*]}" -- "${COMP_WORDS[COMP_CUR]}"))
}

# Enable tab completion for commit messages
complete -F _repo_commit_completions repo-commit repo-save

echo "✅ SpinCirc shortcuts loaded! Type 'repo-help' for usage guide."