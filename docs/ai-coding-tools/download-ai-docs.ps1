# ============================================================================
# AI Coding Tools Documentation Downloader (Windows PowerShell Edition)
# Downloads documentation for ALL major AI coding assistants and IDEs
# ============================================================================

param(
    [switch]$CloneOnly,      # Only clone GitHub repos
    [switch]$DownloadOnly,   # Only download web docs
    [switch]$All             # Download everything
)

$ErrorActionPreference = "Continue"

# Base directory
$BaseDir = $PSScriptRoot
if (-not $BaseDir) { $BaseDir = Get-Location }

Write-Host "================================================" -ForegroundColor Blue
Write-Host "  AI Coding Tools Documentation Downloader     " -ForegroundColor Blue
Write-Host "  Windows PowerShell Edition                   " -ForegroundColor Blue
Write-Host "================================================" -ForegroundColor Blue
Write-Host ""

# Function to clone from GitHub
function Clone-Docs {
    param(
        [string]$Name,
        [string]$Repo,
        [string]$Subdir = ""
    )

    $dir = Join-Path $BaseDir $Name

    Write-Host "[$(Get-Date -Format 'HH:mm:ss')] Cloning $Name..." -ForegroundColor Green
    Write-Host "  Repository: $Repo" -ForegroundColor Yellow
    Write-Host "  Destination: $dir" -ForegroundColor Yellow

    if (Test-Path $dir) {
        Write-Host "  Directory exists, pulling latest..." -ForegroundColor Yellow
        Push-Location $dir
        git pull --quiet 2>$null
        Pop-Location
    } else {
        git clone --quiet --depth 1 $Repo $dir 2>$null
    }

    if ($Subdir -and (Test-Path (Join-Path $dir $Subdir))) {
        Write-Host "  Docs subfolder: $Subdir" -ForegroundColor Cyan
    }

    Write-Host "  Done: $Name" -ForegroundColor Green
    Write-Host ""
}

# Function to create doc reference
function Create-DocRef {
    param(
        [string]$Name,
        [string]$Url,
        [string]$Category,
        [string]$Description
    )

    $dir = Join-Path $BaseDir $Category
    $file = Join-Path $dir "$Name.md"

    $content = @"
# $Name Documentation

**Official URL**: $Url
**Category**: $Category
**Downloaded**: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')

## Description

$Description

## Quick Access

- [Official Documentation]($Url)

## Notes

This is a reference file. Visit the official URL for full documentation.
Some documentation sites require JavaScript or authentication.

## Local Files

If documentation was downloaded, check the `$Name` folder in this directory.
"@

    $content | Out-File -FilePath $file -Encoding utf8
    Write-Host "  Created reference: $file" -ForegroundColor Cyan
}

# ============================================================================
# SECTION 1: GitHub-based Documentation (Reliable cloning)
# ============================================================================

Write-Host "=== Cloning GitHub-based Documentation ===" -ForegroundColor Blue
Write-Host ""

# Continue.dev - has excellent docs
Clone-Docs -Name "01-major-assistants/continue-dev" -Repo "https://github.com/continuedev/continue.git" -Subdir "docs"

# Aider - terminal AI coding
Clone-Docs -Name "02-terminal-cli/aider" -Repo "https://github.com/paul-gauthier/aider.git" -Subdir "docs"

# Shell GPT
Clone-Docs -Name "02-terminal-cli/shell-gpt" -Repo "https://github.com/TheR1D/shell_gpt.git"

# AI Shell
Clone-Docs -Name "02-terminal-cli/ai-shell" -Repo "https://github.com/BuilderIO/ai-shell.git"

# GPT Engineer
Clone-Docs -Name "05-specialized-tools/gpt-engineer" -Repo "https://github.com/gpt-engineer-org/gpt-engineer.git"

# Sweep AI
Clone-Docs -Name "05-specialized-tools/sweep-ai" -Repo "https://github.com/sweepai/sweep.git"

# Mentat
Clone-Docs -Name "05-specialized-tools/mentat" -Repo "https://github.com/AbanteAI/mentat.git"

# CodeGPT
Clone-Docs -Name "05-specialized-tools/codegpt" -Repo "https://github.com/appleboy/CodeGPT.git"

# Ollama
Clone-Docs -Name "06-open-source-llms/ollama" -Repo "https://github.com/ollama/ollama.git"

# Jan AI
Clone-Docs -Name "06-open-source-llms/jan-ai" -Repo "https://github.com/janhq/jan.git"

# LocalAI
Clone-Docs -Name "06-open-source-llms/local-ai" -Repo "https://github.com/mudler/LocalAI.git"

# AutoGPT
Clone-Docs -Name "08-resources/autogpt" -Repo "https://github.com/Significant-Gravitas/AutoGPT.git"

# LangChain
Clone-Docs -Name "08-resources/langchain" -Repo "https://github.com/langchain-ai/langchain.git" -Subdir "docs"

# LlamaIndex
Clone-Docs -Name "08-resources/llamaindex" -Repo "https://github.com/run-llama/llama_index.git" -Subdir "docs"

# ============================================================================
# SECTION 2: Create Reference Files for Web-based Docs
# ============================================================================

Write-Host "=== Creating Documentation References ===" -ForegroundColor Blue
Write-Host ""

# Major Assistants
Create-DocRef -Name "kilo-code" -Url "https://kilocode.ai/docs" -Category "01-major-assistants" -Description "Kilo Code - Open source AI coding assistant with VSCode extension"
Create-DocRef -Name "claude-code" -Url "https://docs.anthropic.com/en/docs/claude-code" -Category "01-major-assistants" -Description "Claude Code - Anthropic's official CLI for Claude, terminal and IDE coding assistant"
Create-DocRef -Name "github-copilot" -Url "https://docs.github.com/en/copilot" -Category "01-major-assistants" -Description "GitHub Copilot - AI pair programmer by GitHub/Microsoft"
Create-DocRef -Name "cursor" -Url "https://docs.cursor.com" -Category "01-major-assistants" -Description "Cursor - AI-first code editor built on VSCode"
Create-DocRef -Name "codeium" -Url "https://codeium.com/docs" -Category "01-major-assistants" -Description "Codeium - Free AI code completion tool"
Create-DocRef -Name "tabnine" -Url "https://docs.tabnine.com" -Category "01-major-assistants" -Description "Tabnine - AI code completion with team learning"
Create-DocRef -Name "codewhisperer" -Url "https://docs.aws.amazon.com/codewhisperer" -Category "01-major-assistants" -Description "Amazon CodeWhisperer - AWS AI coding companion (now Amazon Q)"
Create-DocRef -Name "cody" -Url "https://sourcegraph.com/docs/cody" -Category "01-major-assistants" -Description "Sourcegraph Cody - AI coding assistant with codebase context"
Create-DocRef -Name "blackbox-ai" -Url "https://www.blackbox.ai" -Category "01-major-assistants" -Description "Blackbox AI - AI-powered code generation and chat"
Create-DocRef -Name "trae" -Url "https://trae.ai" -Category "01-major-assistants" -Description "Trae - AI coding assistant"

# Terminal/CLI Tools
Create-DocRef -Name "warp-ai" -Url "https://docs.warp.dev/features/ai" -Category "02-terminal-cli" -Description "Warp AI - Modern terminal with built-in AI assistance"

# IDE Integrations
Create-DocRef -Name "jetbrains-ai" -Url "https://www.jetbrains.com/help/idea/ai-assistant.html" -Category "03-ide-integrations" -Description "JetBrains AI Assistant - AI features in IntelliJ IDEA and other JetBrains IDEs"
Create-DocRef -Name "vs-intellicode" -Url "https://learn.microsoft.com/en-us/visualstudio/intellicode" -Category "03-ide-integrations" -Description "Visual Studio IntelliCode - Microsoft AI for Visual Studio"
Create-DocRef -Name "replit-ai" -Url "https://docs.replit.com/power-ups/ghostwriter" -Category "03-ide-integrations" -Description "Replit AI (Ghostwriter) - AI assistant in Replit IDE"
Create-DocRef -Name "pieces" -Url "https://docs.pieces.app" -Category "03-ide-integrations" -Description "Pieces for Developers - AI-powered code snippets and context"
Create-DocRef -Name "windsurf" -Url "https://codeium.com/windsurf" -Category "03-ide-integrations" -Description "Windsurf - AI IDE by Codeium"

# API Documentation
Create-DocRef -Name "openai-api" -Url "https://platform.openai.com/docs" -Category "04-api-docs" -Description "OpenAI API - GPT-4, Codex, Assistants API, and more"
Create-DocRef -Name "claude-api" -Url "https://docs.anthropic.com" -Category "04-api-docs" -Description "Anthropic Claude API - Claude models API documentation"
Create-DocRef -Name "gemini-api" -Url "https://ai.google.dev/docs" -Category "04-api-docs" -Description "Google Gemini API - Google's AI models and Code Assist"
Create-DocRef -Name "mistral-api" -Url "https://docs.mistral.ai" -Category "04-api-docs" -Description "Mistral AI - Open source models API"
Create-DocRef -Name "cohere-api" -Url "https://docs.cohere.com" -Category "04-api-docs" -Description "Cohere - Language AI platform"
Create-DocRef -Name "together-api" -Url "https://docs.together.ai" -Category "04-api-docs" -Description "Together AI - Open source models hosting"

# Specialized Tools
Create-DocRef -Name "phind" -Url "https://www.phind.com" -Category "05-specialized-tools" -Description "Phind - AI-powered search engine for developers"
Create-DocRef -Name "amazon-q" -Url "https://docs.aws.amazon.com/amazonq" -Category "05-specialized-tools" -Description "Amazon Q Developer - AWS AI assistant (successor to CodeWhisperer)"

# Open Source LLMs
Create-DocRef -Name "lm-studio" -Url "https://lmstudio.ai" -Category "06-open-source-llms" -Description "LM Studio - Desktop app for running local LLMs"
Create-DocRef -Name "code-llama" -Url "https://ai.meta.com/llama" -Category "06-open-source-llms" -Description "Code Llama - Meta's code-specialized LLM"

# Code Review
Create-DocRef -Name "codacy" -Url "https://docs.codacy.com" -Category "07-code-review" -Description "Codacy - Automated code review with AI"
Create-DocRef -Name "snyk-code" -Url "https://docs.snyk.io/scan-using-snyk/snyk-code" -Category "07-code-review" -Description "Snyk Code (DeepCode) - AI-powered security analysis"
Create-DocRef -Name "coderabbit" -Url "https://coderabbit.ai" -Category "07-code-review" -Description "CodeRabbit - AI code review for PRs"
Create-DocRef -Name "semgrep" -Url "https://semgrep.dev/docs" -Category "07-code-review" -Description "Semgrep - Static analysis with AI rules"

# Resources
Create-DocRef -Name "huggingface" -Url "https://huggingface.co/docs" -Category "08-resources" -Description "Hugging Face - Hub for AI models and documentation"

Write-Host ""
Write-Host "================================================" -ForegroundColor Blue
Write-Host "  Documentation Download Complete!              " -ForegroundColor Green
Write-Host "================================================" -ForegroundColor Blue
Write-Host ""
Write-Host "Location: $BaseDir" -ForegroundColor Yellow
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1. Review the cloned repositories for docs/" -ForegroundColor White
Write-Host "  2. Check reference files for web documentation URLs" -ForegroundColor White
Write-Host "  3. Run this script again to update repos" -ForegroundColor White
Write-Host ""
