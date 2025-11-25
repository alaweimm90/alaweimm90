# Auto-Workflow PowerShell Script for YOLO Mode Automation
# This script automates common development tasks without confirmations

param(
    [string]$Action = "all"
)

function Install-Dependencies {
    Write-Host "Installing dependencies..."
    if (Test-Path "package.json") {
        npm install
    }
    if (Test-Path "pnpm-lock.yaml") {
        pnpm install
    }
    if (Test-Path "requirements.txt") {
        pip install -r requirements.txt
    }
}

function Build-Project {
    Write-Host "Building project..."
    if (Test-Path "turbo.json") {
        turbo build
    } elseif (Test-Path "package.json") {
        npm run build
    } else {
        Write-Host "No build script found."
    }
}

function Run-Tests {
    Write-Host "Running tests..."
    if (Test-Path "package.json") {
        npm test
    }
    if (Test-Path "pnpm-lock.yaml") {
        pnpm test
    }
}

function Git-AutoCommit {
    Write-Host "Auto-committing changes..."
    git add .
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    git commit -m "Auto-commit: $timestamp"
}

function Deploy-Project {
    Write-Host "Deploying project..."
    if (Test-Path "package.json") {
        npm run deploy
    } elseif (Test-Path "docker-compose.yml") {
        docker-compose up -d
    } else {
        Write-Host "No deployment script found."
    }
}

function Clean-Rebuild {
    Write-Host "Cleaning and rebuilding..."
    Remove-Item -Recurse -Force node_modules -ErrorAction SilentlyContinue
    Install-Dependencies
    Build-Project
}

switch ($Action) {
    "install" { Install-Dependencies }
    "build" { Build-Project }
    "test" { Run-Tests }
    "commit" { Git-AutoCommit }
    "deploy" { Deploy-Project }
    "clean" { Clean-Rebuild }
    "all" {
        Install-Dependencies
        Build-Project
        Run-Tests
        Git-AutoCommit
        Deploy-Project
    }
    default { Write-Host "Usage: .\auto-workflow.ps1 -Action [install|build|test|commit|deploy|clean|all]" }
}

Write-Host "Automation complete."