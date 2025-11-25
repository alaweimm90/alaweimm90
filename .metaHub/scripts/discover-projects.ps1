# Project Discovery & Inventory Script
# Scans all organizations and populates projects-registry.json

param(
    [string]$RegistryFile = ".metaHub\projects-registry.json",
    [switch]$DryRun = $false,
    [switch]$Verbose = $false
)

Write-Host "[*] Multi-Organization Project Discovery" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Load existing registry
$registry = Get-Content $RegistryFile | ConvertFrom-Json

# Organizations to scan
$orgsToScan = @(
    ".config\organizations\alaweimm90-business",
    ".config\organizations\alaweimm90-science",
    ".config\organizations\alaweimm90-tools",
    ".config\organizations\AlaweinOS",
    ".config\organizations\.personal",
    ".config\organizations\MeatheadPhysicist"
)

$discoveredProjects = @()
$projectCounter = 1

foreach ($orgPath in $orgsToScan) {
    if (-not (Test-Path $orgPath)) {
        Write-Host "[!] Organization path not found: $orgPath" -ForegroundColor Yellow
        continue
    }

    $orgName = Split-Path $orgPath -Leaf
    Write-Host "[>>] Scanning organization: $orgName" -ForegroundColor Magenta

    # Find all package.json files (indicates a project)
    $packageJsonFiles = Get-ChildItem -Path $orgPath -Filter "package.json" -Recurse -ErrorAction SilentlyContinue

    # Find all docker-compose.yml files
    $dockerComposeFiles = Get-ChildItem -Path $orgPath -Filter "docker-compose.yml" -Recurse -ErrorAction SilentlyContinue

    # Find all directories with .git folders (git repos)
    $gitRepos = Get-ChildItem -Path $orgPath -Filter ".git" -Recurse -Directory -ErrorAction SilentlyContinue | ForEach-Object { $_.Parent.FullName }

    # Combine all indicators of projects
    $projectPaths = @()
    $projectPaths += $packageJsonFiles | ForEach-Object { $_.Directory.FullName }
    $projectPaths += $dockerComposeFiles | ForEach-Object { $_.Directory.FullName }
    $projectPaths += $gitRepos

    # Deduplicate
    $projectPaths = $projectPaths | Select-Object -Unique | Where-Object { $_ -ne $null }

    foreach ($projectPath in $projectPaths) {
        $projectName = Split-Path $projectPath -Leaf
        $relativePath = Resolve-Path -Relative $projectPath

        # Detect tech stack
        $techStack = @()
        if (Test-Path "$projectPath\package.json") {
            $packageJson = Get-Content "$projectPath\package.json" | ConvertFrom-Json
            if ($packageJson.dependencies -or $packageJson.devDependencies) {
                if ($packageJson.dependencies.react -or $packageJson.devDependencies.react) { $techStack += "react" }
                if ($packageJson.dependencies.vue -or $packageJson.devDependencies.vue) { $techStack += "vue" }
                if ($packageJson.dependencies.next -or $packageJson.devDependencies.next) { $techStack += "nextjs" }
                if ($packageJson.dependencies.express) { $techStack += "express" }
                if ($packageJson.dependencies.fastify) { $techStack += "fastify" }
                if ($packageJson.dependencies.nest -or $packageJson.dependencies."@nestjs/core") { $techStack += "nestjs" }
                if ($packageJson.dependencies.typescript -or $packageJson.devDependencies.typescript) { $techStack += "typescript" }
                if ($packageJson.dependencies.prisma -or $packageJson.devDependencies.prisma) { $techStack += "prisma" }
                if ($packageJson.dependencies.postgres -or $packageJson.dependencies.pg) { $techStack += "postgres" }
                if ($packageJson.dependencies.mongodb -or $packageJson.dependencies.mongoose) { $techStack += "mongodb" }
            }
        }

        # Check for Python projects
        if (Test-Path "$projectPath\requirements.txt" -or Test-Path "$projectPath\pyproject.toml") {
            $techStack += "python"
            if (Test-Path "$projectPath\manage.py") { $techStack += "django" }
            if (Get-Content "$projectPath\requirements.txt" -ErrorAction SilentlyContinue | Select-String "flask") { $techStack += "flask" }
            if (Get-Content "$projectPath\requirements.txt" -ErrorAction SilentlyContinue | Select-String "fastapi") { $techStack += "fastapi" }
        }

        # Check for Go projects
        if (Test-Path "$projectPath\go.mod") {
            $techStack += "go"
        }

        # Check for Rust projects
        if (Test-Path "$projectPath\Cargo.toml") {
            $techStack += "rust"
        }

        # Check for containerization
        $hasDocker = Test-Path "$projectPath\Dockerfile"
        $hasDockerCompose = Test-Path "$projectPath\docker-compose.yml"
        $isContainerized = $hasDocker -or $hasDockerCompose

        # Check for CI/CD
        $hasCICD = Test-Path "$projectPath\.github\workflows" -or `
                   Test-Path "$projectPath\.gitlab-ci.yml" -or `
                   Test-Path "$projectPath\Jenkinsfile"

        # Determine category
        $category = "unknown"
        if ($projectName -match "api|backend|service") {
            $category = "backend-service"
        } elseif ($projectName -match "frontend|web|app") {
            $category = "frontend-app"
        } elseif ($projectName -match "lib|shared|common") {
            $category = "library"
        } elseif ($projectName -match "tool|cli|script") {
            $category = "tool"
        } elseif ($projectName -match "infra|deploy") {
            $category = "infrastructure"
        }

        # Create project entry
        $project = @{
            id = "proj-{0:D3}" -f $projectCounter
            name = $projectName
            organizationId = ($registry.organizations.PSObject.Properties | Where-Object { $_.Value.path -eq $orgPath.Replace("\", "/") } | Select-Object -First 1).Value.id
            status = "unknown"  # User will need to fill this in
            category = $category
            techStack = @($techStack)
            containerized = $isContainerized
            cicd = $hasCICD
            priority = "medium"  # Default
            healthScore = 5  # Default
            path = $relativePath
            repository = ""  # User will need to fill this in
            isSubmodule = Test-Path "$projectPath\.git"
            dependencies = @()
            metrics = @{
                users = 0
                uptime = 0.0
                lastDeployment = $null
                deploymentFrequency = "unknown"
            }
            styleEnforcement = @{
                codeStyle = ".metaHub/conventions/CODING_STYLE.md"
                writingStyle = ".metaHub/conventions/WRITING_STYLE.md"
                aiConventions = ".metaHub/conventions/AI_AGENT_RULES.md"
            }
        }

        $discoveredProjects += $project
        $projectCounter++

        if ($Verbose) {
            Write-Host "  [+] Found: $projectName" -ForegroundColor Green
            Write-Host "      Path: $relativePath"
            Write-Host "      Tech: $($techStack -join ', ')"
            Write-Host "      Docker: $isContainerized | CI/CD: $hasCICD"
            Write-Host ""
        } else {
            Write-Host "  [+] $projectName" -ForegroundColor Green
        }
    }

    Write-Host ""
}

Write-Host "[OK] Discovery complete!" -ForegroundColor Green
Write-Host "[>>] Found $($discoveredProjects.Count) projects across $($orgsToScan.Count) organizations" -ForegroundColor Cyan
Write-Host ""

if (-not $DryRun) {
    # Update registry
    $registry.projects = $discoveredProjects
    $registry.metadata.totalProjects = $discoveredProjects.Count
    $registry.metadata.containerizationRate = ($discoveredProjects | Where-Object { $_.containerized } | Measure-Object).Count / $discoveredProjects.Count
    $registry.metadata.cicdRate = ($discoveredProjects | Where-Object { $_.cicd } | Measure-Object).Count / $discoveredProjects.Count
    $registry.lastUpdated = (Get-Date -Format "yyyy-MM-ddTHH:mm:ssZ")

    # Save registry
    $registry | ConvertTo-Json -Depth 10 | Set-Content $RegistryFile

    Write-Host "[OK] Registry updated: $RegistryFile" -ForegroundColor Green
    Write-Host ""
    Write-Host "[>>] Summary:" -ForegroundColor Yellow
    Write-Host "  - Total Projects: $($discoveredProjects.Count)"
    Write-Host "  - Containerized: $(($discoveredProjects | Where-Object { $_.containerized } | Measure-Object).Count) ($([math]::Round($registry.metadata.containerizationRate * 100, 1))%)"
    Write-Host "  - With CI/CD: $(($discoveredProjects | Where-Object { $_.cicd } | Measure-Object).Count) ($([math]::Round($registry.metadata.cicdRate * 100, 1))%)"
    Write-Host ""
    Write-Host "[>>] Next steps:" -ForegroundColor Cyan
    Write-Host "  1. Review $RegistryFile"
    Write-Host "  2. Update 'status' fields (production/staging/etc.)"
    Write-Host "  3. Update 'priority' fields (critical/high/medium/low)"
    Write-Host "  4. Fill in 'repository' URLs"
    Write-Host "  5. Update 'metrics' with actual values"
} else {
    Write-Host "[>>] Dry run - no changes made" -ForegroundColor Yellow
    Write-Host "[>>] Run without -DryRun to update registry" -ForegroundColor Yellow
}
