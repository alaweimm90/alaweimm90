# Project Discovery & Inventory Script (Fixed - Excludes node_modules)
# Scans all organizations and populates projects-registry.json

param(
    [string]$RegistryFile = ".metaHub\projects-registry.json",
    [switch]$DryRun = $false,
    [switch]$Verbose = $false
)

Write-Host "[*] Multi-Organization Project Discovery (Fixed)" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""

# Load existing registry
$registryContent = Get-Content $RegistryFile -Raw | ConvertFrom-Json

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

# Function to check if path should be excluded
function Should-Exclude($path) {
    $excludePatterns = @("node_modules", "dist", "build", ".git", ".vite", "coverage", ".cache")
    foreach ($pattern in $excludePatterns) {
        if ($path -like "*\$pattern\*") {
            return $true
        }
    }
    return $false
}

foreach ($orgPath in $orgsToScan) {
    if (-not (Test-Path $orgPath)) {
        Write-Host "[!] Organization path not found: $orgPath" -ForegroundColor Yellow
        continue
    }

    $orgName = Split-Path $orgPath -Leaf
    Write-Host "[>>] Scanning organization: $orgName" -ForegroundColor Magenta

    # Find all directories at depth 1-2 (not deep scanning into node_modules)
    $projectCandidates = Get-ChildItem -Path $orgPath -Directory -Depth 2 -ErrorAction SilentlyContinue |
        Where-Object { -not (Should-Exclude $_.FullName) }

    foreach ($dir in $projectCandidates) {
        # Check if this looks like a project (has package.json, Dockerfile, or docker-compose.yml)
        $hasPackageJson = Test-Path "$($dir.FullName)\package.json"
        $hasDockerfile = Test-Path "$($dir.FullName)\Dockerfile"
        $hasDockerCompose = Test-Path "$($dir.FullName)\docker-compose.yml"
        $hasGoMod = Test-Path "$($dir.FullName)\go.mod"
        $hasRequirements = Test-Path "$($dir.FullName)\requirements.txt"
        $hasCargo = Test-Path "$($dir.FullName)\Cargo.toml"

        if (-not ($hasPackageJson -or $hasDockerfile -or $hasDockerCompose -or $hasGoMod -or $hasRequirements -or $hasCargo)) {
            continue
        }

        $projectName = $dir.Name
        $relativePath = Resolve-Path -Relative $dir.FullName

        # Detect tech stack
        $techStack = @()
        if ($hasPackageJson) {
            try {
                $packageJson = Get-Content "$($dir.FullName)\package.json" | ConvertFrom-Json
                if ($packageJson.dependencies) {
                    if ($packageJson.dependencies.react) { $techStack += "react" }
                    if ($packageJson.dependencies.vue) { $techStack += "vue" }
                    if ($packageJson.dependencies.next) { $techStack += "nextjs" }
                    if ($packageJson.dependencies.express) { $techStack += "express" }
                    if ($packageJson.dependencies.fastify) { $techStack += "fastify" }
                    if ($packageJson.dependencies."@nestjs/core") { $techStack += "nestjs" }
                    if ($packageJson.dependencies.typescript -or $packageJson.devDependencies.typescript) { $techStack += "typescript" }
                    if ($packageJson.dependencies.prisma) { $techStack += "prisma" }
                    if ($packageJson.dependencies.pg -or $packageJson.dependencies.postgres) { $techStack += "postgres" }
                    if ($packageJson.dependencies.mongodb -or $packageJson.dependencies.mongoose) { $techStack += "mongodb" }
                }
            } catch {
                Write-Host "  [!] Error reading package.json in $projectName" -ForegroundColor Yellow
            }
        }

        if ($hasRequirements -or (Test-Path "$($dir.FullName)\pyproject.toml")) {
            $techStack += "python"
        }
        if ($hasGoMod) { $techStack += "go" }
        if ($hasCargo) { $techStack += "rust" }

        # Check for containerization
        $isContainerized = $hasDockerfile -or $hasDockerCompose

        # Check for CI/CD
        $hasCICD = (Test-Path "$($dir.FullName)\.github\workflows") -or `
                   (Test-Path "$($dir.FullName)\.gitlab-ci.yml")

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

        # Get organization ID
        $orgId = ($registryContent.organizations.PSObject.Properties |
                  Where-Object { $_.Value.path -eq $orgPath.Replace("\", "/") } |
                  Select-Object -First 1).Value.id

        # Create project entry
        $project = @{
            id = "proj-{0:D3}" -f $projectCounter
            name = $projectName
            organizationId = $orgId
            status = "unknown"
            category = $category
            techStack = @($techStack)
            containerized = $isContainerized
            cicd = $hasCICD
            priority = "medium"
            healthScore = 5
            path = $relativePath.Replace("\", "/")
            repository = ""
            isSubmodule = $false
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
            Write-Host "  [+] $projectName ($category)" -ForegroundColor Green
        }
    }

    Write-Host ""
}

Write-Host "[OK] Discovery complete!" -ForegroundColor Green
Write-Host "[>>] Found $($discoveredProjects.Count) projects across $($orgsToScan.Count) organizations" -ForegroundColor Cyan
Write-Host ""

if (-not $DryRun) {
    # Update registry
    $registryContent.projects = $discoveredProjects
    $registryContent.metadata.totalProjects = $discoveredProjects.Count

    if ($discoveredProjects.Count -gt 0) {
        $registryContent.metadata.containerizationRate = [math]::Round(
            ($discoveredProjects | Where-Object { $_.containerized } | Measure-Object).Count / $discoveredProjects.Count, 2)
        $registryContent.metadata.cicdRate = [math]::Round(
            ($discoveredProjects | Where-Object { $_.cicd } | Measure-Object).Count / $discoveredProjects.Count, 2)
    }

    $registryContent.lastUpdated = (Get-Date -Format "yyyy-MM-ddTHH:mm:ssZ")

    # Save registry
    $json = $registryContent | ConvertTo-Json -Depth 10
    Set-Content -Path $RegistryFile -Value $json

    Write-Host "[OK] Registry updated: $RegistryFile" -ForegroundColor Green
    Write-Host ""
    Write-Host "[>>] Summary:" -ForegroundColor Yellow
    Write-Host "  - Total Projects: $($discoveredProjects.Count)"
    Write-Host "  - Containerized: $(($discoveredProjects | Where-Object { $_.containerized } | Measure-Object).Count) ($([math]::Round($registryContent.metadata.containerizationRate * 100, 1))%)"
    Write-Host "  - With CI/CD: $(($discoveredProjects | Where-Object { $_.cicd } | Measure-Object).Count) ($([math]::Round($registryContent.metadata.cicdRate * 100, 1))%)"
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
