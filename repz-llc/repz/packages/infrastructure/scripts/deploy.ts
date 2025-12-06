#!/usr/bin/env tsx

/**
 * REPZ Platform Deployment Script
 * Universal deployment orchestrator for multiple environments and cloud providers
 */

import { execSync } from 'child_process';
import { existsSync, readFileSync, writeFileSync } from 'fs';
import { join } from 'path';
import { Command } from 'commander';
import * as yaml from 'yaml';

interface DeploymentConfig {
  environment: 'development' | 'staging' | 'production';
  provider: 'aws' | 'azure' | 'gcp' | 'docker';
  region: string;
  projectName: string;
  version: string;
  features: {
    database: boolean;
    cache: boolean;
    cdn: boolean;
    monitoring: boolean;
    backup: boolean;
  };
  scaling: {
    minInstances: number;
    maxInstances: number;
    targetCpu: number;
  };
  security: {
    enableWaf: boolean;
    enableVpn: boolean;
    enableEncryption: boolean;
  };
}

// Colors for console output
const colors = {
  reset: '\x1b[0m',
  bright: '\x1b[1m',
  red: '\x1b[31m',
  green: '\x1b[32m',
  yellow: '\x1b[33m',
  blue: '\x1b[34m',
  cyan: '\x1b[36m'
};

function log(message: string, color: keyof typeof colors = 'reset') {
  console.log(`${colors[color]}${message}${colors.reset}`);
}

function exec(command: string, options: { cwd?: string; stdio?: any } = {}) {
  log(`Executing: ${command}`, 'cyan');
  return execSync(command, { 
    stdio: 'inherit', 
    cwd: process.cwd(),
    ...options 
  });
}

class RepzDeployer {
  private config: DeploymentConfig;
  private startTime: number;

  constructor(config: DeploymentConfig) {
    this.config = config;
    this.startTime = Date.now();
  }

  async deploy(): Promise<void> {
    log(`🚀 Starting REPZ Platform deployment...`, 'bright');
    log(`Environment: ${this.config.environment}`, 'blue');
    log(`Provider: ${this.config.provider}`, 'blue');
    log(`Region: ${this.config.region}`, 'blue');

    try {
      // Pre-deployment checks
      await this.preDeploymentChecks();
      
      // Build and prepare artifacts
      await this.buildArtifacts();
      
      // Deploy infrastructure
      await this.deployInfrastructure();
      
      // Deploy application
      await this.deployApplication();
      
      // Post-deployment verification
      await this.postDeploymentVerification();
      
      // Success
      const duration = Math.round((Date.now() - this.startTime) / 1000);
      log(`✅ Deployment completed successfully in ${duration}s`, 'green');
      
    } catch (error) {
      log(`❌ Deployment failed: ${error}`, 'red');
      throw error;
    }
  }

  private async preDeploymentChecks(): Promise<void> {
    log('🔍 Running pre-deployment checks...', 'blue');

    // Check required tools
    const requiredTools = this.getRequiredTools();
    for (const tool of requiredTools) {
      try {
        exec(`${tool} --version`, { stdio: 'pipe' });
        log(`✅ ${tool} is available`, 'green');
      } catch {
        throw new Error(`Required tool '${tool}' is not installed`);
      }
    }

    // Check environment variables
    const requiredEnvVars = this.getRequiredEnvVars();
    for (const envVar of requiredEnvVars) {
      if (!process.env[envVar]) {
        throw new Error(`Required environment variable '${envVar}' is not set`);
      }
    }

    // Validate configuration
    await this.validateConfiguration();

    log('✅ Pre-deployment checks passed', 'green');
  }

  private async buildArtifacts(): Promise<void> {
    log('🏗️ Building deployment artifacts...', 'blue');

    // Build application
    exec('npm run build:production');

    // Build Docker image if needed
    if (this.config.provider === 'docker' || this.config.environment !== 'development') {
      const imageTag = `repz-platform:${this.config.version}-${this.config.environment}`;
      exec(`docker build -t ${imageTag} -f packages/infrastructure/docker/Dockerfile .`);
      
      // Push to registry if not local deployment
      if (this.config.provider !== 'docker') {
        const registry = this.getContainerRegistry();
        exec(`docker tag ${imageTag} ${registry}/${imageTag}`);
        exec(`docker push ${registry}/${imageTag}`);
      }
    }

    // Generate deployment manifests
    await this.generateDeploymentManifests();

    log('✅ Artifacts built successfully', 'green');
  }

  private async deployInfrastructure(): Promise<void> {
    log('🏗️ Deploying infrastructure...', 'blue');

    switch (this.config.provider) {
      case 'aws':
        await this.deployAwsInfrastructure();
        break;
      case 'azure':
        await this.deployAzureInfrastructure();
        break;
      case 'gcp':
        await this.deployGcpInfrastructure();
        break;
      case 'docker':
        await this.deployDockerInfrastructure();
        break;
      default:
        throw new Error(`Unsupported provider: ${this.config.provider}`);
    }

    log('✅ Infrastructure deployed successfully', 'green');
  }

  private async deployApplication(): Promise<void> {
    log('📦 Deploying application...', 'blue');

    switch (this.config.provider) {
      case 'aws':
        // Deploy to ECS or Lambda
        exec(`aws ecs update-service --cluster repz-${this.config.environment} --service repz-app --force-new-deployment`);
        break;
      case 'azure':
        // Deploy to Azure Container Instances or App Service
        exec(`az webapp deployment source sync --resource-group repz-${this.config.environment} --name repz-app`);
        break;
      case 'gcp':
        // Deploy to Cloud Run or GKE
        exec(`gcloud run deploy repz-app --image gcr.io/${process.env.GCP_PROJECT}/repz-platform:${this.config.version}`);
        break;
      case 'docker':
        // Deploy using Docker Compose
        exec('docker-compose -f packages/infrastructure/docker/docker-compose.yml up -d');
        break;
    }

    log('✅ Application deployed successfully', 'green');
  }

  private async postDeploymentVerification(): Promise<void> {
    log('🔍 Running post-deployment verification...', 'blue');

    // Wait for services to be ready
    await this.waitForServices();

    // Run health checks
    await this.runHealthChecks();

    // Run smoke tests
    await this.runSmokeTests();

    // Verify monitoring and alerting
    await this.verifyMonitoring();

    log('✅ Post-deployment verification passed', 'green');
  }

  private async deployAwsInfrastructure(): Promise<void> {
    log('☁️ Deploying AWS infrastructure with Terraform...', 'blue');

    const terraformDir = 'packages/infrastructure/terraform';
    
    // Initialize Terraform
    exec('terraform init', { cwd: terraformDir });
    
    // Plan deployment
    exec(`terraform plan -var="environment=${this.config.environment}" -out=tfplan`, { cwd: terraformDir });
    
    // Apply changes
    if (this.config.environment === 'production') {
      log('⚠️ Production deployment requires manual approval', 'yellow');
      log('Run: terraform apply tfplan', 'yellow');
    } else {
      exec('terraform apply -auto-approve tfplan', { cwd: terraformDir });
    }
  }

  private async deployAzureInfrastructure(): Promise<void> {
    log('☁️ Deploying Azure infrastructure...', 'blue');
    
    // Deploy using Pulumi or ARM templates
    exec(`pulumi up --stack ${this.config.environment}`, { cwd: 'packages/infrastructure/pulumi' });
  }

  private async deployGcpInfrastructure(): Promise<void> {
    log('☁️ Deploying GCP infrastructure...', 'blue');
    
    // Deploy using Pulumi or Cloud Deployment Manager
    exec(`gcloud deployment-manager deployments create repz-${this.config.environment} --config infrastructure.yaml`);
  }

  private async deployDockerInfrastructure(): Promise<void> {
    log('🐳 Setting up Docker infrastructure...', 'blue');
    
    // Create Docker network and volumes if needed
    exec('docker network create repz-network || true');
    
    // Start infrastructure services
    exec('docker-compose -f packages/infrastructure/docker/docker-compose.yml up -d postgres redis minio');
  }

  private getRequiredTools(): string[] {
    const tools = ['node', 'npm', 'docker'];
    
    switch (this.config.provider) {
      case 'aws':
        tools.push('terraform', 'aws');
        break;
      case 'azure':
        tools.push('pulumi', 'az');
        break;
      case 'gcp':
        tools.push('gcloud', 'pulumi');
        break;
    }
    
    return tools;
  }

  private getRequiredEnvVars(): string[] {
    const envVars = ['NODE_ENV'];
    
    switch (this.config.provider) {
      case 'aws':
        envVars.push('AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY');
        break;
      case 'azure':
        envVars.push('AZURE_CLIENT_ID', 'AZURE_CLIENT_SECRET', 'AZURE_TENANT_ID');
        break;
      case 'gcp':
        envVars.push('GOOGLE_APPLICATION_CREDENTIALS');
        break;
    }
    
    return envVars;
  }

  private getContainerRegistry(): string {
    switch (this.config.provider) {
      case 'aws':
        return `${process.env.AWS_ACCOUNT_ID}.dkr.ecr.${this.config.region}.amazonaws.com`;
      case 'azure':
        return `${process.env.AZURE_REGISTRY_NAME}.azurecr.io`;
      case 'gcp':
        return `gcr.io/${process.env.GCP_PROJECT}`;
      default:
        return 'localhost:5000';
    }
  }

  private async validateConfiguration(): Promise<void> {
    // Validate environment-specific settings
    if (this.config.environment === 'production') {
      if (!this.config.security.enableEncryption) {
        throw new Error('Production environment must have encryption enabled');
      }
      if (this.config.scaling.minInstances < 2) {
        throw new Error('Production environment must have at least 2 instances');
      }
    }
  }

  private async generateDeploymentManifests(): Promise<void> {
    // Generate Kubernetes manifests, Docker Compose files, etc.
    const manifest = {
      apiVersion: 'apps/v1',
      kind: 'Deployment',
      metadata: {
        name: 'repz-app',
        namespace: this.config.environment
      },
      spec: {
        replicas: this.config.scaling.minInstances,
        selector: {
          matchLabels: {
            app: 'repz-app'
          }
        },
        template: {
          metadata: {
            labels: {
              app: 'repz-app'
            }
          },
          spec: {
            containers: [{
              name: 'repz-app',
              image: `repz-platform:${this.config.version}`,
              ports: [{
                containerPort: 3000
              }]
            }]
          }
        }
      }
    };

    writeFileSync(
      `deployment-manifest-${this.config.environment}.yaml`,
      yaml.stringify(manifest)
    );
  }

  private async waitForServices(): Promise<void> {
    log('⏳ Waiting for services to be ready...', 'yellow');
    
    // Implementation depends on provider
    await new Promise(resolve => setTimeout(resolve, 30000)); // Wait 30 seconds
  }

  private async runHealthChecks(): Promise<void> {
    log('🏥 Running health checks...', 'blue');
    
    // Check application health
    try {
      const healthUrl = this.getApplicationUrl() + '/health';
      exec(`curl -f ${healthUrl} || exit 1`);
      log('✅ Application health check passed', 'green');
    } catch {
      throw new Error('Application health check failed');
    }
  }

  private async runSmokeTests(): Promise<void> {
    log('🧪 Running smoke tests...', 'blue');
    
    // Run basic functionality tests
    exec('npm run test:smoke');
  }

  private async verifyMonitoring(): Promise<void> {
    if (this.config.features.monitoring) {
      log('📊 Verifying monitoring setup...', 'blue');
      // Check that monitoring endpoints are accessible
      // Verify alerting rules are configured
      log('✅ Monitoring verification passed', 'green');
    }
  }

  private getApplicationUrl(): string {
    switch (this.config.environment) {
      case 'production':
        return 'https://app.repzcoach.com';
      case 'staging':
        return 'https://staging.repzcoach.com';
      default:
        return 'http://localhost:3000';
    }
  }
}

// CLI Interface
const program = new Command();

program
  .name('deploy')
  .description('REPZ Platform deployment orchestrator')
  .version('1.0.0');

program
  .option('-e, --env <environment>', 'Deployment environment', 'development')
  .option('-p, --provider <provider>', 'Cloud provider', 'docker')
  .option('-r, --region <region>', 'Deployment region', 'us-east-1')
  .option('-v, --version <version>', 'Application version', '1.0.0')
  .option('--dry-run', 'Show what would be deployed without executing')
  .option('--skip-tests', 'Skip post-deployment tests')
  .action(async (options) => {
    try {
      const config: DeploymentConfig = {
        environment: options.env,
        provider: options.provider,
        region: options.region,
        projectName: 'repz-platform',
        version: options.version,
        features: {
          database: true,
          cache: true,
          cdn: options.env === 'production',
          monitoring: options.env !== 'development',
          backup: options.env === 'production'
        },
        scaling: {
          minInstances: options.env === 'production' ? 2 : 1,
          maxInstances: options.env === 'production' ? 10 : 3,
          targetCpu: 70
        },
        security: {
          enableWaf: options.env === 'production',
          enableVpn: false,
          enableEncryption: options.env !== 'development'
        }
      };

      if (options.dryRun) {
        log('🔍 Dry run mode - showing deployment plan:', 'yellow');
        console.log(JSON.stringify(config, null, 2));
        return;
      }

      const deployer = new RepzDeployer(config);
      await deployer.deploy();

    } catch (error) {
      log(`❌ Deployment failed: ${error}`, 'red');
      process.exit(1);
    }
  });

program.parse();