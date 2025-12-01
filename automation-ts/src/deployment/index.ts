import * as path from 'path';
import * as fs from 'fs';
import { getAutomationPath, readYamlFile } from '../utils/file';
import type { DeploymentProject, DeploymentTemplate } from '../types';

// Deployment registry data
const deploymentRegistry: DeploymentProject[] = [
  // Science projects
  { name: 'QMatSim', path: 'organizations/alaweimm90-science/QMatSim', type: 'scientific', organization: 'alaweimm90-science', technologies: ['python', 'numpy', 'quantum'], status: 'active' },
  { name: 'QubeML', path: 'organizations/alaweimm90-science/QubeML', type: 'scientific', organization: 'alaweimm90-science', technologies: ['python', 'pytorch', 'ml'], status: 'active' },
  { name: 'SpinCirc', path: 'organizations/alaweimm90-science/SpinCirc', type: 'scientific', organization: 'alaweimm90-science', technologies: ['python', 'cirq'], status: 'active' },
  { name: 'MagLogic', path: 'organizations/alaweimm90-science/MagLogic', type: 'scientific', organization: 'alaweimm90-science', technologies: ['python', 'sympy'], status: 'active' },
  { name: 'SciComp', path: 'organizations/alaweimm90-science/SciComp', type: 'scientific', organization: 'alaweimm90-science', technologies: ['python', 'scipy'], status: 'active' },
  { name: 'TalAI', path: 'organizations/alaweimm90-science/TalAI', type: 'ai-service', organization: 'alaweimm90-science', technologies: ['python', 'transformers'], status: 'active' },
  { name: 'SimCore', path: 'organizations/AlaweinOS/SimCore', type: 'scientific', organization: 'AlaweinOS', technologies: ['python', 'simulation'], status: 'active' },

  // Business projects
  { name: 'Repz', path: 'organizations/alaweimm90-business/Repz', type: 'web-app', organization: 'alaweimm90-business', technologies: ['typescript', 'react', 'node'], status: 'active' },

  // Platform projects
  { name: 'MEZAN', path: 'organizations/AlaweinOS/MEZAN', type: 'platform', organization: 'AlaweinOS', technologies: ['typescript', 'python'], status: 'active' },
  { name: 'Attributa', path: 'organizations/AlaweinOS/Attributa', type: 'web-app', organization: 'AlaweinOS', technologies: ['typescript', 'react'], status: 'active' },
  { name: 'Optilibria', path: 'organizations/AlaweinOS/Optilibria', type: 'library', organization: 'AlaweinOS', technologies: ['python', 'optimization'], status: 'active' },
  { name: 'LLMWorks', path: 'organizations/AlaweinOS/LLMWorks', type: 'ai-service', organization: 'AlaweinOS', technologies: ['python', 'llm'], status: 'active' },
  { name: 'QMLab', path: 'organizations/AlaweinOS/QMLab', type: 'scientific', organization: 'AlaweinOS', technologies: ['python', 'quantum'], status: 'active' },

  // Infrastructure
  { name: 'automation', path: 'automation', type: 'infrastructure', organization: 'root', technologies: ['python', 'yaml'], status: 'active' },
  { name: 'automation-ts', path: 'automation-ts', type: 'infrastructure', organization: 'root', technologies: ['typescript', 'node'], status: 'active' },
];

// Deployment templates
const deploymentTemplates: DeploymentTemplate[] = [
  { name: 'docker-compose', description: 'Docker Compose deployment', platform: 'docker', files: ['docker-compose.yml', 'Dockerfile', '.dockerignore'] },
  { name: 'kubernetes', description: 'Kubernetes deployment', platform: 'k8s', files: ['deployment.yaml', 'service.yaml', 'ingress.yaml'] },
  { name: 'netlify', description: 'Netlify static deployment', platform: 'netlify', files: ['netlify.toml', '_redirects'] },
  { name: 'vercel', description: 'Vercel deployment', platform: 'vercel', files: ['vercel.json'] },
  { name: 'github-actions', description: 'GitHub Actions CI/CD', platform: 'github', files: ['.github/workflows/ci.yml', '.github/workflows/deploy.yml'] },
  { name: 'python-package', description: 'Python package setup', platform: 'pypi', files: ['pyproject.toml', 'setup.py', 'MANIFEST.in'] },
  { name: 'npm-package', description: 'NPM package setup', platform: 'npm', files: ['package.json', '.npmignore', '.npmrc'] },
  { name: 'terraform', description: 'Terraform infrastructure', platform: 'terraform', files: ['main.tf', 'variables.tf', 'outputs.tf'] },
];

/**
 * List all deployment projects
 */
export function listProjects(): DeploymentProject[] {
  return deploymentRegistry;
}

/**
 * Get projects by organization
 */
export function getProjectsByOrganization(org: string): DeploymentProject[] {
  return deploymentRegistry.filter(p => p.organization === org);
}

/**
 * Get projects by type
 */
export function getProjectsByType(type: string): DeploymentProject[] {
  return deploymentRegistry.filter(p => p.type === type);
}

/**
 * Get project statistics
 */
export function getProjectStats(): {
  total: number;
  byType: Record<string, number>;
  byOrganization: Record<string, number>;
  byTechnology: Record<string, number>;
} {
  const byType: Record<string, number> = {};
  const byOrganization: Record<string, number> = {};
  const byTechnology: Record<string, number> = {};

  for (const project of deploymentRegistry) {
    // Count by type
    byType[project.type] = (byType[project.type] || 0) + 1;

    // Count by organization
    byOrganization[project.organization] = (byOrganization[project.organization] || 0) + 1;

    // Count by technology
    for (const tech of project.technologies) {
      byTechnology[tech] = (byTechnology[tech] || 0) + 1;
    }
  }

  return {
    total: deploymentRegistry.length,
    byType,
    byOrganization,
    byTechnology
  };
}

/**
 * List deployment templates
 */
export function listTemplates(): DeploymentTemplate[] {
  return deploymentTemplates;
}

/**
 * Get template by name
 */
export function getTemplate(name: string): DeploymentTemplate | null {
  return deploymentTemplates.find(t => t.name === name) || null;
}

/**
 * Check if a project exists on disk
 */
export function projectExists(projectPath: string): boolean {
  const basePath = path.dirname(getAutomationPath());
  const fullPath = path.join(basePath, projectPath);
  return fs.existsSync(fullPath);
}

/**
 * Get deployment configuration for a project
 */
export function getDeploymentConfig(projectPath: string): Record<string, unknown> | null {
  const basePath = path.dirname(getAutomationPath());
  const fullPath = path.join(basePath, projectPath);

  // Check for various config files
  const configFiles = [
    'deploy.yaml',
    'deploy.yml',
    '.deploy.yaml',
    'deployment.yaml'
  ];

  for (const configFile of configFiles) {
    const configPath = path.join(fullPath, configFile);
    const config = readYamlFile<Record<string, unknown>>(configPath);
    if (config) {
      return config;
    }
  }

  return null;
}

/**
 * Generate deployment files from template
 */
export function generateDeploymentFiles(
  templateName: string,
  targetPath: string,
  _variables: Record<string, string> = {}
): { success: boolean; files: string[]; error?: string } {
  const template = getTemplate(templateName);

  if (!template) {
    return { success: false, files: [], error: `Template not found: ${templateName}` };
  }

  const generatedFiles: string[] = [];

  // In a real implementation, this would generate actual files
  // For now, we just return what would be generated
  for (const file of template.files) {
    generatedFiles.push(path.join(targetPath, file));
  }

  return { success: true, files: generatedFiles };
}
