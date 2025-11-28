import path from 'node:path';
import { resolveTargetDir, parsePlaceholders, parseFlag, parseOption } from './config.js';
import {
  findManifests,
  readJson,
  copyTemplateFiles,
  writeTemplateMeta,
  validateTemplate,
  validateContent,
  type TemplateManifest,
} from './fs.js';
import { install, type InstallType } from './install.js';

const TEMPLATES_DIR = path.join(process.cwd(), 'templates', 'devops');

interface DiscoveredTemplate {
  manifest: TemplateManifest;
  manifestPath: string;
  templateDir: string;
}

/**
 * Discover all templates from manifests
 */
function discoverTemplates(): DiscoveredTemplate[] {
  const manifestPaths = findManifests(TEMPLATES_DIR);
  const templates: DiscoveredTemplate[] = [];

  for (const manifestPath of manifestPaths) {
    const manifest = readJson<TemplateManifest>(manifestPath);
    if (manifest) {
      templates.push({
        manifest,
        manifestPath,
        templateDir: path.dirname(manifestPath),
      });
    }
  }

  return templates;
}

/**
 * Filter templates by search pattern
 */
function filterTemplates(
  templates: DiscoveredTemplate[],
  search?: string,
  category?: string
): DiscoveredTemplate[] {
  return templates.filter((t) => {
    if (category && t.manifest.category !== category) {
      return false;
    }
    if (search && search !== '*') {
      const pattern = search.toLowerCase();
      const matches =
        t.manifest.name.toLowerCase().includes(pattern) ||
        t.manifest.description.toLowerCase().includes(pattern) ||
        t.manifest.tags.some((tag) => tag.toLowerCase().includes(pattern));
      if (!matches) return false;
    }
    return true;
  });
}

/**
 * List templates
 */
function listTemplates(templates: DiscoveredTemplate[]): void {
  console.log('\nAvailable Templates:\n');
  for (const t of templates) {
    console.log(`  ${t.manifest.name} (${t.manifest.category})`);
    console.log(`    ${t.manifest.description}`);
    console.log(`    Tags: ${t.manifest.tags.join(', ')}`);
    console.log(`    Version: ${t.manifest.version}\n`);
  }
  console.log(`Total: ${templates.length} templates`);
}

/**
 * Copy a template to target directory
 */
function copyTemplate(
  template: DiscoveredTemplate,
  targetDir: string,
  vars: Record<string, string>
): boolean {
  console.log(`\nCopying template: ${template.manifest.name}`);
  console.log(`  Target: ${targetDir}`);

  // Validate first
  const validation = validateTemplate(template.manifest, template.templateDir);
  if (!validation.valid) {
    console.error('Validation errors:');
    for (const error of validation.errors) {
      console.error(`  - ${error}`);
    }
    return false;
  }

  // Copy files
  copyTemplateFiles(template.manifest, template.templateDir, targetDir, vars);

  // Write metadata
  writeTemplateMeta(targetDir, template.manifest);

  console.log('Copy complete!');
  return true;
}

/**
 * Validate all templates
 */
function validateAllTemplates(templates: DiscoveredTemplate[]): void {
  console.log('\nValidating Templates:\n');
  let passed = 0;
  let failed = 0;

  for (const t of templates) {
    const validation = validateTemplate(t.manifest, t.templateDir);
    if (validation.valid) {
      console.log(`  ✓ ${t.manifest.name}`);
      passed++;
    } else {
      console.log(`  ✗ ${t.manifest.name}`);
      for (const error of validation.errors) {
        console.log(`      - ${error}`);
      }
      failed++;
    }

    // Also validate content of required files
    for (const file of t.manifest.requiredFiles) {
      const filePath = path.join(t.templateDir, file);
      const contentValidation = validateContent(filePath);
      if (!contentValidation.valid) {
        console.log(`      Content: ${file}`);
        for (const error of contentValidation.errors) {
          console.log(`        - ${error}`);
        }
      }
    }
  }

  console.log(`\nResults: ${passed} passed, ${failed} failed`);
}

/**
 * Main CLI entry point
 */
function main(): void {
  const args = process.argv.slice(2);

  // Parse arguments
  const targetDir = resolveTargetDir(args);
  const search = parseOption(args, 'search');
  const category = parseOption(args, 'category');
  const copyName = parseOption(args, 'copy');
  const apply = parseFlag(args, 'apply');
  const validate = parseFlag(args, 'validate');
  const installType = parseOption(args, 'install') as InstallType | undefined;
  const vars = parsePlaceholders(args);

  // Discover templates
  const allTemplates = discoverTemplates();
  const templates = filterTemplates(allTemplates, search, category);

  // Handle validate mode
  if (validate) {
    validateAllTemplates(templates);
    return;
  }

  // Handle copy mode
  if (copyName) {
    const template = allTemplates.find((t) => t.manifest.name === copyName);
    if (!template) {
      console.error(`Template not found: ${copyName}`);
      console.error('Use --search=* to list available templates');
      process.exit(1);
    }

    if (!apply) {
      console.log('\nDry run - would copy:');
      console.log(`  Template: ${template.manifest.name}`);
      console.log(`  Target: ${targetDir}`);
      console.log(`  Files: ${template.manifest.requiredFiles.join(', ')}`);
      console.log('\nUse --apply=true to actually copy');
      return;
    }

    const success = copyTemplate(template, targetDir, vars);
    if (!success) {
      process.exit(1);
    }

    // Run install if requested
    if (installType) {
      console.log(`\nRunning ${installType} install...`);
      install(targetDir, installType);
    }

    return;
  }

  // Default: list templates
  listTemplates(templates);
}

main();
