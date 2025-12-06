/**
 * ORCHEX Governance Integration
 * Enforces governance policies during task orchestration
 */

import * as fs from 'node:fs';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';

// ESM dirname equivalent
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// ============================================================================
// Types
// ============================================================================

export interface GovernanceCheck {
  passed: boolean;
  violations: GovernanceViolation[];
  warnings: string[];
  timestamp: string;
}

export interface GovernanceViolation {
  type: 'forbidden_file' | 'forbidden_dir' | 'missing_readme' | 'policy_violation';
  path: string;
  rule: string;
  severity: 'error' | 'warning';
  suggestion?: string;
}

export interface GovernancePolicy {
  version: string;
  enforcement: {
    level: 'warning' | 'error';
    actions: Record<string, string>;
  };
  forbidden: {
    file_patterns: string[];
    directory_patterns: string[];
    specific_items: string[];
  };
}

// ============================================================================
// Policy Loader
// ============================================================================

const POLICY_PATH = path.resolve(__dirname, '../../../.metaHub/policies/root-structure.yaml');
const EVIDENCE_PATH = path.resolve(__dirname, '../../../.ORCHEX/evidence');

let cachedPolicy: GovernancePolicy | null = null;

/**
 * Load governance policy from YAML
 */
export function loadPolicy(): GovernancePolicy | null {
  if (cachedPolicy) return cachedPolicy;

  try {
    // Simple YAML parsing for the policy (key structures only)
    if (!fs.existsSync(POLICY_PATH)) {
      console.warn('Governance policy not found at', POLICY_PATH);
      return null;
    }

    const content = fs.readFileSync(POLICY_PATH, 'utf-8');

    // Extract key patterns using regex (avoiding yaml dependency)
    const forbidden: GovernancePolicy['forbidden'] = {
      file_patterns: extractListItems(content, 'file_patterns'),
      directory_patterns: extractListItems(content, 'directory_patterns'),
      specific_items: extractListItems(content, 'specific_items'),
    };

    const versionMatch = content.match(/version:\s*["']?([^"'\n]+)/);
    const levelMatch = content.match(/level:\s*["']?([^"'\n]+)/);

    cachedPolicy = {
      version: versionMatch?.[1] || '1.0',
      enforcement: {
        level: (levelMatch?.[1] as 'warning' | 'error') || 'warning',
        actions: {},
      },
      forbidden,
    };

    return cachedPolicy;
  } catch (error) {
    console.error('Failed to load governance policy:', error);
    return null;
  }
}

/**
 * Extract list items from YAML-like content
 */
function extractListItems(content: string, sectionName: string): string[] {
  const sectionRegex = new RegExp(`${sectionName}:\\s*\\n([\\s\\S]*?)(?=\\n\\s*\\w+:|$)`);
  const match = content.match(sectionRegex);

  if (!match) return [];

  const items: string[] = [];
  const lines = match[1].split('\n');

  for (const line of lines) {
    const itemMatch = line.match(/^\s*-\s*["']?([^"'#\n]+)/);
    if (itemMatch) {
      items.push(itemMatch[1].trim());
    }
  }

  return items;
}

// ============================================================================
// Governance Checks
// ============================================================================

/**
 * Check if a file path violates governance policies
 */
export function checkFilePath(filePath: string): GovernanceViolation[] {
  const policy = loadPolicy();
  if (!policy) return [];

  const violations: GovernanceViolation[] = [];
  const fileName = path.basename(filePath);
  const isRoot = path.dirname(filePath) === '.' || path.dirname(filePath) === '';

  // Only check root-level files for forbidden patterns
  if (isRoot) {
    // Check file patterns
    for (const pattern of policy.forbidden.file_patterns) {
      if (matchPattern(fileName, pattern)) {
        violations.push({
          type: 'forbidden_file',
          path: filePath,
          rule: pattern,
          severity: policy.enforcement.level,
          suggestion: `Move ${fileName} to appropriate subdirectory or add to .gitignore`,
        });
      }
    }

    // Check specific items
    if (policy.forbidden.specific_items.includes(fileName)) {
      violations.push({
        type: 'forbidden_file',
        path: filePath,
        rule: fileName,
        severity: 'error',
        suggestion: `Remove ${fileName} - it's forbidden at root level`,
      });
    }
  }

  return violations;
}

/**
 * Check if a directory path violates governance policies
 */
export function checkDirectoryPath(dirPath: string): GovernanceViolation[] {
  const policy = loadPolicy();
  if (!policy) return [];

  const violations: GovernanceViolation[] = [];
  const dirName = path.basename(dirPath);
  const isRoot = path.dirname(dirPath) === '.' || path.dirname(dirPath) === '';

  if (isRoot) {
    // Check directory patterns
    for (const pattern of policy.forbidden.directory_patterns) {
      if (matchPattern(dirName, pattern)) {
        violations.push({
          type: 'forbidden_dir',
          path: dirPath,
          rule: pattern,
          severity: policy.enforcement.level,
          suggestion: `Remove or rename directory ${dirName}`,
        });
      }
    }

    // Check specific items
    if (policy.forbidden.specific_items.includes(dirName)) {
      violations.push({
        type: 'forbidden_dir',
        path: dirPath,
        rule: dirName,
        severity: 'error',
        suggestion: `Directory ${dirName} is forbidden - migrate to appropriate location`,
      });
    }
  }

  return violations;
}

/**
 * Simple glob pattern matching
 */
function matchPattern(name: string, pattern: string): boolean {
  // Convert glob to regex
  const regex = pattern.replace(/\./g, '\\.').replace(/\*/g, '.*').replace(/\?/g, '.');

  return new RegExp(`^${regex}$`).test(name);
}

// ============================================================================
// Pre-Task Governance Check
// ============================================================================

/**
 * Run governance check before task execution
 */
export function preTaskCheck(taskDescription: string, affectedPaths: string[]): GovernanceCheck {
  const violations: GovernanceViolation[] = [];
  const warnings: string[] = [];

  for (const filePath of affectedPaths) {
    // Check if it's a directory or file
    const isDir = filePath.endsWith('/') || filePath.endsWith('\\');

    if (isDir) {
      violations.push(...checkDirectoryPath(filePath));
    } else {
      violations.push(...checkFilePath(filePath));
    }
  }

  // Add warnings for potential issues
  if (affectedPaths.some((p) => p.includes('temp'))) {
    warnings.push('Task involves temporary files - ensure cleanup after completion');
  }

  if (affectedPaths.some((p) => p.match(/\.(env|key|pem)$/))) {
    warnings.push('Task may involve sensitive files - verify security policies');
  }

  const check: GovernanceCheck = {
    passed: violations.filter((v) => v.severity === 'error').length === 0,
    violations,
    warnings,
    timestamp: new Date().toISOString(),
  };

  // Log to evidence
  logEvidence('pre-task-check', { taskDescription, check });

  return check;
}

/**
 * Run governance check after task completion
 */
export function postTaskCheck(taskDescription: string, modifiedPaths: string[]): GovernanceCheck {
  const violations: GovernanceViolation[] = [];
  const warnings: string[] = [];

  // Check all modified paths
  for (const filePath of modifiedPaths) {
    const isDir = filePath.endsWith('/') || filePath.endsWith('\\');

    if (isDir) {
      violations.push(...checkDirectoryPath(filePath));

      // Check for README in new directories
      const readmePath = path.join(filePath, 'README.md');
      if (!fs.existsSync(readmePath)) {
        violations.push({
          type: 'missing_readme',
          path: filePath,
          rule: 'All tool directories should have README.md',
          severity: 'warning',
          suggestion: `Add README.md to ${filePath}`,
        });
      }
    } else {
      violations.push(...checkFilePath(filePath));
    }
  }

  const check: GovernanceCheck = {
    passed: violations.filter((v) => v.severity === 'error').length === 0,
    violations,
    warnings,
    timestamp: new Date().toISOString(),
  };

  // Log to evidence
  logEvidence('post-task-check', { taskDescription, check });

  return check;
}

// ============================================================================
// Evidence Logging
// ============================================================================

/**
 * Log governance evidence for audit trail
 */
function logEvidence(eventType: string, data: Record<string, unknown>): void {
  try {
    if (!fs.existsSync(EVIDENCE_PATH)) {
      fs.mkdirSync(EVIDENCE_PATH, { recursive: true });
    }

    const evidenceFile = path.join(
      EVIDENCE_PATH,
      `governance-${new Date().toISOString().split('T')[0]}.jsonl`
    );

    const entry = {
      timestamp: new Date().toISOString(),
      eventType,
      ...data,
    };

    fs.appendFileSync(evidenceFile, JSON.stringify(entry) + '\n');
  } catch (error) {
    console.warn('Failed to log governance evidence:', error);
  }
}

// ============================================================================
// Exports
// ============================================================================

export const governance = {
  loadPolicy,
  checkFilePath,
  checkDirectoryPath,
  preTaskCheck,
  postTaskCheck,
};

export default governance;
