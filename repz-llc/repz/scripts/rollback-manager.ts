#!/usr/bin/env node
/**
 * Enterprise Rollback Manager
 * Provides automated rollback capabilities for the refactoring process
 */

import { execSync } from 'child_process';
import * as fs from 'fs';
import * as path from 'path';

interface RollbackPoint {
  id: string;
  timestamp: string;
  phase: string;
  gitCommit: string;
  description: string;
  metrics: {
    componentCount: number;
    buildSize: number;
    testCoverage: number;
  };
}

class RollbackManager {
  private rollbacksDir = path.join(process.cwd(), '.rollbacks');
  private metricsFile = path.join(this.rollbacksDir, 'metrics.json');
  
  constructor() {
    this.ensureRollbacksDir();
  }
  
  private ensureRollbacksDir(): void {
    if (!fs.existsSync(this.rollbacksDir)) {
      fs.mkdirSync(this.rollbacksDir, { recursive: true });
    }
  }
  
  // Create a rollback point
  async createRollbackPoint(phase: string, description: string): Promise<string> {
    const timestamp = new Date().toISOString();
    const id = `rollback-${phase}-${Date.now()}`;
    const gitCommit = execSync('git rev-parse HEAD').toString().trim();
    
    // Create git tag
    execSync(`git tag -a ${id} -m "Rollback point: ${description}"`);
    
    // Collect metrics
    const metrics = await this.collectMetrics();
    
    const rollbackPoint: RollbackPoint = {
      id,
      timestamp,
      phase,
      gitCommit,
      description,
      metrics
    };
    
    // Save rollback point metadata
    const rollbackFile = path.join(this.rollbacksDir, `${id}.json`);
    fs.writeFileSync(rollbackFile, JSON.stringify(rollbackPoint, null, 2));
    
    console.log(`✅ Rollback point created: ${id}`);
    console.log(`📊 Metrics: ${metrics.componentCount} components, ${metrics.buildSize}KB build`);
    
    return id;
  }
  
  // Execute rollback to a specific point
  async rollback(rollbackId: string): Promise<void> {
    const rollbackFile = path.join(this.rollbacksDir, `${rollbackId}.json`);
    
    if (!fs.existsSync(rollbackFile)) {
      throw new Error(`Rollback point ${rollbackId} not found`);
    }
    
    const rollbackPoint: RollbackPoint = JSON.parse(
      fs.readFileSync(rollbackFile, 'utf8')
    );
    
    console.log(`🔄 Rolling back to: ${rollbackPoint.description}`);
    console.log(`📅 Created: ${rollbackPoint.timestamp}`);
    
    // Disable all feature flags first
    await this.disableAllFeatures();
    
    // Reset git to rollback point
    execSync(`git reset --hard ${rollbackId}`);
    
    // Reinstall dependencies if needed
    execSync('npm install');
    
    // Run post-rollback validation
    await this.validateRollback(rollbackPoint);
    
    console.log(`✅ Rollback to ${rollbackId} completed successfully`);
  }
  
  // List available rollback points
  listRollbackPoints(): RollbackPoint[] {
    const rollbackFiles = fs.readdirSync(this.rollbacksDir)
      .filter(file => file.endsWith('.json') && file.startsWith('rollback-'))
      .map(file => {
        const content = fs.readFileSync(path.join(this.rollbacksDir, file), 'utf8');
        return JSON.parse(content) as RollbackPoint;
      })
      .sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime());
    
    return rollbackFiles;
  }
  
  // Emergency rollback to baseline
  async emergencyRollback(): Promise<void> {
    console.log('🚨 EMERGENCY ROLLBACK INITIATED');
    
    // Disable all features immediately
    await this.disableAllFeatures();
    
    // Reset to baseline tag
    try {
      execSync('git reset --hard refactor-baseline-20250803-074223');
      console.log('✅ Reset to baseline successful');
    } catch (error) {
      console.error('❌ Failed to reset to baseline:', error);
      // Fallback to main branch
      execSync('git reset --hard origin/main');
      console.log('✅ Reset to main branch as fallback');
    }
    
    // Clean install
    execSync('npm ci');
    
    // Verify system health
    await this.runHealthCheck();
    
    console.log('✅ Emergency rollback completed');
  }
  
  private async collectMetrics(): Promise<RollbackPoint['metrics']> {
    // Count components
    const componentCount = this.countComponents();
    
    // Get build size (approximate)
    let buildSize = 0;
    try {
      execSync('npm run build:dev', { stdio: 'pipe' });
      const distDir = path.join(process.cwd(), 'dist');
      if (fs.existsSync(distDir)) {
        buildSize = this.calculateDirSize(distDir);
      }
    } catch (error) {
      console.warn('Could not calculate build size:', error);
    }
    
    // Get test coverage (if available)
    let testCoverage = 0;
    try {
      const coverageResult = execSync('npm run test:coverage -- --reporter=json', { stdio: 'pipe' });
      const coverage = JSON.parse(coverageResult.toString());
      testCoverage = coverage.total?.statements?.pct || 0;
    } catch (error) {
      console.warn('Could not calculate test coverage:', error);
    }
    
    return {
      componentCount,
      buildSize: Math.round(buildSize / 1024), // Convert to KB
      testCoverage
    };
  }
  
  private countComponents(): number {
    const srcDir = path.join(process.cwd(), 'src', 'components');
    return this.countTsxFiles(srcDir);
  }
  
  private countTsxFiles(dir: string): number {
    if (!fs.existsSync(dir)) return 0;
    
    let count = 0;
    const items = fs.readdirSync(dir);
    
    for (const item of items) {
      const fullPath = path.join(dir, item);
      const stat = fs.statSync(fullPath);
      
      if (stat.isDirectory()) {
        count += this.countTsxFiles(fullPath);
      } else if (item.endsWith('.tsx')) {
        count++;
      }
    }
    
    return count;
  }
  
  private calculateDirSize(dir: string): number {
    if (!fs.existsSync(dir)) return 0;
    
    let size = 0;
    const items = fs.readdirSync(dir);
    
    for (const item of items) {
      const fullPath = path.join(dir, item);
      const stat = fs.statSync(fullPath);
      
      if (stat.isDirectory()) {
        size += this.calculateDirSize(fullPath);
      } else {
        size += stat.size;
      }
    }
    
    return size;
  }
  
  private async disableAllFeatures(): Promise<void> {
    try {
      const { disableAllFeatures } = await import('../src/config/feature-flags');
      disableAllFeatures();
    } catch (error) {
      console.warn('Could not disable feature flags:', error);
    }
  }
  
  private async validateRollback(rollbackPoint: RollbackPoint): Promise<void> {
    console.log('🔍 Validating rollback...');
    
    try {
      // Type check
      execSync('npm run type-check', { stdio: 'pipe' });
      console.log('✅ TypeScript validation passed');
      
      // Lint check
      execSync('npm run lint', { stdio: 'pipe' });
      console.log('✅ Linting validation passed');
      
      // Build test
      execSync('npm run build:dev', { stdio: 'pipe' });
      console.log('✅ Build validation passed');
      
    } catch (error) {
      console.error('❌ Rollback validation failed:', error);
      throw new Error('Rollback validation failed');
    }
  }
  
  private async runHealthCheck(): Promise<void> {
    console.log('🏥 Running system health check...');
    
    const checks = [
      { name: 'TypeScript', command: 'npm run type-check' },
      { name: 'Linting', command: 'npm run lint' },
      { name: 'Build', command: 'npm run build:dev' },
      { name: 'Tests', command: 'npm run test:run' }
    ];
    
    for (const check of checks) {
      try {
        execSync(check.command, { stdio: 'pipe' });
        console.log(`✅ ${check.name} check passed`);
      } catch (error) {
        console.error(`❌ ${check.name} check failed:`, error);
      }
    }
  }
}

// CLI interface
if (import.meta.url === `file://${process.argv[1]}`) {
  const manager = new RollbackManager();
  const command = process.argv[2];
  
  switch (command) {
    case 'create':
      const phase = process.argv[3] || 'unknown';
      const description = process.argv[4] || 'Manual rollback point';
      manager.createRollbackPoint(phase, description);
      break;
      
    case 'rollback':
      const rollbackId = process.argv[3];
      if (!rollbackId) {
        console.error('Please provide rollback ID');
        process.exit(1);
      }
      manager.rollback(rollbackId);
      break;
      
    case 'list':
      const points = manager.listRollbackPoints();
      console.log('Available rollback points:');
      points.forEach(point => {
        console.log(`  ${point.id} - ${point.description} (${point.timestamp})`);
      });
      break;
      
    case 'emergency':
      manager.emergencyRollback();
      break;
      
    default:
      console.log('Usage:');
      console.log('  npm run rollback create <phase> <description>');
      console.log('  npm run rollback rollback <rollback-id>');
      console.log('  npm run rollback list');
      console.log('  npm run rollback emergency');
  }
}

export default RollbackManager;