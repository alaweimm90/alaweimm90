import { describe, it, expect } from 'vitest';

// Mock the coder planning functionality
interface PlannedFile {
  path: string;
  content: string;
  action: 'ADD' | 'UPDATE';
}

function planNodeService(vars: Record<string, string>): PlannedFile[] {
  const projectName = vars.PROJECT_NAME || 'my-service';
  const files: PlannedFile[] = [];

  // Service source
  files.push({
    path: 'service/src/index.ts',
    content: `// ${projectName} service`,
    action: 'ADD',
  });

  // Package.json
  files.push({
    path: 'service/package.json',
    content: JSON.stringify({ name: projectName }),
    action: 'ADD',
  });

  // CI workflow
  files.push({
    path: 'ci/ci.yml',
    content: `name: CI - ${projectName}`,
    action: 'ADD',
  });

  // K8s manifests
  files.push({
    path: 'k8s/deployment.yaml',
    content: `name: ${projectName}`,
    action: 'ADD',
  });

  files.push({
    path: 'k8s/service.yaml',
    content: `name: ${projectName}`,
    action: 'ADD',
  });

  return files;
}

describe('DevOps CLI - Coder', () => {
  describe('planNodeService', () => {
    it('should plan minimum required files', () => {
      const files = planNodeService({ PROJECT_NAME: 'test-app' });
      expect(files.length).toBeGreaterThanOrEqual(5);
    });

    it('should include service, CI, and K8s files', () => {
      const files = planNodeService({ PROJECT_NAME: 'test-app' });
      const paths = files.map((f) => f.path);

      expect(paths).toContain('service/src/index.ts');
      expect(paths).toContain('service/package.json');
      expect(paths).toContain('ci/ci.yml');
      expect(paths).toContain('k8s/deployment.yaml');
      expect(paths).toContain('k8s/service.yaml');
    });

    it('should use project name in content', () => {
      const files = planNodeService({ PROJECT_NAME: 'my-app' });
      const packageJson = files.find((f) => f.path === 'service/package.json');

      expect(packageJson).toBeDefined();
      expect(packageJson?.content).toContain('my-app');
    });

    it('should default project name if not provided', () => {
      const files = planNodeService({});
      const packageJson = files.find((f) => f.path === 'service/package.json');

      expect(packageJson?.content).toContain('my-service');
    });
  });
});
