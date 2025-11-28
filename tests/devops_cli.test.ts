import { describe, it, expect } from 'vitest';
import path from 'node:path';
import { findManifests, readJson, type TemplateManifest } from '../tools/devops/fs.js';

const TEMPLATES_DIR = path.join(process.cwd(), 'templates', 'devops');

describe('DevOps CLI - Template Discovery', () => {
  it('should find template manifests', () => {
    const manifests = findManifests(TEMPLATES_DIR);
    expect(manifests.length).toBeGreaterThan(0);
    expect(manifests.every((m) => m.endsWith('template.json'))).toBe(true);
  });

  it('should read valid manifest JSON', () => {
    const manifests = findManifests(TEMPLATES_DIR);
    const manifest = readJson<TemplateManifest>(manifests[0]);
    expect(manifest).not.toBeNull();
    expect(manifest?.name).toBeDefined();
    expect(manifest?.version).toBeDefined();
  });

  it('should find expected templates', () => {
    const manifests = findManifests(TEMPLATES_DIR);
    const names = manifests.map((m) => {
      const manifest = readJson<TemplateManifest>(m);
      return manifest?.name;
    });
    expect(names).toContain('github-actions-node-ci');
    expect(names).toContain('k8s-deployment-service');
    expect(names).toContain('demo-k8s-node');
  });
});
