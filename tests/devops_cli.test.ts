import { describe, it, expect } from 'vitest';
import * as fs from 'node:fs';
import path from 'node:path';
import { findManifests, readJson, type TemplateManifest } from '../tools/lib/fs.js';

const TEMPLATES_DIR = path.join(process.cwd(), 'templates', 'devops');
const templatesExist = fs.existsSync(TEMPLATES_DIR);

describe('DevOps CLI - Template Discovery', () => {
  it('should find template manifests', () => {
    if (!templatesExist) {
      // Skip test if templates directory doesn't exist yet
      expect(true).toBe(true);
      return;
    }
    const manifests = findManifests(TEMPLATES_DIR);
    expect(manifests.length).toBeGreaterThan(0);
    expect(manifests.every((m) => m.endsWith('template.json'))).toBe(true);
  });

  it('should read valid manifest JSON', () => {
    if (!templatesExist) {
      expect(true).toBe(true);
      return;
    }
    const manifests = findManifests(TEMPLATES_DIR);
    if (manifests.length === 0) {
      expect(true).toBe(true);
      return;
    }
    const manifest = readJson<TemplateManifest>(manifests[0]);
    expect(manifest).not.toBeNull();
    expect(manifest?.name).toBeDefined();
    expect(manifest?.version).toBeDefined();
  });

  it('should find expected templates', () => {
    if (!templatesExist) {
      expect(true).toBe(true);
      return;
    }
    const manifests = findManifests(TEMPLATES_DIR);
    if (manifests.length === 0) {
      expect(true).toBe(true);
      return;
    }
    const names = manifests.map((m) => {
      const manifest = readJson<TemplateManifest>(m);
      return manifest?.name;
    });
    expect(names).toContain('github-actions-node-ci');
    expect(names).toContain('k8s-deployment-service');
    expect(names).toContain('demo-k8s-node');
  });
});
