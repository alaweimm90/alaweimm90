import {
  listProjects,
  getProjectsByOrganization,
  getProjectsByType,
  getProjectStats,
  listTemplates,
  getTemplate
} from '../deployment';

describe('Deployment', () => {
  describe('listProjects', () => {
    it('should return array of projects', () => {
      const projects = listProjects();

      expect(Array.isArray(projects)).toBe(true);
      expect(projects.length).toBeGreaterThan(0);
    });

    it('should have required fields on each project', () => {
      const projects = listProjects();

      for (const project of projects) {
        expect(project).toHaveProperty('name');
        expect(project).toHaveProperty('path');
        expect(project).toHaveProperty('type');
        expect(project).toHaveProperty('organization');
        expect(project).toHaveProperty('technologies');
      }
    });
  });

  describe('getProjectsByOrganization', () => {
    it('should filter by organization', () => {
      const projects = getProjectsByOrganization('alaweimm90-science');

      expect(projects.length).toBeGreaterThan(0);
      for (const project of projects) {
        expect(project.organization).toBe('alaweimm90-science');
      }
    });

    it('should return empty array for unknown org', () => {
      const projects = getProjectsByOrganization('unknown-org');
      expect(projects).toHaveLength(0);
    });
  });

  describe('getProjectsByType', () => {
    it('should filter by type', () => {
      const projects = getProjectsByType('scientific');

      expect(projects.length).toBeGreaterThan(0);
      for (const project of projects) {
        expect(project.type).toBe('scientific');
      }
    });
  });

  describe('getProjectStats', () => {
    it('should return statistics', () => {
      const stats = getProjectStats();

      expect(stats).toHaveProperty('total');
      expect(stats).toHaveProperty('byType');
      expect(stats).toHaveProperty('byOrganization');
      expect(stats).toHaveProperty('byTechnology');
      expect(stats.total).toBeGreaterThan(0);
    });
  });

  describe('listTemplates', () => {
    it('should return array of templates', () => {
      const templates = listTemplates();

      expect(Array.isArray(templates)).toBe(true);
      expect(templates.length).toBeGreaterThan(0);
    });
  });

  describe('getTemplate', () => {
    it('should return template by name', () => {
      const template = getTemplate('docker-compose');

      expect(template).not.toBeNull();
      expect(template?.name).toBe('docker-compose');
      expect(template?.platform).toBe('docker');
    });

    it('should return null for unknown template', () => {
      const template = getTemplate('unknown-template');
      expect(template).toBeNull();
    });
  });
});
