class RBAC {
  constructor({ logger }) {
    this.logger = logger;
    this.roles = new Map();
    this.defineDefaults();
  }

  defineDefaults() {
    this.roles.set('admin', { allow: ['*'] });
    this.roles.set('maintainer', {
      allow: ['build', 'deploy', 'workflow:*', 'security:*', 'orchestrate'],
    });
    this.roles.set('developer', { allow: ['code:*', 'test:*', 'workflow:*', 'orchestrate'] });
    this.roles.set('tester', { allow: ['test:*', 'workflow:*'] });
    this.roles.set('viewer', { allow: ['monitor', 'status'] });
  }

  hasPermission(role, action, resource) {
    const def = this.roles.get(role) || { allow: [] };
    const parts = action.split(':');
    const wildcard = `${parts[0]}:*`;
    return def.allow.includes('*') || def.allow.includes(action) || def.allow.includes(wildcard);
  }
}

module.exports = { RBAC };
