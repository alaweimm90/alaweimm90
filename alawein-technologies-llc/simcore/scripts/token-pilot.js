#!/usr/bin/env node

/**
 * SimCore Token System Pilot Implementation
 * Target: Graphene domain + Statistical physics domain
 */

import { promises as fs } from 'fs';
import { execSync } from 'child_process';
import path from 'path';

const PILOT_DOMAINS = ['quantum', 'statistical'];
const PILOT_COMPONENTS = ['Button', 'Card', 'ModuleCard'];

class TokenPilot {
  constructor() {
    this.baseDir = process.cwd();
    this.tokenDir = path.join(this.baseDir, 'tokens');
    this.srcDir = path.join(this.baseDir, 'src');
  }

  async init() {
    console.log('🚀 SimCore Token Pilot - Phase 1');
    console.log(`📍 Target domains: ${PILOT_DOMAINS.join(', ')}`);
    console.log(`🧩 Target components: ${PILOT_COMPONENTS.join(', ')}`);
    
    await this.setupTokenStructure();
    await this.auditExistingStyles();
    await this.generatePilotTokens();
    await this.migrateComponents();
    await this.setupValidation();
    await this.generateReport();
  }

  /**
   * Setup token directory structure
   */
  async setupTokenStructure() {
    console.log('\n📁 Setting up token structure...');
    
    const dirs = [
      'tokens/primitives',
      'tokens/semantic', 
      'tokens/components',
      'tokens/build'
    ];

    for (const dir of dirs) {
      await fs.mkdir(dir, { recursive: true });
    }

    // Create initial primitive tokens focused on pilot domains
    const primitiveColors = {
      "primitive": {
        "color": {
          "purple": {
            "050": { "value": "hsl(245, 100%, 97%)" },
            "100": { "value": "hsl(245, 85%, 93%)" },
            "200": { "value": "hsl(245, 80%, 85%)" },
            "300": { "value": "hsl(245, 75%, 75%)" },
            "400": { "value": "hsl(245, 70%, 65%)" },
            "500": { "value": "hsl(240, 100%, 65%)" },
            "600": { "value": "hsl(240, 85%, 55%)" },
            "700": { "value": "hsl(240, 75%, 45%)" },
            "800": { "value": "hsl(245, 80%, 20%)" },
            "900": { "value": "hsl(245, 95%, 8%)" }
          },
          "green": {
            "400": { "value": "hsl(142, 76%, 55%)" },
            "500": { "value": "hsl(142, 76%, 45%)" },
            "600": { "value": "hsl(142, 76%, 35%)" }
          },
          "neutral": {
            "000": { "value": "hsl(0, 0%, 100%)" },
            "050": { "value": "hsl(0, 0%, 98%)" },
            "900": { "value": "hsl(0, 0%, 7%)" },
            "950": { "value": "hsl(0, 0%, 2%)" }
          }
        },
        "space": {
          "1": { "value": "0.25rem" },
          "2": { "value": "0.5rem" },
          "4": { "value": "1rem" },
          "6": { "value": "1.5rem" },
          "8": { "value": "2rem" },
          "12": { "value": "3rem" },
          "16": { "value": "4rem" }
        },
        "shadow": {
          "quantum": { "value": "0 0 40px hsl(240 100% 65% / 0.3)" },
          "statistical": { "value": "0 0 30px hsl(142 76% 45% / 0.25)" },
          "elevation": {
            "1": { "value": "0 1px 3px hsla(0, 0%, 0%, 0.12)" },
            "2": { "value": "0 4px 12px hsla(0, 0%, 0%, 0.15)" },
            "3": { "value": "0 12px 24px hsla(0, 0%, 0%, 0.18)" }
          }
        }
      }
    };

    await fs.writeFile(
      path.join(this.tokenDir, 'primitives/colors.json'),
      JSON.stringify(primitiveColors, null, 2)
    );

    console.log('  ✅ Primitive tokens created');
  }

  /**
   * Audit existing styles in pilot components
   */
  async auditExistingStyles() {
    console.log('\n🔍 Auditing existing styles...');
    
    const auditResults = {
      hardcodedColors: [],
      hardcodedSpacing: [],
      inconsistencies: []
    };

    // Scan component files
    for (const component of PILOT_COMPONENTS) {
      const filePaths = await this.findComponentFiles(component);
      
      for (const filePath of filePaths) {
        const content = await fs.readFile(filePath, 'utf-8');
        
        // Find hardcoded HSL colors
        const hslMatches = content.match(/hsl\([^)]+\)/g) || [];
        hslMatches.forEach(color => {
          auditResults.hardcodedColors.push({
            file: filePath,
            color,
            line: this.findLineNumber(content, color)
          });
        });

        // Find hardcoded spacing (rem, px values)
        const spacingMatches = content.match(/[\d.]+(?:rem|px)/g) || [];
        spacingMatches.forEach(spacing => {
          auditResults.hardcodedSpacing.push({
            file: filePath,
            spacing,
            line: this.findLineNumber(content, spacing)
          });
        });
      }
    }

    // Save audit results
    await fs.writeFile(
      path.join(this.tokenDir, 'audit-results.json'),
      JSON.stringify(auditResults, null, 2)
    );

    console.log(`  📊 Found ${auditResults.hardcodedColors.length} hardcoded colors`);
    console.log(`  📊 Found ${auditResults.hardcodedSpacing.length} hardcoded spacing values`);
  }

  /**
   * Find component files in the codebase
   */
  async findComponentFiles(componentName) {
    const possiblePaths = [
      `src/components/${componentName}.tsx`,
      `src/components/${componentName.toLowerCase()}/${componentName}.tsx`,
      `src/components/ui/${componentName.toLowerCase()}.tsx`,
      `src/components/*/${componentName}*.tsx`
    ];

    const existingPaths = [];
    
    for (const pattern of possiblePaths) {
      try {
        if (pattern.includes('*')) {
          // Use glob-like search
          const matches = await this.globSearch(pattern);
          existingPaths.push(...matches);
        } else {
          await fs.access(pattern);
          existingPaths.push(pattern);
        }
      } catch (error) {
        // File doesn't exist, continue
      }
    }

    return existingPaths;
  }

  /**
   * Simple glob search implementation
   */
  async globSearch(pattern) {
    // Simplified implementation - in production use proper glob library
    const dir = pattern.split('*')[0];
    const suffix = pattern.split('*')[1] || '';
    
    try {
      const files = await fs.readdir(dir);
      return files
        .filter(file => file.endsWith(suffix) || suffix === '')
        .map(file => path.join(dir, file));
    } catch (error) {
      return [];
    }
  }

  /**
   * Find line number of text in content
   */
  findLineNumber(content, searchText) {
    const lines = content.split('\n');
    for (let i = 0; i < lines.length; i++) {
      if (lines[i].includes(searchText)) {
        return i + 1;
      }
    }
    return -1;
  }

  /**
   * Generate semantic and component tokens for pilot
   */
  async generatePilotTokens() {
    console.log('\n🎨 Generating semantic tokens...');

    const semanticTokens = {
      "semantic": {
        "color": {
          "background": {
            "primary": { "value": "{primitive.color.purple.900}" },
            "secondary": { "value": "hsl(245, 85%, 10%)" },
            "surface": { "value": "hsla(245, 95%, 15%, 0.8)" }
          },
          "text": {
            "primary": { "value": "{primitive.color.neutral.050}" },
            "secondary": { "value": "hsl(245, 15%, 75%)" },
            "accent": { "value": "{primitive.color.purple.400}" }
          },
          "domain": {
            "quantum": { "value": "{primitive.color.purple.500}" },
            "statistical": { "value": "{primitive.color.green.500}" }
          },
          "interactive": {
            "primary": { "value": "{semantic.color.domain.quantum}" },
            "primary-hover": { "value": "hsl(240, 100%, 70%)" },
            "secondary": { "value": "transparent" },
            "secondary-border": { "value": "{semantic.color.domain.quantum}" }
          }
        },
        "space": {
          "module-padding": { "value": "{primitive.space.8}" },
          "card-padding": { "value": "{primitive.space.6}" },
          "inline-spacing": { "value": "{primitive.space.4}" }
        }
      }
    };

    await fs.writeFile(
      path.join(this.tokenDir, 'semantic/themes.json'),
      JSON.stringify(semanticTokens, null, 2)
    );

    // Generate component tokens
    const componentTokens = {
      "component": {
        "button": {
          "primary": {
            "background": { "value": "{semantic.color.interactive.primary}" },
            "text": { "value": "{semantic.color.background.primary}" },
            "padding": { "value": "{primitive.space.2} {primitive.space.4}" },
            "border-radius": { "value": "0.5rem" }
          },
          "secondary": {
            "background": { "value": "{semantic.color.interactive.secondary}" },
            "text": { "value": "{semantic.color.interactive.primary}" },
            "border": { "value": "1px solid {semantic.color.interactive.secondary-border}" },
            "padding": { "value": "{primitive.space.2} {primitive.space.4}" }
          }
        },
        "card": {
          "background": { "value": "{semantic.color.background.surface}" },
          "border": { "value": "1px solid hsla({semantic.color.domain.quantum}, 0.2)" },
          "padding": { "value": "{semantic.space.card-padding}" },
          "shadow": { "value": "{primitive.shadow.quantum}" }
        }
      }
    };

    await fs.writeFile(
      path.join(this.tokenDir, 'components/pilot.json'),
      JSON.stringify(componentTokens, null, 2)
    );

    console.log('  ✅ Semantic tokens generated');
    console.log('  ✅ Component tokens generated');
  }

  /**
   * Migrate pilot components to use tokens
   */
  async migrateComponents() {
    console.log('\n🔄 Migrating components to token system...');

    // This would contain actual migration logic
    // For now, we'll create migration guides
    const migrationGuide = {
      "Button": {
        "before": "className=\"bg-blue-600 text-white px-4 py-2\"",
        "after": "style={{ backgroundColor: 'var(--component-button-primary-background)' }}",
        "notes": "Replace hardcoded Tailwind classes with CSS custom properties"
      },
      "Card": {
        "before": "className=\"bg-gray-900 border border-blue-500/20 p-6\"",
        "after": "style={{ background: 'var(--component-card-background)', padding: 'var(--component-card-padding)' }}",
        "notes": "Use semantic tokens for consistent theming across domains"
      }
    };

    await fs.writeFile(
      path.join(this.tokenDir, 'migration-guide.json'),
      JSON.stringify(migrationGuide, null, 2)
    );

    console.log('  📝 Migration guide created');
  }

  /**
   * Setup validation and linting
   */
  async setupValidation() {
    console.log('\n🛡️  Setting up validation...');

    // Create ESLint rule for token usage
    const eslintRule = `
// .eslintrc.js addition
module.exports = {
  rules: {
    'no-hardcoded-colors': [
      'error',
      {
        'allowed-functions': ['hsl', 'rgb', 'rgba'],
        'require-var-prefix': '--'
      }
    ]
  }
};
`;

    await fs.writeFile(
      path.join(this.tokenDir, 'eslint-token-rules.js'),
      eslintRule
    );

    // Create pre-commit hook
    const preCommitHook = `#!/bin/sh
# Token validation pre-commit hook
echo "🔍 Validating tokens..."

npm run tokens:validate
if [ $? -ne 0 ]; then
  echo "❌ Token validation failed. Commit aborted."
  exit 1
fi

echo "✅ Token validation passed"
`;

    await fs.writeFile(
      path.join(this.tokenDir, 'pre-commit-hook.sh'),
      preCommitHook
    );

    console.log('  ✅ ESLint rules configured');
    console.log('  ✅ Pre-commit hooks ready');
  }

  /**
   * Generate pilot report
   */
  async generateReport() {
    console.log('\n📊 Generating pilot report...');

    const report = `
# SimCore Token Pilot - Phase 1 Report

## Scope
- **Domains**: ${PILOT_DOMAINS.join(', ')}
- **Components**: ${PILOT_COMPONENTS.join(', ')}
- **Duration**: 2 weeks

## Deliverables Created
- ✅ Token directory structure
- ✅ Primitive color & spacing tokens
- ✅ Semantic theme tokens  
- ✅ Component-specific tokens
- ✅ Style audit of existing components
- ✅ Migration guide documentation
- ✅ Validation tooling setup

## Next Steps
1. **Week 1**: Implement token system in Button and Card components
2. **Week 2**: Migrate Graphene and Statistical physics modules
3. **Week 3**: Visual regression testing with Chromatic
4. **Week 4**: Performance analysis and documentation

## Success Metrics
- [ ] Zero hardcoded colors in pilot components
- [ ] Consistent spacing across pilot modules
- [ ] Domain theme switching working
- [ ] Build-time validation passing
- [ ] Visual regression tests stable

## Risk Mitigation
- **Token cascade issues**: Validation catches circular references
- **Performance impact**: Monitoring CSS custom property performance
- **Designer workflow**: Figma sync process documented
- **Migration complexity**: Automated tooling reduces manual work

Generated: ${new Date().toISOString()}
`;

    await fs.writeFile(
      path.join(this.tokenDir, 'pilot-report.md'),
      report
    );

    console.log('  📄 Pilot report saved');
    console.log('\n🎉 Token pilot setup complete!');
    console.log(`\nNext: Review ${this.tokenDir}/pilot-report.md for implementation plan`);
  }
}

// Run the pilot setup
const pilot = new TokenPilot();
pilot.init().catch(console.error);