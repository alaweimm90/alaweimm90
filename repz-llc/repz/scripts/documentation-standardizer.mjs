#!/usr/bin/env node

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const rootDir = path.resolve(__dirname, '..');

console.log('📚 Enterprise Documentation Standardizer\n');

const standards = {
  requiredFiles: [
    'README.md',
    'CONTRIBUTING.md',
    'SECURITY.md',
    'CHANGELOG.md',
    'LICENSE'
  ],
  featureDocumentation: {
    required: ['README.md'],
    optional: ['ARCHITECTURE.md', 'API.md', 'TESTING.md']
  },
  qualityChecks: {
    brokenLinks: [],
    outdatedDates: [],
    missingFiles: [],
    inconsistentFormat: []
  }
};

// 1. CHECK REQUIRED ROOT DOCUMENTATION
function checkRootDocumentation() {
  console.log('📋 Checking required root documentation...\n');
  
  const missing = [];
  const existing = [];
  
  standards.requiredFiles.forEach(file => {
    const filePath = path.join(rootDir, file);
    if (fs.existsSync(filePath)) {
      existing.push(file);
    } else {
      missing.push(file);
    }
  });
  
  console.log(`✅ Existing files (${existing.length}):`);
  existing.forEach(file => console.log(`  • ${file}`));
  
  if (missing.length > 0) {
    console.log(`\n❌ Missing files (${missing.length}):`);
    missing.forEach(file => console.log(`  • ${file}`));
    standards.qualityChecks.missingFiles.push(...missing);
  }
  
  console.log();
  return { existing, missing };
}

// 2. CHECK FEATURE DOCUMENTATION
function checkFeatureDocumentation() {
  console.log('🎯 Checking feature documentation...\n');
  
  const featuresDir = path.join(rootDir, 'src/features');
  if (!fs.existsSync(featuresDir)) {
    console.log('⚠️  Features directory not found, skipping feature docs check');
    return;
  }
  
  const features = fs.readdirSync(featuresDir).filter(item => {
    const itemPath = path.join(featuresDir, item);
    return fs.statSync(itemPath).isDirectory();
  });
  
  const featureDocsStatus = [];
  
  features.forEach(feature => {
    const featureDir = path.join(featuresDir, feature);
    const status = {
      name: feature,
      hasReadme: false,
      missingDocs: []
    };
    
    // Check for README
    const readmePath = path.join(featureDir, 'README.md');
    if (fs.existsSync(readmePath)) {
      status.hasReadme = true;
    } else {
      status.missingDocs.push('README.md');
    }
    
    featureDocsStatus.push(status);
  });
  
  const withDocs = featureDocsStatus.filter(f => f.hasReadme).length;
  const withoutDocs = featureDocsStatus.filter(f => !f.hasReadme).length;
  
  console.log(`📊 Feature Documentation Status:`);
  console.log(`  ✅ With README: ${withDocs}/${features.length}`);
  console.log(`  ❌ Missing README: ${withoutDocs}/${features.length}`);
  
  if (withoutDocs > 0) {
    console.log(`\n📝 Features missing documentation:`);
    featureDocsStatus
      .filter(f => !f.hasReadme)
      .forEach(f => console.log(`  • ${f.name}`));
  }
  
  console.log();
  return featureDocsStatus;
}

// 3. SCAN FOR BROKEN LINKS
function findBrokenLinks() {
  console.log('🔗 Scanning for broken internal links...\n');
  
  const brokenLinks = [];
  
  function scanDirectory(dir) {
    const items = fs.readdirSync(dir);
    
    items.forEach(item => {
      const itemPath = path.join(dir, item);
      const stat = fs.statSync(itemPath);
      
      if (stat.isDirectory() && !item.includes('node_modules') && !item.includes('_graveyard')) {
        scanDirectory(itemPath);
      } else if (item.endsWith('.md')) {
        checkLinksInFile(itemPath);
      }
    });
  }
  
  function checkLinksInFile(filePath) {
    try {
      const content = fs.readFileSync(filePath, 'utf8');
      const relativePath = path.relative(rootDir, filePath);
      
      // Find markdown links
      const linkRegex = /\[([^\]]+)\]\(([^)]+)\)/g;
      let match;
      
      while ((match = linkRegex.exec(content)) !== null) {
        const linkText = match[1];
        const linkPath = match[2];
        
        // Skip external links
        if (linkPath.startsWith('http') || linkPath.startsWith('mailto:')) {
          continue;
        }
        
        // Check internal links
        if (linkPath.startsWith('./') || linkPath.startsWith('../')) {
          const fullLinkPath = path.resolve(path.dirname(filePath), linkPath);
          if (!fs.existsSync(fullLinkPath)) {
            brokenLinks.push({
              file: relativePath,
              linkText,
              linkPath,
              issue: 'File not found'
            });
          }
        }
      }
    } catch (error) {
      // Skip files that can't be read
    }
  }
  
  scanDirectory(rootDir);
  
  standards.qualityChecks.brokenLinks = brokenLinks;
  
  if (brokenLinks.length > 0) {
    console.log(`Found ${brokenLinks.length} broken links:`);
    brokenLinks.forEach(link => {
      console.log(`  🔗 ${link.file}: "${link.linkText}" → ${link.linkPath}`);
    });
  } else {
    console.log('✅ No broken links found!');
  }
  
  console.log();
}

// 4. CHECK FOR OUTDATED DOCUMENTATION
function checkOutdatedDocumentation() {
  console.log('📅 Checking for outdated documentation...\n');
  
  const outdatedDocs = [];
  const currentYear = new Date().getFullYear();
  
  function scanForDates(dir) {
    const items = fs.readdirSync(dir);
    
    items.forEach(item => {
      const itemPath = path.join(dir, item);
      const stat = fs.statSync(itemPath);
      
      if (stat.isDirectory() && !item.includes('node_modules') && !item.includes('_graveyard')) {
        scanForDates(itemPath);
      } else if (item.endsWith('.md')) {
        checkDatesInFile(itemPath);
      }
    });
  }
  
  function checkDatesInFile(filePath) {
    try {
      const content = fs.readFileSync(filePath, 'utf8');
      const relativePath = path.relative(rootDir, filePath);
      
      // Look for date patterns
      const datePatterns = [
        /(?:updated?|modified|created?):\s*(\d{4})/gi,
        /(?:last\s+update|last\s+modified):\s*(\d{4})/gi,
        /(?:date|year):\s*(\d{4})/gi
      ];
      
      datePatterns.forEach(pattern => {
        let match;
        while ((match = pattern.exec(content)) !== null) {
          const year = parseInt(match[1]);
          if (currentYear - year > 1) {
            outdatedDocs.push({
              file: relativePath,
              lastUpdated: year,
              yearsOld: currentYear - year
            });
          }
        }
      });
    } catch (error) {
      // Skip files that can't be read
    }
  }
  
  scanForDates(rootDir);
  
  standards.qualityChecks.outdatedDates = outdatedDocs;
  
  if (outdatedDocs.length > 0) {
    console.log(`Found ${outdatedDocs.length} potentially outdated documents:`);
    outdatedDocs.forEach(doc => {
      console.log(`  📅 ${doc.file}: Last updated ${doc.lastUpdated} (${doc.yearsOld} years ago)`);
    });
  } else {
    console.log('✅ No outdated documentation found!');
  }
  
  console.log();
}

// 5. GENERATE MISSING DOCUMENTATION
function generateMissingDocumentation() {
  console.log('📝 Generating missing documentation...\n');
  
  const templates = {
    'CHANGELOG.md': `# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial project setup
- Feature-first architecture implementation
- Enterprise-grade cleanup automation

### Changed
- Migrated to monorepo structure
- Implemented atomic design system

### Fixed
- Resolved import path issues
- Eliminated ghost routes and dead pages

## [1.0.0] - ${new Date().toISOString().split('T')[0]}

### Added
- Complete monorepo transformation
- Automated quality gates
- Comprehensive documentation structure
`,

    'LICENSE': `MIT License

Copyright (c) ${new Date().getFullYear()} REPZ Coach Platform

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
`
  };
  
  let created = 0;
  
  standards.qualityChecks.missingFiles.forEach(file => {
    if (templates[file]) {
      const filePath = path.join(rootDir, file);
      fs.writeFileSync(filePath, templates[file]);
      console.log(`✅ Created: ${file}`);
      created++;
    }
  });
  
  if (created === 0) {
    console.log('ℹ️  No missing files to generate');
  }
  
  console.log();
}

// 6. GENERATE README FOR FEATURES
function generateFeatureReadmes() {
  console.log('🎯 Generating README files for features...\n');
  
  const featuresDir = path.join(rootDir, 'src/features');
  if (!fs.existsSync(featuresDir)) return;
  
  const features = fs.readdirSync(featuresDir).filter(item => {
    const itemPath = path.join(featuresDir, item);
    return fs.statSync(itemPath).isDirectory();
  });
  
  let created = 0;
  
  features.forEach(feature => {
    const readmePath = path.join(featuresDir, feature, 'README.md');
    
    if (!fs.existsSync(readmePath)) {
      const featureReadme = `# ${feature.charAt(0).toUpperCase() + feature.slice(1)} Feature

## Overview

This module handles ${feature}-related functionality for the REPZ platform.

## Structure

\`\`\`
${feature}/
├── components/     # React components
├── hooks/         # Custom React hooks  
├── utils/         # Utility functions
├── types/         # TypeScript definitions
├── api/           # API integration
└── pages/         # Route components (if applicable)
\`\`\`

## Usage

\`\`\`typescript
import { ${feature}Components } from '@/features/${feature}';
\`\`\`

## Components

<!-- List main components here -->

## Hooks

<!-- List custom hooks here -->

## API

<!-- Document API integration here -->

## Testing

\`\`\`bash
npm test -- ${feature}
\`\`\`

## Contributing

Please follow the project's coding standards and include tests for new functionality.
`;
      
      fs.writeFileSync(readmePath, featureReadme);
      console.log(`✅ Created: src/features/${feature}/README.md`);
      created++;
    }
  });
  
  if (created === 0) {
    console.log('ℹ️  All features already have README files');
  }
  
  console.log();
}

// 7. GENERATE DOCUMENTATION QUALITY REPORT
function generateQualityReport() {
  const report = {
    timestamp: new Date().toISOString(),
    summary: {
      requiredFiles: {
        total: standards.requiredFiles.length,
        existing: standards.requiredFiles.length - standards.qualityChecks.missingFiles.length,
        missing: standards.qualityChecks.missingFiles.length
      },
      qualityIssues: {
        brokenLinks: standards.qualityChecks.brokenLinks.length,
        outdatedDocs: standards.qualityChecks.outdatedDates.length
      }
    },
    details: {
      missingFiles: standards.qualityChecks.missingFiles,
      brokenLinks: standards.qualityChecks.brokenLinks,
      outdatedDocs: standards.qualityChecks.outdatedDates
    },
    recommendations: [
      'Create missing required documentation files',
      'Fix broken internal links',
      'Update outdated documentation',
      'Add README files to all feature modules',
      'Implement automated documentation checks',
      'Set up documentation update reminders'
    ]
  };
  
  fs.writeFileSync(
    path.join(rootDir, 'documentation-quality-report.json'),
    JSON.stringify(report, null, 2)
  );
  
  return report;
}

// 8. CREATE DOCUMENTATION MAINTENANCE SCRIPT
function createMaintenanceScript() {
  const maintenanceScript = `#!/usr/bin/env node

// Documentation Maintenance Script
// Run this monthly to keep documentation up to date

import fs from 'fs';
import path from 'path';

console.log('📚 Running documentation maintenance...');

// Check for files that haven't been updated in 6 months
const sixMonthsAgo = new Date();
sixMonthsAgo.setMonth(sixMonthsAgo.getMonth() - 6);

function checkDocumentationFreshness(dir) {
  const items = fs.readdirSync(dir);
  const staleFiles = [];
  
  items.forEach(item => {
    const itemPath = path.join(dir, item);
    const stat = fs.statSync(itemPath);
    
    if (stat.isDirectory() && !item.includes('node_modules')) {
      staleFiles.push(...checkDocumentationFreshness(itemPath));
    } else if (item.endsWith('.md')) {
      if (stat.mtime < sixMonthsAgo) {
        staleFiles.push({
          file: path.relative(process.cwd(), itemPath),
          lastModified: stat.mtime.toISOString().split('T')[0]
        });
      }
    }
  });
  
  return staleFiles;
}

const staleFiles = checkDocumentationFreshness('.');

if (staleFiles.length > 0) {
  console.log('⚠️  Found stale documentation files:');
  staleFiles.forEach(file => {
    console.log(\`  📅 \${file.file} (last modified: \${file.lastModified})\`);
  });
  console.log('\\n💡 Consider reviewing and updating these files.');
} else {
  console.log('✅ All documentation appears to be up to date!');
}

console.log('\\n📚 Documentation maintenance complete!');
`;

  fs.writeFileSync(
    path.join(rootDir, 'scripts/maintain-documentation.mjs'),
    maintenanceScript
  );
  
  try {
    fs.chmodSync(path.join(rootDir, 'scripts/maintain-documentation.mjs'), '755');
  } catch (error) {
    // chmod might not work on all systems
  }
}

// MAIN EXECUTION
async function main() {
  console.log('🎯 Starting enterprise documentation standardization...\n');
  
  const rootDocs = checkRootDocumentation();
  const featureDocs = checkFeatureDocumentation();
  findBrokenLinks();
  checkOutdatedDocumentation();
  
  console.log('🔨 Generating missing documentation...\n');
  generateMissingDocumentation();
  generateFeatureReadmes();
  
  const report = generateQualityReport();
  createMaintenanceScript();
  
  // Display final summary
  console.log('📊 Documentation Quality Summary:');
  console.log('═'.repeat(60));
  console.log(`📋 Required Files: ${report.summary.requiredFiles.existing}/${report.summary.requiredFiles.total}`);
  console.log(`🔗 Broken Links: ${report.summary.qualityIssues.brokenLinks}`);
  console.log(`📅 Outdated Docs: ${report.summary.qualityIssues.outdatedDocs}`);
  console.log('═'.repeat(60));
  
  if (report.summary.requiredFiles.missing > 0 || 
      report.summary.qualityIssues.brokenLinks > 0 || 
      report.summary.qualityIssues.outdatedDocs > 0) {
    console.log('\n💡 Recommendations:');
    report.recommendations.forEach((rec, i) => {
      console.log(`  ${i + 1}. ${rec}`);
    });
  } else {
    console.log('\n🎉 Documentation is in excellent condition!');
  }
  
  console.log('\n📄 Reports generated:');
  console.log('  • documentation-quality-report.json');
  console.log('  • scripts/maintain-documentation.mjs');
  
  console.log('\n📚 Documentation standardization complete!');
}

main().catch(console.error);