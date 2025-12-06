#!/usr/bin/env node

/**
 * Accessibility Testing Script for QMLab
 * 
 * This script runs automated accessibility tests using axe-core
 * and generates comprehensive reports for WCAG 2.1 AA compliance.
 */

const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

const ACCESSIBILITY_RULES = {
  // WCAG 2.1 AA rules
  wcag21aa: [
    'color-contrast',
    'aria-allowed-attr',
    'aria-command-name',
    'aria-hidden-body',
    'aria-hidden-focus',
    'aria-input-field-name',
    'aria-required-attr',
    'aria-roles',
    'aria-valid-attr',
    'aria-valid-attr-value',
    'button-name',
    'bypass',
    'document-title',
    'focus-order-semantics',
    'form-field-multiple-labels',
    'frame-title',
    'heading-order',
    'html-has-lang',
    'html-lang-valid',
    'html-xml-lang-mismatch',
    'image-alt',
    'input-button-name',
    'input-image-alt',
    'label',
    'landmark-banner-is-top-level',
    'landmark-complementary-is-top-level',
    'landmark-contentinfo-is-top-level',
    'landmark-main-is-top-level',
    'landmark-no-duplicate-banner',
    'landmark-no-duplicate-contentinfo',
    'landmark-no-duplicate-main',
    'landmark-one-main',
    'landmark-unique',
    'link-name',
    'list',
    'listitem',
    'meta-refresh',
    'meta-viewport',
    'page-has-heading-one',
    'role-img-alt',
    'scrollable-region-focusable',
    'server-side-image-map',
    'skip-link',
    'tabindex',
    'td-headers-attr',
    'th-has-data-cells',
    'valid-lang'
  ],
  
  // Section 508 specific rules
  section508: [
    'color-contrast',
    'aria-allowed-attr',
    'button-name',
    'frame-title',
    'image-alt',
    'input-image-alt',
    'label',
    'link-name',
    'object-alt',
    'table-fake-caption',
    'td-has-header',
    'th-has-data-cells'
  ]
};

function createTestReport(results) {
  const timestamp = new Date().toISOString();
  const report = {
    timestamp,
    summary: {
      total: results.violations.length + results.passes.length,
      violations: results.violations.length,
      passes: results.passes.length,
      incomplete: results.incomplete.length,
      score: Math.round(((results.passes.length) / (results.violations.length + results.passes.length)) * 100) || 0
    },
    violations: results.violations.map(violation => ({
      id: violation.id,
      impact: violation.impact,
      description: violation.description,
      help: violation.help,
      helpUrl: violation.helpUrl,
      nodes: violation.nodes.length,
      wcagLevel: getWCAGLevel(violation.tags),
      section508: violation.tags.includes('section508')
    })),
    compliance: {
      wcagA: results.violations.filter(v => v.tags.includes('wcag2a')).length === 0,
      wcagAA: results.violations.filter(v => v.tags.includes('wcag2aa')).length === 0,
      wcagAAA: results.violations.filter(v => v.tags.includes('wcag2aaa')).length === 0,
      section508: results.violations.filter(v => v.tags.includes('section508')).length === 0
    }
  };

  return report;
}

function getWCAGLevel(tags) {
  if (tags.includes('wcag2aaa')) return 'AAA';
  if (tags.includes('wcag2aa')) return 'AA';
  if (tags.includes('wcag2a')) return 'A';
  return 'Unknown';
}

function generateHTMLReport(report) {
  return `
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>QMLab Accessibility Report</title>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }
    .container { max-width: 1200px; margin: 0 auto; background: white; border-radius: 8px; padding: 30px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
    .header { border-bottom: 2px solid #e1e5e9; padding-bottom: 20px; margin-bottom: 30px; }
    .score { font-size: 48px; font-weight: bold; color: ${report.summary.score >= 90 ? '#28a745' : report.summary.score >= 70 ? '#ffc107' : '#dc3545'}; }
    .summary { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }
    .stat { text-align: center; padding: 20px; background: #f8f9fa; border-radius: 6px; }
    .violations { margin-top: 30px; }
    .violation { border: 1px solid #dee2e6; border-radius: 6px; margin-bottom: 15px; overflow: hidden; }
    .violation-header { background: #f8f9fa; padding: 15px; border-bottom: 1px solid #dee2e6; }
    .violation-body { padding: 15px; }
    .impact-critical { border-left: 4px solid #dc3545; }
    .impact-serious { border-left: 4px solid #fd7e14; }
    .impact-moderate { border-left: 4px solid #ffc107; }
    .impact-minor { border-left: 4px solid #20c997; }
    .compliance { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; margin: 20px 0; }
    .compliance-item { text-align: center; padding: 15px; border-radius: 6px; }
    .pass { background: #d4edda; color: #155724; }
    .fail { background: #f8d7da; color: #721c24; }
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <h1>QMLab Accessibility Report</h1>
      <p>Generated on ${new Date(report.timestamp).toLocaleString()}</p>
      <div class="score">${report.summary.score}%</div>
    </div>
    
    <div class="summary">
      <div class="stat">
        <h3>Total Tests</h3>
        <div style="font-size: 24px; font-weight: bold;">${report.summary.total}</div>
      </div>
      <div class="stat">
        <h3>Violations</h3>
        <div style="font-size: 24px; font-weight: bold; color: #dc3545;">${report.summary.violations}</div>
      </div>
      <div class="stat">
        <h3>Passes</h3>
        <div style="font-size: 24px; font-weight: bold; color: #28a745;">${report.summary.passes}</div>
      </div>
    </div>

    <div class="compliance">
      <div class="compliance-item ${report.compliance.wcagA ? 'pass' : 'fail'}">
        <h4>WCAG 2.1 A</h4>
        <div>${report.compliance.wcagA ? '✓ PASS' : '✗ FAIL'}</div>
      </div>
      <div class="compliance-item ${report.compliance.wcagAA ? 'pass' : 'fail'}">
        <h4>WCAG 2.1 AA</h4>
        <div>${report.compliance.wcagAA ? '✓ PASS' : '✗ FAIL'}</div>
      </div>
      <div class="compliance-item ${report.compliance.section508 ? 'pass' : 'fail'}">
        <h4>Section 508</h4>
        <div>${report.compliance.section508 ? '✓ PASS' : '✗ FAIL'}</div>
      </div>
    </div>

    <div class="violations">
      <h2>Violations (${report.violations.length})</h2>
      ${report.violations.map(violation => `
        <div class="violation impact-${violation.impact}">
          <div class="violation-header">
            <h3>${violation.id}</h3>
            <p><strong>Impact:</strong> ${violation.impact.toUpperCase()} | <strong>WCAG Level:</strong> ${violation.wcagLevel} | <strong>Elements:</strong> ${violation.nodes}</p>
          </div>
          <div class="violation-body">
            <p><strong>Description:</strong> ${violation.description}</p>
            <p><strong>How to fix:</strong> ${violation.help}</p>
            <p><a href="${violation.helpUrl}" target="_blank">Learn more →</a></p>
          </div>
        </div>
      `).join('')}
    </div>
  </div>
</body>
</html>
  `.trim();
}

// Export for use in other scripts
module.exports = {
  ACCESSIBILITY_RULES,
  createTestReport,
  generateHTMLReport
};

console.log('✅ Accessibility testing utilities ready!');
console.log('Use these in your test files or run with the AccessibilityTesting component.');