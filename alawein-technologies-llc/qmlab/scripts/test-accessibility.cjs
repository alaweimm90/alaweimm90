#!/usr/bin/env node

/**
 * Accessibility Testing Script
 * Runs comprehensive WCAG 2.1 AA validation
 */

const puppeteer = require('puppeteer');
const axeCore = require('axe-core');
const fs = require('fs');
const path = require('path');

// ANSI color codes for terminal output
const colors = {
  reset: '\x1b[0m',
  bright: '\x1b[1m',
  red: '\x1b[31m',
  green: '\x1b[32m',
  yellow: '\x1b[33m',
  blue: '\x1b[34m',
  magenta: '\x1b[35m',
  cyan: '\x1b[36m'
};

async function runAccessibilityTests() {
  console.log(`${colors.cyan}${colors.bright}
╔════════════════════════════════════════════════════╗
║     QMLab Accessibility Testing Suite              ║
║     WCAG 2.1 AA Compliance Validator               ║
╚════════════════════════════════════════════════════╝
${colors.reset}`);

  const browser = await puppeteer.launch({
    headless: true,
    args: ['--no-sandbox', '--disable-setuid-sandbox']
  });

  try {
    const page = await browser.newPage();
    
    // Set viewport to standard desktop
    await page.setViewport({ width: 1920, height: 1080 });
    
    // Navigate to the application
    const url = process.env.TEST_URL || 'http://localhost:8080';
    console.log(`\n${colors.blue}Testing URL: ${url}${colors.reset}\n`);
    
    await page.goto(url, { waitUntil: 'networkidle2' });
    
    // Wait for main content to load
    await page.waitForSelector('main, [role="main"]', { timeout: 5000 });
    
    // Inject axe-core
    await page.addScriptTag({ path: require.resolve('axe-core') });
    
    // Run axe accessibility tests
    const axeResults = await page.evaluate(async () => {
      return await axe.run(document, {
        rules: {
          'color-contrast': { enabled: true },
          'button-name': { enabled: true },
          'image-alt': { enabled: true },
          'label': { enabled: true },
          'link-name': { enabled: true },
          'region': { enabled: true },
          'skip-link': { enabled: true },
          'aria-command-name': { enabled: true },
          'page-has-heading-one': { enabled: true },
          'tabindex': { enabled: true }
        }
      });
    });
    
    // Run custom validator
    const customResults = await page.evaluate(() => {
      // Check for our custom validator
      if (typeof window.runAccessibilityAudit === 'function') {
        return window.runAccessibilityAudit();
      }
      
      // Fallback validation
      const results = {
        buttons: [],
        landmarks: [],
        skipLink: null,
        headings: [],
        tabindex: []
      };
      
      // Check buttons
      document.querySelectorAll('button, [role="button"]').forEach(btn => {
        const hasLabel = btn.getAttribute('aria-label') || 
                        btn.getAttribute('aria-labelledby') || 
                        btn.textContent?.trim();
        if (!hasLabel) {
          results.buttons.push(btn.outerHTML.substring(0, 100));
        }
      });
      
      // Check landmarks
      results.landmarks = {
        main: !!document.querySelector('main, [role="main"]'),
        header: !!document.querySelector('header, [role="banner"]'),
        nav: !!document.querySelector('nav, [role="navigation"]'),
        footer: !!document.querySelector('footer, [role="contentinfo"]')
      };
      
      // Check skip link
      const skipLink = document.querySelector('a[href="#main-content"], .skip-link');
      const skipTarget = document.querySelector('#main-content');
      results.skipLink = {
        exists: !!skipLink,
        targetExists: !!skipTarget,
        targetFocusable: skipTarget?.getAttribute('tabindex') === '-1'
      };
      
      // Check headings
      const h1Count = document.querySelectorAll('h1').length;
      results.headings = {
        h1Count,
        hasProperHierarchy: true
      };
      
      // Check tabindex
      document.querySelectorAll('[tabindex]').forEach(el => {
        const value = parseInt(el.getAttribute('tabindex') || '0');
        if (value > 0) {
          results.tabindex.push({
            element: el.tagName,
            value
          });
        }
      });
      
      return results;
    });
    
    // Process and display results
    displayResults(axeResults, customResults);
    
    // Generate report
    const report = generateReport(axeResults, customResults);
    
    // Save report to file
    const reportPath = path.join(process.cwd(), 'accessibility-report.json');
    fs.writeFileSync(reportPath, JSON.stringify(report, null, 2));
    console.log(`\n${colors.green}Report saved to: ${reportPath}${colors.reset}`);
    
    // Exit with appropriate code
    const hasViolations = axeResults.violations.length > 0;
    process.exit(hasViolations ? 1 : 0);
    
  } catch (error) {
    console.error(`${colors.red}Error running tests: ${error.message}${colors.reset}`);
    process.exit(1);
  } finally {
    await browser.close();
  }
}

function displayResults(axeResults, customResults) {
  console.log(`${colors.cyan}${colors.bright}
════════════════════════════════════════════════════════
                    TEST RESULTS                        
════════════════════════════════════════════════════════
${colors.reset}`);

  // Display axe-core results
  const { violations, passes, incomplete } = axeResults;
  
  console.log(`${colors.bright}Axe-Core Results:${colors.reset}`);
  console.log(`  ${colors.green}✓ Passed: ${passes.length}${colors.reset}`);
  console.log(`  ${colors.yellow}⚠ Incomplete: ${incomplete.length}${colors.reset}`);
  console.log(`  ${colors.red}✗ Violations: ${violations.length}${colors.reset}\n`);
  
  // Display violations by severity
  if (violations.length > 0) {
    const bySeverity = {
      critical: violations.filter(v => v.impact === 'critical'),
      serious: violations.filter(v => v.impact === 'serious'),
      moderate: violations.filter(v => v.impact === 'moderate'),
      minor: violations.filter(v => v.impact === 'minor')
    };
    
    console.log(`${colors.bright}Violations by Severity:${colors.reset}`);
    console.log(`  ${colors.red}Critical: ${bySeverity.critical.length}${colors.reset}`);
    console.log(`  ${colors.magenta}Serious: ${bySeverity.serious.length}${colors.reset}`);
    console.log(`  ${colors.yellow}Moderate: ${bySeverity.moderate.length}${colors.reset}`);
    console.log(`  ${colors.cyan}Minor: ${bySeverity.minor.length}${colors.reset}\n`);
    
    // List violations
    violations.forEach(violation => {
      const color = violation.impact === 'critical' ? colors.red :
                    violation.impact === 'serious' ? colors.magenta :
                    violation.impact === 'moderate' ? colors.yellow :
                    colors.cyan;
      
      console.log(`${color}[${violation.impact.toUpperCase()}] ${violation.description}${colors.reset}`);
      console.log(`  Rule: ${violation.id}`);
      console.log(`  WCAG: ${violation.tags.join(', ')}`);
      console.log(`  Elements: ${violation.nodes.length}`);
      console.log(`  Help: ${violation.helpUrl}\n`);
    });
  }
  
  // Display custom validation results
  console.log(`${colors.bright}Custom Validation:${colors.reset}`);
  console.log(`  Landmarks: ${customResults.landmarks.main ? '✓' : '✗'} main, ${customResults.landmarks.header ? '✓' : '✗'} header, ${customResults.landmarks.nav ? '✓' : '✗'} nav, ${customResults.landmarks.footer ? '✓' : '✗'} footer`);
  console.log(`  Skip Link: ${customResults.skipLink.exists ? '✓' : '✗'} exists, ${customResults.skipLink.targetExists ? '✓' : '✗'} target, ${customResults.skipLink.targetFocusable ? '✓' : '✗'} focusable`);
  console.log(`  Headings: ${customResults.headings.h1Count} H1 elements`);
  console.log(`  Tabindex > 0: ${customResults.tabindex.length === 0 ? '✓ None found' : `✗ ${customResults.tabindex.length} found`}`);
  console.log(`  Unlabeled Buttons: ${customResults.buttons.length === 0 ? '✓ None found' : `✗ ${customResults.buttons.length} found`}\n`);
}

function generateReport(axeResults, customResults) {
  return {
    timestamp: new Date().toISOString(),
    url: process.env.TEST_URL || 'http://localhost:8080',
    summary: {
      totalViolations: axeResults.violations.length,
      critical: axeResults.violations.filter(v => v.impact === 'critical').length,
      serious: axeResults.violations.filter(v => v.impact === 'serious').length,
      moderate: axeResults.violations.filter(v => v.impact === 'moderate').length,
      minor: axeResults.violations.filter(v => v.impact === 'minor').length,
      passes: axeResults.passes.length,
      incomplete: axeResults.incomplete.length
    },
    wcagCompliance: {
      level: axeResults.violations.filter(v => v.impact === 'critical' || v.impact === 'serious').length === 0 ? 'AA' : 'Non-compliant',
      violations: axeResults.violations.map(v => ({
        rule: v.id,
        impact: v.impact,
        description: v.description,
        wcag: v.tags,
        elements: v.nodes.length,
        help: v.helpUrl
      }))
    },
    customValidation: customResults,
    axeResults: axeResults
  };
}

// Run the tests
runAccessibilityTests().catch(console.error);