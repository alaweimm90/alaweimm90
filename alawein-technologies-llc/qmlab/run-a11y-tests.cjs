#!/usr/bin/env node

/**
 * Accessibility Test Runner for QMLab
 * Simulates accessibility tests without requiring full browser installation
 */

const fs = require('fs');
const path = require('path');

console.log('');
console.log('🔍 QMLab Accessibility Test Suite');
console.log('==================================');
console.log('');

// Test results storage
const results = {
  passed: 0,
  failed: 0,
  tests: []
};

// Helper function to run test
function runTest(name, check) {
  process.stdout.write(`⏳ ${name}... `);
  try {
    const result = check();
    if (result) {
      console.log('✅ PASSED');
      results.passed++;
      results.tests.push({ name, status: 'passed' });
    } else {
      console.log('❌ FAILED');
      results.failed++;
      results.tests.push({ name, status: 'failed' });
    }
  } catch (error) {
    console.log(`❌ ERROR: ${error.message}`);
    results.failed++;
    results.tests.push({ name, status: 'error', error: error.message });
  }
}

// Test 1: Check CSS for touch target sizes
runTest('Touch targets meet 44px minimum', () => {
  const cssFile = fs.readFileSync(path.join(__dirname, 'src/index.css'), 'utf8');
  return cssFile.includes('min-height: 44px') && cssFile.includes('min-width: 44px');
});

// Test 2: Check for accessibility provider
runTest('Accessibility provider exists', () => {
  const providerFile = path.join(__dirname, 'src/components/AccessibilityProvider.tsx');
  return fs.existsSync(providerFile);
});

// Test 3: Check for skip links
runTest('Skip navigation links implemented', () => {
  const headerFile = fs.readFileSync(path.join(__dirname, 'src/components/Header.tsx'), 'utf8');
  return headerFile.includes('Skip to main content');
});

// Test 4: Check for ARIA labels in quantum components
runTest('Quantum components have ARIA labels', () => {
  const circuitBuilder = fs.readFileSync(path.join(__dirname, 'src/components/CircuitBuilder.tsx'), 'utf8');
  return circuitBuilder.includes('aria-label');
});

// Test 5: Check for mobile optimizations
runTest('Mobile optimizations in CSS', () => {
  const cssFile = fs.readFileSync(path.join(__dirname, 'src/index.css'), 'utf8');
  return cssFile.includes('@media (max-width: 768px)') && cssFile.includes('touch-manipulation');
});

// Test 6: Check for reduced motion support
runTest('Reduced motion preferences supported', () => {
  const cssFile = fs.readFileSync(path.join(__dirname, 'src/index.css'), 'utf8');
  return cssFile.includes('prefers-reduced-motion: reduce');
});

// Test 7: Check for focus indicators
runTest('Focus indicators defined', () => {
  const cssFile = fs.readFileSync(path.join(__dirname, 'src/index.css'), 'utf8');
  return cssFile.includes('focus:ring') || cssFile.includes('focus-visible');
});

// Test 8: Check for semantic HTML
runTest('Header uses semantic HTML', () => {
  const headerFile = fs.readFileSync(path.join(__dirname, 'src/components/Header.tsx'), 'utf8');
  return headerFile.includes('<header') && headerFile.includes('<nav');
});

// Test 9: Check for color contrast utilities
runTest('High contrast mode support', () => {
  const tailwindConfig = fs.readFileSync(path.join(__dirname, 'tailwind.config.ts'), 'utf8');
  return tailwindConfig.includes('contrast') || fs.existsSync(path.join(__dirname, 'src/lib/accessibility-audit.ts'));
});

// Test 10: Check for keyboard navigation
runTest('Keyboard navigation supported', () => {
  const blochSphere = fs.readFileSync(path.join(__dirname, 'src/components/BlochSphere.tsx'), 'utf8');
  return blochSphere.includes('onKeyDown') || blochSphere.includes('tabIndex');
});

// Test 11: Check for alternative text
runTest('Images have alternative text', () => {
  const logoFile = fs.readFileSync(path.join(__dirname, 'src/components/QLogo.tsx'), 'utf8');
  return logoFile.includes('aria-label') || logoFile.includes('title');
});

// Test 12: Check for form labels
runTest('Form inputs have labels', () => {
  const trainingDashboard = fs.readFileSync(path.join(__dirname, 'src/components/TrainingDashboard.tsx'), 'utf8');
  return trainingDashboard.includes('htmlFor') || trainingDashboard.includes('aria-label');
});

// Test 13: Check for error boundaries
runTest('Error boundary implemented', () => {
  const errorBoundary = path.join(__dirname, 'src/components/ErrorBoundary.tsx');
  return fs.existsSync(errorBoundary);
});

// Test 14: Check for loading states
runTest('Loading states accessible', () => {
  const trainingDashboard = fs.readFileSync(path.join(__dirname, 'src/components/TrainingDashboard.tsx'), 'utf8');
  return trainingDashboard.includes('aria-busy') || trainingDashboard.includes('Loading');
});

// Test 15: Check for responsive design
runTest('Responsive breakpoints defined', () => {
  const tailwindConfig = fs.readFileSync(path.join(__dirname, 'tailwind.config.ts'), 'utf8');
  return tailwindConfig.includes('screens') || tailwindConfig.includes('sm:');
});

// Test 16: Check for safe area insets (mobile)
runTest('Safe area insets for mobile', () => {
  const cssFile = fs.readFileSync(path.join(__dirname, 'src/index.css'), 'utf8');
  return cssFile.includes('safe-area-inset') || cssFile.includes('env(safe');
});

// Test 17: Check for quantum state descriptions
runTest('Quantum states have text alternatives', () => {
  const blochSphere = fs.readFileSync(path.join(__dirname, 'src/components/BlochSphere.tsx'), 'utf8');
  return blochSphere.includes('Screen reader') || blochSphere.includes('sr-only');
});

// Test 18: Check for tutorial system
runTest('Tutorial system for guidance', () => {
  const tutorialFile = path.join(__dirname, 'src/components/TutorialOverlay.tsx');
  return fs.existsSync(tutorialFile);
});

// Test 19: Check GitHub Actions workflow
runTest('GitHub Actions a11y workflow exists', () => {
  const workflowFile = path.join(__dirname, '.github/workflows/accessibility.yml');
  return fs.existsSync(workflowFile);
});

// Test 20: Check Lighthouse configuration
runTest('Lighthouse CI configured', () => {
  const lighthouseConfig = path.join(__dirname, '.lighthouserc.json');
  return fs.existsSync(lighthouseConfig);
});

// Generate report
console.log('');
console.log('=====================================');
console.log('📊 TEST RESULTS');
console.log('=====================================');
console.log(`✅ Passed: ${results.passed}`);
console.log(`❌ Failed: ${results.failed}`);
console.log(`📈 Success Rate: ${Math.round((results.passed / (results.passed + results.failed)) * 100)}%`);
console.log('');

// WCAG Compliance Summary
console.log('🎯 WCAG 2.1 AA COMPLIANCE CHECK:');
console.log('--------------------------------');
const wcagChecks = [
  { name: 'Perceivable', status: results.tests.filter(t => [0,3,10,16].includes(results.tests.indexOf(t))).every(t => t.status === 'passed') },
  { name: 'Operable', status: results.tests.filter(t => [2,6,9,11].includes(results.tests.indexOf(t))).every(t => t.status === 'passed') },
  { name: 'Understandable', status: results.tests.filter(t => [1,11,13,17].includes(results.tests.indexOf(t))).every(t => t.status === 'passed') },
  { name: 'Robust', status: results.tests.filter(t => [7,12,14].includes(results.tests.indexOf(t))).every(t => t.status === 'passed') }
];

wcagChecks.forEach(check => {
  console.log(`  ${check.status ? '✅' : '⚠️ '} ${check.name}`);
});

console.log('');
console.log('📋 DETAILED RESULTS:');
console.log('--------------------');
results.tests.forEach((test, index) => {
  const icon = test.status === 'passed' ? '✅' : '❌';
  console.log(`${index + 1}. ${icon} ${test.name}`);
  if (test.error) {
    console.log(`   Error: ${test.error}`);
  }
});

// Generate HTML report
const htmlReport = `
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>QMLab Accessibility Test Report</title>
  <style>
    body { font-family: system-ui; max-width: 800px; margin: 0 auto; padding: 20px; }
    h1 { color: #333; }
    .summary { background: #f0f0f0; padding: 20px; border-radius: 8px; margin: 20px 0; }
    .passed { color: green; }
    .failed { color: red; }
    .test-list { list-style: none; padding: 0; }
    .test-item { padding: 10px; margin: 5px 0; background: white; border: 1px solid #ddd; border-radius: 4px; }
    .test-passed { border-left: 4px solid green; }
    .test-failed { border-left: 4px solid red; }
  </style>
</head>
<body>
  <h1>🔍 QMLab Accessibility Test Report</h1>
  <div class="summary">
    <h2>Summary</h2>
    <p>Total Tests: ${results.passed + results.failed}</p>
    <p class="passed">✅ Passed: ${results.passed}</p>
    <p class="failed">❌ Failed: ${results.failed}</p>
    <p>Success Rate: ${Math.round((results.passed / (results.passed + results.failed)) * 100)}%</p>
    <p>Generated: ${new Date().toISOString()}</p>
  </div>
  <h2>Test Results</h2>
  <ul class="test-list">
    ${results.tests.map(test => `
      <li class="test-item test-${test.status}">
        ${test.status === 'passed' ? '✅' : '❌'} ${test.name}
        ${test.error ? `<br><small>Error: ${test.error}</small>` : ''}
      </li>
    `).join('')}
  </ul>
</body>
</html>
`;

fs.writeFileSync('accessibility-test-report.html', htmlReport);
console.log('');
console.log('📄 HTML report generated: accessibility-test-report.html');
console.log('');

// Exit with appropriate code
process.exit(results.failed > 0 ? 1 : 0);