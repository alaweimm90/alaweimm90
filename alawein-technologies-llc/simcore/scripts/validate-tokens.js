#!/usr/bin/env node

/**
 * Token System Build Integration
 * Runs validation and generates reports during build process
 */

const { execSync } = require('child_process');
const fs = require('fs').promises;
const path = require('path');

async function validateTokenSystem() {
  console.log('🎨 SimCore Token System Validation');
  console.log('=====================================');
  
  try {
    // Read and validate CSS tokens
    const cssPath = path.join(process.cwd(), 'src/index.css');
    const cssContent = await fs.readFile(cssPath, 'utf-8');
    
    // Validation checks
    const checks = {
      primitiveTokens: 0,
      semanticTokens: 0,
      componentTokens: 0,
      hslColorFormat: 0,
      domainVariables: 0
    };
    
    // Count primitive tokens
    const primitiveMatches = cssContent.match(/--primitive-[^:]+:/g);
    checks.primitiveTokens = primitiveMatches ? primitiveMatches.length : 0;
    
    // Count semantic tokens  
    const semanticMatches = cssContent.match(/--semantic-[^:]+:/g);
    checks.semanticTokens = semanticMatches ? semanticMatches.length : 0;
    
    // Count component tokens
    const componentMatches = cssContent.match(/--component-[^:]+:/g);
    checks.componentTokens = componentMatches ? componentMatches.length : 0;
    
    // Validate HSL format
    const hslMatches = cssContent.match(/hsl\([^)]+\)/g);
    checks.hslColorFormat = hslMatches ? hslMatches.length : 0;
    
    // Count domain variables
    const domainMatches = cssContent.match(/--semantic-domain-[^:]+:/g);
    checks.domainVariables = domainMatches ? domainMatches.length : 0;
    
    console.log('📊 Token System Statistics:');
    console.log(`   • Primitive tokens: ${checks.primitiveTokens}`);
    console.log(`   • Semantic tokens: ${checks.semanticTokens}`);
    console.log(`   • Component tokens: ${checks.componentTokens}`);
    console.log(`   • HSL color values: ${checks.hslColorFormat}`);
    console.log(`   • Domain variables: ${checks.domainVariables}`);
    
    // Validation results
    const validationResults = [];
    
    if (checks.primitiveTokens < 15) {
      validationResults.push('⚠️  Consider adding more primitive color tokens for scalability');
    }
    
    if (checks.semanticTokens < 10) {
      validationResults.push('⚠️  Add more semantic tokens for better abstraction');
    }
    
    if (checks.domainVariables < 4) {
      validationResults.push('❌ Missing physics domain variables');
    } else {
      validationResults.push('✅ Physics domain theming configured');
    }
    
    if (checks.hslColorFormat > 0) {
      validationResults.push('✅ HSL color format detected');
    }
    
    // Check for potential issues
    const invalidTokenRefs = cssContent.match(/var\(--[^)]+\)/g)?.filter(ref => {
      const tokenName = ref.replace('var(', '').replace(')', '');
      return !cssContent.includes(`${tokenName}:`);
    });
    
    if (invalidTokenRefs && invalidTokenRefs.length > 0) {
      validationResults.push(`❌ Found ${invalidTokenRefs.length} invalid token references`);
    } else {
      validationResults.push('✅ All token references valid');
    }
    
    console.log('\n🔍 Validation Results:');
    validationResults.forEach(result => console.log(`   ${result}`));
    
    // Generate report
    const report = {
      timestamp: new Date().toISOString(),
      checks,
      validationResults,
      status: validationResults.some(r => r.startsWith('❌')) ? 'FAILED' : 'PASSED'
    };
    
    await fs.writeFile(
      path.join(process.cwd(), 'token-validation-report.json'),
      JSON.stringify(report, null, 2)
    );
    
    console.log(`\n${report.status === 'PASSED' ? '✅' : '❌'} Token validation ${report.status}`);
    console.log('📄 Report saved to token-validation-report.json');
    
    return report.status === 'PASSED';
    
  } catch (error) {
    console.error('❌ Token validation failed:', error.message);
    return false;
  }
}

// Run validation if called directly
if (require.main === module) {
  validateTokenSystem().then(success => {
    process.exit(success ? 0 : 1);
  });
}

module.exports = { validateTokenSystem };