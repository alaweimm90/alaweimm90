#!/usr/bin/env node

/**
 * REPZ Coach Navigation Audit Script
 * Crawls all pages, identifies orphans, 404s, and design system compliance
 * Enterprise-grade site audit for consistency
 */

import { promises as fs } from 'fs';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import { execSync } from 'child_process';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Configuration
const CONFIG = {
  baseUrl: 'http://localhost:8081',
  outputFile: 'audit-results.json',
  timeout: 10000,
  maxDepth: 3,
  excludePatterns: [
    /\.pdf$/,
    /\.zip$/,
    /\.exe$/,
    /\.(jpg|jpeg|png|gif|svg|webp)$/,
    /\.(css|js|woff|woff2|ttf|eot)$/,
    /^mailto:/,
    /^tel:/,
    /^#/,
    /^javascript:/,
    /^data:/,
    /calendly\.com/,
    /stripe\.com/,
    /external-/
  ],
  requiredGlobalCSS: '/global.css',
  requiredDesignTokens: [
    '--color-brand-primary',
    '--font-heading',
    '--spacing-md',
    '--radius-md'
  ]
};

class NavigationAuditor {
  constructor() {
    this.visitedUrls = new Set();
    this.results = {
      timestamp: new Date().toISOString(),
      summary: {
        totalPages: 0,
        validPages: 0,
        errorPages: 0,
        orphanPages: 0,
        missingDesignSystem: 0
      },
      pages: [],
      orphans: [],
      errors: [],
      designSystemIssues: []
    };
  }

  /**
   * Check if URL should be excluded from crawling
   */
  shouldExclude(url) {
    return CONFIG.excludePatterns.some(pattern => pattern.test(url));
  }

  /**
   * Normalize URL for consistent processing
   */
  normalizeUrl(url, baseUrl) {
    try {
      // Handle relative URLs
      if (url.startsWith('/')) {
        return new URL(url, baseUrl).href;
      }
      if (url.startsWith('./') || url.startsWith('../')) {
        return new URL(url, baseUrl).href;
      }
      if (!url.includes('://')) {
        return new URL(url, baseUrl).href;
      }
      return new URL(url).href;
    } catch (error) {
      console.warn(`Invalid URL: ${url}`, error.message);
      return null;
    }
  }

  /**
   * Extract links from HTML content
   */
  extractLinks(html, currentUrl) {
    const links = [];
    const linkRegex = /<a[^>]+href=["']([^"']+)["'][^>]*>/gi;
    let match;

    while ((match = linkRegex.exec(html)) !== null) {
      const href = match[1];
      const normalizedUrl = this.normalizeUrl(href, currentUrl);
      
      if (normalizedUrl && 
          normalizedUrl.startsWith(CONFIG.baseUrl) && 
          !this.shouldExclude(normalizedUrl)) {
        links.push(normalizedUrl);
      }
    }

    return [...new Set(links)]; // Remove duplicates
  }

  /**
   * Check if page includes global CSS
   */
  hasGlobalCSS(html) {
    const cssLinkRegex = /<link[^>]+href=["']([^"']*global\.css[^"']*)["'][^>]*>/i;
    return cssLinkRegex.test(html);
  }

  /**
   * Check if page uses design tokens
   */
  hasDesignTokens(html) {
    const foundTokens = [];
    CONFIG.requiredDesignTokens.forEach(token => {
      if (html.includes(token) || html.includes(`var(${token})`)) {
        foundTokens.push(token);
      }
    });
    return foundTokens;
  }

  /**
   * Fetch and analyze a single page
   */
  async analyzePage(url, depth = 0) {
    if (this.visitedUrls.has(url) || depth > CONFIG.maxDepth) {
      return [];
    }

    this.visitedUrls.add(url);
    console.log(`Analyzing: ${url} (depth: ${depth})`);

    const pageResult = {
      url,
      depth,
      status: null,
      statusText: null,
      title: null,
      hasGlobalCSS: false,
      designTokens: [],
      links: [],
      loadTime: 0,
      error: null
    };

    const startTime = Date.now();

    try {
      // Use a simple HTTP request approach since we're testing a local server
      const response = await this.fetchWithTimeout(url);
      pageResult.status = response.status;
      pageResult.statusText = response.statusText;
      pageResult.loadTime = Date.now() - startTime;

      if (response.ok) {
        const html = await response.text();
        
        // Extract page title
        const titleMatch = html.match(/<title[^>]*>([^<]+)<\/title>/i);
        pageResult.title = titleMatch ? titleMatch[1].trim() : 'No title';

        // Check for global CSS
        pageResult.hasGlobalCSS = this.hasGlobalCSS(html);

        // Check for design tokens
        pageResult.designTokens = this.hasDesignTokens(html);

        // Extract links for further crawling
        pageResult.links = this.extractLinks(html, url);

        this.results.summary.validPages++;
      } else {
        this.results.summary.errorPages++;
        this.results.errors.push({
          url,
          status: response.status,
          statusText: response.statusText
        });
      }
    } catch (error) {
      pageResult.error = error.message;
      this.results.summary.errorPages++;
      this.results.errors.push({
        url,
        error: error.message
      });
    }

    this.results.pages.push(pageResult);
    this.results.summary.totalPages++;

    // Track design system compliance issues
    if (!pageResult.hasGlobalCSS || pageResult.designTokens.length === 0) {
      this.results.summary.missingDesignSystem++;
      this.results.designSystemIssues.push({
        url,
        issues: {
          missingGlobalCSS: !pageResult.hasGlobalCSS,
          missingTokens: CONFIG.requiredDesignTokens.filter(
            token => !pageResult.designTokens.includes(token)
          )
        }
      });
    }

    return pageResult.links;
  }

  /**
   * Fetch with timeout support
   */
  async fetchWithTimeout(url) {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), CONFIG.timeout);

    try {
      const response = await fetch(url, {
        signal: controller.signal,
        headers: {
          'User-Agent': 'REPZ-Audit-Bot/1.0'
        }
      });
      clearTimeout(timeoutId);
      return response;
    } catch (error) {
      clearTimeout(timeoutId);
      throw error;
    }
  }

  /**
   * Crawl all pages starting from the root
   */
  async crawlSite() {
    console.log(`🚀 Starting site audit: ${CONFIG.baseUrl}`);
    console.log(`📝 Output file: ${CONFIG.outputFile}`);
    console.log(`⚙️  Max depth: ${CONFIG.maxDepth}`);
    
    const queue = [{ url: CONFIG.baseUrl, depth: 0 }];
    const processedUrls = new Set();

    while (queue.length > 0) {
      const { url, depth } = queue.shift();

      if (processedUrls.has(url)) continue;
      processedUrls.add(url);

      try {
        const links = await this.analyzePage(url, depth);
        
        // Add new links to queue for next depth level
        if (depth < CONFIG.maxDepth) {
          links.forEach(link => {
            if (!processedUrls.has(link)) {
              queue.push({ url: link, depth: depth + 1 });
            }
          });
        }
      } catch (error) {
        console.error(`Error processing ${url}:`, error.message);
      }
    }
  }

  /**
   * Identify orphan pages by scanning file system
   */
  async findOrphanPages() {
    console.log('\n🔍 Scanning for orphan pages...');
    
    try {
      // Get all React/HTML files that could be pages
      const srcFiles = await this.getAllPageFiles();
      const linkedUrls = new Set(this.results.pages.map(p => p.url));
      
      // Check for pages that exist in code but weren't found via crawling
      const potentialOrphans = srcFiles.filter(file => {
        const possibleUrls = this.fileToUrlPossibilities(file);
        return !possibleUrls.some(url => linkedUrls.has(url));
      });

      this.results.orphans = potentialOrphans.map(file => ({
        file,
        possibleUrls: this.fileToUrlPossibilities(file),
        reason: 'Not reachable via navigation'
      }));

      this.results.summary.orphanPages = this.results.orphans.length;
    } catch (error) {
      console.warn('Could not scan for orphan pages:', error.message);
    }
  }

  /**
   * Get all potential page files
   */
  async getAllPageFiles() {
    const files = [];
    const directories = ['src/pages', 'src/components'];
    
    for (const dir of directories) {
      try {
        const dirPath = join(__dirname, dir);
        await this.scanDirectory(dirPath, files);
      } catch (error) {
        // Directory might not exist, skip
      }
    }
    
    return files;
  }

  /**
   * Recursively scan directory for files
   */
  async scanDirectory(dirPath, files, basePath = '') {
    try {
      const entries = await fs.readdir(dirPath, { withFileTypes: true });
      
      for (const entry of entries) {
        const fullPath = join(dirPath, entry.name);
        const relativePath = join(basePath, entry.name);
        
        if (entry.isDirectory()) {
          await this.scanDirectory(fullPath, files, relativePath);
        } else if (entry.name.match(/\.(tsx?|jsx?)$/)) {
          files.push(relativePath);
        }
      }
    } catch (error) {
      // Ignore errors for individual directories
    }
  }

  /**
   * Convert file path to possible URL paths
   */
  fileToUrlPossibilities(filePath) {
    const urls = [];
    const baseName = filePath.replace(/\.(tsx?|jsx?)$/, '');
    
    // Common React routing patterns
    if (baseName.includes('pages/')) {
      const routePath = baseName.split('pages/')[1];
      urls.push(`${CONFIG.baseUrl}/${routePath.toLowerCase()}`);
      
      if (routePath === 'Index' || routePath === 'index') {
        urls.push(CONFIG.baseUrl);
      }
    }
    
    return urls;
  }

  /**
   * Generate comprehensive report
   */
  generateReport() {
    const report = {
      ...this.results,
      recommendations: this.generateRecommendations()
    };

    return report;
  }

  /**
   * Generate actionable recommendations
   */
  generateRecommendations() {
    const recommendations = [];

    if (this.results.summary.errorPages > 0) {
      recommendations.push({
        priority: 'HIGH',
        category: 'Broken Links',
        issue: `Found ${this.results.summary.errorPages} pages with errors`,
        action: 'Fix broken links and server errors',
        pages: this.results.errors.slice(0, 5) // Show first 5
      });
    }

    if (this.results.summary.missingDesignSystem > 0) {
      recommendations.push({
        priority: 'HIGH',
        category: 'Design System',
        issue: `${this.results.summary.missingDesignSystem} pages missing design tokens`,
        action: `Add <link rel="stylesheet" href="${CONFIG.requiredGlobalCSS}"> to all pages`,
        pages: this.results.designSystemIssues.slice(0, 5)
      });
    }

    if (this.results.summary.orphanPages > 0) {
      recommendations.push({
        priority: 'MEDIUM',
        category: 'Navigation',
        issue: `Found ${this.results.summary.orphanPages} potential orphan pages`,
        action: 'Add navigation links or remove unused pages',
        pages: this.results.orphans.slice(0, 3)
      });
    }

    return recommendations;
  }

  /**
   * Save results to file
   */
  async saveResults() {
    const report = this.generateReport();
    await fs.writeFile(CONFIG.outputFile, JSON.stringify(report, null, 2));
    console.log(`\n📊 Audit results saved to: ${CONFIG.outputFile}`);
    return report;
  }

  /**
   * Print summary to console
   */
  printSummary() {
    const { summary } = this.results;
    
    console.log('\n📋 AUDIT SUMMARY');
    console.log('================');
    console.log(`Total Pages Crawled: ${summary.totalPages}`);
    console.log(`✅ Valid Pages: ${summary.validPages}`);
    console.log(`❌ Error Pages: ${summary.errorPages}`);
    console.log(`🏚️  Orphan Pages: ${summary.orphanPages}`);
    console.log(`🎨 Missing Design System: ${summary.missingDesignSystem}`);
    
    if (summary.errorPages > 0) {
      console.log('\n❌ ERRORS FOUND:');
      this.results.errors.slice(0, 5).forEach(error => {
        console.log(`   • ${error.url} - ${error.status || error.error}`);
      });
    }
    
    if (summary.missingDesignSystem > 0) {
      console.log('\n🎨 DESIGN SYSTEM ISSUES:');
      this.results.designSystemIssues.slice(0, 3).forEach(issue => {
        console.log(`   • ${issue.url}`);
        if (issue.issues.missingGlobalCSS) {
          console.log(`     - Missing global.css`);
        }
        if (issue.issues.missingTokens.length > 0) {
          console.log(`     - Missing tokens: ${issue.issues.missingTokens.join(', ')}`);
        }
      });
    }
  }
}

/**
 * Check if dev server is running
 */
async function checkDevServer() {
  try {
    const response = await fetch(CONFIG.baseUrl, { 
      signal: AbortSignal.timeout(5000) 
    });
    return response.ok;
  } catch (error) {
    return false;
  }
}

/**
 * Main execution function
 */
async function main() {
  console.log('🔍 REPZ Coach Navigation Audit');
  console.log('===============================');
  
  // Check if dev server is running
  const serverRunning = await checkDevServer();
  if (!serverRunning) {
    console.error('❌ Development server not running!');
    console.log('Please start the server with: npm run dev');
    process.exit(1);
  }

  const auditor = new NavigationAuditor();
  
  try {
    // Crawl the site
    await auditor.crawlSite();
    
    // Find orphan pages
    await auditor.findOrphanPages();
    
    // Save results
    const report = await auditor.saveResults();
    
    // Print summary
    auditor.printSummary();
    
    console.log('\n✅ Audit completed successfully!');
    console.log(`📄 Full report: ${CONFIG.outputFile}`);
    
    // Exit with error code if issues found
    const hasIssues = report.summary.errorPages > 0 || report.summary.missingDesignSystem > 0;
    process.exit(hasIssues ? 1 : 0);
    
  } catch (error) {
    console.error('💥 Audit failed:', error.message);
    process.exit(1);
  }
}

// Run the audit
if (import.meta.url === `file://${process.argv[1]}`) {
  main();
}