import { chromium, Browser, Page } from 'playwright';
import * as fs from 'fs';
import * as path from 'path';

interface PerformanceMetrics {
  FCP: number; // First Contentful Paint
  LCP: number; // Largest Contentful Paint
  FID: number; // First Input Delay
  CLS: number; // Cumulative Layout Shift
  TTFB: number; // Time to First Byte
  TTI: number; // Time to Interactive
  TBT: number; // Total Blocking Time
  memoryUsage: number;
  bundleSize: number;
  requestCount: number;
  transferSize: number;
  domNodes: number;
  jsHeapSize: number;
}

interface PerformanceResult {
  app: string;
  url: string;
  timestamp: string;
  metrics: PerformanceMetrics;
  score: number;
  recommendations: string[];
  passed: boolean;
}

class PerformanceMonitor {
  private browser: Browser | null = null;
  private results: PerformanceResult[] = [];

  async initialize(): Promise<void> {
    this.browser = await chromium.launch({
      headless: true,
      args: [
        '--no-sandbox',
        '--disable-setuid-sandbox',
        '--disable-dev-shm-usage',
        '--enable-precise-memory-info',
      ],
    });
  }

  async measurePerformance(url: string, appName: string): Promise<PerformanceResult> {
    if (!this.browser) throw new Error('Browser not initialized');

    const page = await this.browser.newPage();
    const metrics: Partial<PerformanceMetrics> = {};
    const recommendations: string[] = [];

    try {
      // Enable performance monitoring
      await page.route('**/*', (route) => {
        const request = route.request();
        const url = request.url();

        // Block unnecessary resources to improve performance
        if (
          url.includes('google-analytics') ||
          url.includes('facebook') ||
          url.includes('doubleclick')
        ) {
          route.abort();
        } else {
          route.continue();
        }
      });

      // Measure network activity
      let requestCount = 0;
      let transferSize = 0;

      page.on('request', () => requestCount++);
      page.on('response', (response) => {
        const headers = response.headers();
        if (headers['content-length']) {
          transferSize += parseInt(headers['content-length']);
        }
      });

      // Navigate and measure
      const startTime = Date.now();
      await page.goto(url, { waitUntil: 'networkidle' });
      const loadTime = Date.now() - startTime;

      // Get Core Web Vitals
      const webVitals = await page.evaluate(() => {
        return new Promise<any>((resolve) => {
          const observer = new PerformanceObserver((list) => {
            const entries = list.getEntries();
            const vitals: any = {};

            entries.forEach((entry) => {
              if (entry.entryType === 'largest-contentful-paint') {
                vitals.LCP = entry.startTime;
              } else if (entry.entryType === 'first-input') {
                vitals.FID = (entry as any).processingStart - entry.startTime;
              } else if (entry.entryType === 'layout-shift') {
                vitals.CLS = (vitals.CLS || 0) + (entry as any).value;
              }
            });

            resolve(vitals);
          });

          observer.observe({ entryTypes: ['largest-contentful-paint', 'first-input', 'layout-shift'] });

          // Trigger after some time to collect metrics
          setTimeout(() => {
            observer.disconnect();
            resolve({});
          }, 5000);
        });
      });

      // Get additional performance metrics
      const performanceData = await page.evaluate(() => {
        const perfData = performance.getEntriesByType('navigation')[0] as PerformanceNavigationTiming;
        const paintEntries = performance.getEntriesByType('paint');
        const fcpEntry = paintEntries.find((entry) => entry.name === 'first-contentful-paint');

        return {
          FCP: fcpEntry ? fcpEntry.startTime : 0,
          TTFB: perfData.responseStart - perfData.fetchStart,
          domContentLoaded: perfData.domContentLoadedEventEnd - perfData.domContentLoadedEventStart,
          loadComplete: perfData.loadEventEnd - perfData.loadEventStart,
        };
      });

      // Get memory usage
      const memoryData = await page.evaluate(() => {
        if ((performance as any).memory) {
          return {
            jsHeapSize: (performance as any).memory.usedJSHeapSize,
            totalJSHeapSize: (performance as any).memory.totalJSHeapSize,
          };
        }
        return { jsHeapSize: 0, totalJSHeapSize: 0 };
      });

      // Get DOM complexity
      const domData = await page.evaluate(() => {
        return {
          domNodes: document.querySelectorAll('*').length,
          images: document.querySelectorAll('img').length,
          scripts: document.querySelectorAll('script').length,
          stylesheets: document.querySelectorAll('link[rel="stylesheet"]').length,
        };
      });

      // Calculate Time to Interactive (simplified)
      const tti = await page.evaluate(() => {
        return new Promise<number>((resolve) => {
          let lastLongTaskTime = 0;
          const observer = new PerformanceObserver((list) => {
            const entries = list.getEntries();
            entries.forEach((entry) => {
              if (entry.duration > 50) {
                lastLongTaskTime = entry.startTime + entry.duration;
              }
            });
          });

          observer.observe({ entryTypes: ['longtask'] });

          setTimeout(() => {
            observer.disconnect();
            resolve(lastLongTaskTime || performanceData.FCP || 0);
          }, 5000);
        });
      });

      // Compile metrics
      metrics.FCP = performanceData.FCP;
      metrics.LCP = webVitals.LCP || loadTime;
      metrics.FID = webVitals.FID || 0;
      metrics.CLS = webVitals.CLS || 0;
      metrics.TTFB = performanceData.TTFB;
      metrics.TTI = tti;
      metrics.TBT = 0; // Would need more complex calculation
      metrics.memoryUsage = memoryData.jsHeapSize / 1024 / 1024; // Convert to MB
      metrics.requestCount = requestCount;
      metrics.transferSize = transferSize / 1024; // Convert to KB
      metrics.domNodes = domData.domNodes;
      metrics.jsHeapSize = memoryData.jsHeapSize;

      // Generate recommendations
      if (metrics.LCP! > 2500) {
        recommendations.push('Optimize Largest Contentful Paint (LCP) - currently over 2.5s');
      }
      if (metrics.FID! > 100) {
        recommendations.push('Reduce First Input Delay (FID) - currently over 100ms');
      }
      if (metrics.CLS! > 0.1) {
        recommendations.push('Fix layout shifts - CLS is above 0.1');
      }
      if (metrics.TTFB! > 600) {
        recommendations.push('Improve server response time - TTFB is over 600ms');
      }
      if (metrics.requestCount! > 50) {
        recommendations.push(`Reduce number of requests - currently ${metrics.requestCount}`);
      }
      if (metrics.transferSize! > 500) {
        recommendations.push(`Optimize bundle size - currently ${Math.round(metrics.transferSize!)}KB`);
      }
      if (metrics.domNodes! > 1500) {
        recommendations.push(`Reduce DOM complexity - ${metrics.domNodes} nodes found`);
      }
      if (metrics.memoryUsage! > 50) {
        recommendations.push(`High memory usage detected - ${Math.round(metrics.memoryUsage!)}MB`);
      }

      // Calculate score
      const score = this.calculateScore(metrics as PerformanceMetrics);
      const passed = score >= 90;

      return {
        app: appName,
        url,
        timestamp: new Date().toISOString(),
        metrics: metrics as PerformanceMetrics,
        score,
        recommendations,
        passed,
      };
    } finally {
      await page.close();
    }
  }

  private calculateScore(metrics: PerformanceMetrics): number {
    let score = 100;

    // LCP scoring (0-2.5s = 100, 2.5-4s = linear decrease, >4s = 0)
    if (metrics.LCP <= 2500) {
      // Good
    } else if (metrics.LCP <= 4000) {
      score -= ((metrics.LCP - 2500) / 1500) * 25;
    } else {
      score -= 25;
    }

    // FID scoring (0-100ms = 100, 100-300ms = linear decrease, >300ms = 0)
    if (metrics.FID <= 100) {
      // Good
    } else if (metrics.FID <= 300) {
      score -= ((metrics.FID - 100) / 200) * 25;
    } else {
      score -= 25;
    }

    // CLS scoring (0-0.1 = 100, 0.1-0.25 = linear decrease, >0.25 = 0)
    if (metrics.CLS <= 0.1) {
      // Good
    } else if (metrics.CLS <= 0.25) {
      score -= ((metrics.CLS - 0.1) / 0.15) * 25;
    } else {
      score -= 25;
    }

    // TTFB scoring
    if (metrics.TTFB > 600) {
      score -= 10;
    }

    // Bundle size scoring
    if (metrics.transferSize > 1000) {
      score -= 10;
    } else if (metrics.transferSize > 500) {
      score -= 5;
    }

    return Math.max(0, Math.round(score));
  }

  async runPerformanceAudit(apps: string[], baseUrl: string): Promise<void> {
    await this.initialize();

    try {
      for (const app of apps) {
        console.log(`Testing performance for ${app}...`);

        const urls = [
          `${baseUrl}`,
          `${baseUrl}/dashboard`,
          `${baseUrl}/projects`,
        ];

        for (const url of urls) {
          console.log(`  Measuring ${url}...`);
          const result = await this.measurePerformance(url, app);
          this.results.push(result);

          // Run multiple times for consistency
          for (let i = 0; i < 2; i++) {
            await new Promise(resolve => setTimeout(resolve, 1000));
            const additionalResult = await this.measurePerformance(url, app);
            // Average the metrics
            Object.keys(result.metrics).forEach((key) => {
              (result.metrics as any)[key] =
                ((result.metrics as any)[key] + (additionalResult.metrics as any)[key]) / 2;
            });
          }
        }
      }

      await this.generateReport();
    } finally {
      if (this.browser) {
        await this.browser.close();
      }
    }
  }

  async generateReport(): Promise<void> {
    const reportDir = path.join('/home/user/CrazyIdeas/testing/performance/reports');
    if (!fs.existsSync(reportDir)) {
      fs.mkdirSync(reportDir, { recursive: true });
    }

    // Generate JSON report
    const jsonReport = {
      timestamp: new Date().toISOString(),
      results: this.results,
      summary: {
        totalTests: this.results.length,
        passed: this.results.filter((r) => r.passed).length,
        averageScore: Math.round(
          this.results.reduce((sum, r) => sum + r.score, 0) / this.results.length
        ),
        averageMetrics: this.calculateAverageMetrics(),
      },
    };

    fs.writeFileSync(
      path.join(reportDir, 'performance-report.json'),
      JSON.stringify(jsonReport, null, 2)
    );

    // Generate HTML report
    const html = this.generateHTMLReport(jsonReport);
    fs.writeFileSync(path.join(reportDir, 'performance-report.html'), html);

    console.log(`\nPerformance audit complete!`);
    console.log(`Reports generated in: ${reportDir}`);
    console.log(`Average score: ${jsonReport.summary.averageScore}/100`);
  }

  private calculateAverageMetrics(): PerformanceMetrics {
    const totals = this.results.reduce(
      (acc, r) => {
        Object.keys(r.metrics).forEach((key) => {
          acc[key] = (acc[key] || 0) + (r.metrics as any)[key];
        });
        return acc;
      },
      {} as any
    );

    const count = this.results.length;
    Object.keys(totals).forEach((key) => {
      totals[key] = Math.round(totals[key] / count);
    });

    return totals;
  }

  private generateHTMLReport(report: any): string {
    return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Performance Audit Report</title>
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; padding: 20px; background: #f5f5f5; }
    .container { max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; }
    h1 { color: #1a1a1a; margin-bottom: 10px; }
    h2 { color: #333; margin: 30px 0 15px; border-bottom: 2px solid #e0e0e0; padding-bottom: 5px; }
    .timestamp { color: #666; margin-bottom: 30px; }
    .metric-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 15px; margin: 20px 0; }
    .metric-card { background: #f8f9fa; padding: 15px; border-radius: 8px; text-align: center; }
    .metric-value { font-size: 24px; font-weight: bold; margin: 5px 0; }
    .metric-label { color: #666; font-size: 12px; text-transform: uppercase; }
    .score { display: inline-block; padding: 5px 10px; border-radius: 20px; font-weight: bold; }
    .score.good { background: #10b981; color: white; }
    .score.medium { background: #f59e0b; color: white; }
    .score.poor { background: #ef4444; color: white; }
    .good-metric { color: #10b981; }
    .medium-metric { color: #f59e0b; }
    .poor-metric { color: #ef4444; }
    .recommendations { background: #fff3cd; padding: 15px; margin: 15px 0; border-radius: 8px; border-left: 4px solid #ffc107; }
    .recommendations ul { margin: 10px 0 0 20px; }
    table { width: 100%; border-collapse: collapse; margin: 20px 0; }
    th, td { padding: 12px; text-align: left; border-bottom: 1px solid #e0e0e0; }
    th { background: #f5f5f5; font-weight: 600; }
    .chart { margin: 20px 0; height: 200px; background: #f8f9fa; border-radius: 8px; display: flex; align-items: center; justify-content: center; color: #999; }
  </style>
</head>
<body>
  <div class="container">
    <h1>⚡ Performance Audit Report</h1>
    <div class="timestamp">Generated: ${report.timestamp}</div>

    <h2>Summary</h2>
    <div class="metric-grid">
      <div class="metric-card">
        <div class="metric-label">Average Score</div>
        <div class="metric-value ${
          report.summary.averageScore >= 90
            ? 'good-metric'
            : report.summary.averageScore >= 70
            ? 'medium-metric'
            : 'poor-metric'
        }">${report.summary.averageScore}</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Tests Passed</div>
        <div class="metric-value">${report.summary.passed}/${report.summary.totalTests}</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Avg LCP</div>
        <div class="metric-value ${
          report.summary.averageMetrics.LCP <= 2500
            ? 'good-metric'
            : report.summary.averageMetrics.LCP <= 4000
            ? 'medium-metric'
            : 'poor-metric'
        }">${report.summary.averageMetrics.LCP}ms</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Avg FID</div>
        <div class="metric-value ${
          report.summary.averageMetrics.FID <= 100
            ? 'good-metric'
            : report.summary.averageMetrics.FID <= 300
            ? 'medium-metric'
            : 'poor-metric'
        }">${report.summary.averageMetrics.FID}ms</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Avg CLS</div>
        <div class="metric-value ${
          report.summary.averageMetrics.CLS <= 0.1
            ? 'good-metric'
            : report.summary.averageMetrics.CLS <= 0.25
            ? 'medium-metric'
            : 'poor-metric'
        }">${report.summary.averageMetrics.CLS.toFixed(3)}</div>
      </div>
    </div>

    <h2>Detailed Results</h2>
    <table>
      <thead>
        <tr>
          <th>App</th>
          <th>URL</th>
          <th>Score</th>
          <th>LCP</th>
          <th>FID</th>
          <th>CLS</th>
          <th>TTFB</th>
          <th>Requests</th>
          <th>Size (KB)</th>
        </tr>
      </thead>
      <tbody>
        ${report.results
          .map(
            (r: PerformanceResult) => `
          <tr>
            <td>${r.app}</td>
            <td>${r.url}</td>
            <td><span class="score ${r.score >= 90 ? 'good' : r.score >= 70 ? 'medium' : 'poor'}">${r.score}</span></td>
            <td class="${r.metrics.LCP <= 2500 ? 'good-metric' : r.metrics.LCP <= 4000 ? 'medium-metric' : 'poor-metric'}">${Math.round(r.metrics.LCP)}ms</td>
            <td class="${r.metrics.FID <= 100 ? 'good-metric' : r.metrics.FID <= 300 ? 'medium-metric' : 'poor-metric'}">${Math.round(r.metrics.FID)}ms</td>
            <td class="${r.metrics.CLS <= 0.1 ? 'good-metric' : r.metrics.CLS <= 0.25 ? 'medium-metric' : 'poor-metric'}">${r.metrics.CLS.toFixed(3)}</td>
            <td class="${r.metrics.TTFB <= 600 ? 'good-metric' : 'poor-metric'}">${Math.round(r.metrics.TTFB)}ms</td>
            <td>${r.metrics.requestCount}</td>
            <td>${Math.round(r.metrics.transferSize)}</td>
          </tr>
        `
          )
          .join('')}
      </tbody>
    </table>

    <h2>Recommendations</h2>
    ${report.results
      .filter((r: PerformanceResult) => r.recommendations.length > 0)
      .map(
        (r: PerformanceResult) => `
      <div class="recommendations">
        <strong>${r.app} - ${r.url}</strong>
        <ul>
          ${r.recommendations.map((rec: string) => `<li>${rec}</li>`).join('')}
        </ul>
      </div>
    `
      )
      .join('')}

    <h2>Core Web Vitals Compliance</h2>
    <div class="chart">
      <span>Chart placeholder - Use actual charting library in production</span>
    </div>

    <h2>Optimization Priorities</h2>
    <ol style="margin: 20px 0; padding-left: 20px;">
      <li>Optimize images - Use WebP format and lazy loading</li>
      <li>Reduce JavaScript bundle size - Code split and tree shake</li>
      <li>Implement effective caching strategies</li>
      <li>Minimize main thread work</li>
      <li>Reduce server response times (TTFB)</li>
      <li>Eliminate render-blocking resources</li>
      <li>Preconnect to required origins</li>
    </ol>
  </div>
</body>
</html>`;
  }
}

// Export for use in CI/CD
export { PerformanceMonitor, PerformanceResult, PerformanceMetrics };

// Run if called directly
if (require.main === module) {
  const monitor = new PerformanceMonitor();
  const apps = ['ghost-researcher', 'scientific-tinder', 'chaos-engine'];
  const baseUrl = process.env.BASE_URL || 'http://localhost:3000';

  monitor.runPerformanceAudit(apps, baseUrl).catch(console.error);
}