import { test, expect } from '@playwright/test';

test.describe('Ghost Researcher - E2E Tests', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('http://localhost:3001');

    // Login if needed
    if (await page.locator('[data-testid="login-form"]').isVisible()) {
      await page.fill('[name="email"]', 'test@example.com');
      await page.fill('[name="password"]', 'testpassword');
      await page.click('[type="submit"]');
      await page.waitForURL('**/dashboard');
    }
  });

  test.describe('Research Board Workflow', () => {
    test('should create a new research project', async ({ page }) => {
      // Navigate to projects
      await page.click('[data-testid="nav-projects"]');

      // Create new project
      await page.click('[data-testid="create-project-btn"]');
      await page.fill('[name="title"]', 'Quantum Computing Research');
      await page.fill('[name="description"]', 'Exploring quantum algorithms for optimization');
      await page.selectOption('[name="visibility"]', 'team');

      // Add tags
      await page.fill('[data-testid="tag-input"]', 'quantum');
      await page.press('[data-testid="tag-input"]', 'Enter');
      await page.fill('[data-testid="tag-input"]', 'optimization');
      await page.press('[data-testid="tag-input"]', 'Enter');

      await page.click('[data-testid="submit-project"]');

      // Verify creation
      await expect(page).toHaveURL(/.*\/project\/.*/);
      await expect(page.locator('h1')).toContainText('Quantum Computing Research');
    });

    test('should add hypothesis to project', async ({ page }) => {
      // Navigate to existing project
      await page.goto('http://localhost:3001/project/test-project-id');

      // Add hypothesis
      await page.click('[data-testid="add-hypothesis-btn"]');
      await page.fill('[name="hypothesisTitle"]', 'Quantum supremacy in optimization');
      await page.fill('[name="hypothesisDescription"]', 'Quantum computers can solve NP-hard problems exponentially faster');
      await page.click('[data-testid="submit-hypothesis"]');

      // Verify hypothesis appears
      await expect(page.locator('[data-testid="hypothesis-card"]').first()).toContainText('Quantum supremacy');

      // Test real-time update
      const hypothesisCard = page.locator('[data-testid="hypothesis-card"]').first();
      await expect(hypothesisCard).toHaveAttribute('data-status', 'proposed');
    });

    test('should enable real-time collaboration', async ({ page, context }) => {
      // Open second browser tab
      const page2 = await context.newPage();
      await page2.goto('http://localhost:3001/project/test-project-id');

      // Login as different user
      await page2.fill('[name="email"]', 'collaborator@example.com');
      await page2.fill('[name="password"]', 'testpassword');
      await page2.click('[type="submit"]');

      // User 1 adds comment
      await page.click('[data-testid="hypothesis-card"]').first();
      await page.fill('[data-testid="comment-input"]', 'This looks promising!');
      await page.click('[data-testid="post-comment"]');

      // Verify comment appears in real-time for User 2
      await expect(page2.locator('[data-testid="comment"]').first()).toContainText('This looks promising!');

      // Check presence indicators
      await expect(page.locator('[data-testid="online-users"]')).toContainText('2 users online');
      await expect(page2.locator('[data-testid="online-users"]')).toContainText('2 users online');

      await page2.close();
    });

    test('should upload and analyze research papers', async ({ page }) => {
      await page.goto('http://localhost:3001/project/test-project-id');

      // Upload paper
      const [fileChooser] = await Promise.all([
        page.waitForEvent('filechooser'),
        page.click('[data-testid="upload-paper-btn"]')
      ]);

      await fileChooser.setFiles('./test-fixtures/sample-paper.pdf');

      // Wait for processing
      await expect(page.locator('[data-testid="processing-indicator"]')).toBeVisible();
      await expect(page.locator('[data-testid="processing-indicator"]')).toBeHidden({ timeout: 30000 });

      // Verify paper analysis
      await expect(page.locator('[data-testid="paper-card"]').first()).toContainText('sample-paper.pdf');
      await page.click('[data-testid="view-analysis"]');

      await expect(page.locator('[data-testid="paper-insights"]')).toContainText('Key Findings');
      await expect(page.locator('[data-testid="paper-insights"]')).toContainText('Citations');
    });
  });

  test.describe('Literature Search', () => {
    test('should search and filter papers', async ({ page }) => {
      await page.goto('http://localhost:3001/literature');

      // Perform search
      await page.fill('[data-testid="search-input"]', 'quantum computing machine learning');
      await page.click('[data-testid="search-btn"]');

      // Wait for results
      await expect(page.locator('[data-testid="search-results"]')).toBeVisible();
      await expect(page.locator('[data-testid="result-item"]')).toHaveCount(10);

      // Apply filters
      await page.click('[data-testid="filter-year"]');
      await page.selectOption('[data-testid="year-range"]', '2020-2024');

      await page.click('[data-testid="filter-journal"]');
      await page.check('[value="Nature"]');
      await page.check('[value="Science"]');

      await page.click('[data-testid="apply-filters"]');

      // Verify filtered results
      await expect(page.locator('[data-testid="result-count"]')).toContainText('Showing');
      const firstResult = page.locator('[data-testid="result-item"]').first();
      await expect(firstResult).toContainText('2020', '2021', '2022', '2023', '2024');
    });

    test('should save papers to project', async ({ page }) => {
      await page.goto('http://localhost:3001/literature');

      // Search for papers
      await page.fill('[data-testid="search-input"]', 'quantum algorithms');
      await page.click('[data-testid="search-btn"]');

      // Select papers
      await page.check('[data-testid="paper-checkbox-1"]');
      await page.check('[data-testid="paper-checkbox-2"]');

      // Add to project
      await page.click('[data-testid="add-to-project-btn"]');
      await page.selectOption('[data-testid="project-select"]', 'test-project-id');
      await page.click('[data-testid="confirm-add"]');

      // Verify papers added
      await page.goto('http://localhost:3001/project/test-project-id');
      await expect(page.locator('[data-testid="papers-section"]')).toContainText('2 papers');
    });
  });

  test.describe('Analytics Dashboard', () => {
    test('should display research metrics', async ({ page }) => {
      await page.goto('http://localhost:3001/project/test-project-id/analytics');

      // Check metrics display
      await expect(page.locator('[data-testid="hypothesis-count"]')).toBeVisible();
      await expect(page.locator('[data-testid="paper-count"]')).toBeVisible();
      await expect(page.locator('[data-testid="collaboration-score"]')).toBeVisible();

      // Verify charts render
      await expect(page.locator('[data-testid="progress-chart"]')).toBeVisible();
      await expect(page.locator('[data-testid="validation-chart"]')).toBeVisible();

      // Test date range filter
      await page.selectOption('[data-testid="date-range"]', 'last-30-days');
      await expect(page.locator('[data-testid="progress-chart"]')).toHaveAttribute('data-range', 'last-30-days');
    });

    test('should export analytics report', async ({ page }) => {
      await page.goto('http://localhost:3001/project/test-project-id/analytics');

      // Configure export
      await page.click('[data-testid="export-report-btn"]');
      await page.check('[value="hypothesis-summary"]');
      await page.check('[value="collaboration-metrics"]');
      await page.check('[value="paper-analysis"]');

      // Download report
      const [download] = await Promise.all([
        page.waitForEvent('download'),
        page.click('[data-testid="download-report"]')
      ]);

      expect(download.suggestedFilename()).toContain('research-report');
      expect(download.suggestedFilename()).toContain('.pdf');
    });
  });

  test.describe('Accessibility Tests', () => {
    test('should be navigable with keyboard only', async ({ page }) => {
      await page.goto('http://localhost:3001/dashboard');

      // Tab through main navigation
      await page.keyboard.press('Tab');
      await expect(page.locator(':focus')).toHaveAttribute('data-testid', 'nav-dashboard');

      await page.keyboard.press('Tab');
      await expect(page.locator(':focus')).toHaveAttribute('data-testid', 'nav-projects');

      // Navigate to projects using Enter
      await page.keyboard.press('Enter');
      await expect(page).toHaveURL(/.*\/projects/);

      // Tab to create button and activate
      await page.keyboard.press('Tab');
      await page.keyboard.press('Tab');
      await expect(page.locator(':focus')).toHaveAttribute('data-testid', 'create-project-btn');
      await page.keyboard.press('Enter');

      // Verify modal opened
      await expect(page.locator('[role="dialog"]')).toBeVisible();

      // Close with Escape
      await page.keyboard.press('Escape');
      await expect(page.locator('[role="dialog"]')).toBeHidden();
    });

    test('should work with screen reader', async ({ page }) => {
      // Enable screen reader mode
      await page.addInitScript(() => {
        window.localStorage.setItem('screenReaderMode', 'true');
      });

      await page.goto('http://localhost:3001/dashboard');

      // Check ARIA labels
      await expect(page.locator('[role="main"]')).toHaveAttribute('aria-label', 'Research Dashboard');
      await expect(page.locator('[role="navigation"]')).toHaveAttribute('aria-label', 'Main Navigation');

      // Check announcements
      await page.click('[data-testid="create-project-btn"]');
      await expect(page.locator('[role="status"]')).toContainText('Create new project dialog opened');

      await page.keyboard.press('Escape');
      await expect(page.locator('[role="status"]')).toContainText('Dialog closed');
    });

    test('should respect prefers-reduced-motion', async ({ page }) => {
      // Set reduced motion preference
      await page.emulateMedia({ reducedMotion: 'reduce' });

      await page.goto('http://localhost:3001/dashboard');

      // Check that animations are disabled
      const card = page.locator('[data-testid="project-card"]').first();
      await expect(card).toHaveCSS('animation-duration', '0s');
      await expect(card).toHaveCSS('transition-duration', '0s');
    });
  });

  test.describe('Performance Tests', () => {
    test('should load dashboard within performance budget', async ({ page }) => {
      const startTime = Date.now();

      await page.goto('http://localhost:3001/dashboard');
      await page.waitForLoadState('networkidle');

      const loadTime = Date.now() - startTime;
      expect(loadTime).toBeLessThan(3000); // 3 seconds

      // Check Core Web Vitals
      const metrics = await page.evaluate(() => {
        return {
          lcp: performance.getEntriesByType('largest-contentful-paint')[0]?.startTime,
          fid: performance.getEntriesByType('first-input')[0]?.processingStart,
          cls: performance.getEntriesByType('layout-shift').reduce((sum, entry) => sum + entry.value, 0)
        };
      });

      expect(metrics.lcp).toBeLessThan(2500); // LCP < 2.5s
      if (metrics.fid) expect(metrics.fid).toBeLessThan(100); // FID < 100ms
      expect(metrics.cls).toBeLessThan(0.1); // CLS < 0.1
    });

    test('should handle large datasets efficiently', async ({ page }) => {
      // Navigate to project with many hypotheses
      await page.goto('http://localhost:3001/project/large-project-id');

      // Verify virtual scrolling is active
      await expect(page.locator('[data-testid="hypothesis-list"]')).toHaveAttribute('data-virtualized', 'true');

      // Scroll performance
      const scrollStart = Date.now();
      await page.evaluate(() => {
        const element = document.querySelector('[data-testid="hypothesis-list"]');
        element?.scrollTo(0, 10000);
      });
      const scrollTime = Date.now() - scrollStart;

      expect(scrollTime).toBeLessThan(100); // Smooth scrolling

      // Verify only visible items are rendered
      const renderedItems = await page.locator('[data-testid="hypothesis-card"]').count();
      expect(renderedItems).toBeLessThan(50); // Should use windowing
    });
  });

  test.describe('Error Handling', () => {
    test('should handle network errors gracefully', async ({ page, context }) => {
      // Simulate offline
      await context.setOffline(true);

      await page.goto('http://localhost:3001/dashboard');

      // Check offline indicator
      await expect(page.locator('[data-testid="offline-banner"]')).toBeVisible();
      await expect(page.locator('[data-testid="offline-banner"]')).toContainText('You are offline');

      // Try to create project
      await page.click('[data-testid="create-project-btn"]');
      await page.fill('[name="title"]', 'Test Project');
      await page.click('[data-testid="submit-project"]');

      // Should show error
      await expect(page.locator('[role="alert"]')).toContainText('Unable to create project while offline');

      // Go back online
      await context.setOffline(false);

      // Should auto-retry
      await expect(page.locator('[data-testid="offline-banner"]')).toBeHidden();
      await expect(page.locator('[role="alert"]')).toContainText('Project created successfully');
    });

    test('should handle API errors', async ({ page }) => {
      // Intercept API call and return error
      await page.route('**/api/projects', route => {
        route.fulfill({
          status: 500,
          body: JSON.stringify({ error: 'Internal Server Error' })
        });
      });

      await page.goto('http://localhost:3001/projects');

      // Should show error message
      await expect(page.locator('[role="alert"]')).toContainText('Failed to load projects');
      await expect(page.locator('[data-testid="retry-btn"]')).toBeVisible();

      // Remove route intercept
      await page.unroute('**/api/projects');

      // Click retry
      await page.click('[data-testid="retry-btn"]');

      // Should load successfully
      await expect(page.locator('[data-testid="project-card"]')).toHaveCount(3);
    });
  });
});