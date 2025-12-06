import { test, expect } from '@playwright/test';

test.describe('Chaos Engine - E2E Tests', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('http://localhost:3003');

    // Login if needed
    if (await page.locator('[data-testid="login-form"]').isVisible()) {
      await page.fill('[name="email"]', 'innovator@example.com');
      await page.fill('[name="password"]', 'testpassword');
      await page.click('[type="submit"]');
      await page.waitForURL('**/generator');
    }
  });

  test.describe('Idea Generation Workflow', () => {
    test('should generate ideas from domain collision', async ({ page }) => {
      await page.goto('http://localhost:3003/generator');

      // Select domains
      await page.click('[data-testid="domain-ai"]');
      await page.click('[data-testid="domain-healthcare"]');
      await page.click('[data-testid="domain-finance"]');

      // Verify selection
      await expect(page.locator('[data-testid="selected-count"]')).toContainText('3 domains selected');

      // Set chaos level
      await page.locator('[data-testid="chaos-slider"]').fill('75');
      await expect(page.locator('[data-testid="chaos-description"]')).toContainText('High Chaos');

      // Set innovation bias
      await page.locator('[data-testid="innovation-slider"]').fill('85');

      // Generate ideas
      await page.click('[data-testid="collide-btn"]');

      // Wait for generation animation
      await expect(page.locator('[data-testid="collision-animation"]')).toBeVisible();
      await expect(page.locator('[data-testid="collision-animation"]')).toBeHidden({ timeout: 10000 });

      // Verify ideas generated
      await expect(page.locator('[data-testid="idea-card"]')).toHaveCount(5);
      await expect(page.locator('[data-testid="idea-card"]').first()).toContainText('Score:');
      await expect(page.locator('[data-testid="idea-card"]').first()).toContainText('Potential:');
    });

    test('should allow custom domain input', async ({ page }) => {
      await page.goto('http://localhost:3003/generator');

      // Add custom domain
      await page.click('[data-testid="add-custom-domain"]');
      await page.fill('[name="customDomainName"]', 'Nanotechnology');
      await page.fill('[name="customDomainDescription"]', 'Molecular-scale engineering and materials');
      await page.fill('[data-testid="trend-input"]', 'Quantum dots');
      await page.keyboard.press('Enter');
      await page.fill('[data-testid="trend-input"]', 'Molecular machines');
      await page.keyboard.press('Enter');

      await page.click('[data-testid="save-custom-domain"]');

      // Use custom domain in generation
      await page.click('[data-testid="domain-nanotechnology"]');
      await page.click('[data-testid="domain-computing"]');

      await page.click('[data-testid="collide-btn"]');

      await expect(page.locator('[data-testid="idea-card"]').first()).toContainText('Nanotechnology');
    });

    test('should filter generated ideas', async ({ page }) => {
      // Generate ideas first
      await page.goto('http://localhost:3003/generator');
      await page.click('[data-testid="quick-generate"]'); // Quick generate with random domains

      await expect(page.locator('[data-testid="idea-card"]')).toHaveCount(5);

      // Filter by score
      await page.locator('[data-testid="min-score"]').fill('80');
      await expect(page.locator('[data-testid="idea-card"]:visible')).toHaveCount(2);

      // Filter by feasibility
      await page.selectOption('[data-testid="feasibility-filter"]', 'high');
      await expect(page.locator('[data-testid="idea-card"]:visible')).toHaveCount(1);

      // Clear filters
      await page.click('[data-testid="clear-filters"]');
      await expect(page.locator('[data-testid="idea-card"]:visible')).toHaveCount(5);
    });
  });

  test.describe('Idea Refinement', () => {
    test('should refine generated ideas', async ({ page }) => {
      await page.goto('http://localhost:3003/generator');

      // Quick generate
      await page.click('[data-testid="quick-generate"]');
      await expect(page.locator('[data-testid="idea-card"]')).toHaveCount(5);

      // Select idea to refine
      const firstIdea = page.locator('[data-testid="idea-card"]').first();
      const initialScore = await firstIdea.locator('[data-testid="idea-score"]').textContent();

      await firstIdea.locator('[data-testid="refine-btn"]').click();

      // Select refinement options
      await page.check('[value="improve-feasibility"]');
      await page.check('[value="enhance-uniqueness"]');
      await page.check('[value="market-optimization"]');

      // Add constraints
      await page.fill('[name="budget-constraint"]', '100000');
      await page.selectOption('[name="timeline"]', '6-months');

      // Apply refinements
      await page.click('[data-testid="apply-refinements"]');

      // Wait for refinement
      await expect(page.locator('[data-testid="refinement-spinner"]')).toBeVisible();
      await expect(page.locator('[data-testid="refinement-spinner"]')).toBeHidden();

      // Verify refinement results
      const refinedScore = await firstIdea.locator('[data-testid="idea-score"]').textContent();
      expect(parseInt(refinedScore!)).toBeGreaterThan(parseInt(initialScore!));
      await expect(firstIdea).toContainText('Refined');
    });

    test('should suggest refinement strategies', async ({ page }) => {
      await page.goto('http://localhost:3003/generator');
      await page.click('[data-testid="quick-generate"]');

      // Open refinement panel for low-scoring idea
      const ideas = await page.locator('[data-testid="idea-card"]').all();
      let lowScoreIdea;
      for (const idea of ideas) {
        const score = await idea.locator('[data-testid="idea-score"]').textContent();
        if (parseInt(score!) < 70) {
          lowScoreIdea = idea;
          break;
        }
      }

      if (lowScoreIdea) {
        await lowScoreIdea.locator('[data-testid="refine-btn"]').click();

        // Check for suggestions
        await expect(page.locator('[data-testid="refinement-suggestions"]')).toBeVisible();
        await expect(page.locator('[data-testid="suggestion-item"]')).toHaveCount(3);
        await expect(page.locator('[data-testid="suggestion-item"]').first()).toContainText('Consider');
      }
    });
  });

  test.describe('Business Plan Generation', () => {
    test('should generate business plan from idea', async ({ page }) => {
      await page.goto('http://localhost:3003/generator');
      await page.click('[data-testid="quick-generate"]');

      // Select high-scoring idea
      const firstIdea = page.locator('[data-testid="idea-card"]').first();
      await firstIdea.locator('[data-testid="generate-plan-btn"]').click();

      // Configure business plan
      await page.selectOption('[name="planType"]', 'startup');
      await page.selectOption('[name="industry"]', 'tech');
      await page.fill('[name="targetMarket"]', 'B2B SaaS');
      await page.fill('[name="initialInvestment"]', '500000');
      await page.selectOption('[name="timeHorizon"]', '5-years');

      // Add team members
      await page.click('[data-testid="add-team-member"]');
      await page.fill('[name="member-1-role"]', 'CEO');
      await page.fill('[name="member-1-expertise"]', 'Business Strategy, Sales');

      await page.click('[data-testid="add-team-member"]');
      await page.fill('[name="member-2-role"]', 'CTO');
      await page.fill('[name="member-2-expertise"]', 'AI, Software Architecture');

      // Generate plan
      await page.click('[data-testid="generate-business-plan"]');

      // Wait for generation
      await expect(page.locator('[data-testid="plan-generation-progress"]')).toBeVisible();
      await expect(page.locator('[data-testid="plan-generation-progress"]')).toBeHidden({ timeout: 15000 });

      // Verify business plan
      await expect(page).toHaveURL(/.*\/business-plan\/.*/);
      await expect(page.locator('[data-testid="plan-section-executive-summary"]')).toBeVisible();
      await expect(page.locator('[data-testid="plan-section-market-analysis"]')).toBeVisible();
      await expect(page.locator('[data-testid="plan-section-financial-projections"]')).toBeVisible();
      await expect(page.locator('[data-testid="plan-section-go-to-market"]')).toBeVisible();
    });

    test('should export business plan', async ({ page }) => {
      // Navigate to existing business plan
      await page.goto('http://localhost:3003/business-plan/test-plan-123');

      // Export as PDF
      const [download] = await Promise.all([
        page.waitForEvent('download'),
        page.click('[data-testid="export-pdf"]')
      ]);

      expect(download.suggestedFilename()).toContain('business-plan');
      expect(download.suggestedFilename()).toEndWith('.pdf');

      // Export as DOCX
      const [download2] = await Promise.all([
        page.waitForEvent('download'),
        page.click('[data-testid="export-docx"]')
      ]);

      expect(download2.suggestedFilename()).toEndWith('.docx');
    });
  });

  test.describe('Pitch Deck Creation', () => {
    test('should create pitch deck from idea', async ({ page }) => {
      await page.goto('http://localhost:3003/generator');
      await page.click('[data-testid="quick-generate"]');

      const firstIdea = page.locator('[data-testid="idea-card"]').first();
      await firstIdea.locator('[data-testid="create-pitch-btn"]').click();

      // Configure pitch deck
      await page.selectOption('[name="audience"]', 'investors');
      await page.selectOption('[name="pitchLength"]', '10-slides');
      await page.check('[value="problem-solution"]');
      await page.check('[value="market-opportunity"]');
      await page.check('[value="business-model"]');
      await page.check('[value="financial-projections"]');
      await page.check('[value="team"]');

      // Set design preferences
      await page.selectOption('[name="template"]', 'modern');
      await page.fill('[name="primaryColor"]', '#2563eb');

      // Generate pitch deck
      await page.click('[data-testid="generate-pitch"]');

      // Wait for generation
      await expect(page.locator('[data-testid="pitch-generation-progress"]')).toBeVisible();
      await expect(page.locator('[data-testid="pitch-generation-progress"]')).toBeHidden({ timeout: 10000 });

      // Verify pitch deck
      await expect(page).toHaveURL(/.*\/pitch-deck\/.*/);
      await expect(page.locator('[data-testid="slide-thumbnail"]')).toHaveCount(10);
      await expect(page.locator('[data-testid="current-slide"]')).toContainText('Problem');
    });

    test('should edit pitch deck slides', async ({ page }) => {
      await page.goto('http://localhost:3003/pitch-deck/test-deck-123');

      // Edit title slide
      await page.click('[data-testid="slide-thumbnail-1"]');
      await page.fill('[data-testid="slide-title"]', 'Revolutionary AI Healthcare Platform');
      await page.fill('[data-testid="slide-subtitle"]', 'Transforming patient care with AI');

      // Edit market slide
      await page.click('[data-testid="slide-thumbnail-3"]');
      await page.fill('[data-testid="market-size"]', '$150B');
      await page.fill('[data-testid="growth-rate"]', '25% CAGR');

      // Add custom slide
      await page.click('[data-testid="add-slide"]');
      await page.selectOption('[name="slideType"]', 'custom');
      await page.fill('[data-testid="custom-slide-title"]', 'Competitive Advantage');
      await page.fill('[data-testid="custom-slide-content"]', 'Our unique AI algorithms...');

      // Save changes
      await page.click('[data-testid="save-pitch"]');
      await expect(page.locator('[role="alert"]')).toContainText('Pitch deck saved');
    });

    test('should present pitch deck', async ({ page }) => {
      await page.goto('http://localhost:3003/pitch-deck/test-deck-123');

      // Start presentation
      await page.click('[data-testid="present-btn"]');

      // Verify full-screen presentation
      await expect(page.locator('[data-testid="presentation-mode"]')).toBeVisible();
      await expect(page.locator('[data-testid="slide-content"]')).toBeVisible();

      // Navigate slides
      await page.keyboard.press('ArrowRight');
      await expect(page.locator('[data-testid="slide-number"]')).toContainText('2 / 10');

      await page.keyboard.press('ArrowLeft');
      await expect(page.locator('[data-testid="slide-number"]')).toContainText('1 / 10');

      // Exit presentation
      await page.keyboard.press('Escape');
      await expect(page.locator('[data-testid="presentation-mode"]')).toBeHidden();
    });
  });

  test.describe('Idea Comparison and Voting', () => {
    test('should compare multiple ideas', async ({ page }) => {
      await page.goto('http://localhost:3003/generator');
      await page.click('[data-testid="quick-generate"]');

      // Select ideas for comparison
      await page.locator('[data-testid="idea-card"]').nth(0).locator('[data-testid="select-compare"]').check();
      await page.locator('[data-testid="idea-card"]').nth(1).locator('[data-testid="select-compare"]').check();
      await page.locator('[data-testid="idea-card"]').nth(2).locator('[data-testid="select-compare"]').check();

      // Open comparison view
      await page.click('[data-testid="compare-selected"]');

      // Verify comparison table
      await expect(page.locator('[data-testid="comparison-table"]')).toBeVisible();
      await expect(page.locator('[data-testid="comparison-row"]')).toHaveCount(3);

      // Check comparison metrics
      await expect(page.locator('[data-testid="comparison-metrics"]')).toContainText('Score');
      await expect(page.locator('[data-testid="comparison-metrics"]')).toContainText('Feasibility');
      await expect(page.locator('[data-testid="comparison-metrics"]')).toContainText('Market Potential');
      await expect(page.locator('[data-testid="comparison-metrics"]')).toContainText('Innovation');

      // Highlight winner
      await expect(page.locator('[data-testid="best-idea"]')).toHaveClass(/highlight/);
    });

    test('should enable voting on ideas', async ({ page, context }) => {
      await page.goto('http://localhost:3003/ideas/gallery');

      // Vote on idea
      const firstIdea = page.locator('[data-testid="gallery-idea"]').first();
      const initialVotes = await firstIdea.locator('[data-testid="vote-count"]').textContent();

      await firstIdea.locator('[data-testid="upvote-btn"]').click();

      // Verify vote counted
      await expect(firstIdea.locator('[data-testid="vote-count"]')).toContainText(String(parseInt(initialVotes!) + 1));
      await expect(firstIdea.locator('[data-testid="upvote-btn"]')).toHaveClass(/voted/);

      // Open second session to verify real-time update
      const page2 = await context.newPage();
      await page2.goto('http://localhost:3003/ideas/gallery');

      await expect(page2.locator('[data-testid="gallery-idea"]').first().locator('[data-testid="vote-count"]'))
        .toContainText(String(parseInt(initialVotes!) + 1));

      await page2.close();
    });

    test('should show idea leaderboard', async ({ page }) => {
      await page.goto('http://localhost:3003/ideas/leaderboard');

      // Check leaderboard display
      await expect(page.locator('[data-testid="leaderboard-table"]')).toBeVisible();
      await expect(page.locator('[data-testid="leaderboard-entry"]')).toHaveCount(10);

      // Check sorting
      const scores = await page.locator('[data-testid="idea-total-score"]').allTextContents();
      const numScores = scores.map(s => parseInt(s));
      expect(numScores).toEqual([...numScores].sort((a, b) => b - a));

      // Filter by time period
      await page.selectOption('[data-testid="time-filter"]', 'this-week');
      await expect(page.locator('[data-testid="leaderboard-entry"]').first()).toContainText('This week');
    });
  });

  test.describe('History and Analytics', () => {
    test('should track generation history', async ({ page }) => {
      await page.goto('http://localhost:3003/generator');

      // Generate ideas multiple times
      await page.click('[data-testid="quick-generate"]');
      await page.waitForTimeout(1000);
      await page.click('[data-testid="clear-ideas"]');

      await page.click('[data-testid="domain-ai"]');
      await page.click('[data-testid="domain-space"]');
      await page.click('[data-testid="collide-btn"]');
      await page.waitForTimeout(1000);

      // View history
      await page.click('[data-testid="view-history"]');

      // Verify history entries
      await expect(page.locator('[data-testid="history-entry"]')).toHaveCount(2);
      await expect(page.locator('[data-testid="history-entry"]').first()).toContainText('AI + Space');

      // Load previous generation
      await page.locator('[data-testid="history-entry"]').last().locator('[data-testid="load-generation"]').click();
      await expect(page.locator('[data-testid="idea-card"]')).toHaveCount(5);
    });

    test('should display generation analytics', async ({ page }) => {
      await page.goto('http://localhost:3003/analytics');

      // Check statistics
      await expect(page.locator('[data-testid="total-ideas-generated"]')).toBeVisible();
      await expect(page.locator('[data-testid="avg-idea-score"]')).toBeVisible();
      await expect(page.locator('[data-testid="top-domains"]')).toBeVisible();

      // Check charts
      await expect(page.locator('[data-testid="ideas-over-time-chart"]')).toBeVisible();
      await expect(page.locator('[data-testid="domain-distribution-chart"]')).toBeVisible();
      await expect(page.locator('[data-testid="score-distribution-chart"]')).toBeVisible();

      // Filter analytics
      await page.selectOption('[data-testid="analytics-period"]', 'last-30-days');
      await expect(page.locator('[data-testid="ideas-over-time-chart"]')).toHaveAttribute('data-period', 'last-30-days');
    });
  });

  test.describe('Collaboration Features', () => {
    test('should share ideas with team', async ({ page }) => {
      await page.goto('http://localhost:3003/generator');
      await page.click('[data-testid="quick-generate"]');

      // Share idea
      const firstIdea = page.locator('[data-testid="idea-card"]').first();
      await firstIdea.locator('[data-testid="share-btn"]').click();

      // Select recipients
      await page.fill('[data-testid="recipient-email"]', 'teammate@example.com');
      await page.keyboard.press('Enter');
      await page.fill('[data-testid="recipient-email"]', 'investor@example.com');
      await page.keyboard.press('Enter');

      // Add message
      await page.fill('[data-testid="share-message"]', 'Check out this amazing idea!');

      // Set permissions
      await page.check('[value="can-comment"]');
      await page.check('[value="can-vote"]');

      // Send
      await page.click('[data-testid="send-share"]');

      // Verify shared
      await expect(page.locator('[role="alert"]')).toContainText('Idea shared successfully');
      await expect(firstIdea).toContainText('Shared');
    });

    test('should enable team brainstorming', async ({ page, context }) => {
      // Create brainstorming session
      await page.goto('http://localhost:3003/brainstorm/create');

      await page.fill('[name="sessionName"]', 'Q1 Innovation Sprint');
      await page.fill('[name="sessionGoal"]', 'Generate 20 disruptive ideas');
      await page.selectOption('[name="duration"]', '60');

      // Invite participants
      await page.fill('[data-testid="invite-email"]', 'team1@example.com');
      await page.keyboard.press('Enter');
      await page.fill('[data-testid="invite-email"]', 'team2@example.com');
      await page.keyboard.press('Enter');

      await page.click('[data-testid="start-session"]');

      // Verify session started
      await expect(page).toHaveURL(/.*\/brainstorm\/session\/.*/);
      await expect(page.locator('[data-testid="session-timer"]')).toBeVisible();
      await expect(page.locator('[data-testid="participants-list"]')).toContainText('3 participants');

      // Simulate another participant joining
      const page2 = await context.newPage();
      await page2.goto(page.url());

      // Both should see real-time updates
      await page.click('[data-testid="domain-ai"]');
      await expect(page2.locator('[data-testid="domain-ai"]')).toHaveClass(/selected/);

      await page2.close();
    });
  });

  test.describe('Mobile Experience', () => {
    test('should work on mobile devices', async ({ page }) => {
      await page.setViewportSize({ width: 375, height: 667 });
      await page.goto('http://localhost:3003/generator');

      // Check mobile layout
      await expect(page.locator('[data-testid="mobile-domains-carousel"]')).toBeVisible();

      // Swipe through domains
      const carousel = page.locator('[data-testid="mobile-domains-carousel"]');
      await carousel.dispatchEvent('touchstart', { touches: [{ clientX: 300, clientY: 200 }] });
      await carousel.dispatchEvent('touchmove', { touches: [{ clientX: 100, clientY: 200 }] });
      await carousel.dispatchEvent('touchend');

      // Select domains
      await page.click('[data-testid="domain-ai"]');
      await page.click('[data-testid="domain-healthcare"]');

      // Generate on mobile
      await page.click('[data-testid="mobile-collide-btn"]');

      // Verify mobile-optimized results
      await expect(page.locator('[data-testid="mobile-idea-card"]')).toHaveCount(5);
      await expect(page.locator('[data-testid="mobile-idea-card"]')).toHaveCSS('width', '343px'); // With padding
    });
  });

  test.describe('Performance', () => {
    test('should handle rapid generation requests', async ({ page }) => {
      await page.goto('http://localhost:3003/generator');

      // Rapid generation attempts
      for (let i = 0; i < 5; i++) {
        await page.click('[data-testid="quick-generate"]');
        await page.waitForTimeout(100);
      }

      // Should queue or debounce properly
      await expect(page.locator('[data-testid="generation-queue"]')).toBeVisible();
      await expect(page.locator('[data-testid="generation-queue"]')).toContainText('Processing');

      // Wait for completion
      await expect(page.locator('[data-testid="generation-queue"]')).toBeHidden({ timeout: 15000 });
      await expect(page.locator('[data-testid="idea-card"]')).toHaveCount(5);
    });

    test('should handle large domain combinations efficiently', async ({ page }) => {
      await page.goto('http://localhost:3003/generator');

      // Select many domains
      const domains = ['ai', 'healthcare', 'finance', 'space', 'energy', 'education', 'gaming', 'agriculture'];
      for (const domain of domains) {
        await page.click(`[data-testid="domain-${domain}"]`);
      }

      const startTime = Date.now();
      await page.click('[data-testid="collide-btn"]');

      // Should complete within reasonable time
      await expect(page.locator('[data-testid="idea-card"]')).toHaveCount(10, { timeout: 10000 });
      const endTime = Date.now();

      expect(endTime - startTime).toBeLessThan(10000);
    });
  });
});