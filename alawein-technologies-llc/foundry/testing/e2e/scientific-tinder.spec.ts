import { test, expect } from '@playwright/test';

test.describe('Scientific Tinder - E2E Tests', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('http://localhost:3002');

    // Login if needed
    if (await page.locator('[data-testid="login-form"]').isVisible()) {
      await page.fill('[name="email"]', 'researcher@example.com');
      await page.fill('[name="password"]', 'testpassword');
      await page.click('[type="submit"]');
      await page.waitForURL('**/discover');
    }
  });

  test.describe('Profile Setup', () => {
    test('should complete researcher profile', async ({ page }) => {
      await page.goto('http://localhost:3002/profile/setup');

      // Fill basic info
      await page.fill('[name="name"]', 'Dr. Jane Smith');
      await page.fill('[name="institution"]', 'MIT');
      await page.fill('[name="bio"]', 'Quantum computing researcher focused on error correction');

      // Select expertise
      await page.click('[data-testid="expertise-select"]');
      await page.click('[data-value="quantum-computing"]');
      await page.click('[data-value="machine-learning"]');
      await page.click('body'); // Close dropdown

      // Add interests
      await page.fill('[data-testid="interest-input"]', 'Quantum ML');
      await page.press('[data-testid="interest-input"]', 'Enter');
      await page.fill('[data-testid="interest-input"]', 'Error Correction');
      await page.press('[data-testid="interest-input"]', 'Enter');

      // Set availability
      await page.selectOption('[name="availability"]', 'part-time');
      await page.selectOption('[name="timezone"]', 'America/New_York');

      // Upload publications
      const [fileChooser] = await Promise.all([
        page.waitForEvent('filechooser'),
        page.click('[data-testid="upload-publications"]')
      ]);
      await fileChooser.setFiles('./test-fixtures/publications.pdf');

      await page.click('[data-testid="save-profile"]');

      // Verify profile created
      await expect(page).toHaveURL(/.*\/profile/);
      await expect(page.locator('[data-testid="profile-completeness"]')).toContainText('100%');
    });

    test('should set collaboration preferences', async ({ page }) => {
      await page.goto('http://localhost:3002/profile/preferences');

      // Set collaboration goals
      await page.check('[value="research-papers"]');
      await page.check('[value="grant-proposals"]');
      await page.check('[value="open-source"]');

      // Set team size preference
      await page.click('[data-testid="team-size-small"]');

      // Set project duration
      await page.selectOption('[name="projectDuration"]', '3-6-months');

      // Set matching criteria weights
      await page.locator('[data-testid="expertise-weight"]').fill('80');
      await page.locator('[data-testid="interests-weight"]').fill('60');
      await page.locator('[data-testid="availability-weight"]').fill('40');

      await page.click('[data-testid="save-preferences"]');

      // Verify preferences saved
      await expect(page.locator('[role="alert"]')).toContainText('Preferences updated');
    });
  });

  test.describe('Matching System', () => {
    test('should display and interact with match cards', async ({ page }) => {
      await page.goto('http://localhost:3002/discover');

      // Wait for matches to load
      await expect(page.locator('[data-testid="match-card"]')).toBeVisible();

      // Check match details
      const matchCard = page.locator('[data-testid="match-card"]').first();
      await expect(matchCard).toContainText('Match Score');
      await expect(matchCard).toContainText('Expertise');
      await expect(matchCard).toContainText('Common Interests');

      // View detailed profile
      await page.click('[data-testid="view-details"]');
      await expect(page.locator('[data-testid="researcher-modal"]')).toBeVisible();
      await expect(page.locator('[data-testid="publication-list"]')).toBeVisible();
      await expect(page.locator('[data-testid="collaboration-history"]')).toBeVisible();

      // Close modal
      await page.keyboard.press('Escape');
      await expect(page.locator('[data-testid="researcher-modal"]')).toBeHidden();
    });

    test('should handle swipe gestures', async ({ page, browserName }) => {
      // Skip for non-chromium browsers due to touch event limitations
      test.skip(browserName !== 'chromium', 'Touch events work best in Chromium');

      await page.goto('http://localhost:3002/discover');

      const card = page.locator('[data-testid="match-card"]').first();
      const cardText = await card.textContent();

      // Swipe right (accept)
      await card.dispatchEvent('touchstart', { touches: [{ clientX: 100, clientY: 200 }] });
      await card.dispatchEvent('touchmove', { touches: [{ clientX: 300, clientY: 200 }] });
      await card.dispatchEvent('touchend');

      // Verify card changed
      await expect(page.locator('[data-testid="match-card"]').first()).not.toContainText(cardText);

      // Check action recorded
      await expect(page.locator('[data-testid="swipe-indicator"]')).toContainText('Interested');
    });

    test('should use keyboard controls for matching', async ({ page }) => {
      await page.goto('http://localhost:3002/discover');

      const firstCardName = await page.locator('[data-testid="researcher-name"]').first().textContent();

      // Use arrow keys
      await page.keyboard.press('ArrowRight'); // Accept
      await expect(page.locator('[data-testid="researcher-name"]').first()).not.toContainText(firstCardName);

      const secondCardName = await page.locator('[data-testid="researcher-name"]').first().textContent();

      await page.keyboard.press('ArrowLeft'); // Reject
      await expect(page.locator('[data-testid="researcher-name"]').first()).not.toContainText(secondCardName);

      await page.keyboard.press('ArrowUp'); // Super like
      await expect(page.locator('[data-testid="super-like-animation"]')).toBeVisible();
    });

    test('should filter matches', async ({ page }) => {
      await page.goto('http://localhost:3002/discover');

      // Open filters
      await page.click('[data-testid="filter-btn"]');

      // Filter by expertise
      await page.click('[data-testid="filter-expertise"]');
      await page.check('[value="machine-learning"]');
      await page.check('[value="data-science"]');

      // Filter by institution type
      await page.selectOption('[name="institutionType"]', 'university');

      // Filter by h-index
      await page.fill('[name="minHIndex"]', '10');

      // Apply filters
      await page.click('[data-testid="apply-filters"]');

      // Verify filtered results
      await expect(page.locator('[data-testid="filter-badge"]')).toContainText('3 filters');
      const matchCard = page.locator('[data-testid="match-card"]').first();
      await expect(matchCard).toContainText('Machine Learning', 'Data Science');
    });
  });

  test.describe('Mutual Matches and Communication', () => {
    test('should show mutual match notification', async ({ page }) => {
      await page.goto('http://localhost:3002/discover');

      // Simulate mutual match (using test data that creates mutual match)
      await page.click('[data-testid="test-mutual-match"]');

      // Check notification
      await expect(page.locator('[data-testid="match-notification"]')).toBeVisible();
      await expect(page.locator('[data-testid="match-notification"]')).toContainText("It's a Match!");

      // Navigate to matches
      await page.click('[data-testid="view-matches"]');
      await expect(page).toHaveURL(/.*\/matches/);
      await expect(page.locator('[data-testid="mutual-match-card"]').first()).toBeVisible();
    });

    test('should initiate collaboration chat', async ({ page }) => {
      await page.goto('http://localhost:3002/matches');

      // Click on a match to chat
      await page.click('[data-testid="mutual-match-card"]').first();

      // Send initial message
      await page.fill('[data-testid="message-input"]', 'Hi! I am interested in collaborating on quantum ML research.');
      await page.keyboard.press('Enter');

      // Verify message sent
      await expect(page.locator('[data-testid="message-bubble"]').last()).toContainText('quantum ML research');

      // Send collaboration proposal
      await page.click('[data-testid="propose-collaboration"]');
      await page.fill('[name="projectTitle"]', 'Quantum Error Correction with ML');
      await page.fill('[name="projectDescription"]', 'Using machine learning to optimize quantum error correction codes');
      await page.selectOption('[name="duration"]', '6-months');
      await page.click('[data-testid="send-proposal"]');

      // Verify proposal sent
      await expect(page.locator('[data-testid="proposal-card"]')).toBeVisible();
      await expect(page.locator('[data-testid="proposal-status"]')).toContainText('Pending');
    });

    test('should manage collaboration requests', async ({ page }) => {
      await page.goto('http://localhost:3002/collaborations');

      // Check incoming requests
      await page.click('[data-testid="tab-incoming"]');
      await expect(page.locator('[data-testid="request-card"]')).toHaveCount(2);

      // Accept a request
      const firstRequest = page.locator('[data-testid="request-card"]').first();
      await firstRequest.locator('[data-testid="accept-request"]').click();

      // Add availability for collaboration
      await page.fill('[data-testid="availability-hours"]', '10');
      await page.selectOption('[data-testid="start-date"]', '2024-02-01');
      await page.click('[data-testid="confirm-collaboration"]');

      // Verify collaboration started
      await expect(page.locator('[role="alert"]')).toContainText('Collaboration accepted');
      await page.click('[data-testid="tab-active"]');
      await expect(page.locator('[data-testid="active-collaboration"]')).toHaveCount(1);
    });
  });

  test.describe('Project Management', () => {
    test('should create collaborative project', async ({ page }) => {
      await page.goto('http://localhost:3002/projects/create');

      // Fill project details
      await page.fill('[name="title"]', 'Quantum-Classical Hybrid Algorithms');
      await page.fill('[name="description"]', 'Developing hybrid algorithms combining quantum and classical computing');

      // Add collaborators
      await page.click('[data-testid="add-collaborator"]');
      await page.fill('[data-testid="search-collaborator"]', 'Dr. Alice Johnson');
      await page.click('[data-testid="select-collaborator-1"]');

      // Set milestones
      await page.click('[data-testid="add-milestone"]');
      await page.fill('[name="milestone-1-title"]', 'Literature Review');
      await page.selectOption('[name="milestone-1-duration"]', '2-weeks');

      await page.click('[data-testid="add-milestone"]');
      await page.fill('[name="milestone-2-title"]', 'Algorithm Development');
      await page.selectOption('[name="milestone-2-duration"]', '1-month');

      // Set repository
      await page.fill('[name="github-repo"]', 'https://github.com/user/quantum-hybrid');

      await page.click('[data-testid="create-project"]');

      // Verify project created
      await expect(page).toHaveURL(/.*\/project\/.*/);
      await expect(page.locator('h1')).toContainText('Quantum-Classical Hybrid');
    });

    test('should track project progress', async ({ page }) => {
      await page.goto('http://localhost:3002/project/test-project-123');

      // Update milestone status
      await page.click('[data-testid="milestone-1"]');
      await page.selectOption('[name="status"]', 'in-progress');
      await page.fill('[data-testid="progress"]', '60');
      await page.click('[data-testid="update-milestone"]');

      // Add project update
      await page.click('[data-testid="add-update"]');
      await page.fill('[data-testid="update-title"]', 'Completed initial research phase');
      await page.fill('[data-testid="update-description"]', 'Reviewed 25 papers on quantum-classical hybrid approaches');
      await page.click('[data-testid="post-update"]');

      // Verify updates
      await expect(page.locator('[data-testid="progress-bar"]')).toHaveAttribute('aria-valuenow', '60');
      await expect(page.locator('[data-testid="project-update"]').first()).toContainText('Completed initial research');
    });
  });

  test.describe('Analytics and Insights', () => {
    test('should display matching analytics', async ({ page }) => {
      await page.goto('http://localhost:3002/analytics');

      // Check statistics
      await expect(page.locator('[data-testid="total-swipes"]')).toBeVisible();
      await expect(page.locator('[data-testid="match-rate"]')).toBeVisible();
      await expect(page.locator('[data-testid="avg-match-score"]')).toBeVisible();

      // Check match distribution chart
      await expect(page.locator('[data-testid="match-distribution-chart"]')).toBeVisible();

      // Filter by date range
      await page.selectOption('[data-testid="date-range"]', 'last-month');
      await expect(page.locator('[data-testid="match-distribution-chart"]')).toHaveAttribute('data-range', 'last-month');
    });

    test('should show collaboration success metrics', async ({ page }) => {
      await page.goto('http://localhost:3002/analytics/collaborations');

      // Check collaboration metrics
      await expect(page.locator('[data-testid="successful-collaborations"]')).toBeVisible();
      await expect(page.locator('[data-testid="avg-project-duration"]')).toBeVisible();
      await expect(page.locator('[data-testid="publication-count"]')).toBeVisible();

      // View detailed collaboration
      await page.click('[data-testid="collaboration-detail-1"]');
      await expect(page.locator('[data-testid="collaboration-timeline"]')).toBeVisible();
      await expect(page.locator('[data-testid="collaboration-outputs"]')).toBeVisible();
    });
  });

  test.describe('Mobile Experience', () => {
    test('should work on mobile devices', async ({ page }) => {
      // Set mobile viewport
      await page.setViewportSize({ width: 375, height: 667 });

      await page.goto('http://localhost:3002/discover');

      // Check mobile layout
      await expect(page.locator('[data-testid="mobile-nav"]')).toBeVisible();
      await expect(page.locator('[data-testid="match-card"]')).toHaveCSS('width', '375px');

      // Test swipe gestures
      const card = page.locator('[data-testid="match-card"]').first();
      await card.dispatchEvent('touchstart', { touches: [{ clientX: 50, clientY: 300 }] });
      await card.dispatchEvent('touchmove', { touches: [{ clientX: 250, clientY: 300 }] });
      await card.dispatchEvent('touchend');

      // Verify swipe worked
      await expect(page.locator('[data-testid="swipe-feedback"]')).toBeVisible();
    });

    test('should have responsive navigation', async ({ page }) => {
      await page.setViewportSize({ width: 375, height: 667 });
      await page.goto('http://localhost:3002');

      // Open mobile menu
      await page.click('[data-testid="menu-toggle"]');
      await expect(page.locator('[data-testid="mobile-menu"]')).toBeVisible();

      // Navigate using mobile menu
      await page.click('[data-testid="mobile-nav-matches"]');
      await expect(page).toHaveURL(/.*\/matches/);
      await expect(page.locator('[data-testid="mobile-menu"]')).toBeHidden();
    });
  });

  test.describe('Accessibility', () => {
    test('should support keyboard navigation throughout', async ({ page }) => {
      await page.goto('http://localhost:3002/discover');

      // Tab to first interactive element
      await page.keyboard.press('Tab');
      await expect(page.locator(':focus')).toHaveAttribute('data-testid', 'filter-btn');

      // Tab to match card
      await page.keyboard.press('Tab');
      await expect(page.locator(':focus')).toHaveAttribute('data-testid', 'match-card');

      // Use space to view details
      await page.keyboard.press('Space');
      await expect(page.locator('[data-testid="researcher-modal"]')).toBeVisible();

      // Tab through modal content
      await page.keyboard.press('Tab');
      await expect(page.locator(':focus')).toHaveAttribute('data-testid', 'modal-close');

      // Close with Escape
      await page.keyboard.press('Escape');
      await expect(page.locator('[data-testid="researcher-modal"]')).toBeHidden();
    });

    test('should announce match actions to screen readers', async ({ page }) => {
      await page.goto('http://localhost:3002/discover');

      // Accept a match
      await page.click('[data-testid="accept-btn"]');

      // Check for screen reader announcement
      await expect(page.locator('[role="status"]')).toContainText('Accepted match with');

      // Reject a match
      await page.click('[data-testid="reject-btn"]');
      await expect(page.locator('[role="status"]')).toContainText('Passed on match with');
    });
  });
});