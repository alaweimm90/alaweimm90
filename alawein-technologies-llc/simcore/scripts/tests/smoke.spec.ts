import { test, expect } from '@playwright/test';

test('index renders key areas', async ({ page }) => {
  await page.goto('/');
  await expect(page.locator('#main-content')).toBeVisible();
});

test('routes: 404 renders', async ({ page }) => {
  await page.goto('/non-existent');
  await expect(page.getByRole('heading')).toBeVisible();
});

