import { Page, Locator } from '@playwright/test';

export class HomePage {
  readonly page: Page;
  readonly heroTitle: Locator;
  readonly shopButton: Locator;
  readonly navShop: Locator;
  readonly navAbout: Locator;
  readonly cartIcon: Locator;
  readonly cartBadge: Locator;

  constructor(page: Page) {
    this.page = page;
    this.heroTitle = page.locator('h1, [data-testid="hero-title"]').first();
    this.shopButton = page.getByRole('link', { name: /shop/i }).first();
    this.navShop = page.locator('nav').getByRole('link', { name: /shop/i }).first();
    this.navAbout = page.locator('nav').getByRole('link', { name: /about/i }).first();
    this.cartIcon = page.locator('[data-testid="cart-icon"], [aria-label*="cart" i]').first();
    this.cartBadge = page.locator('[data-testid="cart-badge"]');
  }

  async goto() {
    await this.page.goto('/');
  }

  async clickShop() {
    await this.shopButton.click();
  }

  async navigateToAbout() {
    await this.navAbout.click();
  }

  async openCart() {
    await this.cartIcon.click();
  }

  async getCartCount(): Promise<string | null> {
    const badge = this.cartBadge;
    return await badge.textContent();
  }

  async waitForPageLoad() {
    await this.page.waitForLoadState('networkidle');
  }
}
