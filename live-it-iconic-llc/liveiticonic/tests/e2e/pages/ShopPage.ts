import { Page, Locator } from '@playwright/test';

export class ShopPage {
  readonly page: Page;
  readonly productCards: Locator;
  readonly filterCategory: Locator;
  readonly searchInput: Locator;
  readonly addToCartButton: Locator;
  readonly priceFilter: Locator;
  readonly sortDropdown: Locator;
  readonly cartIcon: Locator;
  readonly cartBadge: Locator;

  constructor(page: Page) {
    this.page = page;
    this.productCards = page.locator('[data-testid="product-card"], .product-card');
    this.filterCategory = page.locator('[data-testid="filter-category"]');
    this.searchInput = page.locator('[data-testid="search-input"], input[placeholder*="search" i]').first();
    this.addToCartButton = page.locator('[data-testid="add-to-cart"], button:has-text("Add to Cart")').first();
    this.priceFilter = page.locator('[data-testid="price-filter"]');
    this.sortDropdown = page.locator('[data-testid="sort-dropdown"]');
    this.cartIcon = page.locator('[data-testid="cart-icon"], [aria-label*="cart" i]').first();
    this.cartBadge = page.locator('[data-testid="cart-badge"]');
  }

  async goto() {
    await this.page.goto('/shop');
  }

  async getProductCount() {
    return await this.productCards.count();
  }

  async searchProducts(query: string) {
    if (await this.searchInput.isVisible()) {
      await this.searchInput.fill(query);
      await this.page.waitForLoadState('networkidle');
    }
  }

  async clickProductAt(index: number) {
    await this.productCards.nth(index).click();
  }

  async addFirstProductToCart() {
    const addButton = this.page.locator('[data-testid="add-to-cart"]').first();
    await addButton.click();
  }

  async filterByCategory(category: string) {
    const categoryFilter = this.page.locator(`[data-testid="filter-category"] >> text="${category}"`);
    await categoryFilter.click();
    await this.page.waitForLoadState('networkidle');
  }

  async sortBy(sortOption: string) {
    if (await this.sortDropdown.isVisible()) {
      await this.sortDropdown.click();
      await this.page.locator(`text="${sortOption}"`).click();
      await this.page.waitForLoadState('networkidle');
    }
  }

  async getFirstProductName(): Promise<string | null> {
    return await this.productCards.first().locator('[data-testid="product-name"]').textContent();
  }

  async getFirstProductPrice(): Promise<string | null> {
    return await this.productCards.first().locator('[data-testid="product-price"]').textContent();
  }

  async openCart() {
    await this.cartIcon.click();
  }
}
