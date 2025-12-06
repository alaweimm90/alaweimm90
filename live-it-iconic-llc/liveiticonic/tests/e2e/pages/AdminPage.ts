import { Page, Locator } from '@playwright/test';

export class AdminPage {
  readonly page: Page;
  readonly dashboardTitle: Locator;
  readonly ordersLink: Locator;
  readonly productsLink: Locator;
  readonly usersLink: Locator;
  readonly analyticsLink: Locator;
  readonly settingsLink: Locator;
  readonly logoutButton: Locator;
  readonly ordersTable: Locator;
  readonly productsList: Locator;
  readonly usersList: Locator;
  readonly addProductButton: Locator;
  readonly searchInput: Locator;
  readonly filterDropdown: Locator;
  readonly statusBadge: Locator;
  readonly editButtons: Locator;
  readonly deleteButtons: Locator;

  constructor(page: Page) {
    this.page = page;
    this.dashboardTitle = page.locator('h1, [data-testid="dashboard-title"]').first();
    this.ordersLink = page.getByRole('link', { name: /orders/i }).first();
    this.productsLink = page.getByRole('link', { name: /products/i }).first();
    this.usersLink = page.getByRole('link', { name: /users/i }).first();
    this.analyticsLink = page.getByRole('link', { name: /analytics|reports/i }).first();
    this.settingsLink = page.getByRole('link', { name: /settings/i }).first();
    this.logoutButton = page.getByRole('button', { name: /logout|sign out/i }).first();
    this.ordersTable = page.locator('[data-testid="orders-table"]');
    this.productsList = page.locator('[data-testid="products-list"]');
    this.usersList = page.locator('[data-testid="users-list"]');
    this.addProductButton = page.getByRole('button', { name: /add product/i }).first();
    this.searchInput = page.locator('[data-testid="search-input"], input[placeholder*="search" i]').first();
    this.filterDropdown = page.locator('[data-testid="filter-dropdown"]');
    this.statusBadge = page.locator('[data-testid="status-badge"]');
    this.editButtons = page.locator('[data-testid="edit-button"]');
    this.deleteButtons = page.locator('[data-testid="delete-button"]');
  }

  async goto() {
    await this.page.goto('/admin');
  }

  async gotoOrders() {
    await this.ordersLink.click();
    await this.page.waitForLoadState('networkidle');
  }

  async gotoProducts() {
    await this.productsLink.click();
    await this.page.waitForLoadState('networkidle');
  }

  async gotoUsers() {
    await this.usersLink.click();
    await this.page.waitForLoadState('networkidle');
  }

  async gotoAnalytics() {
    await this.analyticsLink.click();
    await this.page.waitForLoadState('networkidle');
  }

  async gotoSettings() {
    await this.settingsLink.click();
    await this.page.waitForLoadState('networkidle');
  }

  async logout() {
    await this.logoutButton.click();
  }

  async searchFor(query: string) {
    if (await this.searchInput.isVisible()) {
      await this.searchInput.fill(query);
      await this.page.waitForLoadState('networkidle');
    }
  }

  async filterBy(filterValue: string) {
    if (await this.filterDropdown.isVisible()) {
      await this.filterDropdown.click();
      await this.page.locator(`text="${filterValue}"`).click();
      await this.page.waitForLoadState('networkidle');
    }
  }

  async editItemAt(index: number) {
    await this.editButtons.nth(index).click();
  }

  async deleteItemAt(index: number) {
    await this.deleteButtons.nth(index).click();
  }

  async addNewProduct() {
    await this.addProductButton.click();
  }

  async isDashboardVisible(): Promise<boolean> {
    return await this.dashboardTitle.isVisible();
  }
}
