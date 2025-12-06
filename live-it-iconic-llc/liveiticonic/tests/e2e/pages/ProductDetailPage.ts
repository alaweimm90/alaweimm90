import { Page, Locator } from '@playwright/test';

export class ProductDetailPage {
  readonly page: Page;
  readonly productTitle: Locator;
  readonly productPrice: Locator;
  readonly productDescription: Locator;
  readonly addToCartButton: Locator;
  readonly quantityInput: Locator;
  readonly sizeSelector: Locator;
  readonly colorSelector: Locator;
  readonly backButton: Locator;
  readonly rating: Locator;
  readonly reviews: Locator;

  constructor(page: Page) {
    this.page = page;
    this.productTitle = page.locator('h1, [data-testid="product-title"]').first();
    this.productPrice = page.locator('[data-testid="product-price"]');
    this.productDescription = page.locator('[data-testid="product-description"]');
    this.addToCartButton = page.getByRole('button', { name: /add to cart/i }).first();
    this.quantityInput = page.locator('[data-testid="quantity-input"], input[type="number"]').first();
    this.sizeSelector = page.locator('[data-testid="size-selector"]');
    this.colorSelector = page.locator('[data-testid="color-selector"]');
    this.backButton = page.getByRole('button', { name: /back/i }).first();
    this.rating = page.locator('[data-testid="product-rating"]');
    this.reviews = page.locator('[data-testid="product-reviews"]');
  }

  async getProductTitle(): Promise<string | null> {
    return await this.productTitle.textContent();
  }

  async getProductPrice(): Promise<string | null> {
    return await this.productPrice.textContent();
  }

  async selectSize(size: string) {
    await this.sizeSelector.click();
    await this.page.locator(`text="${size}"`).click();
  }

  async selectColor(color: string) {
    await this.colorSelector.click();
    await this.page.locator(`[data-color="${color}"]`).click();
  }

  async setQuantity(quantity: number) {
    await this.quantityInput.clear();
    await this.quantityInput.fill(quantity.toString());
  }

  async addToCart() {
    await this.addToCartButton.click();
  }

  async goBack() {
    await this.backButton.click();
  }

  async isAddToCartButtonEnabled(): Promise<boolean> {
    return await this.addToCartButton.isEnabled();
  }

  async getRating(): Promise<string | null> {
    return await this.rating.textContent();
  }
}
