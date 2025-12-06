import { Page, Locator } from '@playwright/test';

export class CartPage {
  readonly page: Page;
  readonly cartItems: Locator;
  readonly emptyCartMessage: Locator;
  readonly checkoutButton: Locator;
  readonly continueShoppingButton: Locator;
  readonly cartTotal: Locator;
  readonly removeButtons: Locator;
  readonly quantityInputs: Locator;
  readonly increaseButtons: Locator;
  readonly decreaseButtons: Locator;
  readonly promoCodeInput: Locator;
  readonly applyPromoButton: Locator;
  readonly subtotal: Locator;
  readonly tax: Locator;
  readonly shipping: Locator;

  constructor(page: Page) {
    this.page = page;
    this.cartItems = page.locator('[data-testid="cart-item"]');
    this.emptyCartMessage = page.locator('[data-testid="empty-cart-message"]');
    this.checkoutButton = page.getByRole('button', { name: /checkout/i }).first();
    this.continueShoppingButton = page.getByRole('button', { name: /continue shopping/i }).first();
    this.cartTotal = page.locator('[data-testid="cart-total"]');
    this.removeButtons = page.locator('[data-testid="remove-item"]');
    this.quantityInputs = page.locator('[data-testid="quantity-input"]');
    this.increaseButtons = page.locator('[data-testid="increase-quantity"]');
    this.decreaseButtons = page.locator('[data-testid="decrease-quantity"]');
    this.promoCodeInput = page.locator('[data-testid="promo-code-input"]');
    this.applyPromoButton = page.getByRole('button', { name: /apply/i });
    this.subtotal = page.locator('[data-testid="subtotal"]');
    this.tax = page.locator('[data-testid="tax"]');
    this.shipping = page.locator('[data-testid="shipping"]');
  }

  async goto() {
    await this.page.goto('/cart');
  }

  async getItemCount(): Promise<number> {
    return await this.cartItems.count();
  }

  async isCartEmpty(): Promise<boolean> {
    return await this.emptyCartMessage.isVisible();
  }

  async removeFirstItem() {
    await this.removeButtons.first().click();
  }

  async removeItemAt(index: number) {
    await this.removeButtons.nth(index).click();
  }

  async increaseQuantityAt(index: number) {
    await this.increaseButtons.nth(index).click();
  }

  async decreaseQuantityAt(index: number) {
    await this.decreaseButtons.nth(index).click();
  }

  async setQuantityAt(index: number, quantity: number) {
    await this.quantityInputs.nth(index).clear();
    await this.quantityInputs.nth(index).fill(quantity.toString());
  }

  async getCartTotal(): Promise<string | null> {
    return await this.cartTotal.textContent();
  }

  async getSubtotal(): Promise<string | null> {
    return await this.subtotal.textContent();
  }

  async applyPromoCode(code: string) {
    await this.promoCodeInput.fill(code);
    await this.applyPromoButton.click();
  }

  async proceedToCheckout() {
    await this.checkoutButton.click();
  }

  async continueShopping() {
    await this.continueShoppingButton.click();
  }
}
