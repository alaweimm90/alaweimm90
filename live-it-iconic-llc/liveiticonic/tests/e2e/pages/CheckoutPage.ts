import { Page, Locator } from '@playwright/test';

export class CheckoutPage {
  readonly page: Page;
  readonly emailInput: Locator;
  readonly firstNameInput: Locator;
  readonly lastNameInput: Locator;
  readonly addressInput: Locator;
  readonly cityInput: Locator;
  readonly stateInput: Locator;
  readonly zipInput: Locator;
  readonly countryInput: Locator;
  readonly phoneInput: Locator;
  readonly continueButton: Locator;
  readonly backButton: Locator;
  readonly paymentMethodSelect: Locator;
  readonly cardNumberInput: Locator;
  readonly cardExpiryInput: Locator;
  readonly cardCvcInput: Locator;
  readonly placeOrderButton: Locator;
  readonly orderSummary: Locator;
  readonly shippingMethod: Locator;
  readonly billingAddressCheckbox: Locator;

  constructor(page: Page) {
    this.page = page;
    this.emailInput = page.getByLabel(/email/i).first();
    this.firstNameInput = page.getByLabel(/first name/i).first();
    this.lastNameInput = page.getByLabel(/last name/i).first();
    this.addressInput = page.getByLabel(/address/i).first();
    this.cityInput = page.getByLabel(/city/i).first();
    this.stateInput = page.getByLabel(/state|province/i).first();
    this.zipInput = page.getByLabel(/zip|postal/i).first();
    this.countryInput = page.getByLabel(/country/i).first();
    this.phoneInput = page.getByLabel(/phone/i).first();
    this.continueButton = page.getByRole('button', { name: /continue/i }).first();
    this.backButton = page.getByRole('button', { name: /back/i }).first();
    this.paymentMethodSelect = page.locator('[data-testid="payment-method"]');
    this.cardNumberInput = page.locator('input[placeholder*="card number" i], input[placeholder*="4242" i]').first();
    this.cardExpiryInput = page.locator('input[placeholder*="MM/YY" i], input[placeholder*="expiry" i]').first();
    this.cardCvcInput = page.locator('input[placeholder*="CVC" i], input[placeholder*="CVV" i]').first();
    this.placeOrderButton = page.getByRole('button', { name: /place order|pay|complete/i }).first();
    this.orderSummary = page.locator('[data-testid="order-summary"]');
    this.shippingMethod = page.locator('[data-testid="shipping-method"]');
    this.billingAddressCheckbox = page.locator('[data-testid="billing-address-same"]');
  }

  async goto() {
    await this.page.goto('/checkout');
  }

  async fillShippingInfo(shippingInfo: {
    email: string;
    firstName: string;
    lastName: string;
    address: string;
    city: string;
    state: string;
    zip: string;
    country?: string;
    phone?: string;
  }) {
    await this.emailInput.fill(shippingInfo.email);
    await this.firstNameInput.fill(shippingInfo.firstName);
    await this.lastNameInput.fill(shippingInfo.lastName);
    await this.addressInput.fill(shippingInfo.address);
    await this.cityInput.fill(shippingInfo.city);
    await this.stateInput.fill(shippingInfo.state);
    await this.zipInput.fill(shippingInfo.zip);

    if (shippingInfo.country && (await this.countryInput.isVisible())) {
      await this.countryInput.fill(shippingInfo.country);
    }

    if (shippingInfo.phone && (await this.phoneInput.isVisible())) {
      await this.phoneInput.fill(shippingInfo.phone);
    }
  }

  async fillPaymentInfo(paymentInfo: {
    cardNumber: string;
    expiryDate: string;
    cvc: string;
    cardholderName?: string;
  }) {
    if (await this.cardNumberInput.isVisible()) {
      await this.cardNumberInput.fill(paymentInfo.cardNumber);
    }
    if (await this.cardExpiryInput.isVisible()) {
      await this.cardExpiryInput.fill(paymentInfo.expiryDate);
    }
    if (await this.cardCvcInput.isVisible()) {
      await this.cardCvcInput.fill(paymentInfo.cvc);
    }
  }

  async selectShippingMethod(method: string) {
    if (await this.shippingMethod.isVisible()) {
      await this.page.locator(`[data-shipping="${method}"]`).click();
    }
  }

  async continueToPaymentStep() {
    await this.continueButton.click();
  }

  async placeOrder() {
    await this.placeOrderButton.click();
  }

  async goBack() {
    await this.backButton.click();
  }

  async setSameBillingAddress(same: boolean) {
    if (await this.billingAddressCheckbox.isVisible()) {
      const isChecked = await this.billingAddressCheckbox.isChecked();
      if (isChecked !== same) {
        await this.billingAddressCheckbox.click();
      }
    }
  }
}
