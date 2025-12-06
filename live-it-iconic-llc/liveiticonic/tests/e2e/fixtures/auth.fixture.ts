import { test as base, expect } from '@playwright/test';
import { HomePage } from '../pages/HomePage';
import { ShopPage } from '../pages/ShopPage';
import { CartPage } from '../pages/CartPage';
import { CheckoutPage } from '../pages/CheckoutPage';
import { ProductDetailPage } from '../pages/ProductDetailPage';
import { AdminPage } from '../pages/AdminPage';

type TestFixtures = {
  homePage: HomePage;
  shopPage: ShopPage;
  cartPage: CartPage;
  checkoutPage: CheckoutPage;
  productDetailPage: ProductDetailPage;
  adminPage: AdminPage;
  authenticatedPage: void;
};

export const test = base.extend<TestFixtures>({
  homePage: async ({ page }, provide) => {
    const homePage = new HomePage(page);
    await provide(homePage);
  },

  shopPage: async ({ page }, provide) => {
    const shopPage = new ShopPage(page);
    await provide(shopPage);
  },

  cartPage: async ({ page }, provide) => {
    const cartPage = new CartPage(page);
    await provide(cartPage);
  },

  checkoutPage: async ({ page }, provide) => {
    const checkoutPage = new CheckoutPage(page);
    await provide(checkoutPage);
  },

  productDetailPage: async ({ page }, provide) => {
    const productDetailPage = new ProductDetailPage(page);
    await provide(productDetailPage);
  },

  adminPage: async ({ page }, provide) => {
    const adminPage = new AdminPage(page);
    await provide(adminPage);
  },

  authenticatedPage: async ({ page }, provide) => {
    // Authenticate before test runs
    // This can be expanded with actual authentication logic
    await page.context().addCookies([
      {
        name: 'auth_token',
        value: 'test-token',
        domain: 'localhost',
        path: '/',
        httpOnly: true,
        secure: false,
        sameSite: 'Lax',
        expires: Date.now() / 1000 + 86400, // 24 hours
      },
    ]);

    await provide();

    // Cleanup if needed
    await page.context().clearCookies();
  },
});

export { expect };
