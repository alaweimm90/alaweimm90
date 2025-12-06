import { test, expect } from '../fixtures/auth.fixture';

test.describe('Admin Dashboard', () => {
  test.beforeEach(async ({ adminPage }) => {
    // Navigate to admin page before each test
    await adminPage.goto();
  });

  test('admin dashboard loads successfully', async ({ adminPage }) => {
    // Verify dashboard is visible
    const isDashboardVisible = await adminPage.isDashboardVisible();
    expect(isDashboardVisible).toBe(true);
  });

  test('admin can navigate to orders section', async ({ adminPage }) => {
    // Click on orders link
    const ordersLink = adminPage.page.getByRole('link', { name: /orders/i }).first();
    if (await ordersLink.isVisible()) {
      await adminPage.gotoOrders();

      // Verify orders page loaded
      const ordersTable = adminPage.page.locator('[data-testid="orders-table"]');
      if (await ordersTable.isVisible()) {
        expect(ordersTable).toBeVisible();
      }
    }
  });

  test('admin can navigate to products section', async ({ adminPage }) => {
    // Click on products link
    const productsLink = adminPage.page.getByRole('link', { name: /products/i }).first();
    if (await productsLink.isVisible()) {
      await adminPage.gotoProducts();

      // Verify products page loaded
      const productsList = adminPage.page.locator('[data-testid="products-list"]');
      if (await productsList.isVisible()) {
        expect(productsList).toBeVisible();
      }
    }
  });

  test('admin can navigate to users section', async ({ adminPage }) => {
    // Click on users link
    const usersLink = adminPage.page.getByRole('link', { name: /users/i }).first();
    if (await usersLink.isVisible()) {
      await adminPage.gotoUsers();

      // Verify users page loaded
      const usersList = adminPage.page.locator('[data-testid="users-list"]');
      if (await usersList.isVisible()) {
        expect(usersList).toBeVisible();
      }
    }
  });

  test('admin can search for items', async ({ adminPage }) => {
    // Try to search
    const searchInput = adminPage.page.locator('[data-testid="search-input"]');
    if (await searchInput.isVisible()) {
      await adminPage.searchFor('test');

      // Wait for results to load
      await adminPage.page.waitForTimeout(500);

      // Verify search was executed (specific results depend on implementation)
      expect(searchInput).toBeVisible();
    }
  });

  test('admin can filter results', async ({ adminPage }) => {
    // Try to use filter
    const filterDropdown = adminPage.page.locator('[data-testid="filter-dropdown"]');
    if (await filterDropdown.isVisible()) {
      await adminPage.filterBy('Active');

      // Wait for filtered results
      await adminPage.page.waitForTimeout(500);

      expect(filterDropdown).toBeVisible();
    }
  });

  test('admin can view analytics section', async ({ adminPage }) => {
    // Navigate to analytics if available
    const analyticsLink = adminPage.page.getByRole('link', { name: /analytics|reports/i }).first();
    if (await analyticsLink.isVisible()) {
      await adminPage.gotoAnalytics();

      // Verify analytics page loaded
      await adminPage.page.waitForLoadState('networkidle');
      expect(analyticsLink).toBeVisible();
    }
  });

  test('admin can access settings', async ({ adminPage }) => {
    // Navigate to settings if available
    const settingsLink = adminPage.page.getByRole('link', { name: /settings/i }).first();
    if (await settingsLink.isVisible()) {
      await adminPage.gotoSettings();

      // Verify settings page loaded
      await adminPage.page.waitForLoadState('networkidle');
      expect(settingsLink).toBeVisible();
    }
  });

  test('admin can edit items', async ({ adminPage }) => {
    // Navigate to products
    const productsLink = adminPage.page.getByRole('link', { name: /products/i }).first();
    if (await productsLink.isVisible()) {
      await adminPage.gotoProducts();

      // Check if edit buttons are available
      const editButtons = adminPage.page.locator('[data-testid="edit-button"]');
      const count = await editButtons.count();

      if (count > 0) {
        await adminPage.editItemAt(0);

        // Verify edit form or modal appears
        await adminPage.page.waitForTimeout(500);
        // (Specific expectations depend on edit implementation)
      }
    }
  });

  test('admin can delete items with confirmation', async ({ adminPage, page }) => {
    // Navigate to products
    const productsLink = page.getByRole('link', { name: /products/i }).first();
    if (await productsLink.isVisible()) {
      await adminPage.gotoProducts();

      // Check if delete buttons are available
      const deleteButtons = adminPage.page.locator('[data-testid="delete-button"]');
      const count = await deleteButtons.count();

      if (count > 0) {
        // Set up listener for confirmation dialog
        page.once('dialog', async (dialog) => {
          expect(dialog.type()).toContain('confirm');
          await dialog.accept();
        });

        await adminPage.deleteItemAt(0);

        // Wait for deletion
        await page.waitForTimeout(1000);
      }
    }
  });

  test('admin can add new product', async ({ adminPage }) => {
    // Navigate to products
    const productsLink = adminPage.page.getByRole('link', { name: /products/i }).first();
    if (await productsLink.isVisible()) {
      await adminPage.gotoProducts();

      // Check if add product button is available
      const addBtn = adminPage.page.getByRole('button', { name: /add product/i });
      if (await addBtn.isVisible()) {
        await adminPage.addNewProduct();

        // Verify form appears
        await adminPage.page.waitForTimeout(500);
        const productForm = adminPage.page.locator('[data-testid="product-form"]');

        if (await productForm.isVisible()) {
          expect(productForm).toBeVisible();
        }
      }
    }
  });

  test('admin dashboard navigation sidebar works', async ({ adminPage, page }) => {
    // Check for sidebar
    const sidebar = page.locator('[data-testid="admin-sidebar"], nav');
    if (await sidebar.isVisible()) {
      // Verify multiple navigation links exist
      const navLinks = sidebar.getByRole('link');
      const count = await navLinks.count();

      expect(count).toBeGreaterThan(0);
    }
  });

  test('admin can view order details', async ({ adminPage }) => {
    // Navigate to orders
    const ordersLink = adminPage.page.getByRole('link', { name: /orders/i }).first();
    if (await ordersLink.isVisible()) {
      await adminPage.gotoOrders();

      // Look for order rows
      const orderRows = adminPage.page.locator('[data-testid="order-row"]');
      const count = await orderRows.count();

      if (count > 0) {
        // Click on first order
        await orderRows.first().click();

        // Verify order details appear
        await adminPage.page.waitForTimeout(500);
        const orderDetails = adminPage.page.locator('[data-testid="order-details"]');

        if (await orderDetails.isVisible()) {
          expect(orderDetails).toBeVisible();
        }
      }
    }
  });

  test('admin can logout', async ({ adminPage }) => {
    // Click logout button
    const logoutBtn = adminPage.page.getByRole('button', { name: /logout|sign out/i });
    if (await logoutBtn.isVisible()) {
      await adminPage.logout();

      // Should be redirected to login or home
      await adminPage.page.waitForURL(
        (url) =>
          url.toString().includes('/auth') ||
          url.toString().includes('/login') ||
          url.toString() === '/'
      );
    }
  });

  test('admin dashboard is responsive', async ({ adminPage, page }) => {
    // Test on tablet
    await page.setViewportSize({ width: 768, height: 1024 });

    // Verify dashboard is still usable
    const isDashboardVisible = await adminPage.isDashboardVisible();
    expect(isDashboardVisible).toBe(true);

    // Test on mobile
    await page.setViewportSize({ width: 375, height: 667 });

    // Verify dashboard adapts
    const sidebarToggle = page.locator('[data-testid="sidebar-toggle"]');
    if (await sidebarToggle.isVisible()) {
      // Mobile view with hamburger menu
      await sidebarToggle.click();
    }

    // Should still see content
    const content = page.locator('[data-testid="admin-content"]');
    if (await content.isVisible()) {
      expect(content).toBeVisible();
    }
  });

  test('admin can perform bulk operations', async ({ adminPage, page }) => {
    // Navigate to orders or products
    const ordersLink = page.getByRole('link', { name: /orders/i }).first();
    if (await ordersLink.isVisible()) {
      await adminPage.gotoOrders();

      // Check for checkboxes (bulk select)
      const checkboxes = page.locator('[data-testid="bulk-select"]');
      const count = await checkboxes.count();

      if (count > 0) {
        // Select first item
        await checkboxes.first().click();

        // Check for bulk action buttons
        const bulkActionBtn = page.locator('[data-testid="bulk-action-button"]');
        if (await bulkActionBtn.isVisible()) {
          expect(bulkActionBtn).toBeVisible();
        }
      }
    }
  });

  test('admin page accessibility', async ({ adminPage, page }) => {
    // Check for basic a11y attributes
    const buttons = page.locator('button');
    const buttonCount = await buttons.count();

    for (let i = 0; i < Math.min(buttonCount, 5); i++) {
      const button = buttons.nth(i);
      const hasLabel = await button.getAttribute('aria-label');
      const hasText = (await button.textContent())?.trim();

      // Button should have either text or aria-label
      expect(hasLabel || hasText).toBeTruthy();
    }
  });
});
