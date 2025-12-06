import { test, expect } from '@playwright/test';

test.describe('Comprehensive Nutrition Management', () => {
  test.beforeEach(async ({ page }) => {
    // Use Adaptive+ tier auth for nutrition features
    await page.context().storageState({ 
      path: 'tests/fixtures/auth/adaptive-auth.json' 
    });
    await page.goto('/dashboard');
  });

  test('complete food database search and nutrition tracking', async ({ page }) => {
    // Step 1: Navigate to nutrition management
    await page.click('[data-testid="nutrition-menu"]');
    await page.click('[data-testid="food-database-link"]');
    await expect(page).toHaveURL('/nutrition/food-database');

    // Step 2: Search comprehensive food database (40k+ items)
    await page.fill('[data-testid="food-search-input"]', 'chicken breast');
    await page.click('[data-testid="search-button"]');

    // Verify search results
    await expect(page.locator('[data-testid="search-results"]')).toBeVisible();
    await expect(page.locator('[data-testid="food-item"]')).toHaveCount({ min: 5 });

    // Verify detailed nutrition information
    await page.click('[data-testid="food-item"]:first-child');
    await expect(page.locator('[data-testid="nutrition-panel"]')).toBeVisible();
    await expect(page.locator('[data-testid="calories-per-100g"]')).toBeVisible();
    await expect(page.locator('[data-testid="protein-content"]')).toBeVisible();
    await expect(page.locator('[data-testid="carbs-content"]')).toBeVisible();
    await expect(page.locator('[data-testid="fat-content"]')).toBeVisible();
    await expect(page.locator('[data-testid="micronutrients-panel"]')).toBeVisible();

    // Step 3: Add food to daily log
    await page.fill('[data-testid="serving-size-input"]', '200');
    await page.selectOption('[data-testid="meal-type-select"]', 'lunch');
    await page.click('[data-testid="add-to-log-button"]');

    // Verify food added to log
    await expect(page.locator('[data-testid="food-logged-message"]')).toBeVisible();
    await expect(page.locator('[data-testid="updated-daily-totals"]')).toBeVisible();

    // Step 4: View daily nutrition summary
    await page.click('[data-testid="daily-log-tab"]');
    
    await expect(page.locator('[data-testid="daily-calories"]')).toBeVisible();
    await expect(page.locator('[data-testid="daily-protein"]')).toBeVisible();
    await expect(page.locator('[data-testid="daily-carbs"]')).toBeVisible();
    await expect(page.locator('[data-testid="daily-fat"]')).toBeVisible();

    // Verify macro ratio visualization
    await expect(page.locator('[data-testid="macro-chart"]')).toBeVisible();
    await expect(page.locator('[data-testid="calorie-goal-progress"]')).toBeVisible();

    // Step 5: Advanced search filters
    await page.click('[data-testid="advanced-filters-button"]');
    
    // Filter by dietary preferences
    await page.check('[data-testid="filter-high-protein"]');
    await page.check('[data-testid="filter-low-carb"]');
    await page.selectOption('[data-testid="food-category-filter"]', 'meat-poultry');
    
    await page.click('[data-testid="apply-filters-button"]');

    // Verify filtered results
    await expect(page.locator('[data-testid="filtered-results"]')).toBeVisible();
    await expect(page.locator('[data-testid="results-count"]')).toContainText('high protein');
  });

  test('recipe creation with auto nutrition calculation', async ({ page }) => {
    await page.goto('/nutrition/recipes');

    // Step 1: Create new recipe
    await page.click('[data-testid="create-recipe-button"]');
    
    await page.fill('[data-testid="recipe-name-input"]', 'Power Bowl');
    await page.fill('[data-testid="recipe-description"]', 'High protein post-workout meal');
    await page.selectOption('[data-testid="recipe-category"]', 'main-dish');
    await page.fill('[data-testid="servings-count"]', '2');

    // Step 2: Add ingredients with quantities
    await page.click('[data-testid="add-ingredient-button"]');
    await page.fill('[data-testid="ingredient-search-1"]', 'quinoa cooked');
    await page.click('[data-testid="ingredient-result-1"]');
    await page.fill('[data-testid="ingredient-quantity-1"]', '200');

    await page.click('[data-testid="add-ingredient-button"]');
    await page.fill('[data-testid="ingredient-search-2"]', 'chicken breast grilled');
    await page.click('[data-testid="ingredient-result-2"]');
    await page.fill('[data-testid="ingredient-quantity-2"]', '150');

    await page.click('[data-testid="add-ingredient-button"]');
    await page.fill('[data-testid="ingredient-search-3"]', 'avocado');
    await page.click('[data-testid="ingredient-result-3"]');
    await page.fill('[data-testid="ingredient-quantity-3"]', '100');

    // Step 3: Auto calculate nutrition
    await page.click('[data-testid="calculate-nutrition-button"]');

    // Verify automatic nutrition calculation
    await expect(page.locator('[data-testid="recipe-calories"]')).toBeVisible();
    await expect(page.locator('[data-testid="recipe-protein"]')).toBeVisible();
    await expect(page.locator('[data-testid="recipe-carbs"]')).toBeVisible();
    await expect(page.locator('[data-testid="recipe-fat"]')).toBeVisible();
    
    // Verify per-serving calculations
    await expect(page.locator('[data-testid="per-serving-calories"]')).toBeVisible();
    await expect(page.locator('[data-testid="per-serving-protein"]')).toBeVisible();

    // Step 4: Add cooking instructions
    await page.fill('[data-testid="cooking-instructions"]', 
      '1. Cook quinoa according to package directions\n2. Grill chicken breast until internal temp reaches 165°F\n3. Slice avocado\n4. Combine all ingredients in bowl');
    
    await page.fill('[data-testid="prep-time"]', '15');
    await page.fill('[data-testid="cook-time"]', '20');

    // Step 5: Save recipe
    await page.click('[data-testid="save-recipe-button"]');

    // Verify recipe saved
    await expect(page.locator('[data-testid="recipe-saved-message"]')).toBeVisible();
    await expect(page).toHaveURL('/nutrition/recipes/power-bowl');

    // Step 6: Add recipe to meal plan
    await page.click('[data-testid="add-to-meal-plan-button"]');
    await page.selectOption('[data-testid="meal-plan-day"]', 'monday');
    await page.selectOption('[data-testid="meal-plan-type"]', 'dinner');
    await page.click('[data-testid="confirm-add-to-plan-button"]');

    await expect(page.locator('[data-testid="added-to-meal-plan"]')).toBeVisible();
  });

  test('weekly meal planning and scheduling', async ({ page }) => {
    await page.goto('/nutrition/meal-planning');

    // Step 1: View weekly meal plan calendar
    await expect(page.locator('[data-testid="meal-plan-calendar"]')).toBeVisible();
    await expect(page.locator('[data-testid="current-week"]')).toBeVisible();

    // Step 2: Add meals to specific days
    await page.click('[data-testid="monday-breakfast-slot"]');
    
    // Search and select meal
    await page.fill('[data-testid="meal-search"]', 'protein pancakes');
    await page.click('[data-testid="meal-result-protein-pancakes"]');
    await page.fill('[data-testid="meal-servings"]', '1');
    await page.click('[data-testid="add-meal-button"]');

    // Verify meal added to calendar
    await expect(page.locator('[data-testid="monday-breakfast-meal"]')).toContainText('Protein Pancakes');

    // Step 3: Add multiple meals for complete day
    await page.click('[data-testid="monday-lunch-slot"]');
    await page.fill('[data-testid="meal-search"]', 'power bowl');
    await page.click('[data-testid="meal-result-power-bowl"]');
    await page.click('[data-testid="add-meal-button"]');

    await page.click('[data-testid="monday-dinner-slot"]');
    await page.fill('[data-testid="meal-search"]', 'salmon teriyaki');
    await page.click('[data-testid="meal-result-salmon-teriyaki"]');
    await page.click('[data-testid="add-meal-button"]');

    // Step 4: View daily nutrition totals
    await page.click('[data-testid="monday-nutrition-summary"]');
    
    await expect(page.locator('[data-testid="daily-calorie-total"]')).toBeVisible();
    await expect(page.locator('[data-testid="daily-protein-total"]')).toBeVisible();
    await expect(page.locator('[data-testid="macro-distribution-chart"]')).toBeVisible();
    
    // Verify against goals
    await expect(page.locator('[data-testid="calorie-goal-status"]')).toBeVisible();
    await expect(page.locator('[data-testid="protein-goal-status"]')).toBeVisible();

    // Step 5: Copy day to other days
    await page.click('[data-testid="copy-day-button"]');
    await page.check('[data-testid="copy-to-wednesday"]');
    await page.check('[data-testid="copy-to-friday"]');
    await page.click('[data-testid="confirm-copy-button"]');

    // Verify meals copied
    await expect(page.locator('[data-testid="wednesday-breakfast-meal"]')).toContainText('Protein Pancakes');
    await expect(page.locator('[data-testid="friday-breakfast-meal"]')).toContainText('Protein Pancakes');

    // Step 6: Generate weekly nutrition report
    await page.click('[data-testid="weekly-report-button"]');
    
    await expect(page.locator('[data-testid="weekly-calories-average"]')).toBeVisible();
    await expect(page.locator('[data-testid="weekly-protein-average"]')).toBeVisible();
    await expect(page.locator('[data-testid="weekly-nutrition-chart"]')).toBeVisible();
    await expect(page.locator('[data-testid="goal-adherence-percentage"]')).toBeVisible();
  });

  test('automated grocery list generation', async ({ page }) => {
    await page.goto('/nutrition/meal-planning');

    // Ensure we have a planned week
    await page.click('[data-testid="quick-plan-week-button"]');
    await page.selectOption('[data-testid="plan-template"]', 'balanced-macro');
    await page.click('[data-testid="generate-plan-button"]');

    // Step 1: Generate grocery list from meal plan
    await page.click('[data-testid="generate-grocery-list"]');
    
    // Verify grocery list generation options
    await expect(page.locator('[data-testid="grocery-list-options"]')).toBeVisible();
    await page.check('[data-testid="include-current-week"]');
    await page.check('[data-testid="include-next-week"]');
    await page.check('[data-testid="consolidate-similar-items"]');
    
    await page.click('[data-testid="generate-list-button"]');

    // Step 2: Review generated grocery list
    await expect(page).toHaveURL('/nutrition/grocery-list');
    await expect(page.locator('[data-testid="grocery-list-items"]')).toBeVisible();

    // Verify categorized grocery list
    await expect(page.locator('[data-testid="category-proteins"]')).toBeVisible();
    await expect(page.locator('[data-testid="category-vegetables"]')).toBeVisible();
    await expect(page.locator('[data-testid="category-grains"]')).toBeVisible();
    await expect(page.locator('[data-testid="category-dairy"]')).toBeVisible();

    // Verify item quantities calculated
    await expect(page.locator('[data-testid="item-chicken-breast"]')).toContainText('lbs');
    await expect(page.locator('[data-testid="item-quinoa"]')).toContainText('cups');

    // Step 3: Customize grocery list
    await page.click('[data-testid="edit-list-button"]');
    
    // Add custom items
    await page.click('[data-testid="add-custom-item-button"]');
    await page.fill('[data-testid="custom-item-name"]', 'Coconut Oil');
    await page.fill('[data-testid="custom-item-quantity"]', '1 jar');
    await page.selectOption('[data-testid="custom-item-category"]', 'pantry');
    await page.click('[data-testid="add-item-button"]');

    // Remove items not needed
    await page.click('[data-testid="remove-item-avocado"]');
    await page.click('[data-testid="confirm-remove-button"]');

    // Modify quantities
    await page.click('[data-testid="edit-quantity-salmon"]');
    await page.fill('[data-testid="quantity-input-salmon"]', '2');
    await page.click('[data-testid="save-quantity-button"]');

    // Step 4: Export grocery list
    await page.click('[data-testid="export-list-button"]');
    
    // Test different export formats
    await page.click('[data-testid="export-pdf"]');
    await expect(page.locator('[data-testid="pdf-generated"]')).toBeVisible();

    await page.click('[data-testid="export-email"]');
    await page.fill('[data-testid="email-address"]', 'user@example.com');
    await page.click('[data-testid="send-email-button"]');
    await expect(page.locator('[data-testid="email-sent"]')).toBeVisible();

    // Step 5: Shopping mode
    await page.click('[data-testid="shopping-mode-button"]');
    
    // Verify mobile-optimized shopping interface
    await expect(page.locator('[data-testid="shopping-checklist"]')).toBeVisible();
    await expect(page.locator('[data-testid="check-off-items"]')).toBeVisible();

    // Test checking off items
    await page.click('[data-testid="check-chicken-breast"]');
    await page.click('[data-testid="check-quinoa"]');
    
    await expect(page.locator('[data-testid="progress-bar"]')).toBeVisible();
    await expect(page.locator('[data-testid="items-remaining"]')).toContainText('remaining');

    // Step 6: Store integration (if available)
    await page.click('[data-testid="store-integration-button"]');
    
    // Test store locator
    await page.fill('[data-testid="zip-code-input"]', '90210');
    await page.click('[data-testid="find-stores-button"]');
    
    await expect(page.locator('[data-testid="nearby-stores"]')).toBeVisible();
    await expect(page.locator('[data-testid="store-whole-foods"]')).toBeVisible();
    
    // Select store and check availability
    await page.click('[data-testid="select-store-whole-foods"]');
    await expect(page.locator('[data-testid="item-availability"]')).toBeVisible();
  });

  test('nutrition goal setting and tracking', async ({ page }) => {
    await page.goto('/nutrition/goals');

    // Step 1: Set personalized nutrition goals
    await page.click('[data-testid="set-goals-button"]');
    
    // Personal information
    await page.fill('[data-testid="current-weight"]', '180');
    await page.fill('[data-testid="target-weight"]', '175');
    await page.selectOption('[data-testid="activity-level"]', 'moderately-active');
    await page.selectOption('[data-testid="primary-goal"]', 'fat-loss');

    // Macro preferences
    await page.selectOption('[data-testid="macro-approach"]', 'custom');
    await page.fill('[data-testid="protein-percentage"]', '30');
    await page.fill('[data-testid="carbs-percentage"]', '40');
    await page.fill('[data-testid="fat-percentage"]', '30');

    await page.click('[data-testid="calculate-goals-button"]');

    // Verify calculated goals
    await expect(page.locator('[data-testid="daily-calorie-goal"]')).toBeVisible();
    await expect(page.locator('[data-testid="daily-protein-goal"]')).toBeVisible();
    await expect(page.locator('[data-testid="daily-carbs-goal"]')).toBeVisible();
    await expect(page.locator('[data-testid="daily-fat-goal"]')).toBeVisible();

    // Step 2: Track progress over time
    await page.click('[data-testid="view-progress-button"]');
    
    await expect(page.locator('[data-testid="progress-chart"]')).toBeVisible();
    await expect(page.locator('[data-testid="weight-trend"]')).toBeVisible();
    await expect(page.locator('[data-testid="goal-adherence-chart"]')).toBeVisible();

    // Step 3: Adjust goals based on progress
    await page.click('[data-testid="adjust-goals-button"]');
    
    await page.fill('[data-testid="new-target-weight"]', '173');
    await page.selectOption('[data-testid="timeline-adjustment"]', 'accelerate');
    
    await page.click('[data-testid="update-goals-button"]');
    
    await expect(page.locator('[data-testid="goals-updated"]')).toBeVisible();
  });

  test('nutrition access control for core tier users', async ({ page }) => {
    // Test with Core tier user
    await page.context().storageState({ 
      path: 'tests/fixtures/auth/core-auth.json' 
    });
    
    await page.goto('/nutrition/food-database');
    
    // Verify limited access
    await expect(page.locator('[data-testid="basic-food-search"]')).toBeVisible();
    await expect(page.locator('[data-testid="advanced-features-locked"]')).toBeVisible();
    
    // Verify no access to advanced features
    await expect(page.locator('[data-testid="meal-planning"]')).not.toBeVisible();
    await expect(page.locator('[data-testid="grocery-list-generation"]')).not.toBeVisible();
    await expect(page.locator('[data-testid="recipe-creation"]')).not.toBeVisible();
    
    // Verify upgrade prompts
    await expect(page.locator('[data-testid="upgrade-to-adaptive-button"]')).toBeVisible();
  });
});