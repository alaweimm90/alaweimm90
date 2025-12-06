import { test, expect } from '@playwright/test';

test.describe('Advanced Protocol Management (PEDs)', () => {
  test.beforeEach(async ({ page }) => {
    // Use Performance+ tier auth for PEDs access
    await page.context().storageState({ 
      path: 'tests/fixtures/auth/performance-auth.json' 
    });
    await page.goto('/dashboard');
  });

  test('complete PEDs protocol request and approval flow', async ({ page }) => {
    // Step 1: Navigate to PEDs management
    await page.click('[data-testid="advanced-protocols-menu"]');
    await page.click('[data-testid="peds-management-link"]');
    await expect(page).toHaveURL('/protocols/peds');

    // Step 2: View available protocols
    await expect(page.locator('[data-testid="peds-protocols-list"]')).toBeVisible();
    await expect(page.locator('[data-testid="protocol-testosterone"]')).toBeVisible();
    await expect(page.locator('[data-testid="protocol-growth-hormone"]')).toBeVisible();
    await expect(page.locator('[data-testid="protocol-sarms"]')).toBeVisible();

    // Step 3: Select testosterone protocol
    await page.click('[data-testid="protocol-testosterone"]');
    await expect(page).toHaveURL('/protocols/peds/testosterone');

    // Verify protocol details displayed
    await expect(page.locator('[data-testid="protocol-name"]')).toHaveText('Testosterone Enanthate');
    await expect(page.locator('[data-testid="protocol-description"]')).toBeVisible();
    await expect(page.locator('[data-testid="protocol-dosage"]')).toBeVisible();
    await expect(page.locator('[data-testid="protocol-duration"]')).toBeVisible();
    await expect(page.locator('[data-testid="protocol-side-effects"]')).toBeVisible();

    // Step 4: Request medical consultation
    await page.click('[data-testid="request-consultation-button"]');
    
    // Fill consultation request form
    await page.fill('[data-testid="medical-history-textarea"]', 'No significant medical history');
    await page.fill('[data-testid="current-medications-textarea"]', 'None');
    await page.fill('[data-testid="goals-textarea"]', 'Increase lean muscle mass and strength');
    await page.check('[data-testid="lab-work-consent-checkbox"]');
    await page.check('[data-testid="medical-supervision-consent-checkbox"]');

    await page.click('[data-testid="submit-consultation-request-button"]');

    // Verify consultation request submitted
    await expect(page.locator('[data-testid="consultation-status"]')).toHaveText('Pending Medical Review');
    await expect(page.locator('[data-testid="consultation-id"]')).toBeVisible();

    // Step 5: Simulate medical approval (in test environment)
    // This would normally require medical professional login
    await page.goto('/admin/medical-approvals');
    await page.click('[data-testid="approve-consultation-button"]');
    
    // Add medical notes
    await page.fill('[data-testid="medical-notes-textarea"]', 'Patient cleared for testosterone protocol. Monitor liver function and lipid panel.');
    await page.selectOption('[data-testid="monitoring-frequency-select"]', 'monthly');
    await page.click('[data-testid="confirm-approval-button"]');

    // Step 6: Protocol activation
    await page.goto('/protocols/peds');
    await expect(page.locator('[data-testid="protocol-status-testosterone"]')).toHaveText('Approved');
    
    await page.click('[data-testid="activate-protocol-testosterone"]');
    await expect(page.locator('[data-testid="protocol-status-testosterone"]')).toHaveText('Active');

    // Step 7: Daily tracking setup
    await page.click('[data-testid="setup-tracking-button"]');
    
    // Configure tracking parameters
    await page.check('[data-testid="track-injection-sites"]');
    await page.check('[data-testid="track-side-effects"]');
    await page.check('[data-testid="track-biomarkers"]');
    await page.selectOption('[data-testid="injection-frequency-select"]', 'twice-weekly');
    
    await page.click('[data-testid="save-tracking-config-button"]');

    // Step 8: Log first dose
    await page.click('[data-testid="log-dose-button"]');
    
    await page.fill('[data-testid="dose-amount-input"]', '200');
    await page.selectOption('[data-testid="injection-site-select"]', 'vastus-lateralis-right');
    await page.fill('[data-testid="notes-textarea"]', 'First injection, no immediate side effects');
    
    await page.click('[data-testid="save-dose-log-button"]');

    // Verify dose logged
    await expect(page.locator('[data-testid="dose-log-entry"]')).toBeVisible();
    await expect(page.locator('[data-testid="next-dose-date"]')).toBeVisible();

    // Step 9: Side effects monitoring
    await page.click('[data-testid="report-side-effects-button"]');
    
    // Test side effects form
    await page.check('[data-testid="side-effect-acne"]');
    await page.selectOption('[data-testid="severity-acne"]', 'mild');
    await page.fill('[data-testid="side-effect-notes"]', 'Minor acne on back, manageable');
    
    await page.click('[data-testid="submit-side-effects-button"]');

    // Step 10: Safety alerts system
    // Simulate concerning side effects
    await page.click('[data-testid="report-side-effects-button"]');
    await page.check('[data-testid="side-effect-chest-pain"]');
    await page.selectOption('[data-testid="severity-chest-pain"]', 'severe');
    
    await page.click('[data-testid="submit-side-effects-button"]');

    // Verify safety alert triggered
    await expect(page.locator('[data-testid="safety-alert"]')).toBeVisible();
    await expect(page.locator('[data-testid="safety-alert"]')).toContainText('Seek immediate medical attention');
    await expect(page.locator('[data-testid="emergency-contact-info"]')).toBeVisible();
  });

  test('bioregulators protocol management', async ({ page }) => {
    await page.goto('/protocols/bioregulators');

    // Verify bioregulators list
    await expect(page.locator('[data-testid="bioregulator-bpc157"]')).toBeVisible();
    await expect(page.locator('[data-testid="bioregulator-tb500"]')).toBeVisible();
    await expect(page.locator('[data-testid="bioregulator-ghk-cu"]')).toBeVisible();

    // Select BPC-157 protocol
    await page.click('[data-testid="bioregulator-bpc157"]');
    
    // Verify protocol information
    await expect(page.locator('[data-testid="protocol-healing-properties"]')).toBeVisible();
    await expect(page.locator('[data-testid="protocol-dosage-schedule"]')).toBeVisible();
    await expect(page.locator('[data-testid="protocol-injection-guide"]')).toBeVisible();

    // Start protocol
    await page.click('[data-testid="start-protocol-button"]');
    
    // Configure injection schedule
    await page.selectOption('[data-testid="dosage-select"]', '250mcg');
    await page.selectOption('[data-testid="frequency-select"]', 'daily');
    await page.selectOption('[data-testid="duration-select"]', '4-weeks');
    
    await page.click('[data-testid="confirm-protocol-button"]');

    // Verify protocol started
    await expect(page.locator('[data-testid="protocol-status"]')).toHaveText('Active');
    await expect(page.locator('[data-testid="remaining-doses"]')).toHaveText('28 doses remaining');
  });

  test('advanced cycling schemes management', async ({ page }) => {
    await page.goto('/protocols/cycling');

    // View pre-built cycling schemes
    await expect(page.locator('[data-testid="cycle-beginner-test"]')).toBeVisible();
    await expect(page.locator('[data-testid="cycle-intermediate-stack"]')).toBeVisible();
    await expect(page.locator('[data-testid="cycle-advanced-recomp"]')).toBeVisible();

    // Create custom cycling scheme
    await page.click('[data-testid="create-custom-cycle-button"]');
    
    // Add compounds to cycle
    await page.click('[data-testid="add-compound-button"]');
    await page.selectOption('[data-testid="compound-select"]', 'testosterone-enanthate');
    await page.fill('[data-testid="compound-dosage"]', '500');
    await page.selectOption('[data-testid="compound-frequency"]', 'twice-weekly');
    
    await page.click('[data-testid="add-compound-button"]');
    await page.selectOption('[data-testid="compound-select-2"]', 'nandrolone-decanoate');
    await page.fill('[data-testid="compound-dosage-2"]', '300');
    await page.selectOption('[data-testid="compound-frequency-2"]', 'weekly');

    // Set cycle duration
    await page.fill('[data-testid="cycle-duration"]', '16');
    
    // Configure PCT
    await page.check('[data-testid="include-pct-checkbox"]');
    await page.selectOption('[data-testid="pct-compound"]', 'clomiphene');
    await page.fill('[data-testid="pct-duration"]', '4');

    // Save custom cycle
    await page.fill('[data-testid="cycle-name-input"]', 'Custom Test/Deca Cycle');
    await page.click('[data-testid="save-cycle-button"]');

    // Verify cycle saved
    await expect(page.locator('[data-testid="saved-cycle"]')).toContainText('Custom Test/Deca Cycle');
  });

  test('medical oversight integration', async ({ page }) => {
    await page.goto('/protocols/medical-oversight');

    // View assigned medical professional
    await expect(page.locator('[data-testid="assigned-doctor"]')).toBeVisible();
    await expect(page.locator('[data-testid="doctor-credentials"]')).toBeVisible();
    await expect(page.locator('[data-testid="next-appointment"]')).toBeVisible();

    // Schedule follow-up consultation
    await page.click('[data-testid="schedule-followup-button"]');
    
    await page.selectOption('[data-testid="appointment-type"]', 'protocol-review');
    await page.selectOption('[data-testid="appointment-date"]', '2024-02-15');
    await page.selectOption('[data-testid="appointment-time"]', '14:00');
    
    await page.click('[data-testid="confirm-appointment-button"]');

    // Verify appointment scheduled
    await expect(page.locator('[data-testid="upcoming-appointment"]')).toBeVisible();
    await expect(page.locator('[data-testid="appointment-confirmation"]')).toContainText('February 15, 2024');

    // Upload lab results
    await page.click('[data-testid="upload-lab-results-button"]');
    
    // Simulate file upload
    const fileInput = page.locator('[data-testid="lab-results-file-input"]');
    await fileInput.setInputFiles('tests/fixtures/sample-lab-results.pdf');
    
    await page.fill('[data-testid="lab-results-notes"]', 'Latest bloodwork results after 8 weeks on protocol');
    await page.click('[data-testid="submit-lab-results-button"]');

    // Verify lab results uploaded
    await expect(page.locator('[data-testid="lab-results-uploaded"]')).toBeVisible();
    await expect(page.locator('[data-testid="lab-results-status"]')).toHaveText('Pending Review');
  });

  test('emergency protocols and safety measures', async ({ page }) => {
    await page.goto('/protocols/safety');

    // Verify emergency contact information
    await expect(page.locator('[data-testid="emergency-contacts"]')).toBeVisible();
    await expect(page.locator('[data-testid="poison-control-number"]')).toBeVisible();
    await expect(page.locator('[data-testid="medical-emergency-number"]')).toBeVisible();

    // Test emergency protocol activation
    await page.click('[data-testid="activate-emergency-protocol-button"]');
    
    await page.selectOption('[data-testid="emergency-type"]', 'severe-adverse-reaction');
    await page.fill('[data-testid="emergency-symptoms"]', 'Severe chest pain and difficulty breathing');
    
    await page.click('[data-testid="confirm-emergency-button"]');

    // Verify emergency protocol activated
    await expect(page.locator('[data-testid="emergency-alert"]')).toBeVisible();
    await expect(page.locator('[data-testid="emergency-instructions"]')).toContainText('Call 911 immediately');
    await expect(page.locator('[data-testid="emergency-medical-contact"]')).toBeVisible();

    // Verify protocol suspension
    await expect(page.locator('[data-testid="protocol-suspended"]')).toBeVisible();
    await expect(page.locator('[data-testid="suspension-reason"]')).toContainText('Emergency protocol activated');
  });

  test('protocol access control for non-performance tiers', async ({ page }) => {
    // Test with Core tier user
    await page.context().storageState({ 
      path: 'tests/fixtures/auth/core-auth.json' 
    });
    
    await page.goto('/protocols/peds');
    
    // Verify access denied
    await expect(page.locator('[data-testid="upgrade-required-message"]')).toBeVisible();
    await expect(page.locator('[data-testid="upgrade-to-performance-button"]')).toBeVisible();
    
    // Verify no protocol content visible
    await expect(page.locator('[data-testid="peds-protocols-list"]')).not.toBeVisible();
  });
});