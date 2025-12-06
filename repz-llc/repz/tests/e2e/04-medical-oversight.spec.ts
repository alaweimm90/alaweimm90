import { test, expect } from '@playwright/test';

test.describe('Medical Oversight Integration', () => {
  test.beforeEach(async ({ page }) => {
    // Start with Performance+ tier client auth
    await page.context().storageState({ 
      path: 'tests/fixtures/auth/performance-auth.json' 
    });
    await page.goto('/dashboard');
  });

  test('complete medical professional verification and approval workflow', async ({ page }) => {
    // Switch to medical professional perspective
    await page.context().storageState({ 
      path: 'tests/fixtures/auth/medical-auth.json' 
    });
    
    // Step 1: Medical professional platform onboarding
    await page.goto('/medical/onboarding');
    
    // Verify medical credentials form
    await expect(page.locator('[data-testid="medical-credentials-form"]')).toBeVisible();
    
    await page.fill('[data-testid="medical-license-number"]', 'MD123456789');
    await page.selectOption('[data-testid="medical-specialty"]', 'endocrinology');
    await page.selectOption('[data-testid="license-state"]', 'california');
    await page.fill('[data-testid="years-experience"]', '15');
    
    // Upload credentials
    await page.setInputFiles('[data-testid="license-upload"]', 'tests/fixtures/sample-medical-license.pdf');
    await page.setInputFiles('[data-testid="board-certification"]', 'tests/fixtures/sample-board-cert.pdf');
    
    await page.fill('[data-testid="professional-bio"]', 'Board-certified endocrinologist with 15 years experience in hormone optimization and performance medicine.');
    
    // Agree to platform terms
    await page.check('[data-testid="medical-terms-checkbox"]');
    await page.check('[data-testid="hipaa-compliance-checkbox"]');
    await page.check('[data-testid="liability-agreement-checkbox"]');
    
    await page.click('[data-testid="submit-verification-button"]');
    
    // Step 2: Platform approval process (simulated admin approval)
    await page.goto('/admin/medical-verification');
    
    await expect(page.locator('[data-testid="pending-verification"]')).toBeVisible();
    await page.click('[data-testid="review-application-MD123456789"]');
    
    // Review credentials
    await expect(page.locator('[data-testid="license-document"]')).toBeVisible();
    await expect(page.locator('[data-testid="certification-document"]')).toBeVisible();
    await expect(page.locator('[data-testid="license-verification-status"]')).toHaveText('Valid');
    
    // Approve medical professional
    await page.fill('[data-testid="approval-notes"]', 'Credentials verified. License active and in good standing.');
    await page.click('[data-testid="approve-medical-professional"]');
    
    await expect(page.locator('[data-testid="approval-confirmed"]')).toBeVisible();
    
    // Step 3: Medical professional dashboard access
    await page.context().storageState({ 
      path: 'tests/fixtures/auth/medical-auth.json' 
    });
    
    await page.goto('/medical/dashboard');
    
    await expect(page.locator('[data-testid="medical-dashboard"]')).toBeVisible();
    await expect(page.locator('[data-testid="verification-status"]')).toHaveText('Verified');
    await expect(page.locator('[data-testid="patient-queue"]')).toBeVisible();
    await expect(page.locator('[data-testid="consultation-requests"]')).toBeVisible();
  });

  test('client consultation scheduling and protocol review', async ({ page }) => {
    // Step 1: Client requests consultation
    await page.goto('/protocols/consultation');
    
    await page.click('[data-testid="request-consultation-button"]');
    
    // Fill consultation request
    await page.selectOption('[data-testid="consultation-type"]', 'protocol-approval');
    await page.selectOption('[data-testid="requested-protocol"]', 'testosterone-enanthate');
    await page.fill('[data-testid="medical-history"]', 'No significant medical history. Previous lab work normal.');
    await page.fill('[data-testid="current-medications"]', 'None');
    await page.fill('[data-testid="consultation-goals"]', 'Optimize hormone levels for performance and recovery');
    
    // Upload recent lab work
    await page.setInputFiles('[data-testid="lab-work-upload"]', 'tests/fixtures/sample-bloodwork.pdf');
    
    await page.check('[data-testid="consent-medical-review"]');
    await page.click('[data-testid="submit-consultation-request"]');
    
    // Verify consultation request submitted
    await expect(page.locator('[data-testid="consultation-submitted"]')).toBeVisible();
    await expect(page.locator('[data-testid="consultation-id"]')).toBeVisible();
    const consultationId = await page.locator('[data-testid="consultation-id"]').textContent();
    
    // Step 2: Medical professional reviews consultation
    await page.context().storageState({ 
      path: 'tests/fixtures/auth/medical-auth.json' 
    });
    
    await page.goto('/medical/consultations');
    
    await expect(page.locator('[data-testid="pending-consultations"]')).toBeVisible();
    await page.click(`[data-testid="review-consultation-${consultationId}"]`);
    
    // Review client information
    await expect(page.locator('[data-testid="client-medical-history"]')).toBeVisible();
    await expect(page.locator('[data-testid="requested-protocol-details"]')).toBeVisible();
    await expect(page.locator('[data-testid="uploaded-lab-work"]')).toBeVisible();
    
    // Step 3: Medical clearance decision
    await page.click('[data-testid="medical-clearance-tab"]');
    
    // Review protocol safety
    await expect(page.locator('[data-testid="protocol-contraindications"]')).toBeVisible();
    await expect(page.locator('[data-testid="dosage-recommendations"]')).toBeVisible();
    await expect(page.locator('[data-testid="monitoring-requirements"]')).toBeVisible();
    
    // Provide medical clearance
    await page.selectOption('[data-testid="clearance-status"]', 'approved');
    await page.fill('[data-testid="dosage-modification"]', 'Start with 200mg weekly, split into 2 doses');
    await page.fill('[data-testid="monitoring-schedule"]', 'Lab work at 6 weeks, 12 weeks, then quarterly');
    await page.fill('[data-testid="medical-notes"]', 'Patient is good candidate for TRT. Monitor lipids and hematocrit closely.');
    
    // Set follow-up requirements
    await page.check('[data-testid="require-followup-6weeks"]');
    await page.check('[data-testid="require-lab-monitoring"]');
    
    await page.click('[data-testid="submit-medical-clearance"]');
    
    await expect(page.locator('[data-testid="clearance-submitted"]')).toBeVisible();
    
    // Step 4: Client receives approval notification
    await page.context().storageState({ 
      path: 'tests/fixtures/auth/performance-auth.json' 
    });
    
    await page.goto('/protocols/status');
    
    await expect(page.locator('[data-testid="protocol-status-approved"]')).toBeVisible();
    await expect(page.locator('[data-testid="medical-clearance-details"]')).toBeVisible();
    await expect(page.locator('[data-testid="modified-dosage"]')).toContainText('200mg weekly');
    await expect(page.locator('[data-testid="monitoring-requirements"]')).toContainText('6 weeks');
  });

  test('ongoing medical monitoring and safety oversight', async ({ page }) => {
    // Assume client has active protocol
    await page.goto('/protocols/monitoring');
    
    // Step 1: Client reports side effects
    await page.click('[data-testid="report-side-effects-button"]');
    
    await page.check('[data-testid="side-effect-oily-skin"]');
    await page.selectOption('[data-testid="severity-oily-skin"]', 'mild');
    await page.check('[data-testid="side-effect-mood-changes"]');
    await page.selectOption('[data-testid="severity-mood-changes"]', 'moderate');
    
    await page.fill('[data-testid="side-effect-details"]', 'Increased oily skin on face and back. Some irritability in evenings.');
    await page.fill('[data-testid="onset-date"]', '2024-01-15');
    
    await page.click('[data-testid="submit-side-effects-report"]');
    
    // Step 2: Medical professional receives alert
    await page.context().storageState({ 
      path: 'tests/fixtures/auth/medical-auth.json' 
    });
    
    await page.goto('/medical/alerts');
    
    await expect(page.locator('[data-testid="side-effect-alert"]')).toBeVisible();
    await expect(page.locator('[data-testid="alert-severity-moderate"]')).toBeVisible();
    
    await page.click('[data-testid="review-side-effect-alert"]');
    
    // Review reported side effects
    await expect(page.locator('[data-testid="reported-symptoms"]')).toBeVisible();
    await expect(page.locator('[data-testid="symptom-timeline"]')).toBeVisible();
    
    // Step 3: Medical response to side effects
    await page.selectOption('[data-testid="response-action"]', 'continue-with-modifications');
    await page.fill('[data-testid="modification-instructions"]', 'Reduce frequency to once weekly. Add liver support supplement.');
    await page.fill('[data-testid="medical-advice"]', 'Side effects are common and manageable. Schedule follow-up in 2 weeks.');
    
    await page.check('[data-testid="schedule-followup-checkbox"]');
    await page.fill('[data-testid="followup-date"]', '2024-02-01');
    
    await page.click('[data-testid="submit-medical-response"]');
    
    // Step 4: Client receives medical guidance
    await page.context().storageState({ 
      path: 'tests/fixtures/auth/performance-auth.json' 
    });
    
    await page.goto('/protocols/medical-guidance');
    
    await expect(page.locator('[data-testid="medical-response"]')).toBeVisible();
    await expect(page.locator('[data-testid="modified-protocol"]')).toContainText('once weekly');
    await expect(page.locator('[data-testid="medical-advice"]')).toContainText('manageable');
    await expect(page.locator('[data-testid="scheduled-followup"]')).toContainText('February 1');
  });

  test('emergency protocol activation and medical intervention', async ({ page }) => {
    await page.goto('/protocols/active');
    
    // Step 1: Client reports severe adverse reaction
    await page.click('[data-testid="emergency-report-button"]');
    
    await page.selectOption('[data-testid="emergency-type"]', 'severe-adverse-reaction');
    await page.check('[data-testid="symptom-chest-pain"]');
    await page.check('[data-testid="symptom-difficulty-breathing"]');
    await page.check('[data-testid="symptom-severe-headache"]');
    
    await page.fill('[data-testid="emergency-details"]', 'Severe chest pain started 2 hours ago, getting worse. Some difficulty breathing.');
    await page.selectOption('[data-testid="pain-scale"]', '8');
    
    await page.click('[data-testid="submit-emergency-report"]');
    
    // Step 2: Immediate emergency response
    await expect(page.locator('[data-testid="emergency-alert"]')).toBeVisible();
    await expect(page.locator('[data-testid="call-911-instruction"]')).toBeVisible();
    await expect(page.locator('[data-testid="emergency-contact-numbers"]')).toBeVisible();
    
    // Verify protocol suspension
    await expect(page.locator('[data-testid="protocol-suspended"]')).toBeVisible();
    await expect(page.locator('[data-testid="suspension-reason"]')).toContainText('Emergency reported');
    
    // Step 3: Medical professional emergency notification
    await page.context().storageState({ 
      path: 'tests/fixtures/auth/medical-auth.json' 
    });
    
    await page.goto('/medical/emergencies');
    
    await expect(page.locator('[data-testid="critical-alert"]')).toBeVisible();
    await expect(page.locator('[data-testid="emergency-severity-critical"]')).toBeVisible();
    
    await page.click('[data-testid="review-emergency"]');
    
    // Review emergency details
    await expect(page.locator('[data-testid="reported-symptoms"]')).toBeVisible();
    await expect(page.locator('[data-testid="pain-severity"]')).toContainText('8/10');
    await expect(page.locator('[data-testid="protocol-history"]')).toBeVisible();
    
    // Step 4: Medical professional emergency response
    await page.check('[data-testid="contacted-emergency-services"]');
    await page.check('[data-testid="contacted-patient"]');
    await page.fill('[data-testid="emergency-actions"]', 'Patient transported to ER. Advised discontinuation of protocol. Will follow up with hospital.');
    
    await page.selectOption('[data-testid="protocol-status"]', 'permanently-discontinued');
    await page.fill('[data-testid="discontinuation-reason"]', 'Severe adverse reaction - chest pain and respiratory symptoms');
    
    await page.click('[data-testid="submit-emergency-response"]');
    
    // Step 5: Follow-up protocol
    await page.fill('[data-testid="hospital-followup"]', 'Contact Dr. Smith at City General ER');
    await page.check('[data-testid="schedule-post-emergency-consultation"]');
    
    await page.click('[data-testid="complete-emergency-protocol"]');
    
    await expect(page.locator('[data-testid="emergency-protocol-completed"]')).toBeVisible();
  });

  test('medical documentation and compliance tracking', async ({ page }) => {
    await page.context().storageState({ 
      path: 'tests/fixtures/auth/medical-auth.json' 
    });
    
    await page.goto('/medical/documentation');
    
    // Step 1: Review compliance requirements
    await expect(page.locator('[data-testid="hipaa-compliance-status"]')).toBeVisible();
    await expect(page.locator('[data-testid="medical-liability-status"]')).toBeVisible();
    await expect(page.locator('[data-testid="documentation-requirements"]')).toBeVisible();
    
    // Step 2: Generate medical reports
    await page.click('[data-testid="generate-report-button"]');
    
    await page.selectOption('[data-testid="report-type"]', 'patient-consultations');
    await page.fill('[data-testid="report-date-from"]', '2024-01-01');
    await page.fill('[data-testid="report-date-to"]', '2024-01-31');
    
    await page.click('[data-testid="generate-report-submit"]');
    
    // Verify report generation
    await expect(page.locator('[data-testid="report-generated"]')).toBeVisible();
    await expect(page.locator('[data-testid="report-download-link"]')).toBeVisible();
    
    // Step 3: Audit trail review
    await page.click('[data-testid="audit-trail-tab"]');
    
    await expect(page.locator('[data-testid="consultation-audit-log"]')).toBeVisible();
    await expect(page.locator('[data-testid="approval-audit-log"]')).toBeVisible();
    await expect(page.locator('[data-testid="emergency-audit-log"]')).toBeVisible();
    
    // Verify audit entries contain required information
    await expect(page.locator('[data-testid="audit-entry"]').first()).toContainText('timestamp');
    await expect(page.locator('[data-testid="audit-entry"]').first()).toContainText('action');
    await expect(page.locator('[data-testid="audit-entry"]').first()).toContainText('medical-professional');
  });

  test('medical oversight access control and permissions', async ({ page }) => {
    // Test client cannot access medical functions
    await page.context().storageState({ 
      path: 'tests/fixtures/auth/performance-auth.json' 
    });
    
    await page.goto('/medical/dashboard');
    
    await expect(page.locator('[data-testid="access-denied"]')).toBeVisible();
    await expect(page.locator('[data-testid="insufficient-permissions"]')).toBeVisible();
    
    // Test coach cannot access medical functions
    await page.context().storageState({ 
      path: 'tests/fixtures/auth/coach-auth.json' 
    });
    
    await page.goto('/medical/consultations');
    
    await expect(page.locator('[data-testid="access-denied"]')).toBeVisible();
    
    // Test unverified medical professional has limited access
    await page.context().storageState({ 
      path: 'tests/fixtures/auth/unverified-medical-auth.json' 
    });
    
    await page.goto('/medical/dashboard');
    
    await expect(page.locator('[data-testid="verification-required"]')).toBeVisible();
    await expect(page.locator('[data-testid="consultation-requests"]')).not.toBeVisible();
    await expect(page.locator('[data-testid="patient-queue"]')).not.toBeVisible();
  });

  test('HIPAA compliance and data protection', async ({ page }) => {
    await page.context().storageState({ 
      path: 'tests/fixtures/auth/medical-auth.json' 
    });
    
    await page.goto('/medical/privacy');
    
    // Step 1: Verify HIPAA compliance measures
    await expect(page.locator('[data-testid="hipaa-compliance-badge"]')).toBeVisible();
    await expect(page.locator('[data-testid="data-encryption-status"]')).toHaveText('AES-256');
    await expect(page.locator('[data-testid="access-logging-status"]')).toHaveText('Enabled');
    
    // Step 2: Test audit logging
    await page.click('[data-testid="view-patient-data"]');
    
    // Verify access is logged
    await page.goto('/medical/audit-log');
    await expect(page.locator('[data-testid="data-access-log"]').first()).toContainText('patient data accessed');
    
    // Step 3: Test data retention policies
    await page.goto('/medical/data-retention');
    
    await expect(page.locator('[data-testid="retention-policy"]')).toBeVisible();
    await expect(page.locator('[data-testid="automatic-deletion-schedule"]')).toBeVisible();
    await expect(page.locator('[data-testid="backup-encryption-status"]')).toHaveText('Encrypted');
  });
});