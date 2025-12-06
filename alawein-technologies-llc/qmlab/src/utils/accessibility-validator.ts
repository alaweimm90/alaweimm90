/**
 * WCAG 2.1 AA Accessibility Validator
 * Comprehensive runtime validation for accessibility compliance
 */

export interface AccessibilityIssue {
  severity: 'critical' | 'serious' | 'moderate' | 'minor';
  element: Element | null;
  message: string;
  wcagCriteria: string;
  remediation: string;
}

export class AccessibilityValidator {
  private issues: AccessibilityIssue[] = [];

  /**
   * Run comprehensive accessibility audit
   */
  public async runAudit(): Promise<AccessibilityIssue[]> {
    this.issues = [];
    
    // Critical: Button accessibility
    this.validateButtons();
    
    // Critical: Landmark structure
    this.validateLandmarks();
    
    // Critical: ARIA command names
    this.validateAriaCommands();
    
    // Serious: Focus management
    this.validateFocusManagement();
    
    // Serious: Color contrast
    await this.validateColorContrast();
    
    // Moderate: Heading hierarchy
    this.validateHeadingHierarchy();
    
    // Moderate: Skip links
    this.validateSkipLinks();
    
    // Moderate: Form labels
    this.validateFormLabels();
    
    // Minor: Image alt text
    this.validateImageAltText();
    
    return this.issues;
  }

  /**
   * Validate all buttons have accessible names
   */
  private validateButtons(): void {
    const buttons = document.querySelectorAll('button, [role="button"]');
    
    buttons.forEach(button => {
      const hasAriaLabel = button.hasAttribute('aria-label');
      const hasAriaLabelledby = button.hasAttribute('aria-labelledby');
      const hasTextContent = button.textContent?.trim().length > 0;
      const hasSrOnly = button.querySelector('.sr-only');
      
      if (!hasAriaLabel && !hasAriaLabelledby && !hasTextContent && !hasSrOnly) {
        this.issues.push({
          severity: 'critical',
          element: button,
          message: `Button without accessible name: ${button.outerHTML.substring(0, 100)}`,
          wcagCriteria: 'WCAG 4.1.2 - Name, Role, Value',
          remediation: 'Add aria-label, aria-labelledby, or visible text content'
        });
      }
    });
  }

  /**
   * Validate landmark structure
   */
  private validateLandmarks(): void {
    const main = document.querySelector('main, [role="main"]');
    const header = document.querySelector('header, [role="banner"]');
    const nav = document.querySelector('nav, [role="navigation"]');
    const footer = document.querySelector('footer, [role="contentinfo"]');
    
    if (!main) {
      this.issues.push({
        severity: 'serious',
        element: null,
        message: 'Missing main landmark',
        wcagCriteria: 'WCAG 1.3.1 - Info and Relationships',
        remediation: 'Add <main> element or role="main"'
      });
    }
    
    // Check if all content is within landmarks
    const allContent = document.body.querySelectorAll('*:not(script):not(style):not(meta):not(link)');
    allContent.forEach(element => {
      if (element.textContent?.trim() && !this.isWithinLandmark(element)) {
        this.issues.push({
          severity: 'moderate',
          element: element,
          message: 'Content outside of landmark regions',
          wcagCriteria: 'Best Practice - Landmark Regions',
          remediation: 'Wrap content in appropriate landmark elements'
        });
      }
    });
  }

  /**
   * Validate ARIA command elements have names
   */
  private validateAriaCommands(): void {
    const ariaElements = document.querySelectorAll('[role="button"], [role="menuitem"], [role="tab"], [role="link"]');
    
    ariaElements.forEach(element => {
      const hasAriaLabel = element.hasAttribute('aria-label');
      const hasAriaLabelledby = element.hasAttribute('aria-labelledby');
      const hasAccessibleName = element.textContent?.trim().length > 0;
      
      if (!hasAriaLabel && !hasAriaLabelledby && !hasAccessibleName) {
        this.issues.push({
          severity: 'serious',
          element: element,
          message: `ARIA element without accessible name: role="${element.getAttribute('role')}"`,
          wcagCriteria: 'WCAG 4.1.2 - Name, Role, Value',
          remediation: 'Add aria-label or aria-labelledby attribute'
        });
      }
    });
  }

  /**
   * Validate focus management
   */
  private validateFocusManagement(): void {
    // Check for positive tabindex
    const positiveTabindex = document.querySelectorAll('[tabindex]:not([tabindex="0"]):not([tabindex="-1"])');
    positiveTabindex.forEach(element => {
      const tabindexValue = element.getAttribute('tabindex');
      if (parseInt(tabindexValue || '0') > 0) {
        this.issues.push({
          severity: 'serious',
          element: element,
          message: `Positive tabindex found: tabindex="${tabindexValue}"`,
          wcagCriteria: 'WCAG 2.4.3 - Focus Order',
          remediation: 'Use tabindex="0" or "-1" only, rely on DOM order'
        });
      }
    });
    
    // Check focusable elements have focus indicators
    const focusableElements = document.querySelectorAll('a, button, input, select, textarea, [tabindex="0"]');
    focusableElements.forEach(element => {
      const styles = window.getComputedStyle(element);
      // This is a simplified check - in production you'd check :focus-visible styles
      if (!styles.outlineStyle || styles.outlineStyle === 'none') {
        console.warn('Element may lack focus indicator:', element);
      }
    });
  }

  /**
   * Validate color contrast ratios
   */
  private async validateColorContrast(): Promise<void> {
    // Get all text elements
    const textElements = document.querySelectorAll('p, span, div, h1, h2, h3, h4, h5, h6, a, button, label');
    
    for (const element of textElements) {
      if (!element.textContent?.trim()) continue;
      
      const styles = window.getComputedStyle(element);
      const fontSize = parseFloat(styles.fontSize);
      const fontWeight = styles.fontWeight;
      const isLargeText = fontSize >= 18 || (fontSize >= 14 && parseInt(fontWeight) >= 700);
      
      const contrast = await this.getContrastRatio(element as HTMLElement);
      const requiredRatio = isLargeText ? 3 : 4.5;
      
      if (contrast < requiredRatio) {
        this.issues.push({
          severity: 'serious',
          element: element,
          message: `Insufficient color contrast: ${contrast.toFixed(2)}:1 (required: ${requiredRatio}:1)`,
          wcagCriteria: 'WCAG 1.4.3 - Contrast (Minimum)',
          remediation: `Increase contrast ratio to at least ${requiredRatio}:1`
        });
      }
    }
  }

  /**
   * Validate heading hierarchy
   */
  private validateHeadingHierarchy(): void {
    const headings = document.querySelectorAll('h1, h2, h3, h4, h5, h6');
    const h1Count = document.querySelectorAll('h1').length;
    
    if (h1Count === 0) {
      this.issues.push({
        severity: 'moderate',
        element: null,
        message: 'No H1 heading found on page',
        wcagCriteria: 'WCAG 1.3.1 - Info and Relationships',
        remediation: 'Add a single H1 that matches the page title'
      });
    } else if (h1Count > 1) {
      this.issues.push({
        severity: 'moderate',
        element: null,
        message: `Multiple H1 headings found (${h1Count})`,
        wcagCriteria: 'Best Practice - Heading Hierarchy',
        remediation: 'Use only one H1 per page'
      });
    }
    
    // Check heading order
    let lastLevel = 0;
    headings.forEach(heading => {
      const level = parseInt(heading.tagName.substring(1));
      if (level > lastLevel + 1) {
        this.issues.push({
          severity: 'minor',
          element: heading,
          message: `Heading level skipped: H${lastLevel} to H${level}`,
          wcagCriteria: 'WCAG 1.3.1 - Info and Relationships',
          remediation: 'Maintain sequential heading hierarchy'
        });
      }
      lastLevel = level;
    });
  }

  /**
   * Validate skip links
   */
  private validateSkipLinks(): void {
    const skipLink = document.querySelector('a[href="#main-content"], .skip-link, .skip-to-main');
    
    if (!skipLink) {
      this.issues.push({
        severity: 'moderate',
        element: null,
        message: 'No skip link found',
        wcagCriteria: 'WCAG 2.4.1 - Bypass Blocks',
        remediation: 'Add skip link as first focusable element'
      });
    } else {
      const target = document.querySelector('#main-content');
      if (!target) {
        this.issues.push({
          severity: 'moderate',
          element: skipLink,
          message: 'Skip link target #main-content not found',
          wcagCriteria: 'WCAG 2.4.1 - Bypass Blocks',
          remediation: 'Add id="main-content" to main content area'
        });
      } else if (!target.hasAttribute('tabindex')) {
        this.issues.push({
          severity: 'minor',
          element: target,
          message: 'Skip link target missing tabindex="-1"',
          wcagCriteria: 'Best Practice - Skip Links',
          remediation: 'Add tabindex="-1" to main content for focus management'
        });
      }
    }
  }

  /**
   * Validate form labels
   */
  private validateFormLabels(): void {
    const inputs = document.querySelectorAll('input:not([type="hidden"]), select, textarea');
    
    inputs.forEach(input => {
      const hasLabel = input.closest('label') || 
                      document.querySelector(`label[for="${input.id}"]`);
      const hasAriaLabel = input.hasAttribute('aria-label');
      const hasAriaLabelledby = input.hasAttribute('aria-labelledby');
      
      if (!hasLabel && !hasAriaLabel && !hasAriaLabelledby) {
        this.issues.push({
          severity: 'serious',
          element: input,
          message: `Form control without label: ${input.outerHTML.substring(0, 100)}`,
          wcagCriteria: 'WCAG 3.3.2 - Labels or Instructions',
          remediation: 'Add <label>, aria-label, or aria-labelledby'
        });
      }
    });
  }

  /**
   * Validate image alt text
   */
  private validateImageAltText(): void {
    const images = document.querySelectorAll('img');
    
    images.forEach(img => {
      if (!img.hasAttribute('alt')) {
        const isDecorative = img.hasAttribute('role') && img.getAttribute('role') === 'presentation';
        
        if (!isDecorative) {
          this.issues.push({
            severity: 'moderate',
            element: img,
            message: `Image missing alt attribute: ${img.src}`,
            wcagCriteria: 'WCAG 1.1.1 - Non-text Content',
            remediation: 'Add alt="" for decorative or alt="description" for informative images'
          });
        }
      }
    });
  }

  /**
   * Helper: Check if element is within a landmark
   */
  private isWithinLandmark(element: Element): boolean {
    const landmarks = ['main', 'header', 'nav', 'footer', 'aside', 
                       '[role="main"]', '[role="banner"]', '[role="navigation"]', 
                       '[role="contentinfo"]', '[role="complementary"]'];
    
    return landmarks.some(selector => element.closest(selector) !== null);
  }

  /**
   * Helper: Calculate contrast ratio between text and background
   */
  private async getContrastRatio(element: HTMLElement): Promise<number> {
    const styles = window.getComputedStyle(element);
    const textColor = styles.color;
    const bgColor = this.getBackgroundColor(element);
    
    return this.calculateContrastRatio(textColor, bgColor);
  }

  /**
   * Helper: Get effective background color
   */
  private getBackgroundColor(element: HTMLElement): string {
    let current = element;
    while (current) {
      const styles = window.getComputedStyle(current);
      const bgColor = styles.backgroundColor;
      
      if (bgColor !== 'rgba(0, 0, 0, 0)' && bgColor !== 'transparent') {
        return bgColor;
      }
      
      current = current.parentElement as HTMLElement;
    }
    
    return 'rgb(255, 255, 255)'; // Default to white
  }

  /**
   * Helper: Calculate WCAG contrast ratio
   */
  private calculateContrastRatio(color1: string, color2: string): number {
    const rgb1 = this.parseColor(color1);
    const rgb2 = this.parseColor(color2);
    
    const l1 = this.relativeLuminance(rgb1);
    const l2 = this.relativeLuminance(rgb2);
    
    const lighter = Math.max(l1, l2);
    const darker = Math.min(l1, l2);
    
    return (lighter + 0.05) / (darker + 0.05);
  }

  /**
   * Helper: Parse color string to RGB
   */
  private parseColor(color: string): [number, number, number] {
    const match = color.match(/\d+/g);
    if (match) {
      return [parseInt(match[0]), parseInt(match[1]), parseInt(match[2])];
    }
    return [0, 0, 0];
  }

  /**
   * Helper: Calculate relative luminance
   */
  private relativeLuminance([r, g, b]: [number, number, number]): number {
    const sRGB = [r, g, b].map(val => {
      val = val / 255;
      return val <= 0.03928 ? val / 12.92 : Math.pow((val + 0.055) / 1.055, 2.4);
    });
    
    return 0.2126 * sRGB[0] + 0.7152 * sRGB[1] + 0.0722 * sRGB[2];
  }

  /**
   * Generate accessibility report
   */
  public generateReport(): string {
    const grouped = this.groupBySeverity();
    
    return `
ACCESSIBILITY AUDIT REPORT
==========================
Date: ${new Date().toISOString()}
URL: ${window.location.href}

SUMMARY
-------
Critical: ${grouped.critical.length}
Serious: ${grouped.serious.length}
Moderate: ${grouped.moderate.length}
Minor: ${grouped.minor.length}
TOTAL: ${this.issues.length}

${grouped.critical.length > 0 ? `
CRITICAL ISSUES
---------------
${grouped.critical.map(issue => this.formatIssue(issue)).join('\n\n')}
` : ''}

${grouped.serious.length > 0 ? `
SERIOUS ISSUES
--------------
${grouped.serious.map(issue => this.formatIssue(issue)).join('\n\n')}
` : ''}

${grouped.moderate.length > 0 ? `
MODERATE ISSUES
---------------
${grouped.moderate.map(issue => this.formatIssue(issue)).join('\n\n')}
` : ''}

${grouped.minor.length > 0 ? `
MINOR ISSUES
------------
${grouped.minor.map(issue => this.formatIssue(issue)).join('\n\n')}
` : ''}

VALIDATION COMPLETE
-------------------
${this.issues.length === 0 ? '✅ No accessibility issues found!' : '⚠️ Please address the issues above for WCAG 2.1 AA compliance.'}
    `.trim();
  }

  /**
   * Helper: Group issues by severity
   */
  private groupBySeverity() {
    return {
      critical: this.issues.filter(i => i.severity === 'critical'),
      serious: this.issues.filter(i => i.severity === 'serious'),
      moderate: this.issues.filter(i => i.severity === 'moderate'),
      minor: this.issues.filter(i => i.severity === 'minor')
    };
  }

  /**
   * Helper: Format single issue
   */
  private formatIssue(issue: AccessibilityIssue): string {
    return `
• ${issue.message}
  WCAG: ${issue.wcagCriteria}
  Fix: ${issue.remediation}
  ${issue.element ? `Element: ${issue.element.tagName.toLowerCase()}${issue.element.className ? `.${issue.element.className.split(' ')[0]}` : ''}` : ''}
    `.trim();
  }
}

// Export singleton instance
export const accessibilityValidator = new AccessibilityValidator();

// Add to window for console access
if (typeof window !== 'undefined') {
  (window as any).accessibilityValidator = accessibilityValidator;
  (window as any).runAccessibilityAudit = async () => {
    const issues = await accessibilityValidator.runAudit();
    console.log(accessibilityValidator.generateReport());
    return issues;
  };
}