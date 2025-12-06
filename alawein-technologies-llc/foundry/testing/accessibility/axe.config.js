module.exports = {
  // Axe-core configuration for accessibility testing
  rules: {
    // WCAG 2.1 Level AA compliance
    'color-contrast': { enabled: true },
    'image-alt': { enabled: true },
    'label': { enabled: true },
    'aria-allowed-attr': { enabled: true },
    'aria-hidden-body': { enabled: true },
    'aria-hidden-focus': { enabled: true },
    'aria-input-field-name': { enabled: true },
    'aria-required-attr': { enabled: true },
    'aria-required-children': { enabled: true },
    'aria-required-parent': { enabled: true },
    'aria-roledescription': { enabled: true },
    'aria-roles': { enabled: true },
    'aria-toggle-field-name': { enabled: true },
    'aria-valid-attr': { enabled: true },
    'aria-valid-attr-value': { enabled: true },
    'button-name': { enabled: true },
    'bypass': { enabled: true },
    'document-title': { enabled: true },
    'duplicate-id': { enabled: true },
    'duplicate-id-aria': { enabled: true },
    'form-field-multiple-labels': { enabled: true },
    'frame-focusable-content': { enabled: true },
    'frame-title': { enabled: true },
    'html-has-lang': { enabled: true },
    'html-lang-valid': { enabled: true },
    'html-xml-lang-mismatch': { enabled: true },
    'input-button-name': { enabled: true },
    'input-image-alt': { enabled: true },
    'label-content-name-mismatch': { enabled: true },
    'label-title-only': { enabled: true },
    'link-name': { enabled: true },
    'list': { enabled: true },
    'listitem': { enabled: true },
    'marquee': { enabled: true },
    'meta-refresh': { enabled: true },
    'nested-interactive': { enabled: true },
    'no-autoplay-audio': { enabled: true },
    'object-alt': { enabled: true },
    'role-img-alt': { enabled: true },
    'scrollable-region-focusable': { enabled: true },
    'select-name': { enabled: true },
    'server-side-image-map': { enabled: true },
    'svg-img-alt': { enabled: true },
    'td-has-header': { enabled: true },
    'valid-lang': { enabled: true },
    'video-caption': { enabled: true },

    // Additional WCAG 2.1 rules
    'autocomplete-valid': { enabled: true },
    'avoid-inline-spacing': { enabled: true },

    // Best practices
    'accesskeys': { enabled: true },
    'area-alt': { enabled: true },
    'aria-allowed-role': { enabled: true },
    'aria-dpub-role-fallback': { enabled: true },
    'empty-heading': { enabled: true },
    'focus-order-semantics': { enabled: true },
    'frame-tested': { enabled: true },
    'heading-order': { enabled: true },
    'hidden-content': { enabled: false }, // May have false positives
    'identical-links-same-purpose': { enabled: true },
    'landmark-banner-is-top-level': { enabled: true },
    'landmark-complementary-is-top-level': { enabled: true },
    'landmark-contentinfo-is-top-level': { enabled: true },
    'landmark-main-is-top-level': { enabled: true },
    'landmark-no-duplicate-banner': { enabled: true },
    'landmark-no-duplicate-contentinfo': { enabled: true },
    'landmark-no-duplicate-main': { enabled: true },
    'landmark-one-main': { enabled: true },
    'landmark-unique': { enabled: true },
    'link-in-text-block': { enabled: true },
    'meta-viewport': { enabled: true },
    'meta-viewport-large': { enabled: true },
    'page-has-heading-one': { enabled: true },
    'presentation-role-conflict': { enabled: true },
    'region': { enabled: true },
    'scope-attr-valid': { enabled: true },
    'skip-link': { enabled: true },
    'tabindex': { enabled: true },
    'table-duplicate-name': { enabled: true },
    'table-fake-caption': { enabled: true },
    'target-size': { enabled: true },
    'td-headers-attr': { enabled: true },
    'th-has-data-cells': { enabled: true },
  },

  // Tags for different test levels
  tags: {
    wcag2a: true,
    wcag2aa: true,
    wcag21a: true,
    wcag21aa: true,
    'best-practice': true,
    section508: false,
  },

  // Color contrast settings
  colorContrast: {
    normal: {
      aa: 4.5,
      aaa: 7,
    },
    large: {
      aa: 3,
      aaa: 4.5,
    },
  },

  // Exclude certain selectors from testing
  exclude: [
    // Third-party widgets that we can't control
    '.third-party-widget',
    // Development-only elements
    '[data-testid]',
    // Hidden elements for screen readers only
    '.sr-only',
  ],

  // Custom checks
  checks: [
    {
      id: 'touch-target-size',
      evaluate: function(node) {
        const rect = node.getBoundingClientRect();
        return rect.width >= 44 && rect.height >= 44;
      },
      metadata: {
        impact: 'serious',
        messages: {
          pass: 'Touch target is at least 44x44 pixels',
          fail: 'Touch target is smaller than 44x44 pixels',
        },
      },
    },
    {
      id: 'focus-visible',
      evaluate: function(node) {
        const styles = window.getComputedStyle(node, ':focus');
        return (
          styles.outline !== 'none' ||
          styles.border !== 'none' ||
          styles.boxShadow !== 'none'
        );
      },
      metadata: {
        impact: 'serious',
        messages: {
          pass: 'Element has visible focus indicator',
          fail: 'Element lacks visible focus indicator',
        },
      },
    },
  ],

  // Reporter configuration
  reporter: 'v2',

  // Locale
  locale: 'en',
};