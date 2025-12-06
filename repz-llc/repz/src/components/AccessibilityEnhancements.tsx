import { useEffect } from 'react';

const AccessibilityEnhancements = () => {
  useEffect(() => {
    // Add skip navigation link
    const skipLink = document.createElement('a');
    skipLink.href = '#main-content';
    skipLink.textContent = 'Skip to main content';
    skipLink.className = 'sr-only focus:not-sr-only focus:absolute focus:top-0 focus:left-0 z-50 bg-primary text-primary-foreground px-4 py-2 rounded-br-md';
    skipLink.style.transform = 'translateY(-100%)';
    skipLink.style.transition = 'transform 0.3s';
    
    skipLink.addEventListener('focus', () => {
      skipLink.style.transform = 'translateY(0)';
    });
    
    skipLink.addEventListener('blur', () => {
      skipLink.style.transform = 'translateY(-100%)';
    });
    
    document.body.insertBefore(skipLink, document.body.firstChild);

    // Enhance keyboard navigation
    const handleKeyNavigation = (e: KeyboardEvent) => {
      // Escape key to close modals
      if (e.key === 'Escape') {
        const openModal = document.querySelector('[role="dialog"][aria-hidden="false"]');
        if (openModal) {
          const closeButton = openModal.querySelector('[aria-label*="Close"], [aria-label*="close"]');
          if (closeButton) {
            (closeButton as HTMLElement).click();
          }
        }
      }
    };

    document.addEventListener('keydown', handleKeyNavigation);

    // Add focus indicators for custom components
    const addFocusIndicators = () => {
      const focusableElements = document.querySelectorAll(
        'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
      );
      
      focusableElements.forEach(element => {
        if (!element.classList.contains('focus-indicator-added')) {
          element.classList.add('focus-indicator-added');
          element.addEventListener('focus', () => {
            element.classList.add('custom-focus');
          });
          element.addEventListener('blur', () => {
            element.classList.remove('custom-focus');
          });
        }
      });
    };

    // Run immediately and on DOM changes
    addFocusIndicators();
    const observer = new MutationObserver(() => {
      addFocusIndicators();
    });
    
    observer.observe(document.body, {
      childList: true,
      subtree: true
    });

    return () => {
      document.removeEventListener('keydown', handleKeyNavigation);
      observer.disconnect();
      // Remove skip link
      const existingSkipLink = document.querySelector('a[href="#main-content"]');
      if (existingSkipLink) {
        existingSkipLink.remove();
      }
    };
  }, []);

  return null;
};

export default AccessibilityEnhancements;