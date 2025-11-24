// DOM Elements
const mobileMenuToggle = document.querySelector('.mobile-menu-toggle');
const navMenu = document.querySelector('.nav-menu');
const portfolioItems = document.querySelectorAll('.portfolio-item');
const lightbox = document.getElementById('lightbox');
const lightboxImage = document.getElementById('lightbox-image');
const lightboxTitle = document.getElementById('lightbox-title');
const lightboxDescription = document.getElementById('lightbox-description');
const lightboxClose = document.querySelector('.lightbox-close');
const lightboxOverlay = document.querySelector('.lightbox-overlay');
const contactForm = document.getElementById('contact-form');
const submitButton = document.querySelector('.submit-button');
const buttonText = document.querySelector('.button-text');
const buttonLoading = document.querySelector('.button-loading');
const formSuccess = document.getElementById('form-success');

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    initMobileMenu();
    initPortfolioLightbox();
    initContactForm();
    initSmoothScrolling();
    initAccessibility();
});

// Mobile Menu Toggle
function initMobileMenu() {
    if (mobileMenuToggle) {
        mobileMenuToggle.addEventListener('click', function() {
            navMenu.classList.toggle('active');
            mobileMenuToggle.classList.toggle('active');
            mobileMenuToggle.setAttribute('aria-expanded',
                mobileMenuToggle.classList.contains('active'));
        });
    }

    // Close mobile menu when clicking outside
    document.addEventListener('click', function(event) {
        if (!event.target.closest('nav') && navMenu.classList.contains('active')) {
            navMenu.classList.remove('active');
            mobileMenuToggle.classList.remove('active');
            mobileMenuToggle.setAttribute('aria-expanded', 'false');
        }
    });

    // Close mobile menu on window resize
    window.addEventListener('resize', function() {
        if (window.innerWidth > 768 && navMenu.classList.contains('active')) {
            navMenu.classList.remove('active');
            mobileMenuToggle.classList.remove('active');
            mobileMenuToggle.setAttribute('aria-expanded', 'false');
        }
    });
}

// Portfolio Lightbox
function initPortfolioLightbox() {
    portfolioItems.forEach(item => {
        item.addEventListener('click', function() {
            const image = this.dataset.image;
            const title = this.dataset.title;
            const description = this.dataset.description;

            if (image && title && description) {
                openLightbox(image, title, description);
            }
        });

        item.addEventListener('keydown', function(event) {
            if (event.key === 'Enter' || event.key === ' ') {
                event.preventDefault();
                this.click();
            }
        });
    });

    // Close lightbox events
    if (lightboxClose) {
        lightboxClose.addEventListener('click', closeLightbox);
    }

    if (lightboxOverlay) {
        lightboxOverlay.addEventListener('click', closeLightbox);
    }

    // Close on Escape key
    document.addEventListener('keydown', function(event) {
        if (event.key === 'Escape' && lightbox.style.display !== 'none') {
            closeLightbox();
        }
    });
}

function openLightbox(imageSrc, title, description) {
    lightboxImage.src = imageSrc;
    lightboxImage.alt = title;
    lightboxTitle.textContent = title;
    lightboxDescription.textContent = description;
    lightbox.style.display = 'flex';
    document.body.style.overflow = 'hidden';

    // Focus management
    lightboxClose.focus();

    // Announce to screen readers
    lightbox.setAttribute('aria-hidden', 'false');
}

function closeLightbox() {
    lightbox.style.display = 'none';
    document.body.style.overflow = 'auto';

    // Return focus to the last focused element
    if (document.activeElement === lightboxClose) {
        // Find the last focused portfolio item
        const focusedItem = document.querySelector('.portfolio-item:focus');
        if (focusedItem) {
            focusedItem.focus();
        }
    }

    lightbox.setAttribute('aria-hidden', 'true');
}

// Contact Form Validation
function initContactForm() {
    if (!contactForm) return;

    contactForm.addEventListener('submit', function(event) {
        event.preventDefault();

        if (validateForm()) {
            submitForm();
        }
    });

    // Real-time validation
    const inputs = contactForm.querySelectorAll('input, textarea');
    inputs.forEach(input => {
        input.addEventListener('blur', function() {
            validateField(this);
        });

        input.addEventListener('input', function() {
            clearFieldError(this);
        });
    });
}

function validateForm() {
    let isValid = true;
    const requiredFields = contactForm.querySelectorAll('[required]');

    requiredFields.forEach(field => {
        if (!validateField(field)) {
            isValid = false;
        }
    });

    return isValid;
}

function validateField(field) {
    const value = field.value.trim();
    const fieldName = field.name;
    const errorElement = document.getElementById(`${fieldName}-error`);

    if (!errorElement) return true;

    clearFieldError(field);

    if (field.hasAttribute('required') && !value) {
        showFieldError(field, `${fieldName.charAt(0).toUpperCase() + fieldName.slice(1)} is required`);
        return false;
    }

    if (field.type === 'email' && value) {
        const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
        if (!emailRegex.test(value)) {
            showFieldError(field, 'Please enter a valid email address');
            return false;
        }
    }

    if (field.name === 'message' && value.length < 10) {
        showFieldError(field, 'Message must be at least 10 characters long');
        return false;
    }

    return true;
}

function showFieldError(field, message) {
    const errorElement = document.getElementById(`${field.name}-error`);
    if (errorElement) {
        errorElement.textContent = message;
        errorElement.style.display = 'block';
    }
    field.setAttribute('aria-invalid', 'true');
}

function clearFieldError(field) {
    const errorElement = document.getElementById(`${field.name}-error`);
    if (errorElement) {
        errorElement.textContent = '';
        errorElement.style.display = 'none';
    }
    field.setAttribute('aria-invalid', 'false');
}

function submitForm() {
    // Show loading state
    submitButton.disabled = true;
    buttonText.style.display = 'none';
    buttonLoading.style.display = 'inline';

    // Simulate form submission (replace with actual submission logic)
    setTimeout(() => {
        // Reset loading state
        submitButton.disabled = false;
        buttonText.style.display = 'inline';
        buttonLoading.style.display = 'none';

        // Show success message
        formSuccess.style.display = 'block';

        // Reset form
        contactForm.reset();

        // Clear any remaining errors
        const errorElements = contactForm.querySelectorAll('.error-message');
        errorElements.forEach(el => {
            el.style.display = 'none';
        });

        // Hide success message after 5 seconds
        setTimeout(() => {
            formSuccess.style.display = 'none';
        }, 5000);

        // Announce success to screen readers
        formSuccess.setAttribute('aria-live', 'polite');
    }, 2000);
}

// Smooth Scrolling for Navigation
function initSmoothScrolling() {
    const navLinks = document.querySelectorAll('a[href^="#"]');

    navLinks.forEach(link => {
        link.addEventListener('click', function(event) {
            const targetId = this.getAttribute('href');
            const targetElement = document.querySelector(targetId);

            if (targetElement) {
                event.preventDefault();

                const headerHeight = document.querySelector('header').offsetHeight;
                const targetPosition = targetElement.offsetTop - headerHeight;

                window.scrollTo({
                    top: targetPosition,
                    behavior: 'smooth'
                });

                // Close mobile menu if open
                if (navMenu.classList.contains('active')) {
                    navMenu.classList.remove('active');
                    mobileMenuToggle.classList.remove('active');
                    mobileMenuToggle.setAttribute('aria-expanded', 'false');
                }
            }
        });
    });
}

// Accessibility Enhancements
function initAccessibility() {
    // Skip to main content link (add to HTML if needed)
    // Add keyboard navigation for portfolio grid
    const portfolioGrid = document.querySelector('.portfolio-grid');
    if (portfolioGrid) {
        portfolioGrid.addEventListener('keydown', function(event) {
            const items = Array.from(portfolioItems);
            const currentIndex = items.indexOf(document.activeElement);

            if (currentIndex === -1) return;

            let newIndex;

            switch (event.key) {
                case 'ArrowRight':
                    newIndex = (currentIndex + 1) % items.length;
                    break;
                case 'ArrowLeft':
                    newIndex = (currentIndex - 1 + items.length) % items.length;
                    break;
                case 'ArrowDown':
                    newIndex = Math.min(currentIndex + 3, items.length - 1);
                    break;
                case 'ArrowUp':
                    newIndex = Math.max(currentIndex - 3, 0);
                    break;
                default:
                    return;
            }

            event.preventDefault();
            items[newIndex].focus();
        });
    }

    // Add focus trap for lightbox
    document.addEventListener('keydown', function(event) {
        if (lightbox.style.display === 'flex') {
            const focusableElements = lightbox.querySelectorAll(
                'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
            );
            const firstElement = focusableElements[0];
            const lastElement = focusableElements[focusableElements.length - 1];

            if (event.key === 'Tab') {
                if (event.shiftKey) {
                    if (document.activeElement === firstElement) {
                        event.preventDefault();
                        lastElement.focus();
                    }
                } else {
                    if (document.activeElement === lastElement) {
                        event.preventDefault();
                        firstElement.focus();
                    }
                }
            }
        }
    });

    // High contrast mode detection
    if (window.matchMedia && window.matchMedia('(prefers-contrast: high)').matches) {
        document.body.classList.add('high-contrast');
    }

    // Reduced motion preference
    if (window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
        document.body.classList.add('reduced-motion');
    }
}

// Performance optimizations
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Lazy loading for images (if needed in the future)
function initLazyLoading() {
    const images = document.querySelectorAll('img[data-src]');

    if ('IntersectionObserver' in window) {
        const imageObserver = new IntersectionObserver((entries, observer) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    const img = entry.target;
                    img.src = img.dataset.src;
                    img.classList.remove('lazy');
                    imageObserver.unobserve(img);
                }
            });
        });

        images.forEach(img => imageObserver.observe(img));
    } else {
        // Fallback for browsers without IntersectionObserver
        images.forEach(img => {
            img.src = img.dataset.src;
        });
    }
}

// Initialize lazy loading if needed
// initLazyLoading();

// Analytics (placeholder for future implementation)
function trackEvent(eventName, parameters = {}) {
    // Placeholder for analytics tracking
    console.log('Event tracked:', eventName, parameters);
}

// Export functions for potential use in other scripts
window.PortfolioTemplate = {
    openLightbox,
    closeLightbox,
    validateForm,
    trackEvent
};