// Analytics tracking function - replace with your analytics provider
function trackEvent(eventName, parameters = {}) {
    // Google Analytics 4 example
    if (typeof gtag !== 'undefined') {
        gtag('event', eventName, parameters);
    }

    // Facebook Pixel example
    if (typeof fbq !== 'undefined') {
        fbq('track', 'Lead', parameters);
    }

    // Console log for development
    console.log('Event tracked:', eventName, parameters);
}

// Form validation
function validateEmail(email) {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return emailRegex.test(email);
}

function showError(form, message) {
    // Remove existing error
    const existingError = form.querySelector('.error-message');
    if (existingError) {
        existingError.remove();
    }

    // Add new error
    const errorDiv = document.createElement('div');
    errorDiv.className = 'error-message';
    errorDiv.textContent = message;
    errorDiv.style.color = '#dc2626';
    errorDiv.style.fontSize = '0.875rem';
    errorDiv.style.marginTop = '0.5rem';
    form.appendChild(errorDiv);

    // Focus on input
    const input = form.querySelector('input[type="email"]');
    if (input) {
        input.focus();
    }
}

function clearError(form) {
    const error = form.querySelector('.error-message');
    if (error) {
        error.remove();
    }
}

// Form submission handlers
function handleFormSubmit(event, formId) {
    event.preventDefault();

    const form = event.target;
    const emailInput = form.querySelector('input[type="email"]');
    const email = emailInput.value.trim();

    // Clear previous errors
    clearError(form);

    // Validate email
    if (!email) {
        showError(form, 'Please enter your email address.');
        return;
    }

    if (!validateEmail(email)) {
        showError(form, 'Please enter a valid email address.');
        return;
    }

    // Track form submission
    trackEvent(`${formId}_submit`, {
        email: email,
        form_location: formId
    });

    // Simulate form submission (replace with actual API call)
    console.log(`Form submitted: ${formId}`, { email });

    // Show success message
    showSuccess(form, 'Thank you! We\'ll be in touch soon.');

    // Reset form
    form.reset();

    // Optional: redirect or show next step
    // window.location.href = '/thank-you';
}

function showSuccess(form, message) {
    const successDiv = document.createElement('div');
    successDiv.className = 'success-message';
    successDiv.textContent = message;
    successDiv.style.color = '#059669';
    successDiv.style.fontSize = '0.875rem';
    successDiv.style.marginTop = '0.5rem';
    form.appendChild(successDiv);

    // Remove success message after 5 seconds
    setTimeout(() => {
        if (successDiv.parentNode) {
            successDiv.remove();
        }
    }, 5000);
}

// Initialize forms
function initializeForms() {
    const forms = document.querySelectorAll('.lead-form');

    forms.forEach((form, index) => {
        const formId = form.id || `form-${index}`;
        form.addEventListener('submit', (event) => handleFormSubmit(event, formId));
    });
}

// Smooth scrolling for anchor links
function initializeSmoothScrolling() {
    const anchorLinks = document.querySelectorAll('a[href^="#"]');

    anchorLinks.forEach(link => {
        link.addEventListener('click', (event) => {
            const targetId = link.getAttribute('href');
            const targetElement = document.querySelector(targetId);

            if (targetElement) {
                event.preventDefault();
                targetElement.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });

                // Track navigation
                trackEvent('navigation_click', {
                    target: targetId,
                    source: 'menu'
                });
            }
        });
    });
}

// Intersection Observer for scroll-based animations/tracking
function initializeScrollTracking() {
    const observerOptions = {
        threshold: 0.5,
        rootMargin: '0px 0px -100px 0px'
    };

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const section = entry.target;
                const sectionId = section.id || section.className.split(' ')[0];

                // Track section view
                trackEvent('section_view', {
                    section: sectionId
                });

                // Add visible class for animations
                section.classList.add('visible');
            }
        });
    }, observerOptions);

    // Observe sections
    const sections = document.querySelectorAll('section');
    sections.forEach(section => {
        observer.observe(section);
    });
}

// A/B testing helper
function getABVariant(element) {
    return element.getAttribute('data-ab-variant') || 'default';
}

// Performance optimization: lazy load images
function initializeLazyLoading() {
    const images = document.querySelectorAll('img[loading="lazy"]');

    if ('IntersectionObserver' in window) {
        const imageObserver = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    const img = entry.target;
                    img.src = img.dataset.src || img.src;
                    img.classList.remove('lazy');
                    imageObserver.unobserve(img);
                }
            });
        });

        images.forEach(img => imageObserver.observe(img));
    }
}

// Initialize everything when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    initializeForms();
    initializeSmoothScrolling();
    initializeScrollTracking();
    initializeLazyLoading();

    // Track page view
    trackEvent('page_view', {
        page_title: document.title,
        page_location: window.location.href
    });
});

// Handle browser back/forward buttons
window.addEventListener('popstate', () => {
    trackEvent('navigation', {
        type: 'browser_navigation',
        to: window.location.href
    });
});

// Error handling
window.addEventListener('error', (event) => {
    trackEvent('javascript_error', {
        message: event.message,
        filename: event.filename,
        lineno: event.lineno,
        colno: event.colno
    });
});

// Service Worker registration (for PWA features)
if ('serviceWorker' in navigator) {
    window.addEventListener('load', () => {
        // navigator.serviceWorker.register('/sw.js');
    });
}