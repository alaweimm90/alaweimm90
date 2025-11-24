// Modern Blog JavaScript
// Handles all interactive functionality for the blog template

document.addEventListener('DOMContentLoaded', function() {
    // Initialize all components
    initReadingProgress();
    initSocialSharing();
    initForms();
    initSearch();
    initMobileMenu();
    initFilters();
    initLoadMore();
    initScrollEffects();
});

// Reading Progress Bar
function initReadingProgress() {
    const progressBar = document.querySelector('.progress-bar');
    if (!progressBar) return;

    function updateProgress() {
        const scrollTop = window.pageYOffset;
        const docHeight = document.documentElement.scrollHeight - window.innerHeight;
        const scrollPercent = (scrollTop / docHeight) * 100;
        progressBar.style.width = Math.min(scrollPercent, 100) + '%';
    }

    window.addEventListener('scroll', updateProgress);
    updateProgress(); // Initial call
}

// Social Sharing
function initSocialSharing() {
    const shareButtons = document.querySelectorAll('.share-btn');
    if (shareButtons.length === 0) return;

    shareButtons.forEach(button => {
        button.addEventListener('click', function(e) {
            e.preventDefault();
            const platform = this.dataset.platform;
            const url = encodeURIComponent(window.location.href);
            const title = encodeURIComponent(document.title);
            const text = encodeURIComponent('Check out this article!');

            let shareUrl = '';

            switch (platform) {
                case 'twitter':
                    shareUrl = `https://twitter.com/intent/tweet?url=${url}&text=${text}`;
                    break;
                case 'linkedin':
                    shareUrl = `https://www.linkedin.com/sharing/share-offsite/?url=${url}`;
                    break;
                case 'facebook':
                    shareUrl = `https://www.facebook.com/sharer/sharer.php?u=${url}`;
                    break;
                case 'copy':
                    copyToClipboard(window.location.href);
                    showNotification('Link copied to clipboard!', 'success');
                    return;
            }

            if (shareUrl) {
                window.open(shareUrl, '_blank', 'width=600,height=400');
            }
        });
    });
}

// Copy to Clipboard Utility
function copyToClipboard(text) {
    if (navigator.clipboard && window.isSecureContext) {
        navigator.clipboard.writeText(text);
    } else {
        // Fallback for older browsers
        const textArea = document.createElement('textarea');
        textArea.value = text;
        textArea.style.position = 'fixed';
        textArea.style.left = '-999999px';
        textArea.style.top = '-999999px';
        document.body.appendChild(textArea);
        textArea.focus();
        textArea.select();
        document.execCommand('copy');
        textArea.remove();
    }
}

// Notification System
function showNotification(message, type = 'info') {
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.textContent = message;
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        background: ${type === 'success' ? '#d4edda' : '#f8d7da'};
        color: ${type === 'success' ? '#155724' : '#721c24'};
        padding: 12px 16px;
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        z-index: 10000;
        font-weight: 500;
        transform: translateX(100%);
        transition: transform 0.3s ease;
    `;

    document.body.appendChild(notification);

    // Animate in
    setTimeout(() => {
        notification.style.transform = 'translateX(0)';
    }, 100);

    // Remove after 3 seconds
    setTimeout(() => {
        notification.style.transform = 'translateX(100%)';
        setTimeout(() => {
            if (notification.parentNode) {
                notification.parentNode.removeChild(notification);
            }
        }, 300);
    }, 3000);
}

// Form Handling
function initForms() {
    // Newsletter Form
    const newsletterForm = document.getElementById('newsletter-form');
    if (newsletterForm) {
        newsletterForm.addEventListener('submit', function(e) {
            e.preventDefault();
            const email = this.querySelector('#newsletter-email').value;
            const messageDiv = document.getElementById('newsletter-message');

            if (validateEmail(email)) {
                // Simulate API call
                showFormMessage(messageDiv, 'Thank you for subscribing! Check your email for confirmation.', 'success');
                this.reset();
            } else {
                showFormMessage(messageDiv, 'Please enter a valid email address.', 'error');
            }
        });
    }

    // Comment Form
    const commentForm = document.getElementById('comment-form');
    if (commentForm) {
        commentForm.addEventListener('submit', function(e) {
            e.preventDefault();
            const name = this.querySelector('#comment-name').value;
            const email = this.querySelector('#comment-email').value;
            const message = this.querySelector('#comment-message').value;
            const messageDiv = document.getElementById('comment-message');

            if (name && validateEmail(email) && message) {
                // Simulate adding comment
                addComment(name, message);
                showFormMessage(messageDiv, 'Comment posted successfully!', 'success');
                this.reset();
            } else {
                showFormMessage(messageDiv, 'Please fill in all required fields with valid information.', 'error');
            }
        });
    }
}

// Email Validation
function validateEmail(email) {
    const re = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return re.test(email);
}

// Form Message Display
function showFormMessage(container, message, type) {
    container.textContent = message;
    container.className = `form-message ${type}`;
    container.style.display = 'block';

    // Auto-hide success messages
    if (type === 'success') {
        setTimeout(() => {
            container.style.display = 'none';
        }, 5000);
    }
}

// Add Comment (Static Implementation)
function addComment(name, message) {
    const commentsList = document.querySelector('.comments-list');
    if (!commentsList) return;

    const commentHTML = `
        <div class="comment">
            <div class="comment-avatar">
                <img src="https://via.placeholder.com/50x50/4A90E2/FFFFFF?text=${name.charAt(0).toUpperCase()}" alt="${name}">
            </div>
            <div class="comment-content">
                <div class="comment-header">
                    <span class="comment-author">${name}</span>
                    <time datetime="${new Date().toISOString()}">${new Date().toLocaleDateString()}</time>
                </div>
                <p>${message}</p>
            </div>
        </div>
    `;

    commentsList.insertAdjacentHTML('afterbegin', commentHTML);

    // Update comment count
    const countElement = document.querySelector('.comment-count');
    if (countElement) {
        const currentCount = parseInt(countElement.textContent) || 0;
        countElement.textContent = `(${currentCount + 1})`;
    }
}

// Search Functionality (Static Simulation)
function initSearch() {
    const searchForm = document.getElementById('search-form');
    const searchInput = document.getElementById('search-input');
    const clearSearch = document.getElementById('clear-search');
    const resultsList = document.getElementById('results-list');
    const noResults = document.getElementById('no-results');
    const resultsCount = document.querySelector('.result-count');

    if (!searchForm) return;

    // Clear search
    if (clearSearch) {
        clearSearch.addEventListener('click', function() {
            searchInput.value = '';
            searchInput.focus();
            hideResults();
        });
    }

    // Search input handler
    if (searchInput) {
        searchInput.addEventListener('input', function() {
            const query = this.value.trim();
            clearSearch.style.display = query ? 'block' : 'none';

            if (query.length > 2) {
                performSearch(query);
            } else {
                hideResults();
            }
        });
    }

    // Form submission
    searchForm.addEventListener('submit', function(e) {
        e.preventDefault();
        const query = searchInput.value.trim();
        if (query) {
            performSearch(query);
        }
    });

    function performSearch(query) {
        // Simulate search results
        const mockResults = [
            { title: 'The Future of Web Development', excerpt: 'Explore the latest trends...', category: 'Technology' },
            { title: 'CSS Grid Layout Guide', excerpt: 'Complete guide to CSS Grid...', category: 'Tutorial' },
            { title: 'Typography in Web Design', excerpt: 'Mastering clean typography...', category: 'Design' }
        ];

        const filteredResults = mockResults.filter(result =>
            result.title.toLowerCase().includes(query.toLowerCase()) ||
            result.excerpt.toLowerCase().includes(query.toLowerCase())
        );

        displayResults(filteredResults, query);
    }

    function displayResults(results, query) {
        if (results.length > 0) {
            resultsCount.textContent = `(${results.length})`;
            noResults.style.display = 'none';
            resultsList.style.display = 'block';

            // Update results (simplified - in real app, would rebuild the list)
            const resultItems = resultsList.querySelectorAll('.search-result-item');
            resultItems.forEach((item, index) => {
                if (index < results.length) {
                    const title = item.querySelector('h3');
                    const excerpt = item.querySelector('.result-excerpt');
                    if (title) title.textContent = results[index].title;
                    if (excerpt) excerpt.textContent = results[index].excerpt;
                    item.style.display = 'flex';
                } else {
                    item.style.display = 'none';
                }
            });
        } else {
            hideResults();
            noResults.style.display = 'block';
        }
    }

    function hideResults() {
        resultsCount.textContent = '(0)';
        resultsList.style.display = 'none';
        noResults.style.display = 'none';
    }
}

// Mobile Menu
function initMobileMenu() {
    const mobileToggle = document.querySelector('.mobile-menu-toggle');
    const mainNav = document.querySelector('.main-nav');

    if (!mobileToggle || !mainNav) return;

    mobileToggle.addEventListener('click', function() {
        const isOpen = mainNav.classList.contains('mobile-open');

        if (isOpen) {
            closeMobileMenu();
        } else {
            openMobileMenu();
        }
    });

    // Close menu when clicking outside
    document.addEventListener('click', function(e) {
        if (!mobileToggle.contains(e.target) && !mainNav.contains(e.target)) {
            closeMobileMenu();
        }
    });

    // Close menu on window resize
    window.addEventListener('resize', function() {
        if (window.innerWidth > 768) {
            closeMobileMenu();
        }
    });

    function openMobileMenu() {
        mainNav.classList.add('mobile-open');
        mobileToggle.classList.add('active');
        document.body.style.overflow = 'hidden';
    }

    function closeMobileMenu() {
        mainNav.classList.remove('mobile-open');
        mobileToggle.classList.remove('active');
        document.body.style.overflow = '';
    }
}

// Filter Toggle (Search Page)
function initFilters() {
    const filterToggle = document.getElementById('filter-toggle');
    const filtersPanel = document.getElementById('filters-panel');
    const filterCount = document.getElementById('filter-count');

    if (!filterToggle || !filtersPanel) return;

    filterToggle.addEventListener('click', function() {
        const isOpen = filtersPanel.style.display !== 'none';

        if (isOpen) {
            filtersPanel.style.display = 'none';
            this.querySelector('svg').style.transform = 'rotate(0deg)';
        } else {
            filtersPanel.style.display = 'grid';
            this.querySelector('svg').style.transform = 'rotate(180deg)';
        }
    });

    // Filter change handler
    const filters = filtersPanel.querySelectorAll('select');
    filters.forEach(filter => {
        filter.addEventListener('change', function() {
            updateFilterCount();
        });
    });

    function updateFilterCount() {
        const activeFilters = Array.from(filters).filter(f => f.value !== '').length;
        if (activeFilters > 0) {
            filterCount.textContent = activeFilters;
            filterCount.style.display = 'inline-block';
        } else {
            filterCount.style.display = 'none';
        }
    }

    // Clear filters
    const clearFilters = document.getElementById('clear-filters');
    if (clearFilters) {
        clearFilters.addEventListener('click', function() {
            filters.forEach(filter => filter.value = '');
            updateFilterCount();
        });
    }
}

// Load More Functionality
function initLoadMore() {
    const loadMoreButtons = document.querySelectorAll('.load-more-btn');

    loadMoreButtons.forEach(button => {
        button.addEventListener('click', function() {
            // Simulate loading
            this.textContent = 'Loading...';
            this.disabled = true;

            setTimeout(() => {
                // In a real app, this would load more content
                showNotification('More articles loaded!', 'success');
                this.textContent = 'Load More Articles';
                this.disabled = false;
            }, 1500);
        });
    });
}

// Scroll Effects
function initScrollEffects() {
    // Smooth scroll for anchor links
    const anchorLinks = document.querySelectorAll('a[href^="#"]');
    anchorLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                e.preventDefault();
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });

    // Add scroll-based animations
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };

    const observer = new IntersectionObserver(function(entries) {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('animate-in');
            }
        });
    }, observerOptions);

    // Observe elements for animation
    const animateElements = document.querySelectorAll('.article-card, .featured-article, .author-article-card');
    animateElements.forEach(el => observer.observe(el));
}

// Popular Search Tags
function initPopularTags() {
    const popularTags = document.querySelectorAll('.popular-tag');

    popularTags.forEach(tag => {
        tag.addEventListener('click', function() {
            const searchTerm = this.dataset.search;
            const searchInput = document.getElementById('search-input');

            if (searchInput && searchTerm) {
                searchInput.value = searchTerm;
                searchInput.dispatchEvent(new Event('input'));
                searchInput.focus();
            }
        });
    });
}

// Keyboard Navigation
document.addEventListener('keydown', function(e) {
    // Escape key to close modals/menus
    if (e.key === 'Escape') {
        // Close mobile menu
        const mainNav = document.querySelector('.main-nav');
        if (mainNav && mainNav.classList.contains('mobile-open')) {
            document.querySelector('.mobile-menu-toggle').click();
        }

        // Close filters
        const filtersPanel = document.getElementById('filters-panel');
        if (filtersPanel && filtersPanel.style.display !== 'none') {
            document.getElementById('filter-toggle').click();
        }
    }
});

// Performance Optimization
// Debounce function for search input
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

// Apply debounce to search if needed
if (document.getElementById('search-input')) {
    const searchInput = document.getElementById('search-input');
    const debouncedSearch = debounce(function() {
        // Search logic here
    }, 300);

    searchInput.addEventListener('input', debouncedSearch);
}

// Lazy Loading Images (if needed in the future)
function initLazyLoading() {
    const images = document.querySelectorAll('img[data-src]');

    if ('IntersectionObserver' in window) {
        const imageObserver = new IntersectionObserver(function(entries) {
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
        // Fallback for older browsers
        images.forEach(img => {
            img.src = img.dataset.src;
        });
    }
}

// Error Handling
window.addEventListener('error', function(e) {
    console.error('JavaScript Error:', e.error);
    // In production, you might want to send this to an error tracking service
});

// Service Worker Registration (for PWA features)
if ('serviceWorker' in navigator) {
    window.addEventListener('load', function() {
        // navigator.serviceWorker.register('/sw.js')
        //     .then(registration => console.log('SW registered'))
        //     .catch(error => console.log('SW registration failed'));
    });
}

// Accessibility Improvements
function initAccessibility() {
    // Add focus management for modals
    // Skip links are already in HTML

    // Announce dynamic content changes to screen readers
    const observer = new MutationObserver(function(mutations) {
        mutations.forEach(function(mutation) {
            if (mutation.type === 'childList' && mutation.addedNodes.length > 0) {
                // Announce new content if needed
                const newContent = Array.from(mutation.addedNodes)
                    .filter(node => node.nodeType === Node.ELEMENT_NODE)
                    .find(node => node.matches('.notification, .comment'));

                if (newContent) {
                    newContent.setAttribute('aria-live', 'polite');
                }
            }
        });
    });

    observer.observe(document.body, {
        childList: true,
        subtree: true
    });
}

// Initialize accessibility features
initAccessibility();
initPopularTags();

// Export functions for potential use in other scripts
window.BlogUtils = {
    showNotification,
    copyToClipboard,
    validateEmail
};