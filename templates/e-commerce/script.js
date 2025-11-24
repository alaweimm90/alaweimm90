// Product data (in a real app, this would come from an API)
const products = [
    { id: 1, name: 'Wireless Headphones', price: 99.99, image: 'https://via.placeholder.com/300x200?text=Product+1' },
    { id: 2, name: 'Smart Watch', price: 199.99, image: 'https://via.placeholder.com/300x200?text=Product+2' },
    { id: 3, name: 'Gaming Laptop', price: 1299.99, image: 'https://via.placeholder.com/300x200?text=Product+3' },
    { id: 4, name: 'Smartphone', price: 699.99, image: 'https://via.placeholder.com/300x200?text=Product+4' }
];

// Cart management
class Cart {
    constructor() {
        this.items = this.loadCart();
        this.updateCartCount();
    }

    loadCart() {
        try {
            const cart = localStorage.getItem('ecommerce-cart');
            return cart ? JSON.parse(cart) : [];
        } catch (error) {
            console.warn('Failed to load cart from localStorage:', error);
            return [];
        }
    }

    saveCart() {
        localStorage.setItem('ecommerce-cart', JSON.stringify(this.items));
        this.updateCartCount();
    }

    addItem(productId, quantity = 1) {
        const product = products.find(p => p.id === parseInt(productId));
        if (!product) return false;

        const existingItem = this.items.find(item => item.id === product.id);
        if (existingItem) {
            existingItem.quantity += quantity;
        } else {
            this.items.push({
                id: product.id,
                name: product.name,
                price: product.price,
                image: product.image,
                quantity: quantity
            });
        }
        this.saveCart();
        this.showNotification(`${product.name} added to cart!`);
        return true;
    }

    removeItem(productId) {
        this.items = this.items.filter(item => item.id !== parseInt(productId));
        this.saveCart();
    }

    updateQuantity(productId, quantity) {
        const parsedQuantity = parseInt(quantity, 10);
        if (isNaN(parsedQuantity) || parsedQuantity < 0) {
            console.warn('Invalid quantity provided:', quantity);
            return;
        }
        const item = this.items.find(item => item.id === parseInt(productId));
        if (item) {
            item.quantity = Math.max(0, parsedQuantity);
            if (item.quantity === 0) {
                this.removeItem(productId);
            } else {
                this.saveCart();
            }
        }
    }

    getTotal() {
        return this.items.reduce((total, item) => total + (item.price * item.quantity), 0);
    }

    getItemCount() {
        return this.items.reduce((count, item) => count + item.quantity, 0);
    }

    clearCart() {
        this.items = [];
        this.saveCart();
    }

    sanitizeText(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    updateCartCount() {
        const cartCountElement = document.getElementById('cart-count');
        if (cartCountElement) {
            cartCountElement.textContent = this.getItemCount();
        }
    }

    showNotification(message) {
        // Create notification element
        const notification = document.createElement('div');
        notification.className = 'notification';
        // Sanitize message to prevent XSS
        notification.textContent = this.sanitizeText(message);
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background: #10b981;
            color: white;
            padding: 1rem;
            border-radius: 8px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            z-index: 1000;
            animation: slideIn 0.3s ease-out;
        `;

        document.body.appendChild(notification);

        // Remove after 3 seconds
        setTimeout(() => {
            notification.style.animation = 'slideOut 0.3s ease-in';
            setTimeout(() => notification.remove(), 300);
        }, 3000);
    }
}

// Form validation utilities
class FormValidator {
    static validateEmail(email) {
        const emailRegex = /^[a-zA-Z0-9.!#$%&'*+/=?^_`{|}~-]+@[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)*$/;
        return emailRegex.test(email);
    }

    static validatePassword(password) {
        return password.length >= 8;
    }

    static validateRequired(value) {
        return value.trim().length > 0;
    }

    static showError(input, message) {
        const errorElement = input.parentElement.querySelector('.error-message');
        if (errorElement) {
            errorElement.textContent = this.sanitizeText(message);
            errorElement.style.display = 'block';
        }
        input.classList.add('error');
        input.setAttribute('aria-invalid', 'true');
    }

    static clearError(input) {
        const errorElement = input.parentElement.querySelector('.error-message');
        if (errorElement) {
            errorElement.style.display = 'none';
        }
        input.classList.remove('error');
        input.setAttribute('aria-invalid', 'false');
    }

    static sanitizeText(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    static validateForm(form) {
        let isValid = true;
        const inputs = form.querySelectorAll('input[required], select[required], textarea[required]');

        inputs.forEach(input => {
            this.clearError(input);

            if (!this.validateRequired(input.value)) {
                this.showError(input, `${input.name || 'This field'} is required`);
                isValid = false;
            } else if (input.type === 'email' && !this.validateEmail(input.value)) {
                this.showError(input, 'Please enter a valid email address');
                isValid = false;
            } else if (input.type === 'password' && !this.validatePassword(input.value)) {
                this.showError(input, 'Password must be at least 8 characters long');
                isValid = false;
            }
        });

        return isValid;
    }
}

// UI utilities
class UI {
    static showLoading(button) {
        button.disabled = true;
        button.innerHTML = '<span class="loading-spinner"></span> Loading...';
        button.classList.add('loading');
    }

    static hideLoading(button, originalText) {
        button.disabled = false;
        button.innerHTML = originalText;
        button.classList.remove('loading');
    }

    static smoothScroll(target) {
        try {
            const element = document.querySelector(target);
            if (element) {
                element.scrollIntoView({ behavior: 'smooth' });
            }
        } catch (error) {
            console.warn('Failed to scroll to element:', target, error);
        }
    }
}

// Initialize cart
const cart = new Cart();

// Initialization functions
function initMobileMenu() {
    const mobileMenuToggle = document.querySelector('.mobile-menu-toggle');
    const navMenu = document.querySelector('.nav-menu');

    if (mobileMenuToggle && navMenu) {
        mobileMenuToggle.addEventListener('click', function() {
            navMenu.classList.toggle('active');
            const isExpanded = navMenu.classList.contains('active');
            mobileMenuToggle.setAttribute('aria-expanded', isExpanded);
        });

        // Keyboard navigation for mobile menu
        document.addEventListener('keydown', function(e) {
            if (e.key === 'Escape' && navMenu.classList.contains('active')) {
                navMenu.classList.remove('active');
                mobileMenuToggle.setAttribute('aria-expanded', 'false');
            }
        });
    }
}

function initCartButtons() {
    document.addEventListener('click', function(e) {
        if (e.target.classList.contains('add-to-cart')) {
            e.preventDefault();
            const productId = e.target.dataset.productId;
            const button = e.target;

            UI.showLoading(button);
            // Remove artificial delay for better performance
            cart.addItem(productId);
            UI.hideLoading(button, 'Add to Cart');
        }
    });
}

function initFormValidation() {
    document.addEventListener('submit', function(e) {
        const form = e.target;
        if (!FormValidator.validateForm(form)) {
            e.preventDefault();
        }
    });
}

function initSmoothScrolling() {
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            e.preventDefault();
            const target = this.getAttribute('href');
            UI.smoothScroll(target);
        });
    });
}

// Event listeners
document.addEventListener('DOMContentLoaded', function() {
    initMobileMenu();
    initCartButtons();
    initFormValidation();
    initSmoothScrolling();

    // Add notification styles dynamically
    const style = document.createElement('style');
    style.textContent = `
        @keyframes slideIn {
            from { transform: translateX(100%); opacity: 0; }
            to { transform: translateX(0); opacity: 1; }
        }
        @keyframes slideOut {
            from { transform: translateX(0); opacity: 1; }
            to { transform: translateX(100%); opacity: 0; }
        }
        .notification {
            font-family: inherit;
        }
        .loading-spinner {
            display: inline-block;
            width: 16px;
            height: 16px;
            border: 2px solid #ffffff;
            border-radius: 50%;
            border-top-color: transparent;
            animation: spin 1s ease-in-out infinite;
            margin-right: 8px;
        }
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
        .error {
            border-color: #ef4444 !important;
        }
        .error-message {
            color: #ef4444;
            font-size: 0.875rem;
            margin-top: 0.25rem;
            display: none;
        }
    `;
    document.head.appendChild(style);
});

// Export for use in other scripts
window.Cart = Cart;
window.FormValidator = FormValidator;
window.UI = UI;