// Test suite for e-commerce script
// Run with: node script.test.js

// Mock DOM environment
const { JSDOM } = require('jsdom');
const dom = new JSDOM('<!DOCTYPE html><html><body></body></html>');
global.document = dom.window.document;
global.window = dom.window;
global.localStorage = {
    data: {},
    getItem: function(key) { return this.data[key] || null; },
    setItem: function(key, value) { this.data[key] = value; },
    removeItem: function(key) { delete this.data[key]; },
    clear: function() { this.data = {}; }
};

// Load the script
require('./script.js');

// Test utilities
function assert(condition, message) {
    if (!condition) {
        throw new Error(`Assertion failed: ${message}`);
    }
}

function assertEquals(actual, expected, message) {
    if (actual !== expected) {
        throw new Error(`${message}: expected ${expected}, got ${actual}`);
    }
}

// Test Suite
class TestSuite {
    constructor() {
        this.tests = [];
        this.passed = 0;
        this.failed = 0;
    }

    test(name, fn) {
        this.tests.push({ name, fn });
    }

    run() {
        console.log('Running e-commerce script tests...\n');
        
        for (const test of this.tests) {
            try {
                // Reset localStorage before each test
                localStorage.clear();
                test.fn();
                console.log(`✓ ${test.name}`);
                this.passed++;
            } catch (error) {
                console.log(`✗ ${test.name}: ${error.message}`);
                this.failed++;
            }
        }

        console.log(`\nResults: ${this.passed} passed, ${this.failed} failed`);
        return this.failed === 0;
    }
}

const suite = new TestSuite();

// Cart Tests
suite.test('Cart initialization creates empty cart', () => {
    const cart = new window.Cart();
    assertEquals(cart.items.length, 0, 'Cart should be empty initially');
    assertEquals(cart.getItemCount(), 0, 'Item count should be 0');
    assertEquals(cart.getTotal(), 0, 'Total should be 0');
});

suite.test('Cart loads from localStorage', () => {
    const testData = [{ id: 1, name: 'Test', price: 10, quantity: 2 }];
    localStorage.setItem('ecommerce-cart', JSON.stringify(testData));
    
    const cart = new window.Cart();
    assertEquals(cart.items.length, 1, 'Should load 1 item');
    assertEquals(cart.getItemCount(), 2, 'Should have 2 total items');
});

suite.test('Cart handles corrupted localStorage gracefully', () => {
    localStorage.setItem('ecommerce-cart', 'invalid json');
    
    const cart = new window.Cart();
    assertEquals(cart.items.length, 0, 'Should fallback to empty cart');
});

suite.test('addItem adds new product to cart', () => {
    const cart = new window.Cart();
    const result = cart.addItem('1', 2);
    
    assert(result, 'addItem should return true for valid product');
    assertEquals(cart.items.length, 1, 'Should have 1 item');
    assertEquals(cart.items[0].quantity, 2, 'Should have correct quantity');
});

suite.test('addItem increases quantity for existing product', () => {
    const cart = new window.Cart();
    cart.addItem('1', 1);
    cart.addItem('1', 2);
    
    assertEquals(cart.items.length, 1, 'Should still have 1 unique item');
    assertEquals(cart.items[0].quantity, 3, 'Should have combined quantity');
});

suite.test('addItem returns false for invalid product', () => {
    const cart = new window.Cart();
    const result = cart.addItem('999');
    
    assert(!result, 'addItem should return false for invalid product');
    assertEquals(cart.items.length, 0, 'Cart should remain empty');
});

suite.test('removeItem removes product from cart', () => {
    const cart = new window.Cart();
    cart.addItem('1');
    cart.removeItem('1');
    
    assertEquals(cart.items.length, 0, 'Cart should be empty after removal');
});

suite.test('updateQuantity updates item quantity', () => {
    const cart = new window.Cart();
    cart.addItem('1', 1);
    cart.updateQuantity('1', 5);
    
    assertEquals(cart.items[0].quantity, 5, 'Quantity should be updated');
});

suite.test('updateQuantity removes item when quantity is 0', () => {
    const cart = new window.Cart();
    cart.addItem('1', 1);
    cart.updateQuantity('1', 0);
    
    assertEquals(cart.items.length, 0, 'Item should be removed when quantity is 0');
});

suite.test('updateQuantity handles invalid input', () => {
    const cart = new window.Cart();
    cart.addItem('1', 1);
    cart.updateQuantity('1', 'invalid');
    
    assertEquals(cart.items[0].quantity, 1, 'Quantity should remain unchanged for invalid input');
});

suite.test('getTotal calculates correct total', () => {
    const cart = new window.Cart();
    cart.addItem('1', 2); // $99.99 * 2 = $199.98
    cart.addItem('2', 1); // $199.99 * 1 = $199.99
    
    const expected = (99.99 * 2) + (199.99 * 1);
    assertEquals(cart.getTotal(), expected, 'Total should be calculated correctly');
});

suite.test('clearCart empties the cart', () => {
    const cart = new window.Cart();
    cart.addItem('1', 2);
    cart.clearCart();
    
    assertEquals(cart.items.length, 0, 'Cart should be empty after clear');
    assertEquals(cart.getItemCount(), 0, 'Item count should be 0');
});

// FormValidator Tests
suite.test('validateEmail accepts valid emails', () => {
    const validEmails = [
        'test@example.com',
        'user.name@domain.co.uk',
        'user+tag@example.org'
    ];
    
    validEmails.forEach(email => {
        assert(window.FormValidator.validateEmail(email), `Should accept valid email: ${email}`);
    });
});

suite.test('validateEmail rejects invalid emails', () => {
    const invalidEmails = [
        'invalid',
        '@example.com',
        'test@',
        'test..test@example.com'
    ];
    
    invalidEmails.forEach(email => {
        assert(!window.FormValidator.validateEmail(email), `Should reject invalid email: ${email}`);
    });
});

suite.test('validatePassword accepts valid passwords', () => {
    const validPasswords = ['password123', 'verylongpassword', '12345678'];
    
    validPasswords.forEach(password => {
        assert(window.FormValidator.validatePassword(password), `Should accept valid password: ${password}`);
    });
});

suite.test('validatePassword rejects short passwords', () => {
    const shortPasswords = ['short', '1234567', ''];
    
    shortPasswords.forEach(password => {
        assert(!window.FormValidator.validatePassword(password), `Should reject short password: ${password}`);
    });
});

suite.test('validateRequired accepts non-empty values', () => {
    const validValues = ['test', '   test   ', '123'];
    
    validValues.forEach(value => {
        assert(window.FormValidator.validateRequired(value), `Should accept non-empty value: ${value}`);
    });
});

suite.test('validateRequired rejects empty values', () => {
    const emptyValues = ['', '   ', '\t\n'];
    
    emptyValues.forEach(value => {
        assert(!window.FormValidator.validateRequired(value), `Should reject empty value: "${value}"`);
    });
});

suite.test('sanitizeText prevents XSS', () => {
    const maliciousInput = '<script>alert("xss")</script>';
    const sanitized = window.FormValidator.sanitizeText(maliciousInput);
    
    assert(!sanitized.includes('<script>'), 'Should remove script tags');
    assert(sanitized.includes('&lt;script&gt;'), 'Should encode HTML entities');
});

// Performance Tests
suite.test('Cart operations are performant', () => {
    const cart = new window.Cart();
    const startTime = Date.now();
    
    // Add 1000 items
    for (let i = 0; i < 1000; i++) {
        cart.addItem('1', 1);
    }
    
    const endTime = Date.now();
    const duration = endTime - startTime;
    
    assert(duration < 100, `Cart operations should be fast (took ${duration}ms)`);
});

// Edge Case Tests
suite.test('Cart handles concurrent modifications', () => {
    const cart = new window.Cart();
    
    // Simulate concurrent adds
    cart.addItem('1', 1);
    cart.addItem('1', 1);
    cart.removeItem('1');
    cart.addItem('1', 1);
    
    assertEquals(cart.items.length, 1, 'Should handle concurrent modifications correctly');
});

suite.test('Cart persists data correctly', () => {
    const cart1 = new window.Cart();
    cart1.addItem('1', 2);
    
    // Create new cart instance (simulating page reload)
    const cart2 = new window.Cart();
    assertEquals(cart2.getItemCount(), 2, 'Data should persist across instances');
});

// Run all tests
const success = suite.run();
process.exit(success ? 0 : 1);