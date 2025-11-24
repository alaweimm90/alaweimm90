// Performance Optimization Utilities
// Implements caching, lazy loading, and performance monitoring

class PerformanceOptimizer {
    constructor() {
        this.cache = new Map();
        this.observers = new Map();
        this.metrics = {
            loadTime: 0,
            renderTime: 0,
            apiCalls: 0,
            cacheHits: 0,
            cacheMisses: 0
        };
        this.init();
    }

    init() {
        this.measurePageLoad();
        this.setupLazyLoading();
        this.setupIntersectionObserver();
        this.preloadCriticalResources();
    }

    // Performance Monitoring
    measurePageLoad() {
        window.addEventListener('load', () => {
            const navigation = performance.getEntriesByType('navigation')[0];
            this.metrics.loadTime = navigation.loadEventEnd - navigation.fetchStart;
            
            // Measure First Contentful Paint
            const paintEntries = performance.getEntriesByType('paint');
            const fcp = paintEntries.find(entry => entry.name === 'first-contentful-paint');
            if (fcp) {
                this.metrics.renderTime = fcp.startTime;
            }

            this.logPerformanceMetrics();
        });
    }

    logPerformanceMetrics() {
        console.log('Performance Metrics:', {
            ...this.metrics,
            memoryUsage: this.getMemoryUsage(),
            connectionType: this.getConnectionType()
        });
    }

    getMemoryUsage() {
        if ('memory' in performance) {
            return {
                used: Math.round(performance.memory.usedJSHeapSize / 1048576) + ' MB',
                total: Math.round(performance.memory.totalJSHeapSize / 1048576) + ' MB',
                limit: Math.round(performance.memory.jsHeapSizeLimit / 1048576) + ' MB'
            };
        }
        return 'Not available';
    }

    getConnectionType() {
        if ('connection' in navigator) {
            return {
                effectiveType: navigator.connection.effectiveType,
                downlink: navigator.connection.downlink + ' Mbps',
                rtt: navigator.connection.rtt + ' ms'
            };
        }
        return 'Not available';
    }

    // Caching System
    setCache(key, value, ttl = 300000) { // 5 minutes default
        const expiry = Date.now() + ttl;
        this.cache.set(key, { value, expiry });
    }

    getCache(key) {
        const cached = this.cache.get(key);
        if (!cached) {
            this.metrics.cacheMisses++;
            return null;
        }

        if (Date.now() > cached.expiry) {
            this.cache.delete(key);
            this.metrics.cacheMisses++;
            return null;
        }

        this.metrics.cacheHits++;
        return cached.value;
    }

    clearExpiredCache() {
        const now = Date.now();
        for (const [key, value] of this.cache.entries()) {
            if (now > value.expiry) {
                this.cache.delete(key);
            }
        }
    }

    // Lazy Loading
    setupLazyLoading() {
        const images = document.querySelectorAll('img[data-src]');
        const imageObserver = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    const img = entry.target;
                    img.src = img.dataset.src;
                    img.removeAttribute('data-src');
                    imageObserver.unobserve(img);
                }
            });
        }, { rootMargin: '50px' });

        images.forEach(img => imageObserver.observe(img));
    }

    setupIntersectionObserver() {
        const sections = document.querySelectorAll('[data-lazy-load]');
        const sectionObserver = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    const section = entry.target;
                    const loadFunction = section.dataset.lazyLoad;
                    if (window[loadFunction]) {
                        window[loadFunction](section);
                    }
                    sectionObserver.unobserve(section);
                }
            });
        }, { rootMargin: '100px' });

        sections.forEach(section => sectionObserver.observe(section));
    }

    // Resource Preloading
    preloadCriticalResources() {
        const criticalResources = [
            { href: '/api/products', as: 'fetch' },
            { href: '/css/critical.css', as: 'style' },
            { href: '/fonts/main.woff2', as: 'font', type: 'font/woff2', crossorigin: 'anonymous' }
        ];

        criticalResources.forEach(resource => {
            const link = document.createElement('link');
            link.rel = 'preload';
            Object.assign(link, resource);
            document.head.appendChild(link);
        });
    }

    // Debouncing and Throttling
    static debounce(func, wait) {
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

    static throttle(func, limit) {
        let inThrottle;
        return function executedFunction(...args) {
            if (!inThrottle) {
                func.apply(this, args);
                inThrottle = true;
                setTimeout(() => inThrottle = false, limit);
            }
        };
    }

    // Virtual Scrolling for Large Lists
    createVirtualList(container, items, itemHeight, renderItem) {
        const containerHeight = container.clientHeight;
        const visibleItems = Math.ceil(containerHeight / itemHeight) + 2;
        let scrollTop = 0;

        const viewport = document.createElement('div');
        viewport.style.height = containerHeight + 'px';
        viewport.style.overflow = 'auto';

        const content = document.createElement('div');
        content.style.height = items.length * itemHeight + 'px';
        content.style.position = 'relative';

        viewport.appendChild(content);
        container.appendChild(viewport);

        const updateVisibleItems = () => {
            const startIndex = Math.floor(scrollTop / itemHeight);
            const endIndex = Math.min(startIndex + visibleItems, items.length);

            // Clear existing items
            content.innerHTML = '';

            // Render visible items
            for (let i = startIndex; i < endIndex; i++) {
                const item = renderItem(items[i], i);
                item.style.position = 'absolute';
                item.style.top = i * itemHeight + 'px';
                item.style.height = itemHeight + 'px';
                content.appendChild(item);
            }
        };

        viewport.addEventListener('scroll', this.throttle(() => {
            scrollTop = viewport.scrollTop;
            updateVisibleItems();
        }, 16)); // ~60fps

        updateVisibleItems();
        return viewport;
    }

    // Image Optimization
    optimizeImage(img, options = {}) {
        const {
            quality = 0.8,
            maxWidth = 1920,
            maxHeight = 1080,
            format = 'webp'
        } = options;

        return new Promise((resolve) => {
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');

            img.onload = () => {
                // Calculate new dimensions
                let { width, height } = img;
                if (width > maxWidth) {
                    height = (height * maxWidth) / width;
                    width = maxWidth;
                }
                if (height > maxHeight) {
                    width = (width * maxHeight) / height;
                    height = maxHeight;
                }

                canvas.width = width;
                canvas.height = height;

                // Draw and compress
                ctx.drawImage(img, 0, 0, width, height);
                const optimizedDataUrl = canvas.toDataURL(`image/${format}`, quality);
                resolve(optimizedDataUrl);
            };
        });
    }

    // Bundle Splitting Simulation
    async loadModule(moduleName) {
        const cacheKey = `module_${moduleName}`;
        const cached = this.getCache(cacheKey);
        
        if (cached) {
            return cached;
        }

        try {
            // Simulate dynamic import
            const module = await import(`./${moduleName}.js`);
            this.setCache(cacheKey, module, 600000); // 10 minutes
            return module;
        } catch (error) {
            console.error(`Failed to load module ${moduleName}:`, error);
            throw error;
        }
    }

    // Performance Budget monitoring
    checkPerformanceBudget() {
        const budget = {
            loadTime: 3000, // 3 seconds
            renderTime: 1000, // 1 second
            memoryUsage: 50 * 1024 * 1024, // 50MB
            cacheHitRatio: 0.8 // 80%
        };

        const violations = [];
        
        if (this.metrics.loadTime > budget.loadTime) {
            violations.push(`Load time exceeded: ${this.metrics.loadTime}ms > ${budget.loadTime}ms`);
        }

        if (this.metrics.renderTime > budget.renderTime) {
            violations.push(`Render time exceeded: ${this.metrics.renderTime}ms > ${budget.renderTime}ms`);
        }

        const cacheHitRatio = this.metrics.cacheHits / (this.metrics.cacheHits + this.metrics.cacheMisses);
        if (cacheHitRatio < budget.cacheHitRatio) {
            violations.push(`Cache hit ratio too low: ${cacheHitRatio.toFixed(2)} < ${budget.cacheHitRatio}`);
        }

        if (violations.length > 0) {
            console.warn('Performance Budget Violations:', violations);
        }

        return violations.length === 0;
    }

    // Service Worker Registration
    registerServiceWorker() {
        if ('serviceWorker' in navigator) {
            navigator.serviceWorker.register('/sw.js')
                .then(registration => {
                    console.log('Service Worker registered:', registration);
                })
                .catch(error => {
                    console.error('Service Worker registration failed:', error);
                });
        }
    }

    // Critical CSS Inlining
    inlineCriticalCSS() {
        const criticalCSS = `
            /* Critical CSS for above-the-fold content */
            body { margin: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }
            .header { background: #fff; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            .hero { min-height: 60vh; display: flex; align-items: center; justify-content: center; }
            .loading { opacity: 0.6; pointer-events: none; }
        `;

        const style = document.createElement('style');
        style.textContent = criticalCSS;
        document.head.insertBefore(style, document.head.firstChild);
    }
}

// Initialize performance optimizer
const performanceOptimizer = new PerformanceOptimizer();

// Clean up expired cache every 5 minutes
setInterval(() => {
    performanceOptimizer.clearExpiredCache();
}, 300000);

// Check performance budget every minute
setInterval(() => {
    performanceOptimizer.checkPerformanceBudget();
}, 60000);

// Export for use in other scripts
window.PerformanceOptimizer = PerformanceOptimizer;
window.performanceOptimizer = performanceOptimizer;