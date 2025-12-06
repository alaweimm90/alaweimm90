// ============================================================================
// HELIOS DASHBOARD - Interactive Application
// Modern, accessible, feature-rich UI for research discovery platform
// ============================================================================

// ============================================================================
// 1. DATA MODELS & CONSTANTS
// ============================================================================

const DOMAINS = {
    quantum: {
        name: 'Quantum Computing',
        icon: '⚛️',
        description: 'Circuit optimization, error correction, and quantum algorithms',
        algorithms: 4,
        benchmarks: 5
    },
    materials: {
        name: 'Materials Science',
        icon: '🧪',
        description: 'Crystal structure prediction and property design',
        algorithms: 4,
        benchmarks: 5
    },
    optimization: {
        name: 'Optimization',
        icon: '🎯',
        description: 'QAP, TSP, and combinatorial problem solving',
        algorithms: 4,
        benchmarks: 5
    },
    ml: {
        name: 'Machine Learning',
        icon: '🧠',
        description: 'Architecture search, training optimization',
        algorithms: 4,
        benchmarks: 5
    },
    nas: {
        name: 'Neural Architecture Search',
        icon: '🏗️',
        description: 'AutoML with zero-cost proxies',
        algorithms: 4,
        benchmarks: 5
    },
    synthesis: {
        name: 'Molecular Synthesis',
        icon: '💊',
        description: 'Drug discovery and ADME prediction',
        algorithms: 4,
        benchmarks: 5
    },
    graph: {
        name: 'Graph Optimization',
        icon: '🌐',
        description: 'Partitioning and community detection',
        algorithms: 4,
        benchmarks: 5
    }
};

const ALGORITHMS = [
    // Quantum Baselines
    { name: 'QAOA', domain: 'quantum', type: 'baseline', quality: 88, speedup: 2.1, tests: 12 },
    { name: 'VQE', domain: 'quantum', type: 'baseline', quality: 85, speedup: 1.8, tests: 11 },
    { name: 'Circuit Optimizer', domain: 'quantum', type: 'baseline', quality: 82, speedup: 1.5, tests: 10 },

    // Quantum Novel
    { name: 'Warm-Started VQE', domain: 'quantum', type: 'novel', quality: 92, speedup: 3.2, tests: 14 },
    { name: 'Adaptive VQE', domain: 'quantum', type: 'novel', quality: 94, speedup: 3.8, tests: 15 },

    // Materials Baselines
    { name: 'Band Gap Predictor', domain: 'materials', type: 'baseline', quality: 87, speedup: 2.5, tests: 13 },
    { name: 'Formation Energy', domain: 'materials', type: 'baseline', quality: 84, speedup: 2.2, tests: 12 },

    // Materials Novel
    { name: 'GNN Structures', domain: 'materials', type: 'novel', quality: 96, speedup: 12.5, tests: 16 },
    { name: 'Transfer Learning', domain: 'materials', type: 'novel', quality: 93, speedup: 8.3, tests: 14 },

    // Optimization
    { name: 'Greedy Solver', domain: 'optimization', type: 'baseline', quality: 80, speedup: 1.2, tests: 9 },
    { name: 'Local Search 2-opt', domain: 'optimization', type: 'baseline', quality: 83, speedup: 1.6, tests: 10 },
    { name: 'Hybrid Genetic-Tabu', domain: 'optimization', type: 'novel', quality: 95, speedup: 5.1, tests: 16 },
    { name: 'Ant Colony', domain: 'optimization', type: 'novel', quality: 92, speedup: 4.2, tests: 14 },

    // ML
    { name: 'Logistic Regression', domain: 'ml', type: 'baseline', quality: 81, speedup: 1.1, tests: 8 },
    { name: 'Random Forest', domain: 'ml', type: 'baseline', quality: 85, speedup: 1.9, tests: 11 },
    { name: 'Knowledge Distillation', domain: 'ml', type: 'novel', quality: 91, speedup: 3.5, tests: 13 },
    { name: 'Meta-Learning MAML', domain: 'ml', type: 'novel', quality: 94, speedup: 4.7, tests: 15 },

    // NAS
    { name: 'Random Search', domain: 'nas', type: 'baseline', quality: 79, speedup: 1.0, tests: 7 },
    { name: 'Zero-Cost Proxies', domain: 'nas', type: 'novel', quality: 93, speedup: 6.2, tests: 15 },

    // Synthesis
    { name: 'Basic Affinity', domain: 'synthesis', type: 'baseline', quality: 82, speedup: 1.3, tests: 9 },
    { name: 'Transformer Generation', domain: 'synthesis', type: 'novel', quality: 97, speedup: 8.33, tests: 17 },

    // Graph
    { name: 'Greedy Clique', domain: 'graph', type: 'baseline', quality: 78, speedup: 1.1, tests: 8 },
    { name: 'GNN Partitioning', domain: 'graph', type: 'novel', quality: 91, speedup: 5.5, tests: 14 }
];

// ============================================================================
// 2. DOM MANIPULATION & TAB SWITCHING
// ============================================================================

document.addEventListener('DOMContentLoaded', () => {
    initializeApp();
});

function initializeApp() {
    setupTabNavigation();
    setupThemeToggle();
    populateLeaderboard();
    populateAlgorithms();
    setupEventListeners();
}

function setupTabNavigation() {
    const navLinks = document.querySelectorAll('.nav-link');

    navLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const tabId = link.getAttribute('data-tab');
            switchTab(tabId);
        });
    });
}

function switchTab(tabId) {
    // Hide all tabs
    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.classList.remove('active');
    });

    // Remove active from all nav links
    document.querySelectorAll('.nav-link').forEach(link => {
        link.classList.remove('active');
    });

    // Show selected tab
    const selectedTab = document.getElementById(tabId);
    if (selectedTab) {
        selectedTab.classList.add('active');

        // Mark nav link as active
        const activeLink = document.querySelector(`[data-tab="${tabId}"]`);
        if (activeLink) {
            activeLink.classList.add('active');
        }
    }
}

// ============================================================================
// 3. THEME TOGGLE (DARK MODE)
// ============================================================================

function setupThemeToggle() {
    const themeToggle = document.getElementById('theme-toggle');

    // Check for saved theme preference
    const savedTheme = localStorage.getItem('theme') || 'light';
    applyTheme(savedTheme);

    themeToggle.addEventListener('click', () => {
        const currentTheme = localStorage.getItem('theme') || 'light';
        const newTheme = currentTheme === 'light' ? 'dark' : 'light';
        applyTheme(newTheme);
        localStorage.setItem('theme', newTheme);
    });
}

function applyTheme(theme) {
    if (theme === 'dark') {
        document.documentElement.style.colorScheme = 'dark';
    } else {
        document.documentElement.style.colorScheme = 'light';
    }
}

// ============================================================================
// 4. LEADERBOARD FUNCTIONALITY
// ============================================================================

function populateLeaderboard() {
    const tbody = document.getElementById('leaderboard-body');

    // Sort by quality descending
    const sorted = [...ALGORITHMS].sort((a, b) => b.quality - a.quality);

    tbody.innerHTML = sorted.map((algo, index) => `
        <tr>
            <td>
                <strong>#${index + 1}</strong>
            </td>
            <td>
                <span class="algorithm-name">${algo.name}</span>
                <span class="badge" style="margin-left: 0.5rem;">${algo.type}</span>
            </td>
            <td>${DOMAINS[algo.domain].name}</td>
            <td>
                <strong style="color: var(--color-success);">${algo.quality}%</strong>
            </td>
            <td>
                <strong style="color: var(--color-primary);">${algo.speedup}x</strong>
            </td>
            <td>${algo.tests}</td>
        </tr>
    `).join('');
}

function updateLeaderboard() {
    const domain = document.getElementById('leaderboard-domain').value;
    const tbody = document.getElementById('leaderboard-body');

    let filtered = ALGORITHMS;
    if (domain) {
        filtered = filtered.filter(a => a.domain === domain);
    }

    const sorted = [...filtered].sort((a, b) => b.quality - a.quality);

    tbody.innerHTML = sorted.map((algo, index) => `
        <tr>
            <td><strong>#${index + 1}</strong></td>
            <td>
                <span>${algo.name}</span>
                <span class="badge" style="margin-left: 0.5rem;">${algo.type}</span>
            </td>
            <td>${DOMAINS[algo.domain].name}</td>
            <td><strong style="color: var(--color-success);">${algo.quality}%</strong></td>
            <td><strong style="color: var(--color-primary);">${algo.speedup}x</strong></td>
            <td>${algo.tests}</td>
        </tr>
    `).join('');
}

function sortLeaderboard(column) {
    // Implement sorting
    console.log('Sort by:', column);
}

// ============================================================================
// 5. ALGORITHMS TABLE
// ============================================================================

function populateAlgorithms() {
    filterAlgorithms();
}

function filterAlgorithms() {
    const domain = document.getElementById('filter-domain').value;
    const type = document.getElementById('filter-type').value;
    const search = document.getElementById('filter-search').value.toLowerCase();

    let filtered = ALGORITHMS;

    if (domain) {
        filtered = filtered.filter(a => a.domain === domain);
    }

    if (type) {
        filtered = filtered.filter(a => a.type === type);
    }

    if (search) {
        filtered = filtered.filter(a =>
            a.name.toLowerCase().includes(search)
        );
    }

    const list = document.getElementById('algorithms-list');

    if (filtered.length === 0) {
        list.innerHTML = '<div style="padding: 2rem; text-align: center; color: var(--color-text-tertiary);">No algorithms found.</div>';
        return;
    }

    list.innerHTML = `
        <div class="algorithms-header">
            <div>Algorithm</div>
            <div>Type</div>
            <div>Quality</div>
            <div>Speedup</div>
        </div>
        ${filtered.map(algo => `
            <div class="algorithm-item">
                <div>
                    <strong>${algo.name}</strong>
                    <div style="font-size: 0.875rem; color: var(--color-text-tertiary); margin-top: 0.25rem;">
                        ${DOMAINS[algo.domain].name}
                    </div>
                </div>
                <div><span class="badge">${algo.type}</span></div>
                <div><strong style="color: var(--color-success);">${algo.quality}%</strong></div>
                <div><strong style="color: var(--color-primary);">${algo.speedup}x</strong></div>
            </div>
        `).join('')}
    `;
}

// ============================================================================
// 6. DOMAIN DETAIL MODAL
// ============================================================================

function viewDomainDetail(domainId) {
    const domain = DOMAINS[domainId];

    const modalBody = document.getElementById('modal-body');
    modalBody.innerHTML = `
        <h2 style="margin-bottom: 0;">${domain.icon} ${domain.name}</h2>
        <p style="color: var(--color-text-secondary); margin-bottom: 1.5rem;">${domain.description}</p>

        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-bottom: 2rem;">
            <div style="padding: 1rem; background-color: var(--color-bg-secondary); border-radius: 0.5rem;">
                <div style="font-size: 1.5rem; font-weight: 700; color: var(--color-primary);">${domain.algorithms}</div>
                <div style="color: var(--color-text-secondary); font-size: 0.875rem;">Algorithms</div>
            </div>
            <div style="padding: 1rem; background-color: var(--color-bg-secondary); border-radius: 0.5rem;">
                <div style="font-size: 1.5rem; font-weight: 700; color: var(--color-secondary);">${domain.benchmarks}</div>
                <div style="color: var(--color-text-secondary); font-size: 0.875rem;">Benchmarks</div>
            </div>
        </div>

        <h3>Research Areas</h3>
        <ul style="color: var(--color-text-secondary);">
            <li>Advanced methodologies and techniques</li>
            <li>Benchmark problems and evaluations</li>
            <li>Performance optimization</li>
            <li>Integration with other domains</li>
        </ul>
    `;

    openModal('domain-detail-modal');
}

function openModal(modalId) {
    const modal = document.getElementById(modalId);
    if (modal) {
        modal.classList.add('active');
        document.body.style.overflow = 'hidden';
    }
}

function closeModal() {
    const modal = document.querySelector('.modal.active');
    if (modal) {
        modal.classList.remove('active');
        document.body.style.overflow = '';
    }
}

// Close modal on backdrop click
document.addEventListener('click', (e) => {
    if (e.target.classList.contains('modal')) {
        closeModal();
    }
});

// Close modal on Escape key
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
        closeModal();
    }
});

// ============================================================================
// 7. HYPOTHESIS VALIDATOR
// ============================================================================

function validateHypothesis() {
    const hypothesis = document.getElementById('hypothesis-text').value;
    const domain = document.getElementById('hypothesis-domain').value;

    if (!hypothesis.trim()) {
        alert('Please enter a hypothesis');
        return;
    }

    if (!domain) {
        alert('Please select a domain');
        return;
    }

    // Simulate validation (in real app, would call API)
    const confidence = Math.random() * 100;
    const quality = Math.random() * 100;

    const resultsDiv = document.getElementById('validator-results');

    resultsDiv.innerHTML = `
        <h3 style="margin-bottom: 1.5rem;">Validation Results</h3>

        <div class="validation-result success">
            <h4 style="margin-bottom: 0.5rem;">✓ Self-Refutation Protocol</h4>
            <p style="margin: 0; font-size: 0.875rem;">Hypothesis survived 5-strategy falsification with ${(Math.random() * 100).toFixed(1)}% confidence</p>
        </div>

        <div class="validation-result">
            <h4 style="margin-bottom: 0.5rem;">200-Question Interrogation</h4>
            <p style="margin: 0; font-size: 0.875rem;">Answered ${Math.floor(Math.random() * 50) + 150} out of 200 questions successfully</p>
        </div>

        <div class="validation-result success">
            <h4 style="margin-bottom: 0.5rem;">✓ Hall of Failures Check</h4>
            <p style="margin: 0; font-size: 0.875rem;">Hypothesis is ${Math.random() > 0.5 ? 'novel' : 'related to'} previous research</p>
        </div>

        <div style="margin-top: 2rem; padding: 1rem; background: var(--color-primary-lighter); border-radius: 0.5rem;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                <span style="font-weight: 600;">Overall Quality</span>
                <span style="font-size: 1.5rem; font-weight: 700; color: var(--color-primary);">${quality.toFixed(1)}%</span>
            </div>
            <div style="height: 8px; background-color: rgba(37, 99, 235, 0.2); border-radius: 0.25rem; overflow: hidden;">
                <div style="height: 100%; width: ${quality}%; background: linear-gradient(90deg, var(--color-primary), var(--color-secondary));"></div>
            </div>
        </div>
    `;

    resultsDiv.style.display = 'block';
}

// ============================================================================
// 8. UTILITY FUNCTIONS
// ============================================================================

function scrollToTab(tabId) {
    switchTab(tabId);
    const element = document.getElementById(tabId);
    if (element) {
        element.scrollIntoView({ behavior: 'smooth' });
    }
}

function setupEventListeners() {
    // Add keyboard navigation support
    document.addEventListener('keydown', (e) => {
        // Alt + number to switch tabs
        const num = parseInt(e.key);
        if (e.altKey && num >= 1 && num <= 5) {
            const tabs = ['overview', 'domains', 'algorithms', 'leaderboard', 'validator'];
            if (num <= tabs.length) {
                switchTab(tabs[num - 1]);
            }
        }
    });
}

// ============================================================================
// 9. ACCESSIBILITY ENHANCEMENTS
// ============================================================================

// Add ARIA live regions for dynamic updates
const createLiveRegion = () => {
    const region = document.createElement('div');
    region.setAttribute('role', 'status');
    region.setAttribute('aria-live', 'polite');
    region.setAttribute('aria-atomic', 'true');
    region.style.position = 'absolute';
    region.style.left = '-10000px';
    document.body.appendChild(region);
    return region;
};

const liveRegion = createLiveRegion();

// Announce tab changes to screen readers
const originalSwitchTab = switchTab;
switchTab = function(tabId) {
    originalSwitchTab(tabId);
    liveRegion.textContent = `Switched to ${tabId.replace('-', ' ')} tab`;
};

console.log('HELIOS Dashboard initialized successfully');
console.log('Keyboard shortcuts: Alt+1 Overview, Alt+2 Domains, Alt+3 Algorithms, Alt+4 Leaderboard, Alt+5 Validator');
