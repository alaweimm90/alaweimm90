// Stationery Template Application
class StationeryApp {
    constructor() {
        this.currentTemplate = null;
        this.currentSettings = {
            theme: 'minimalist',
            primaryColor: '#2c3e50',
            secondaryColor: '#3498db',
            headerText: '',
            footerText: 'Page {page}',
            pageCount: 50
        };

        this.init();
    }

    init() {
        this.bindEvents();
        this.loadSavedSettings();
    }

    bindEvents() {
        // Template selection
        document.querySelectorAll('.customize-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const templateType = e.target.closest('.template-card').dataset.type;
                const style = e.target.closest('.template-card').querySelector('.page-style, .date-style, .planner-style');
                const styleValue = style ? style.value : '';

                this.selectTemplate(templateType, styleValue);
            });
        });

        // Theme selection
        document.getElementById('theme-select').addEventListener('change', (e) => {
            this.updateTheme(e.target.value);
        });

        // Color pickers
        document.getElementById('primary-color').addEventListener('change', (e) => {
            this.updatePrimaryColor(e.target.value);
        });

        document.getElementById('secondary-color').addEventListener('change', (e) => {
            this.updateSecondaryColor(e.target.value);
        });

        // Text inputs
        document.getElementById('header-text').addEventListener('input', (e) => {
            this.currentSettings.headerText = e.target.value;
            this.saveSettings();
        });

        document.getElementById('footer-text').addEventListener('input', (e) => {
            this.currentSettings.footerText = e.target.value;
            this.saveSettings();
        });

        document.getElementById('page-count').addEventListener('change', (e) => {
            this.currentSettings.pageCount = parseInt(e.target.value);
            this.saveSettings();
        });

        // Action buttons
        document.getElementById('preview-btn').addEventListener('click', () => {
            this.showPreview();
        });

        document.getElementById('print-btn').addEventListener('click', () => {
            this.printTemplate();
        });

        document.getElementById('back-btn').addEventListener('click', () => {
            this.showTemplateSelection();
        });

        document.getElementById('edit-btn').addEventListener('click', () => {
            this.showCustomization();
        });

        document.getElementById('download-btn').addEventListener('click', () => {
            this.downloadPDF();
        });
    }

    selectTemplate(type, style) {
        this.currentTemplate = { type, style };
        this.showCustomization();
    }

    showTemplateSelection() {
        document.querySelector('.template-selection').style.display = 'block';
        document.querySelector('.customization-panel').style.display = 'none';
        document.querySelector('.preview-section').style.display = 'none';
    }

    showCustomization() {
        document.querySelector('.template-selection').style.display = 'none';
        document.querySelector('.customization-panel').style.display = 'block';
        document.querySelector('.preview-section').style.display = 'none';

        // Update form with current settings
        document.getElementById('theme-select').value = this.currentSettings.theme;
        document.getElementById('primary-color').value = this.currentSettings.primaryColor;
        document.getElementById('secondary-color').value = this.currentSettings.secondaryColor;
        document.getElementById('header-text').value = this.currentSettings.headerText;
        document.getElementById('footer-text').value = this.currentSettings.footerText;
        document.getElementById('page-count').value = this.currentSettings.pageCount;

        this.updateTheme(this.currentSettings.theme);
    }

    showPreview() {
        document.querySelector('.template-selection').style.display = 'none';
        document.querySelector('.customization-panel').style.display = 'none';
        document.querySelector('.preview-section').style.display = 'block';

        this.generatePreview();
    }

    updateTheme(theme) {
        this.currentSettings.theme = theme;
        document.body.className = `theme-${theme}`;
        this.saveSettings();
    }

    updatePrimaryColor(color) {
        this.currentSettings.primaryColor = color;
        document.documentElement.style.setProperty('--primary-color', color);
        this.saveSettings();
    }

    updateSecondaryColor(color) {
        this.currentSettings.secondaryColor = color;
        document.documentElement.style.setProperty('--secondary-color', color);
        this.saveSettings();
    }

    generatePreview() {
        if (!this.currentTemplate) return;

        const templateFile = this.getTemplateFile();
        const params = new URLSearchParams({
            theme: this.currentSettings.theme,
            primary: this.currentSettings.primaryColor,
            secondary: this.currentSettings.secondaryColor,
            header: encodeURIComponent(this.currentSettings.headerText),
            footer: encodeURIComponent(this.currentSettings.footerText),
            pages: this.currentSettings.pageCount
        });

        const previewUrl = `${templateFile}?${params.toString()}`;
        document.getElementById('template-preview').src = previewUrl;
    }

    getTemplateFile() {
        const { type, style } = this.currentTemplate;

        switch (type) {
            case 'notebook':
                switch (style) {
                    case 'ruled': return 'templates/notebook-ruled.html';
                    case 'dotted': return 'templates/notebook-dotted.html';
                    case 'blank': return 'templates/notebook-blank.html';
                }
                break;
            case 'journal':
                switch (style) {
                    case 'dated': return 'templates/journal-dated.html';
                    case 'undated': return 'templates/journal-undated.html';
                }
                break;
            case 'planner':
                switch (style) {
                    case 'weekly': return 'templates/planner-weekly.html';
                    case 'monthly': return 'templates/planner-monthly.html';
                }
                break;
        }

        return 'templates/notebook-ruled.html'; // default
    }

    printTemplate() {
        const iframe = document.getElementById('template-preview');
        if (iframe && iframe.contentWindow) {
            iframe.contentWindow.print();
        } else {
            // Fallback: open template in new window for printing
            const templateFile = this.getTemplateFile();
            const params = new URLSearchParams({
                theme: this.currentSettings.theme,
                primary: this.currentSettings.primaryColor,
                secondary: this.currentSettings.secondaryColor,
                header: encodeURIComponent(this.currentSettings.headerText),
                footer: encodeURIComponent(this.currentSettings.footerText),
                pages: this.currentSettings.pageCount
            });

            const printWindow = window.open(`${templateFile}?${params.toString()}`, '_blank');
            printWindow.onload = () => {
                printWindow.print();
            };
        }
    }

    downloadPDF() {
        // Note: This would require a server-side solution or a library like jsPDF
        // For now, we'll use the browser's print to PDF functionality
        alert('To download as PDF, use your browser\'s "Print to PDF" feature when printing the template.');
        this.printTemplate();
    }

    saveSettings() {
        try {
            localStorage.setItem('stationerySettings', JSON.stringify(this.currentSettings));
        } catch (e) {
            console.warn('Could not save settings to localStorage');
        }
    }

    loadSavedSettings() {
        try {
            const saved = localStorage.getItem('stationerySettings');
            if (saved) {
                this.currentSettings = { ...this.currentSettings, ...JSON.parse(saved) };
            }
        } catch (e) {
            console.warn('Could not load settings from localStorage');
        }
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new StationeryApp();
});

// Utility functions
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

// Add some decorative elements dynamically
document.addEventListener('DOMContentLoaded', () => {
    // Add floating decorative elements
    const decoratives = ['✨', '🌸', '⭐', '🌙', '🌻', '🍀', '🌟', '💫'];
    const container = document.querySelector('.container');

    for (let i = 0; i < 5; i++) {
        const decorative = document.createElement('div');
        decorative.className = 'floating-decorative';
        decorative.textContent = decoratives[Math.floor(Math.random() * decoratives.length)];
        decorative.style.cssText = `
            position: absolute;
            font-size: 24px;
            opacity: 0.1;
            pointer-events: none;
            z-index: -1;
            left: ${Math.random() * 100}%;
            top: ${Math.random() * 100}%;
            animation: float 10s ease-in-out infinite;
            animation-delay: ${Math.random() * 5}s;
        `;
        container.appendChild(decorative);
    }
});

// Add floating animation
const style = document.createElement('style');
style.textContent = `
    @keyframes float {
        0%, 100% { transform: translateY(0px) rotate(0deg); }
        25% { transform: translateY(-10px) rotate(5deg); }
        50% { transform: translateY(-20px) rotate(0deg); }
        75% { transform: translateY(-10px) rotate(-5deg); }
    }
`;
document.head.appendChild(style);