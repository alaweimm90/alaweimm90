# Page Templates

Reusable HTML templates for creating new brand pages, landing pages, and project sites.

## Available Templates

### 1. Product Landing (`product-landing.html`)

For active SPA products with features, tech stack, and CTAs.

**Best for:** SimCore, QMLab, LLMWorks, Attributa, REPZ, LiveItIconic

**Sections:**
- Hero with tagline, title, description, CTAs
- Feature grid with icons
- Technology stack panel
- Authentication info (optional)

### 2. Research Project (`research-project.html`)

For academic/research projects with publications and research areas.

**Best for:** MagLogic, QMatSim, QubeML, SciComp, SpinCirc

**Sections:**
- Hero with research field and institution
- Research areas grid
- Publications list (optional)
- Project status

### 3. Persona Page (`persona-page.html`)

For content creators, education personas, and personal brands.

**Best for:** MeatheadPhysicist, other content personas

**Sections:**
- Centered hero with avatar
- Content areas grid
- Social links
- Featured content (optional)

### 4. Family Site (`family-site.html`)

For planned family/personal projects with minimal content.

**Best for:** DrMalawein, Rounaq, other family projects

**Sections:**
- Simple hero with status
- "Under development" banner
- Info cards
- Contact section (optional)

## Using Templates

### 1. Copy the template

```bash
cp templates/product-landing.html brands/newproject/index.html
```

### 2. Replace placeholders

All templates use `{{PLACEHOLDER}}` syntax. Replace with actual values:

| Placeholder | Description | Example |
|-------------|-------------|---------|
| `{{PRODUCT_NAME}}` | Product/project name | SimCore |
| `{{TAGLINE}}` | Short tagline | Interactive Scientific Computing |
| `{{DESCRIPTION}}` | 1-2 sentence description | ... |
| `{{PRIMARY_COLOR}}` | Hex color | #22c55e |
| `{{TIER}}` | Template class | scientific, ai-ml, business, etc. |
| `{{DOMAIN_URL}}` | Live domain | https://simcore.dev |
| `{{GITHUB_URL}}` | GitHub repo | https://github.com/... |
| `{{STATUS}}` | Current status | Active, Planned, Research |

### 3. Customize sections

Remove or add sections as needed. All templates use the shared design system.

## Shared Design System

All templates import `../styles/design-system.css` which provides:

- **CSS Variables:** Colors, spacing, typography, shadows
- **Components:** Headers, heroes, buttons, badges, cards, panels, footers
- **Utilities:** Flexbox, spacing, text alignment
- **Template classes:** `.template-scientific`, `.template-ai-ml`, etc.

### Customizing Colors

Override the primary color in your page's `<style>` block:

```css
:root {
  --color-primary: #22c55e; /* Your brand color */
}
```

Or use a template class on `<body>`:

```html
<body class="template-scientific">
```

## File Structure

```
docs/pages/
├── index.html              # Main hub
├── styles/
│   └── design-system.css   # Shared CSS
├── templates/
│   ├── README.md           # This file
│   ├── product-landing.html
│   ├── research-project.html
│   ├── persona-page.html
│   └── family-site.html
├── brands/
│   ├── index.html          # Brand directory
│   ├── simcore/
│   ├── qmlab/
│   └── ...
└── personas/
    └── meathead-physicist/
```

## Adding a New Brand Page

1. Choose the appropriate template
2. Create directory: `mkdir brands/newproject`
3. Copy template: `cp templates/product-landing.html brands/newproject/index.html`
4. Edit placeholders
5. Add to `brands/index.html` directory
6. Update `PROJECT-PLATFORMS-CONFIG.ts` with `brandPageUrl`

## Design Principles

- **Dark theme:** All pages use a dark background (#020617)
- **Consistent spacing:** Use CSS variables (--space-4, --space-6, etc.)
- **Responsive:** Mobile-first with breakpoints at 640px
- **Accessible:** Semantic HTML, good contrast, focus states
- **Fast:** No JavaScript required, minimal CSS
