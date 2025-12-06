# Design System Superprompts Collection

## Table of Contents

1. [Anti-Generic Design Principles](#anti-generic-design-principles)
2. [Authentic Content & Copy](#authentic-content--copy)
3. [Sophisticated Visual Design](#sophisticated-visual-design)
4. [Elegant Animation Patterns](#elegant-animation-patterns)
5. [Component Design Excellence](#component-design-excellence)
6. [Professional Polish Details](#professional-polish-details)
7. [Modern Layout Techniques](#modern-layout-techniques)
8. [Color & Theme Sophistication](#color--theme-sophistication)
9. [Typography Excellence](#typography-excellence)
10. [Anti-Patterns to Avoid](#anti-patterns-to-avoid)
11. [Comprehensive Audit Protocol](#comprehensive-audit-protocol)
12. [Platform Rebranding Framework](#platform-rebranding-framework)

---

## Anti-Generic Design Principles

```
NEVER:
- Use generic Tailwind templates or Bootstrap-like layouts
- Create standard hero sections with "Welcome to [App]" headers
- Implement cookie-cutter card grids or basic flexbox layouts
- Default to generic blue/gray color schemes
- Apply standard rounded corners on everything
- Use template marketplace designs
- Copy Material Design or Ant Design wholesale
```

## Authentic Content & Copy

```
ELIMINATE:
- Fake statistics ("99.9% uptime", "10,000+ users", "5-star rated")
- Generic CTAs ("Get Started", "Learn More", "Sign Up Now")
- Trust badges and fake testimonials
- Lorem ipsum or placeholder content in production
- Corporate jargon ("synergy", "leverage", "cutting-edge", "revolutionary")
- Scammy urgency tactics ("Limited time!", "Only 3 spots left!")
- "AI-powered" badges on everything
- Marketing superlatives without evidence
- Unsubstantiated social proof
```

## Sophisticated Visual Design

### Typography System

```css
/* Mathematical Type Scale */
--type-scale-ratio: 1.25; /* Major Third */
--type-base: 16px;
--type-xs: calc(var(--type-base) / var(--type-scale-ratio));
--type-sm: var(--type-base);
--type-md: calc(var(--type-base) * var(--type-scale-ratio));
--type-lg: calc(var(--type-md) * var(--type-scale-ratio));
--type-xl: calc(var(--type-lg) * var(--type-scale-ratio));

/* Sophisticated Font Stack */
--font-display: 'Instrument Serif', 'Playfair Display', serif;
--font-body: 'Inter var', -apple-system, system-ui, sans-serif;
--font-mono: 'JetBrains Mono', 'Fira Code', monospace;

/* Precise Typography Controls */
--letter-spacing-tight: -0.02em;
--letter-spacing-normal: 0;
--letter-spacing-wide: 0.025em;
--line-height-tight: 1.25;
--line-height-normal: 1.6;
--line-height-relaxed: 1.75;
```

### Color Sophistication

```css
/* Nuanced Color System using HSL */
--color-neutral-50: hsl(210, 20%, 98%);
--color-neutral-100: hsl(210, 18%, 96%);
--color-neutral-200: hsl(210, 16%, 93%);
--color-neutral-300: hsl(210, 14%, 89%);
--color-neutral-400: hsl(210, 12%, 71%);
--color-neutral-500: hsl(210, 10%, 53%);
--color-neutral-600: hsl(210, 12%, 43%);
--color-neutral-700: hsl(210, 14%, 34%);
--color-neutral-800: hsl(210, 18%, 24%);
--color-neutral-900: hsl(210, 24%, 16%);
--color-neutral-950: hsl(210, 30%, 8%);

/* Perceptually Uniform Colors (LAB/LCH) */
--accent-primary: lch(65% 80 260); /* Vibrant blue */
--accent-secondary: lch(75% 65 140); /* Sophisticated green */
--accent-danger: lch(55% 85 30); /* Warning red */
```

### Spacing System

```css
/* Golden Ratio Spacing */
--space-phi: 1.618;
--space-base: 1rem;
--space-xs: calc(var(--space-base) / var(--space-phi) / var(--space-phi));
--space-sm: calc(var(--space-base) / var(--space-phi));
--space-md: var(--space-base);
--space-lg: calc(var(--space-base) * var(--space-phi));
--space-xl: calc(var(--space-lg) * var(--space-phi));
--space-2xl: calc(var(--space-xl) * var(--space-phi));
```

## Elegant Animation Patterns

### Micro-Interactions

```css
/* Sophisticated Timing Functions */
--ease-out-expo: cubic-bezier(0.19, 1, 0.22, 1);
--ease-in-out-quart: cubic-bezier(0.77, 0, 0.175, 1);
--ease-spring: cubic-bezier(0.175, 0.885, 0.32, 1.275);

/* Staggered Animations */
@keyframes fadeInUp {
  from {
    opacity: 0;
    transform: translateY(30px) scale(0.98);
  }
  to {
    opacity: 1;
    transform: translateY(0) scale(1);
  }
}

.stagger-children > * {
  animation: fadeInUp 0.6s var(--ease-out-expo) both;
  animation-delay: calc(var(--stagger-index) * 50ms);
}

/* Magnetic Hover Effect */
.magnetic-hover {
  transition: transform 0.3s var(--ease-out-expo);
}

.magnetic-hover:hover {
  transform: translate(calc(var(--mouse-x) * 0.3px), calc(var(--mouse-y) * 0.3px));
}
```

### Performance-First Animations

```css
/* GPU-Accelerated Properties Only */
.performant-transition {
  will-change: transform, opacity;
  transform: translateZ(0); /* Force GPU layer */
  transition:
    transform 0.3s,
    opacity 0.3s;
}

/* Reduced Motion Support */
@media (prefers-reduced-motion: reduce) {
  *,
  *::before,
  *::after {
    animation-duration: 0.01ms !important;
    animation-iteration-count: 1 !important;
    transition-duration: 0.01ms !important;
    scroll-behavior: auto !important;
  }
}
```

## Component Design Excellence

### Glass Morphism Panel

```css
.glass-panel {
  background: hsl(0 0% 100% / 0.02);
  backdrop-filter: blur(40px) saturate(180%);
  border: 1px solid hsl(0 0% 100% / 0.05);
  box-shadow:
    0 0 0 1px hsl(0 0% 0% / 0.03),
    0 2px 4px hsl(0 0% 0% / 0.04),
    0 8px 16px hsl(0 0% 0% / 0.06),
    0 16px 32px hsl(0 0% 0% / 0.08);
}
```

### Sophisticated Button States

```css
.button-elegant {
  position: relative;
  overflow: hidden;
  transition: all 0.3s var(--ease-out-expo);

  /* Multi-layer shadow */
  box-shadow:
    0 1px 2px hsl(0 0% 0% / 0.05),
    0 2px 4px hsl(0 0% 0% / 0.05),
    0 4px 8px hsl(0 0% 0% / 0.05),
    0 8px 16px hsl(0 0% 0% / 0.05);
}

.button-elegant::before {
  content: '';
  position: absolute;
  inset: 0;
  background: radial-gradient(
    circle at var(--mouse-x) var(--mouse-y),
    hsl(0 0% 100% / 0.1),
    transparent 50%
  );
  opacity: 0;
  transition: opacity 0.3s;
}

.button-elegant:hover::before {
  opacity: 1;
}

/* Ripple Effect */
.button-elegant::after {
  content: '';
  position: absolute;
  left: var(--ripple-x);
  top: var(--ripple-y);
  width: 0;
  height: 0;
  border-radius: 50%;
  background: hsl(0 0% 100% / 0.3);
  transform: translate(-50%, -50%);
  transition:
    width 0.6s,
    height 0.6s;
}

.button-elegant:active::after {
  width: 300px;
  height: 300px;
}
```

## Professional Polish Details

### Custom Selection

```css
::selection {
  background: hsl(var(--accent-primary) / 0.2);
  color: hsl(var(--accent-primary));
}

/* Custom Scrollbar */
::-webkit-scrollbar {
  width: 12px;
  height: 12px;
}

::-webkit-scrollbar-track {
  background: hsl(0 0% 0% / 0.05);
}

::-webkit-scrollbar-thumb {
  background: hsl(0 0% 0% / 0.2);
  border-radius: 6px;
  border: 3px solid transparent;
  background-clip: content-box;
}

::-webkit-scrollbar-thumb:hover {
  background: hsl(0 0% 0% / 0.3);
}
```

### Focus States

```css
/* Custom Focus Ring */
:focus-visible {
  outline: none;
  box-shadow:
    0 0 0 2px hsl(var(--background)),
    0 0 0 4px hsl(var(--accent-primary));
}

/* Keyboard Navigation Indicator */
.keyboard-focus:focus-visible {
  outline: 2px dashed hsl(var(--accent-primary));
  outline-offset: 4px;
}
```

## Modern Layout Techniques

### Advanced Grid System

```css
/* Named Grid Areas */
.app-layout {
  display: grid;
  grid-template-areas:
    'header header header'
    'nav main aside'
    'footer footer footer';
  grid-template-columns: 240px 1fr 320px;
  grid-template-rows: auto 1fr auto;
  min-height: 100vh;
}

/* Fluid Typography */
.fluid-text {
  font-size: clamp(1rem, 2vw + 0.5rem, 1.5rem);
  line-height: clamp(1.5, 2vw + 1.2, 1.75);
}

/* Container Queries */
@container (min-width: 400px) {
  .card {
    grid-template-columns: 150px 1fr;
  }
}

/* Logical Properties */
.international-layout {
  margin-inline: auto;
  padding-block: var(--space-lg);
  border-inline-start: 4px solid hsl(var(--accent-primary));
}
```

## Color & Theme Sophistication

### Dynamic Theme Variables

```css
:root {
  /* Semantic Color Tokens */
  --color-background: hsl(210 30% 98%);
  --color-surface: hsl(210 25% 100%);
  --color-surface-elevated: hsl(210 25% 100%);
  --color-border: hsl(210 20% 94%);
  --color-text-primary: hsl(210 30% 8%);
  --color-text-secondary: hsl(210 20% 35%);
  --color-text-muted: hsl(210 15% 55%);

  /* State Colors */
  --color-hover: hsl(210 40% 96%);
  --color-active: hsl(210 40% 92%);
  --color-selected: hsl(210 100% 96%);
  --color-focus: hsl(210 100% 50% / 0.2);
}

[data-theme='dark'] {
  --color-background: hsl(210 30% 8%);
  --color-surface: hsl(210 25% 12%);
  --color-surface-elevated: hsl(210 25% 16%);
  --color-border: hsl(210 20% 20%);
  --color-text-primary: hsl(210 30% 98%);
  --color-text-secondary: hsl(210 20% 70%);
  --color-text-muted: hsl(210 15% 45%);
}
```

## Typography Excellence

### Variable Font Controls

```css
@font-face {
  font-family: 'Inter var';
  src: url('/fonts/Inter.var.woff2') format('woff2');
  font-weight: 100 900;
  font-display: swap;
  font-optical-sizing: auto;
}

.heading-refined {
  font-variation-settings:
    'wght' 650,
    'slnt' 0,
    'opsz' 32;
  letter-spacing: -0.02em;
  font-kerning: normal;
  font-feature-settings:
    'kern' 1,
    'liga' 1,
    'calt' 1,
    'ss01' 1; /* Contextual alternates */
}

/* Hanging Punctuation */
.prose {
  hanging-punctuation: first allow-end;
  hyphens: auto;
  text-wrap: pretty; /* Chrome 117+ */
}
```

## Anti-Patterns to Avoid

```
NEVER USE:
- "AI-powered" badges without substance
- Floating chat widgets (Intercom, Drift clones)
- Generic icon libraries (use custom or carefully curated)
- Real-time visitor counters
- Particle.js backgrounds
- Matrix rain effects
- Gradient text on gradient backgrounds
- Overused Google Fonts (Roboto, Open Sans, Lato)
- "Handwritten" fonts for personality
- Stock photos of people in suits pointing at screens
- Countdown timers for fake urgency
- Pop-ups within first 5 seconds
- Auto-playing videos with sound
- Infinite scroll without clear endpoints
- Hamburger menus on desktop
- Centered body text over 65 characters wide
- Pure black (#000) on pure white (#FFF)
- Neon colors on dark backgrounds for body text
- Comic Sans, Papyrus, or novelty fonts in professional contexts
```

## Implementation Standards

```
REQUIREMENTS:
1. Semantic HTML before any styling
2. CSS logical properties for internationalization
3. Performance budgets for all animations (<16ms per frame)
4. Respect prefers-reduced-motion
5. Print stylesheets for document interfaces
6. High contrast mode support
7. Focus management for SPAs
8. ARIA labels, not descriptions
9. Progressive enhancement approach
10. Mobile-first responsive design
11. Component-based architecture
12. Design tokens for consistency
13. Accessibility testing at each stage
14. Performance monitoring
15. Cross-browser testing
```

---

## Comprehensive Audit Protocol

### Visual/UX/UI + Performance + Layout Audit Framework

#### 0) Positioning & Information Architecture (Open-Source Focus)

```
1. Eliminate SaaS tropes:
   - No "pricing", "free trial", "trusted by N users"
   - No testimonial carousels
   - Claims must be substantiated (FTC guidance)

2. Primary CTAs for Open-Source:
   - "Run in browser"
   - "Read docs"
   - "Star on GitHub"
   - "Download/Install"
   - "Try examples"
   - "Join community"

3. Open-source IA skeleton:
   Home · Playground · Docs · Examples · Benchmarks ·
   Roadmap · GitHub · Community · Contribute · License · Privacy

4. Repository hygiene:
   - CONTRIBUTING.md
   - Issue/PR templates
   - Code of Conduct
   - Link from site header/footer
```

#### 1) Heuristic Usability Review (Nielsen Norman Group)

```
Score each heuristic (0-3):
1. Visibility of system status
2. Match between system and real world
3. User control and freedom
4. Consistency and standards
5. Error prevention
6. Recognition rather than recall
7. Flexibility and efficiency of use
8. Aesthetic and minimalist design
9. Error recovery
10. Help and documentation

For severity ≥2: problem → violated heuristic → before/after redesign
```

#### 2) Accessibility (WCAG 2.2 AA)

```
Audit Checklist:
- Contrast: text ≥4.5:1, large text ≥3:1, UI elements ≥3:1
- Focus: visible, non-obscured indicators (SC 2.4.11/12/13)
- Target size: ≥24×24 CSS px (SC 2.5.8)
- Reduced motion: honor prefers-reduced-motion
- ARIA patterns: validate against WAI-ARIA APG

Output: Issue table (SC id · element/URL · evidence · fix)
```

#### 3) Interaction Ergonomics

```
- Touch targets: iOS ≥44×44 pt, Material ≥48 dp
- Fitts's law: T = a + b × log₂(1 + D/W)
- Hick-Hyman law: RT = a + b × log₂(n + 1)
```

#### 4) Visual Design & Design Tokens

```json
{
  "typography": {
    "scale": "1.25",
    "base": "16px",
    "fonts": {
      "display": "serif",
      "body": "sans-serif",
      "mono": "monospace"
    }
  },
  "color": {
    "semantic": {
      "primary": "hsl()",
      "secondary": "hsl()",
      "surface": "hsl()",
      "text": "hsl()"
    }
  },
  "spacing": {
    "base": "8px",
    "scale": [4, 8, 12, 16, 24, 32, 48, 64]
  },
  "motion": {
    "duration": {
      "fast": "100ms",
      "normal": "200ms",
      "slow": "300ms"
    },
    "easing": {
      "standard": "cubic-bezier(0.4, 0.0, 0.2, 1)",
      "decelerate": "cubic-bezier(0.0, 0.0, 0.2, 1)",
      "accelerate": "cubic-bezier(0.4, 0.0, 1, 1)"
    }
  }
}
```

#### 5) Performance (Core Web Vitals)

```
Targets:
- LCP ≤2.5s
- INP ≤200ms (replaced FID in 2024)
- CLS ≤0.1
- Lighthouse score ≥90

Optimizations:
- Image: srcset/sizes, loading="lazy", fetchpriority
- Fonts: preload critical WOFF2, font-display
- Resources: limit preconnect/preload
```

#### 6) Prioritized Backlog (RICE Scoring)

```
| Title | Standard | Impact | Effort | Owner | ETA |
|-------|----------|--------|--------|-------|-----|
| Fix contrast | WCAG 2.1 | High | S | Design | 1w |
| Mobile layout | Responsive | High | M | Eng | 2w |
| Focus states | A11y | Med | S | Design | 1w |
```

---

## Platform Rebranding Framework

### Quick Start Protocol

#### Step 1: Platform Discovery (5 minutes)

```
1. What is the platform called and what does it do?
   - Current name:
   - Core functionality (1-2 sentences):

2. Who uses it currently?
   - Primary user type:
   - Main use case:

3. What's driving the rebrand?
   - Why now:

4. Existing materials?
   - Repository/URL:
   - Screenshots/demos:
```

#### Step 2: Strategic Vision

```
Rebranding Objectives:
- Market repositioning
- Feature expansion
- User base growth
- Competition differentiation

Success Metrics:
- User engagement
- Brand recognition
- Conversion rates
- Market share
```

#### Step 3: Design Direction

```
Brand Personality:
□ Professional  □ Innovative  □ Approachable
□ Technical     □ Playful     □ Sophisticated
□ Minimal       □ Bold        □ Experimental

Visual Style:
□ Modern minimalist
□ Geometric/structured
□ Organic/fluid
□ Retro/nostalgic
□ Futuristic/sci-fi
□ Illustrated/artistic
```

#### Step 4: Implementation Paths

##### Path A: Quick Brand Refresh (2-3 days)

```
- New visual identity
- Updated messaging
- Enhanced UX
- Quick wins implementation
```

##### Path B: Strategic Reboot (1-2 weeks)

```
- Architecture analysis
- User journey redesign
- Component system
- Implementation roadmap
```

##### Path C: Full Platform Charter

```
- Technical specification
- Multi-workspace architecture
- Ecosystem strategy
- Migration plan
- Launch strategy
```

### Rebranding Deliverables

```
1. Brand Guidelines
   - Logo system
   - Color palette
   - Typography
   - Voice & tone
   - Design principles

2. Component Library
   - UI components
   - Patterns
   - Templates
   - Documentation

3. Implementation Plan
   - Technical requirements
   - Migration strategy
   - Timeline
   - Resources

4. Launch Strategy
   - Announcement plan
   - User communication
   - Training materials
   - Success metrics
```

---

## Additional Sophistication Patterns

### Noise Texture Overlays

```css
.subtle-texture {
  position: relative;
}

.subtle-texture::before {
  content: '';
  position: absolute;
  inset: 0;
  background-image: repeating-linear-gradient(
    45deg,
    transparent,
    transparent 35px,
    hsl(0 0% 0% / 0.01) 35px,
    hsl(0 0% 0% / 0.01) 70px
  );
  pointer-events: none;
  mix-blend-mode: multiply;
}
```

### Gradient Borders

```css
.gradient-border {
  position: relative;
  background: hsl(var(--background));
  border-radius: 12px;
}

.gradient-border::before {
  content: '';
  position: absolute;
  inset: -1px;
  background: linear-gradient(135deg, hsl(var(--accent-primary)), hsl(var(--accent-secondary)));
  border-radius: inherit;
  z-index: -1;
}
```

### Asymmetric Layouts

```css
.asymmetric-grid {
  display: grid;
  grid-template-columns: 1fr 1.618fr 1fr;
  grid-template-rows: auto 1fr auto;
  gap: var(--space-lg);
}

.hero-asymmetric {
  clip-path: polygon(0 0, 100% 0, 100% calc(100% - 4vw), 0 100%);
}
```

### Command Palette Interface

```css
.command-palette {
  position: fixed;
  inset: 0;
  display: grid;
  place-items: center;
  background: hsl(0 0% 0% / 0.5);
  backdrop-filter: blur(4px);
  animation: fadeIn 0.2s ease-out;
}

.command-input {
  width: min(600px, 90vw);
  padding: var(--space-lg);
  background: hsl(var(--surface));
  border-radius: 12px;
  box-shadow:
    0 25px 50px -12px hsl(0 0% 0% / 0.25),
    0 0 0 1px hsl(0 0% 0% / 0.05);
}
```

---

## Testing & Quality Assurance

### Accessibility Testing Checklist

```
□ Keyboard navigation complete
□ Screen reader tested (NVDA/JAWS/VoiceOver)
□ Color contrast validated
□ Focus indicators visible
□ ARIA labels meaningful
□ Reduced motion respected
□ High contrast mode functional
□ Text resizable to 200%
□ No keyboard traps
□ Error messages clear
```

### Performance Testing

```
□ Lighthouse score ≥90
□ First paint <1s
□ Interactive <3s
□ No layout shifts
□ Images optimized
□ Fonts loaded efficiently
□ JavaScript bundled/minified
□ CSS critical path optimized
□ Service worker caching
□ CDN configured
```

### Cross-Browser Testing

```
□ Chrome (latest)
□ Firefox (latest)
□ Safari (latest)
□ Edge (latest)
□ Mobile Safari
□ Chrome Mobile
□ Samsung Internet
□ Opera Mini (basic support)
```

---

_Last Updated: 2025_
_Version: 1.0.0_
