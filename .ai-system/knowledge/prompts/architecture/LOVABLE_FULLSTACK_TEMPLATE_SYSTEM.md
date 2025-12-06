---
name: 'Lovable Fullstack Template System'
version: '2.0'
category: 'project'
tags:
  [
    'lovable',
    'templates',
    'ui',
    'ux',
    'fullstack',
    'design-system',
    'frontend',
    'backend',
    'unification',
  ]
created: '2024-12-05'
updated: '2024-12-05'
author: 'Meshal Alawein'
credits_budget: 200
---

# Lovable Fullstack Template System

## Complete Documentation & Unification Strategy

---

# TABLE OF CONTENTS

1. [Executive Summary](#1-executive-summary)
2. [Project Overview](#2-project-overview)
3. [Design System Foundation](#3-design-system-foundation)
4. [5 Design Engines](#4-five-design-engines)
5. [Template Catalog (43+ Templates)](#5-template-catalog)
6. [Individual Template Prompts](#6-individual-template-prompts)
7. [Lovable Project Audit Prompt](#7-lovable-project-audit-prompt)
8. [Fullstack Template Unification Strategy](#8-fullstack-template-unification-strategy)
9. [Implementation Roadmap](#9-implementation-roadmap)
10. [Technical Specifications](#10-technical-specifications)
11. [Post-Processing Checklist](#11-post-processing-checklist)
12. [Appendices](#12-appendices)

---

# 1. EXECUTIVE SUMMARY

## 1.1 Purpose

This document provides a complete, production-ready system for generating, auditing, and unifying UI/UX templates using Lovable.dev. The system is designed to:

1. **Maximize 200 Lovable credits** through strategic template generation
2. **Establish 5 distinct design engines** for diverse aesthetic needs
3. **Create 43+ deployment-ready templates** across 12 categories
4. **Enable fullstack unification** with backend integration patterns
5. **Provide audit capabilities** to assess existing Lovable projects

## 1.2 Key Deliverables

| Deliverable          | Status      | Description                                                   |
| -------------------- | ----------- | ------------------------------------------------------------- |
| Master Design Tokens | ✅ Complete | Alawein Design System color, typography, spacing              |
| 5 Design Engines     | ✅ Complete | Glassmorphism, Neumorphism, Brutalist, Cyberpunk, Soft Pastel |
| Template Catalog     | ✅ Complete | 43+ templates across 12 categories                            |
| Individual Prompts   | ✅ Complete | 10 detailed template prompts ready for Lovable                |
| Audit Prompt         | ✅ Complete | Project audit and analysis prompt                             |
| Fullstack Strategy   | ✅ Complete | Backend integration and unification patterns                  |

## 1.3 Credit Allocation Summary

| Category                    | Templates | Credits  | Priority |
| --------------------------- | --------- | -------- | -------- |
| Template Gallery Foundation | 1         | 15-20    | P0       |
| Cyberpunk Engine            | 2         | 18-22    | P1       |
| Glassmorphism Engine        | 2         | 14-18    | P1       |
| Brutalist Engine            | 2         | 12-16    | P2       |
| Neumorphism Engine          | 2         | 12-16    | P2       |
| Soft Pastel Engine          | 2         | 12-16    | P2       |
| Fullstack Templates         | 3         | 25-35    | P1       |
| **Buffer for iterations**   | -         | ~20      | -        |
| **TOTAL**                   | ~14       | **~200** | -        |

---

# 2. PROJECT OVERVIEW

## 2.1 Background

The Lovable Template System was initiated to create a comprehensive library of deployment-ready UI/UX templates that align with the Alawein Design System. The system leverages Lovable.dev's AI-powered generation capabilities to produce high-quality React + TypeScript templates.

## 2.2 Goals

1. **Immediate Deployment**: Templates ready to deploy without modification
2. **Design Consistency**: All templates follow Alawein Design System tokens
3. **Technical Excellence**: React 18, TypeScript, Tailwind CSS, Framer Motion
4. **Accessibility**: WCAG 2.1 AA compliance across all templates
5. **Performance**: Lighthouse scores > 90 for all templates
6. **Fullstack Ready**: Backend integration patterns included

## 2.3 Target Use Cases

- **Scientific Research Portals**: Quantum physics, materials science, optimization
- **Developer Portfolios**: Personal branding, project showcases
- **SaaS Products**: Landing pages, dashboards, admin panels
- **Documentation Sites**: API docs, project wikis, changelogs
- **Interactive Demos**: WebGL visualizations, physics simulations

---

# 3. DESIGN SYSTEM FOUNDATION

## 3.1 Alawein Design System - Color Tokens

```yaml
colors:
  # Primary Palette (Quantum Spectrum)
  primary:
    quantum-purple: '#A855F7' # Primary accent
    plasma-pink: '#EC4899' # Secondary accent
    electron-cyan: '#4CC9F0' # Tertiary accent

  # Background Gradients
  backgrounds:
    void-start: '#0F0F23' # Deep space black
    void-mid: '#1A1B3D' # Midnight purple
    void-end: '#2A1B4D' # Deep violet

  # Surface Colors
  surfaces:
    card-dark: 'rgba(255, 255, 255, 0.02)'
    card-border: 'rgba(255, 255, 255, 0.05)'
    glass: 'rgba(255, 255, 255, 0.03)'

  # Text Colors
  text:
    primary: '#FFFFFF'
    secondary: '#A0A0C0'
    muted: '#808090'
    accent: '#A855F7'

  # Status Colors
  status:
    success: '#10B981' # Emerald
    warning: '#F59E0B' # Amber
    error: '#EF4444' # Red
    info: '#4CC9F0' # Cyan

  # Gradient Definitions
  gradients:
    text-gradient: 'linear-gradient(90deg, #A855F7 0%, #EC4899 50%, #4CC9F0 100%)'
    bg-gradient: 'linear-gradient(135deg, #0F0F23 0%, #1A1B3D 50%, #2A1B4D 100%)'
    orbit-gradient: 'linear-gradient(135deg, #4CC9F0 0%, #A855F7 100%)'
    button-gradient: 'linear-gradient(135deg, #A855F7 0%, #EC4899 100%)'
```

## 3.2 Typography System

```yaml
typography:
  fonts:
    display: "'SF Mono', 'Monaco', 'Inconsolata', 'Fira Code', monospace"
    body: "'Inter var', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"
    mono: "'JetBrains Mono', 'Fira Code', 'SF Mono', monospace"
    serif: "'Instrument Serif', 'Playfair Display', Georgia, serif"

  scale:
    xs: '0.75rem' # 12px
    sm: '0.875rem' # 14px
    base: '1rem' # 16px
    lg: '1.125rem' # 18px
    xl: '1.25rem' # 20px
    2xl: '1.5rem' # 24px
    3xl: '1.875rem' # 30px
    4xl: '2.25rem' # 36px
    5xl: '3rem' # 48px
    6xl: '4rem' # 64px

  weights:
    light: 300
    normal: 400
    medium: 500
    semibold: 600
    bold: 700
```

## 3.3 Animation System

```yaml
animation:
  duration:
    instant: '0ms'
    fast: '150ms'
    normal: '300ms'
    slow: '500ms'
    slower: '700ms'
    slowest: '1000ms'

  easing:
    default: 'cubic-bezier(0.4, 0, 0.2, 1)'
    in: 'cubic-bezier(0.4, 0, 1, 1)'
    out: 'cubic-bezier(0, 0, 0.2, 1)'
    in-out: 'cubic-bezier(0.4, 0, 0.2, 1)'
    spring: 'cubic-bezier(0.175, 0.885, 0.32, 1.275)'
    expo-out: 'cubic-bezier(0.19, 1, 0.22, 1)'

  orbital:
    fast: '10s'
    normal: '12s'
    slow: '15s'
```

## 3.4 Effects System

```yaml
effects:
  shadows:
    sm: '0 1px 2px 0 rgb(0 0 0 / 0.05)'
    default: '0 1px 3px 0 rgb(0 0 0 / 0.1)'
    md: '0 4px 6px -1px rgb(0 0 0 / 0.1)'
    lg: '0 10px 15px -3px rgb(0 0 0 / 0.1)'
    xl: '0 20px 25px -5px rgb(0 0 0 / 0.1)'
    glow-purple: '0 0 20px rgba(168, 85, 247, 0.3)'
    glow-cyan: '0 0 20px rgba(76, 201, 240, 0.3)'
    glow-pink: '0 0 20px rgba(236, 72, 153, 0.3)'

  radius:
    none: '0'
    sm: '0.125rem'
    default: '0.25rem'
    md: '0.375rem'
    lg: '0.5rem'
    xl: '0.75rem'
    2xl: '1rem'
    full: '9999px'

  blur:
    sm: 'blur(4px)'
    default: 'blur(8px)'
    md: 'blur(12px)'
    lg: 'blur(16px)'
    xl: 'blur(24px)'
    glass: 'blur(40px) saturate(180%)'
```

---

# 4. FIVE DESIGN ENGINES

## 4.1 Engine Overview

| Engine        | Aesthetic                  | Primary Use Cases                       | Tailwind Classes                                                   |
| ------------- | -------------------------- | --------------------------------------- | ------------------------------------------------------------------ |
| Glassmorphism | Frosted glass, translucent | Dashboards, weather apps, music players | `backdrop-blur-xl bg-white/10 border border-white/20`              |
| Neumorphism   | Soft, tactile, 3D          | Settings panels, calculators, forms     | `shadow-[8px_8px_16px_#c5ccd6,-8px_-8px_16px_#ffffff]`             |
| Brutalist     | Raw, bold, high contrast   | Task managers, portfolios, statements   | `border-4 border-black rounded-none font-mono`                     |
| Cyberpunk     | Neon, futuristic, HUD      | Dashboards, chat apps, games            | `bg-slate-950 text-cyan-50 shadow-[0_0_20px_rgba(34,211,238,0.3)]` |
| Soft Pastel   | Gentle, friendly, rounded  | File managers, calendars, wellness      | `bg-gradient-to-br from-rose-200 via-lavender-200 to-sky-200`      |

## 4.2 Glassmorphism Engine

### Design Principles

- Frosted glass effect with backdrop blur
- Translucent backgrounds (10-20% opacity)
- Subtle borders (white at 20% opacity)
- Colored glow shadows for depth
- Dark gradient backgrounds for contrast

### CSS Implementation

```css
.glass-card {
  background: rgba(255, 255, 255, 0.1);
  backdrop-filter: blur(24px) saturate(180%);
  -webkit-backdrop-filter: blur(24px) saturate(180%);
  border: 1px solid rgba(255, 255, 255, 0.2);
  border-radius: 16px;
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
}

.glass-card:hover {
  background: rgba(255, 255, 255, 0.15);
  box-shadow: 0 8px 32px rgba(168, 85, 247, 0.2);
  transform: translateY(-4px);
}
```

### Tailwind Configuration

```javascript
// tailwind.config.js
module.exports = {
  theme: {
    extend: {
      backdropBlur: {
        glass: '24px',
      },
      backgroundColor: {
        glass: 'rgba(255, 255, 255, 0.1)',
        'glass-hover': 'rgba(255, 255, 255, 0.15)',
      },
      borderColor: {
        glass: 'rgba(255, 255, 255, 0.2)',
      },
      boxShadow: {
        glass: '0 8px 32px rgba(0, 0, 0, 0.1)',
        'glass-glow': '0 8px 32px rgba(168, 85, 247, 0.2)',
      },
    },
  },
};
```

## 4.3 Neumorphism Engine

### Design Principles

- Soft, extruded appearance
- Light background (#e8eef5 recommended)
- Dual shadows (light and dark)
- Pressed/inset states for interaction
- Subtle, tactile feel

### CSS Implementation

```css
/* Raised element */
.neu-raised {
  background: #e8eef5;
  border-radius: 16px;
  box-shadow:
    8px 8px 16px #c5ccd6,
    -8px -8px 16px #ffffff;
}

/* Pressed/inset element */
.neu-pressed {
  background: #e8eef5;
  border-radius: 16px;
  box-shadow:
    inset 4px 4px 8px #c5ccd6,
    inset -4px -4px 8px #ffffff;
}

/* Toggle switch track */
.neu-toggle-track {
  background: #e8eef5;
  border-radius: 9999px;
  box-shadow:
    inset 2px 2px 4px #c5ccd6,
    inset -2px -2px 4px #ffffff;
}

/* Toggle switch knob */
.neu-toggle-knob {
  background: linear-gradient(145deg, #ffffff, #e6e6e6);
  border-radius: 50%;
  box-shadow:
    4px 4px 8px #c5ccd6,
    -4px -4px 8px #ffffff;
}
```

## 4.4 Brutalist Engine

### Design Principles

- High contrast (black and white primary)
- Thick borders (4px minimum)
- No border radius (sharp corners)
- Monospace typography
- Hard shadows (offset, no blur)
- Yellow accent on interaction

### CSS Implementation

```css
.brutal-card {
  background: #ffffff;
  border: 4px solid #000000;
  border-radius: 0;
  box-shadow: 4px 4px 0px 0px #000000;
  font-family: 'JetBrains Mono', monospace;
}

.brutal-card:hover {
  background: #fde047; /* Yellow */
  transform: translate(-2px, -2px);
  box-shadow: 6px 6px 0px 0px #000000;
}

.brutal-button {
  background: #000000;
  color: #ffffff;
  border: 4px solid #000000;
  padding: 12px 24px;
  font-family: 'JetBrains Mono', monospace;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  cursor: pointer;
}

.brutal-button:hover {
  background: #ffffff;
  color: #000000;
}
```

## 4.5 Cyberpunk Engine

### Design Principles

- Dark backgrounds (slate-950)
- Neon colors (cyan, fuchsia, pink)
- Glowing effects
- Scan lines overlay
- Glitch text effects
- Monospace/terminal aesthetic

### CSS Implementation

```css
.cyber-panel {
  background: #020617; /* slate-950 */
  border: 1px solid rgba(34, 211, 238, 0.3);
  box-shadow:
    0 0 20px rgba(34, 211, 238, 0.2),
    inset 0 0 20px rgba(34, 211, 238, 0.05);
}

.cyber-text {
  color: #22d3ee; /* cyan-400 */
  text-shadow:
    0 0 10px rgba(34, 211, 238, 0.8),
    0 0 20px rgba(34, 211, 238, 0.6),
    0 0 30px rgba(34, 211, 238, 0.4);
}

/* Scan lines overlay */
.cyber-scanlines::after {
  content: '';
  position: absolute;
  inset: 0;
  background: repeating-linear-gradient(
    0deg,
    transparent,
    transparent 2px,
    rgba(0, 0, 0, 0.1) 2px,
    rgba(0, 0, 0, 0.1) 4px
  );
  pointer-events: none;
}

/* Glitch effect */
@keyframes glitch {
  0%,
  90%,
  100% {
    transform: translate(0);
  }
  92% {
    transform: translate(-2px, 1px);
  }
  94% {
    transform: translate(2px, -1px);
  }
  96% {
    transform: translate(-1px, 2px);
  }
  98% {
    transform: translate(1px, -2px);
  }
}

.cyber-glitch {
  animation: glitch 3s infinite;
}
```

## 4.6 Soft Pastel Engine

### Design Principles

- Soft gradient backgrounds
- Rounded corners (2xl/16px)
- Gentle shadows
- Pastel color palette
- Friendly, approachable feel
- White cards with soft shadows

### CSS Implementation

```css
.pastel-bg {
  background: linear-gradient(
    135deg,
    #fecdd3 0%,
    /* rose-200 */ #e9d5ff 50%,
    /* purple-200 */ #bae6fd 100% /* sky-200 */
  );
}

.pastel-card {
  background: #ffffff;
  border-radius: 16px;
  box-shadow:
    0 4px 6px -1px rgba(0, 0, 0, 0.05),
    0 2px 4px -1px rgba(0, 0, 0, 0.03);
}

.pastel-card:hover {
  box-shadow:
    0 10px 15px -3px rgba(0, 0, 0, 0.08),
    0 4px 6px -2px rgba(0, 0, 0, 0.04);
  transform: translateY(-2px);
}

/* Pastel color palette */
:root {
  --pastel-rose: #fecdd3;
  --pastel-pink: #fbcfe8;
  --pastel-purple: #e9d5ff;
  --pastel-lavender: #ddd6fe;
  --pastel-blue: #bfdbfe;
  --pastel-sky: #bae6fd;
  --pastel-mint: #a7f3d0;
  --pastel-peach: #fed7aa;
}
```

---

# 5. TEMPLATE CATALOG

## 5.1 Complete Template List (43+ Templates)

### Category 1: Landing Pages (8 templates)

| #   | Template       | Engine        | Description                     |
| --- | -------------- | ------------- | ------------------------------- |
| 1   | Hero Gradient  | Glassmorphism | Animated gradient hero with CTA |
| 2   | Product Launch | Cyberpunk     | Neon product showcase           |
| 3   | SaaS Landing   | Glassmorphism | Feature-focused SaaS page       |
| 4   | Portfolio Hero | Brutalist     | Bold personal statement         |
| 5   | Event Landing  | Soft Pastel   | Conference/event promotion      |
| 6   | Coming Soon    | Glassmorphism | Waitlist with countdown         |
| 7   | Open Source    | Brutalist     | GitHub project landing          |
| 8   | App Download   | Neumorphism   | Mobile app promotion            |

### Category 2: Dashboards (7 templates)

| #   | Template            | Engine        | Description            |
| --- | ------------------- | ------------- | ---------------------- |
| 9   | Analytics Dashboard | Glassmorphism | Data visualization hub |
| 10  | Admin Panel         | Neumorphism   | Full admin interface   |
| 11  | Metrics Command     | Cyberpunk     | Real-time metrics HUD  |
| 12  | Performance Monitor | Cyberpunk     | System monitoring      |
| 13  | Project Dashboard   | Soft Pastel   | Project management     |
| 14  | Sales Dashboard     | Glassmorphism | Revenue tracking       |
| 15  | DevOps Dashboard    | Brutalist     | CI/CD monitoring       |

### Category 3: E-commerce (5 templates)

| #   | Template       | Engine        | Description         |
| --- | -------------- | ------------- | ------------------- |
| 16  | Product Grid   | Soft Pastel   | Product catalog     |
| 17  | Product Detail | Glassmorphism | Single product page |
| 18  | Shopping Cart  | Neumorphism   | Cart interface      |
| 19  | Checkout Flow  | Soft Pastel   | Multi-step checkout |
| 20  | Order Tracking | Cyberpunk     | Shipment tracking   |

### Category 4: Social/Community (4 templates)

| #   | Template        | Engine        | Description         |
| --- | --------------- | ------------- | ------------------- |
| 21  | User Profile    | Glassmorphism | Social profile page |
| 22  | Activity Feed   | Soft Pastel   | Social feed         |
| 23  | Chat Interface  | Cyberpunk     | Messaging UI        |
| 24  | Community Forum | Brutalist     | Discussion board    |

### Category 5: Productivity (5 templates)

| #   | Template     | Engine      | Description         |
| --- | ------------ | ----------- | ------------------- |
| 25  | Task Manager | Brutalist   | Kanban board        |
| 26  | Calendar App | Soft Pastel | Event calendar      |
| 27  | Note Taking  | Neumorphism | Notes interface     |
| 28  | File Manager | Soft Pastel | File browser        |
| 29  | Code Editor  | Cyberpunk   | Monaco-style editor |

### Category 6: Forms & Auth (4 templates)

| #   | Template        | Engine        | Description  |
| --- | --------------- | ------------- | ------------ |
| 30  | Login/Register  | Glassmorphism | Auth forms   |
| 31  | Multi-step Form | Neumorphism   | Wizard form  |
| 32  | Settings Panel  | Neumorphism   | App settings |
| 33  | Survey Builder  | Soft Pastel   | Form builder |

### Category 7: Content/Media (4 templates)

| #   | Template       | Engine        | Description         |
| --- | -------------- | ------------- | ------------------- |
| 34  | Blog Layout    | Brutalist     | Article listing     |
| 35  | Article Reader | Glassmorphism | Reading experience  |
| 36  | Media Gallery  | Soft Pastel   | Image/video gallery |
| 37  | Music Player   | Glassmorphism | Audio player UI     |

### Category 8: Documentation (3 templates)

| #   | Template     | Engine        | Description        |
| --- | ------------ | ------------- | ------------------ |
| 38  | API Docs     | Cyberpunk     | API documentation  |
| 39  | Project Wiki | Glassmorphism | Documentation site |
| 40  | Changelog    | Brutalist     | Release notes      |

### Category 9: Scientific/Research (3 templates)

| #   | Template           | Engine        | Description        |
| --- | ------------------ | ------------- | ------------------ |
| 41  | Research Portal    | Glassmorphism | Lab website        |
| 42  | Data Visualization | Cyberpunk     | 3D data viz        |
| 43  | Simulation UI      | Cyberpunk     | Physics simulation |

---

# 6. INDIVIDUAL TEMPLATE PROMPTS

## 6.1 Master Template Gallery Prompt

```markdown
# LOVABLE PROMPT: Master Template Gallery

Create a comprehensive template gallery showcasing 5 distinct design engines with the following specifications:

## Project Overview

Build a React + TypeScript template gallery that demonstrates 5 design systems: Glassmorphism, Neumorphism, Brutalist, Cyberpunk, and Soft Pastel. Each engine should have its own section with interactive component examples.

## Technical Stack

- React 18 with TypeScript
- Vite for build tooling
- Tailwind CSS for styling
- Framer Motion for animations
- React Three Fiber + Drei for 3D components (Cyberpunk section)
- Lucide React for icons

## Layout Structure

### 1. Navigation Header

- Fixed position with glass effect
- Logo: "TEMPLATE GALLERY" in gradient text
- Engine selector tabs (5 tabs, one per engine)
- Theme toggle (if applicable)
- GitHub link

### 2. Hero Section

- Dynamic hero that changes based on selected engine
- Large title with engine-specific styling
- Animated background elements
- "Explore Templates" CTA

### 3. Engine Sections (5 total)

#### Section A: Glassmorphism Engine

- Background: Dark gradient (#0F0F23 to #2A1B4D)
- Components to showcase:
  - Glass Card with blur effect
  - Glass Navigation Bar
  - Glass Modal/Dialog
  - Glass Form Inputs
  - Glass Buttons (primary, secondary, ghost)
- Floating animated orbs in background
- Tailwind classes: `backdrop-blur-xl bg-white/10 border border-white/20`

#### Section B: Neumorphism Engine

- Background: Soft gray (#e8eef5)
- Components to showcase:
  - Raised Card
  - Pressed/Inset Card
  - Toggle Switch (with animation)
  - Slider Control
  - Circular Progress
  - Calculator-style Buttons
- Tailwind classes: `shadow-[8px_8px_16px_#c5ccd6,-8px_-8px_16px_#ffffff]`

#### Section C: Brutalist Engine

- Background: White with black accents
- Components to showcase:
  - Thick-bordered Card
  - Hard Shadow Button
  - Monospace Typography Display
  - Table with thick borders
  - Form with stark styling
- Yellow (#FDE047) hover states
- Tailwind classes: `border-4 border-black rounded-none font-mono`

#### Section D: Cyberpunk Engine

- Background: Slate 950 (#020617)
- Components to showcase:
  - Neon-bordered Panel
  - Glitch Text Effect
  - Terminal/Console Component
  - HUD-style Stats Display
  - 3D Holographic Element (React Three Fiber):
    - Rotating wireframe cube (cyan)
    - Inner cube (purple)
    - Glowing icosahedron core
    - 200+ orbiting particles
    - 3 rotating data rings
- Scan lines overlay effect
- Tailwind classes: `bg-slate-950 text-cyan-50 shadow-[0_0_20px_rgba(34,211,238,0.3)]`

#### Section E: Soft Pastel Engine

- Background: Gradient (rose-200 → lavender-200 → sky-200)
- Components to showcase:
  - Rounded Card with soft shadow
  - Pill Buttons
  - Tag/Badge Components
  - Progress Bar
  - Avatar Stack
  - Notification Toast
- Tailwind classes: `bg-gradient-to-br from-rose-200 via-purple-200 to-sky-200 rounded-2xl`

### 4. Interactive Features

- Click any component to see its code
- Copy code button with success feedback
- Live preview toggle
- Responsive preview (mobile/tablet/desktop)

### 5. Footer

- Links to each engine section
- "Built with Lovable" badge
- Social links

## Animations (Framer Motion)

- Page transitions: Fade + slide
- Component hover: Scale 1.02 + shadow increase
- Section scroll: Staggered fade-in
- 3D element: Continuous rotation
- Glitch effect: Random interval trigger

## Color Palette

Primary: #A855F7 (quantum purple)
Secondary: #EC4899 (plasma pink)
Tertiary: #4CC9F0 (electron cyan)
Success: #10B981
Warning: #F59E0B
Error: #EF4444

## Typography

- Headings: 'SF Mono', monospace
- Body: 'Inter', sans-serif
- Code: 'JetBrains Mono', monospace

## Responsive Breakpoints

- Mobile: 320px - 767px
- Tablet: 768px - 1023px
- Desktop: 1024px+

## Accessibility Requirements

- WCAG 2.1 AA compliance
- Keyboard navigation
- Focus indicators
- Screen reader labels
- Reduced motion support

## DO NOT INCLUDE

- User authentication
- Backend API calls
- Database connections
- Payment processing
- Cookie consent banners
- Newsletter popups
- Chat widgets
```

## 6.2 Glassmorphism Templates

### Template: Glass Weather Dashboard

```markdown
# LOVABLE PROMPT: Glass Weather Dashboard

Create a glassmorphism weather dashboard with the following specifications:

## Visual Design

- Dark gradient background (#0F0F23 to #1A1B3D)
- Frosted glass cards: backdrop-blur-xl bg-white/10 border border-white/20
- Colored glow shadows: shadow-cyan-500/20, shadow-amber-500/20
- Animated floating orbs in background (cyan, amber, purple)

## Layout Structure

1. **Header Bar** (glass panel):
   - Location name with pin icon
   - Search input with glass styling
   - Settings gear icon

2. **Main Weather Card** (large, centered):
   - Current temperature (large, 72px font)
   - Weather icon (animated sun/cloud/rain)
   - "Feels like" temperature
   - Weather condition text
   - High/Low temperatures

3. **Hourly Forecast Row**:
   - Horizontal scroll of glass cards
   - Each card: Time, Icon, Temperature
   - Current hour highlighted with glow

4. **Weather Details Grid** (2x3):
   - Humidity (with droplet icon)
   - Wind Speed (with wind icon)
   - UV Index (with sun icon)
   - Visibility (with eye icon)
   - Pressure (with gauge icon)
   - Sunrise/Sunset times

5. **7-Day Forecast**:
   - Vertical list of glass cards
   - Day name, Icon, High/Low temps
   - Precipitation percentage

## Animations

- Floating orbs: slow drift animation (20s cycle)
- Temperature fluctuation: subtle pulse effect
- Cloud layers: parallax movement
- Card hover: lift with increased glow
- Weather icon: gentle bounce animation

## Special Effects

- Animated cloud layers behind main card
- Glowing temperature display
- Gradient progress bars for humidity/UV
- Particle effects for rain/snow conditions

## Technical Requirements

- React + TypeScript
- Tailwind CSS with custom shadows
- Framer Motion for animations
- Responsive (mobile-first)

## DO NOT INCLUDE

- Real API integration (use mock data)
- Location permissions
- Ads or promotions
```

### Template: Glass Music Player

```markdown
# LOVABLE PROMPT: Glass Music Player

Create a glassmorphism music player interface:

## Visual Design

- Dark gradient background with album art blur
- Frosted glass controls: backdrop-blur-xl bg-white/10
- Neon glow accents on active elements
- Translucent overlays

## Layout Structure

1. **Album Art Section**:
   - Large album cover (300x300)
   - Reflection effect below
   - Floating glass frame around art
   - Animated vinyl record behind (optional)

2. **Track Info**:
   - Song title (large, white)
   - Artist name (smaller, gray)
   - Album name (smallest, muted)

3. **Progress Bar**:
   - Glass track with gradient fill
   - Glowing playhead
   - Current time / Total time

4. **Control Panel** (glass pill):
   - Shuffle button
   - Previous track
   - Play/Pause (large, centered, glowing)
   - Next track
   - Repeat button

5. **Volume Control**:
   - Speaker icon
   - Glass slider with glow
   - Mute toggle

6. **Equalizer Visualization**:
   - 8-12 vertical bars
   - Animated heights
   - Gradient colors (cyan to purple)
   - Glass container

7. **Playlist Sidebar** (collapsible):
   - Glass panel
   - Track list with hover states
   - Currently playing indicator

## Animations

- Play button: pulse glow when playing
- Equalizer bars: smooth height transitions
- Album art: subtle float animation
- Progress bar: smooth scrubbing
- Track change: crossfade transition

## Technical Requirements

- React + TypeScript
- Tailwind CSS
- Framer Motion
- Mock audio data (no actual playback needed)
```

## 6.3 Neumorphism Templates

### Template: Neu Settings Panel

```markdown
# LOVABLE PROMPT: Neumorphic Settings Panel

Create a neumorphic settings panel interface:

## Visual Design

- Background: #e8eef5 (soft gray-blue)
- Raised elements: shadow-[8px_8px_16px_#c5ccd6,-8px_-8px_16px_#ffffff]
- Pressed elements: shadow-[inset_4px_4px_8px_#c5ccd6,inset_-4px_-4px_8px_#ffffff]
- Accent color: Soft blue (#6366F1)

## Layout Structure

1. **Header**:
   - "Settings" title (raised panel)
   - Back button (circular, raised)
   - Search input (inset)

2. **Category Navigation** (left sidebar):
   - Raised panel container
   - Category buttons:
     - General (gear icon)
     - Appearance (palette icon)
     - Notifications (bell icon)
     - Privacy (shield icon)
     - Account (user icon)
     - About (info icon)
   - Active category: pressed state

3. **Settings Content Area**:
   - Section headers (raised pills)
   - Setting groups in raised cards

4. **Setting Types**:
   - **Toggle Switch**: Pill-shaped, pressed track, raised knob
   - **Slider**: Inset track, raised handle with shadow
   - **Dropdown**: Raised container, inset when open
   - **Radio Buttons**: Circular, pressed when selected
   - **Text Input**: Inset field
   - **Stepper**: +/- buttons (raised), value display (inset)

5. **Sample Settings**:
   - Dark Mode toggle
   - Font Size slider (12-24px)
   - Language dropdown
   - Notification toggles (Email, Push, SMS)
   - Privacy radio buttons (Public, Friends, Private)
   - Auto-save toggle

6. **Action Buttons** (bottom):
   - "Save Changes" (raised, accent color)
   - "Reset to Defaults" (raised, neutral)

## Animations

- Toggle: smooth slide with shadow transition
- Button press: transition to inset shadow
- Slider: smooth drag with haptic feedback feel
- Hover: subtle shadow intensification

## Technical Requirements

- React + TypeScript
- Tailwind CSS with custom shadows
- Framer Motion for transitions
- Form state management
```

### Template: Neu Recipe App

```markdown
# LOVABLE PROMPT: Neumorphic Recipe Application

Create a neumorphic recipe application:

## Visual Design

- Background: #e8eef5
- Cards: Raised neumorphic shadows
- Accent: Warm orange (#F97316) for food-related elements
- Soft, tactile feel throughout

## Layout Structure

1. **Header**:
   - App logo (raised circular)
   - Search bar (inset)
   - Filter button (raised)
   - Favorites button (raised heart)

2. **Recipe Card Grid** (2-3 columns):
   - Raised cards with rounded corners
   - Food image (inset frame)
   - Recipe name
   - Cooking time pill
   - Difficulty rating (raised dots)
   - Save button (heart, toggles to pressed)

3. **Recipe Detail View**:
   - Large hero image (inset frame)
   - Title and description
   - Stats row: Time, Servings, Calories (raised pills)
4. **Ingredients Section**:
   - Raised container
   - Checkable ingredient list
   - Each item: checkbox (pressed when checked) + text
   - "Add to Shopping List" button

5. **Instructions Section**:
   - Numbered steps in raised cards
   - Step completion checkboxes
   - Timer buttons for timed steps

6. **Nutrition Panel**:
   - Raised card
   - Circular progress indicators (inset tracks)
   - Calories, Protein, Carbs, Fat

## Animations

- Checkbox: satisfying press animation
- Card hover: slight lift
- Step completion: checkmark animation
- Timer: countdown with progress ring

## Technical Requirements

- React + TypeScript
- Tailwind CSS
- Framer Motion
- Mock recipe data
```

## 6.4 Brutalist Templates

### Template: Brutalist Task Manager

```markdown
# LOVABLE PROMPT: Brutalist Task Manager

Create a brutalist task manager with kanban board:

## Visual Design

- Background: White (#FFFFFF)
- Borders: 4px solid black, no border-radius
- Typography: Monospace, uppercase, tight tracking
- Hard shadows: shadow-[4px_4px_0px_0px_rgba(0,0,0,1)]
- Accent: Neon yellow (#FDE047) on hover

## Layout Structure

1. **Header Bar**:
   - "TASK MANAGER" in bold monospace
   - Thick black border bottom
   - Add Task button (black bg, white text)
   - Filter dropdown (thick border)

2. **Kanban Columns** (3-4):
   - Column headers: "TODO", "IN PROGRESS", "REVIEW", "DONE"
   - Black background headers, white text
   - Thick black borders around columns

3. **Task Cards**:
   - White background
   - 4px black border
   - Hard shadow offset
   - Task title (uppercase)
   - Priority tag (colored border)
   - Due date
   - Assignee initials (black circle)
   - Drag handle (6 dots)

4. **Priority Tags**:
   - HIGH: Red border, "!!!" prefix
   - MEDIUM: Yellow border
   - LOW: Gray border

5. **Add Task Modal**:
   - Thick black border
   - No rounded corners
   - Form fields with black borders
   - "CREATE" button (black, yellow on hover)
   - "CANCEL" button (white, black border)

6. **Hover States**:
   - Cards: Yellow background
   - Buttons: Invert colors
   - Links: Underline appears

## Animations

- Card drag: Increase shadow offset
- Button hover: Instant color flip (no transition)
- Modal: Slide in from top (harsh, fast)

## Technical Requirements

- React + TypeScript
- Tailwind CSS
- Drag and drop (react-beautiful-dnd or similar)
- No smooth transitions (brutalist = harsh)
```

### Template: Brutalist Travel Planner

```markdown
# LOVABLE PROMPT: Brutalist Travel Planner

Create a brutalist travel planner interface:

## Visual Design

- High contrast black and white
- Giant typography for destinations
- Thick borders, no curves
- Hard shadows
- Yellow accent on interaction

## Layout Structure

1. **Header**:
   - "TRAVEL" in massive font (120px+)
   - Black bar with white text
   - Search input (thick black border)

2. **Destination Cards** (grid):
   - Giant city names: "PARIS", "TOKYO", "BERLIN"
   - Black and white photography (high contrast)
   - Thick black frame
   - Hard shadow
   - Price in corner (black pill)

3. **Trip Details Panel**:
   - Destination name (huge)
   - Dates in monospace
   - Flight info (table with thick borders)
   - Hotel info (stark layout)
   - Activities list (numbered, bold)

4. **Itinerary View**:
   - Day headers (black bars)
   - Time slots (grid layout)
   - Activity cards (white, black border)
   - Map placeholder (black border frame)

5. **Booking Section**:
   - Form with thick-bordered inputs
   - "BOOK NOW" button (black, full width)
   - Price breakdown (table format)

## Typography

- Headings: 72-120px, uppercase, tight
- Body: 16px monospace
- Labels: 12px uppercase, letter-spacing wide

## Animations

- Hover: Instant yellow background
- Click: Invert colors
- Page transition: Hard cut (no fade)

## Technical Requirements

- React + TypeScript
- Tailwind CSS
- Monospace font (JetBrains Mono)
- Mock travel data
```

## 6.5 Cyberpunk Templates

### Template: Cyberpunk Dashboard

```markdown
# LOVABLE PROMPT: Cyberpunk Dashboard

Create a cyberpunk-style dashboard with HUD elements:

## Visual Design

- Background: Slate 950 (#020617)
- Primary: Cyan (#22D3EE)
- Secondary: Fuchsia (#D946EF)
- Accent: Pink (#EC4899)
- Font: Monospace throughout
- Neon glows: shadow-[0_0_20px_rgba(34,211,238,0.3)]

## Layout Structure

1. **HUD Header**:
   - "SYSTEM://DASHBOARD" in glitch text
   - Status indicators (pulsing dots)
   - Time display (digital clock style)
   - Alert counter with neon badge

2. **Terminal Panel** (left):
   - Black background with scan lines
   - Green/cyan text
   - Typing animation for commands
   - Blinking cursor
   - Command history scroll

3. **Holographic 3D Display** (center):
   - React Three Fiber component
   - Rotating wireframe cube (cyan)
   - Inner cube (purple)
   - Glowing icosahedron core
   - Orbiting particles (200+)
   - 3 rotating data rings

4. **Data Streams** (right):
   - Matrix-style falling characters
   - Hexagon grid background
   - Real-time "data" visualization
   - Neon progress bars

5. **Stats Grid** (bottom):
   - Hexagonal stat cards
   - Glowing borders
   - Animated counters
   - Mini sparkline charts

6. **Alert Panel**:
   - Scrolling alert feed
   - Severity colors (cyan, yellow, red)
   - Glitch effect on new alerts
   - "ACKNOWLEDGE" buttons

## Special Effects

- Scan lines overlay (CSS animation)
- Glitch text effect (random character swap)
- CRT screen curvature (subtle)
- Neon flicker on elements
- Data stream animation

## Animations

- Terminal typing: character by character
- Hologram rotation: continuous
- Particles: orbital motion
- Glitch: random intervals
- Counters: rapid increment

## Technical Requirements

- React + TypeScript
- Tailwind CSS
- Framer Motion
- React Three Fiber + Drei
- Custom glitch text component
```

### Template: Cyberpunk Chat Interface

```markdown
# LOVABLE PROMPT: Cyberpunk Chat Interface

Create a cyberpunk chat interface with neon aesthetics:

## Visual Design

- Background: Dark slate with grid pattern
- Messages: Neon-bordered bubbles
- User: Cyan theme
- Other: Fuchsia theme
- Scan lines overlay

## Layout Structure

1. **Header**:
   - "NEURAL_LINK://CHAT" title
   - Connection status (pulsing indicator)
   - Encryption badge
   - Settings gear (neon outline)

2. **Contact Sidebar**:
   - Dark panel with neon border
   - Contact avatars (hexagonal frames)
   - Online status (glowing dots)
   - Unread count badges
   - Search with neon focus state

3. **Chat Area**:
   - Message bubbles with neon glow
   - Sent messages: Right-aligned, cyan border
   - Received messages: Left-aligned, fuchsia border
   - Timestamps in monospace
   - "Typing..." indicator with dots animation

4. **Message Types**:
   - Text: Neon-bordered bubble
   - Image: Glowing frame
   - File: Hexagonal attachment card
   - Voice: Waveform visualization

5. **Input Area**:
   - Dark input with neon border on focus
   - Attachment button (+ icon)
   - Emoji button
   - Send button (glowing arrow)
   - Voice record button

6. **Special Effects**:
   - New message: Glitch-in animation
   - Typing indicator: Pulsing dots
   - Send: Message flies with trail
   - Notification: Screen flash

## Animations

- Message appear: Glitch effect then stabilize
- Typing dots: Sequential pulse
- Send button: Glow intensify on hover
- Scroll: Smooth with momentum

## Technical Requirements

- React + TypeScript
- Tailwind CSS
- Framer Motion
- Mock chat data
- Glitch text component
```

## 6.6 Soft Pastel Templates

### Template: Pastel File Manager

```markdown
# LOVABLE PROMPT: Soft Pastel File Manager

Create a soft pastel file manager interface:

## Visual Design

- Background: Gradient from rose-200 via lavender-200 to sky-200
- Cards: White with soft shadows
- Accents: Soft purple, pink, blue
- Border radius: 2xl (16px) on everything
- Typography: Rounded, friendly font

## Layout Structure

1. **Header**:
   - "My Files" in soft purple
   - Search bar (white, rounded-full)
   - View toggle (grid/list)
   - Sort dropdown
   - Upload button (gradient pill)

2. **Sidebar Navigation**:
   - White panel with soft shadow
   - Navigation items:
     - All Files (folder icon)
     - Recent (clock icon)
     - Favorites (heart icon)
     - Shared (users icon)
     - Trash (trash icon)
   - Storage usage bar (gradient fill)

3. **Breadcrumb Path**:
   - Pill-shaped segments
   - Soft colors per level
   - Chevron separators

4. **File Grid**:
   - Rounded cards with soft shadows
   - Folder icons: Pastel colored
   - File thumbnails: Rounded corners
   - File name below
   - Size and date (muted text)
   - Checkbox for selection

5. **File Types**:
   - Folders: Lavender
   - Images: Rose
   - Documents: Sky blue
   - Videos: Mint
   - Audio: Peach

6. **Context Menu**:
   - Rounded white panel
   - Soft shadow
   - Icon + label items
   - Dividers between groups

7. **Upload Modal**:
   - Centered white card
   - Drag-drop zone (dashed border)
   - Progress bars (gradient)
   - File list with remove buttons

## Animations

- Card hover: Gentle lift with shadow increase
- Folder open: Soft scale animation
- Upload progress: Smooth fill
- Selection: Soft pulse

## Technical Requirements

- React + TypeScript
- Tailwind CSS
- Framer Motion
- Mock file system data
```

### Template: Pastel Calendar App

```markdown
# LOVABLE PROMPT: Soft Pastel Calendar Application

Create a soft pastel calendar application:

## Visual Design

- Background: Soft gradient (rose to lavender to sky)
- Cards: White with gentle shadows
- Events: Pastel color-coded
- Rounded everything
- Friendly, approachable feel

## Layout Structure

1. **Header**:
   - Month/Year display (large, soft purple)
   - Navigation arrows (rounded buttons)
   - Today button (gradient pill)
   - View switcher (Month/Week/Day)

2. **Mini Calendar** (sidebar):
   - Compact month view
   - Current day highlighted (gradient circle)
   - Days with events: colored dots

3. **Main Calendar Grid**:
   - Day headers (pastel background)
   - Date cells (white, rounded)
   - Today: Gradient border
   - Events as colored pills
   - "+" button on hover

4. **Event Cards**:
   - Rounded pills
   - Color-coded by category:
     - Work: Soft blue
     - Personal: Soft pink
     - Health: Soft green
     - Social: Soft purple
   - Time and title

5. **Event Detail Modal**:
   - White rounded card
   - Event title (large)
   - Date/time with icons
   - Location
   - Description
   - Attendees (avatar stack)
   - Edit/Delete buttons

6. **Add Event Form**:
   - Rounded input fields
   - Color picker (pastel palette)
   - Date/time pickers
   - Repeat options
   - "Save" button (gradient)

## Animations

- Day hover: Soft highlight
- Event click: Gentle scale
- Modal: Fade and slide up
- Month change: Soft crossfade

## Technical Requirements

- React + TypeScript
- Tailwind CSS
- Framer Motion
- Date handling (date-fns)
- Mock event data
```

---

# 7. LOVABLE PROJECT AUDIT PROMPT

Use this prompt to audit any existing Lovable project and get a comprehensive analysis of its frontend and backend components.

```markdown
# LOVABLE PROMPT: Project Audit & Analysis

Analyze this project and provide a comprehensive audit report with the following structure:

## 1. PROJECT OVERVIEW

Provide a summary including:

- Project name and purpose
- Primary functionality
- Target users/audience
- Overall architecture pattern

## 2. FRONTEND ANALYSIS

### 2.1 Technology Stack

List all frontend technologies detected:

- Framework (React, Vue, etc.)
- Language (TypeScript, JavaScript)
- Styling solution (Tailwind, CSS Modules, Styled Components)
- State management
- Animation libraries
- UI component libraries
- Build tools

### 2.2 Component Architecture

Analyze the component structure:

- Total number of components
- Component hierarchy (tree structure)
- Shared/reusable components
- Page components
- Layout components
- Atomic design level (atoms, molecules, organisms)

### 2.3 Styling Analysis

Evaluate the styling approach:

- Design system/tokens used
- Color palette (list all colors with hex codes)
- Typography (fonts, sizes, weights)
- Spacing system
- Responsive breakpoints
- Animation patterns
- CSS architecture (BEM, utility-first, etc.)

### 2.4 Design Engine Classification

Identify which design engine(s) the project uses:

- [ ] Glassmorphism (blur effects, translucent cards)
- [ ] Neumorphism (soft shadows, extruded elements)
- [ ] Brutalist (thick borders, monospace, high contrast)
- [ ] Cyberpunk (neon colors, glows, dark backgrounds)
- [ ] Soft Pastel (gradients, rounded corners, gentle shadows)
- [ ] Other (describe)

### 2.5 Accessibility Audit

Check for accessibility features:

- ARIA labels present
- Keyboard navigation support
- Focus indicators
- Color contrast ratios
- Screen reader compatibility
- Reduced motion support

### 2.6 Performance Analysis

Evaluate performance characteristics:

- Bundle size estimate
- Code splitting implemented
- Lazy loading used
- Image optimization
- Font loading strategy

## 3. BACKEND ANALYSIS (if applicable)

### 3.1 Backend Technology

Identify backend components:

- Server framework (Express, FastAPI, etc.)
- Database (PostgreSQL, MongoDB, Supabase, etc.)
- Authentication method
- API architecture (REST, GraphQL)
- Serverless functions

### 3.2 API Endpoints

List all API endpoints:
| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | /api/... | ... |
| POST | /api/... | ... |

### 3.3 Data Models

Document data structures:

- User model
- Content models
- Relationship mappings

### 3.4 Authentication & Authorization

Analyze auth implementation:

- Auth provider (Supabase, Auth0, custom)
- Session management
- Role-based access control
- Protected routes

## 4. FILE STRUCTURE

Provide the complete file tree:
```

project/
├── src/
│ ├── components/
│ ├── pages/
│ ├── hooks/
│ ├── utils/
│ └── ...
├── public/
└── ...

```

## 5. DEPENDENCIES
List all dependencies with versions:

### Production Dependencies
| Package | Version | Purpose |
|---------|---------|---------|
| react | ^18.x | UI framework |
| ... | ... | ... |

### Development Dependencies
| Package | Version | Purpose |
|---------|---------|---------|
| typescript | ^5.x | Type checking |
| ... | ... | ... |

## 6. QUALITY METRICS

### 6.1 Code Quality
- TypeScript coverage (%)
- ESLint errors/warnings
- Test coverage (if tests exist)
- Documentation quality

### 6.2 Best Practices Compliance
- [ ] Consistent naming conventions
- [ ] Proper error handling
- [ ] Loading states implemented
- [ ] Empty states handled
- [ ] Form validation
- [ ] SEO meta tags

## 7. RECOMMENDATIONS

### 7.1 Improvements Needed
List specific improvements:
1. ...
2. ...
3. ...

### 7.2 Missing Features
Identify gaps:
1. ...
2. ...
3. ...

### 7.3 Optimization Opportunities
Suggest optimizations:
1. ...
2. ...
3. ...

## 8. INTEGRATION READINESS

### 8.1 Export Compatibility
- Can be exported as standalone React app
- Compatible with Next.js migration
- Vite configuration present

### 8.2 Design System Alignment
How well does this align with Alawein Design System:
- Color compatibility: X/10
- Typography compatibility: X/10
- Component pattern compatibility: X/10

### 8.3 Unification Potential
Rate the ease of integrating into unified template system:
- Effort required: Low/Medium/High
- Estimated time: X hours
- Key blockers: ...

## 9. COMPLETE CODE INVENTORY

### 9.1 All Components
List every component with:
- File path
- Props interface
- Dependencies
- Lines of code

### 9.2 All Hooks
List custom hooks:
- Hook name
- Purpose
- Return type

### 9.3 All Utilities
List utility functions:
- Function name
- Purpose
- Parameters

## 10. VISUAL DOCUMENTATION

### 10.1 Screenshots
Describe key screens:
1. Home/Landing
2. Dashboard (if applicable)
3. Forms (if applicable)
4. Modals (if applicable)

### 10.2 Component Gallery
List all UI components with their variants:
- Buttons: primary, secondary, ghost, etc.
- Cards: default, hover, active
- Inputs: text, select, checkbox, etc.
- Navigation: header, sidebar, tabs
- Feedback: alerts, toasts, modals

---

OUTPUT FORMAT:
Provide this audit as a structured markdown document that can be saved and referenced. Include code snippets where relevant to illustrate patterns found.
```

---

# 8. FULLSTACK TEMPLATE UNIFICATION STRATEGY

## 8.1 Unification Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    UNIFIED TEMPLATE SYSTEM                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   FRONTEND   │  │   BACKEND    │  │   SHARED     │          │
│  │   LAYER      │  │   LAYER      │  │   LAYER      │          │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘          │
│         │                 │                 │                    │
│  ┌──────▼───────┐  ┌──────▼───────┐  ┌──────▼───────┐          │
│  │ React 18     │  │ Supabase     │  │ TypeScript   │          │
│  │ TypeScript   │  │ PostgreSQL   │  │ Types        │          │
│  │ Tailwind     │  │ Edge Funcs   │  │ Validators   │          │
│  │ Framer       │  │ Auth         │  │ Utils        │          │
│  │ Three.js     │  │ Storage      │  │ Constants    │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                    5 DESIGN ENGINES                      │    │
│  ├─────────┬─────────┬─────────┬─────────┬─────────────────┤    │
│  │ Glass   │ Neu     │ Brutal  │ Cyber   │ Pastel          │    │
│  └─────────┴─────────┴─────────┴─────────┴─────────────────┘    │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                    43+ TEMPLATES                         │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## 8.2 Backend Integration Patterns

### Pattern 1: Supabase Integration

```typescript
// lib/supabase.ts
import { createClient } from '@supabase/supabase-js';
import type { Database } from './database.types';

const supabaseUrl = import.meta.env.VITE_SUPABASE_URL;
const supabaseAnonKey = import.meta.env.VITE_SUPABASE_ANON_KEY;

export const supabase = createClient<Database>(supabaseUrl, supabaseAnonKey);

// Auth helpers
export const signIn = async (email: string, password: string) => {
  const { data, error } = await supabase.auth.signInWithPassword({
    email,
    password,
  });
  return { data, error };
};

export const signUp = async (email: string, password: string) => {
  const { data, error } = await supabase.auth.signUp({
    email,
    password,
  });
  return { data, error };
};

export const signOut = async () => {
  const { error } = await supabase.auth.signOut();
  return { error };
};

// Data helpers
export const fetchData = async <T>(
  table: string,
  query?: { column: string; value: any }
): Promise<T[]> => {
  let queryBuilder = supabase.from(table).select('*');

  if (query) {
    queryBuilder = queryBuilder.eq(query.column, query.value);
  }

  const { data, error } = await queryBuilder;

  if (error) throw error;
  return data as T[];
};
```

### Pattern 2: API Route Structure

```typescript
// api/routes.ts
export const API_ROUTES = {
  // Auth
  AUTH: {
    LOGIN: '/api/auth/login',
    REGISTER: '/api/auth/register',
    LOGOUT: '/api/auth/logout',
    REFRESH: '/api/auth/refresh',
    ME: '/api/auth/me',
  },

  // Users
  USERS: {
    LIST: '/api/users',
    GET: (id: string) => `/api/users/${id}`,
    UPDATE: (id: string) => `/api/users/${id}`,
    DELETE: (id: string) => `/api/users/${id}`,
  },

  // Content
  CONTENT: {
    LIST: '/api/content',
    GET: (id: string) => `/api/content/${id}`,
    CREATE: '/api/content',
    UPDATE: (id: string) => `/api/content/${id}`,
    DELETE: (id: string) => `/api/content/${id}`,
  },

  // Files
  FILES: {
    UPLOAD: '/api/files/upload',
    GET: (id: string) => `/api/files/${id}`,
    DELETE: (id: string) => `/api/files/${id}`,
  },
};
```

### Pattern 3: Data Fetching Hooks

```typescript
// hooks/useQuery.ts
import { useState, useEffect } from 'react';
import { supabase } from '@/lib/supabase';

interface UseQueryOptions<T> {
  table: string;
  select?: string;
  filter?: { column: string; value: any };
  orderBy?: { column: string; ascending?: boolean };
  limit?: number;
  enabled?: boolean;
}

interface UseQueryResult<T> {
  data: T[] | null;
  isLoading: boolean;
  error: Error | null;
  refetch: () => Promise<void>;
}

export function useQuery<T>({
  table,
  select = '*',
  filter,
  orderBy,
  limit,
  enabled = true,
}: UseQueryOptions<T>): UseQueryResult<T> {
  const [data, setData] = useState<T[] | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  const fetchData = async () => {
    if (!enabled) return;

    setIsLoading(true);
    setError(null);

    try {
      let query = supabase.from(table).select(select);

      if (filter) {
        query = query.eq(filter.column, filter.value);
      }

      if (orderBy) {
        query = query.order(orderBy.column, { ascending: orderBy.ascending ?? true });
      }

      if (limit) {
        query = query.limit(limit);
      }

      const { data: result, error: queryError } = await query;

      if (queryError) throw queryError;

      setData(result as T[]);
    } catch (err) {
      setError(err as Error);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
  }, [table, select, filter?.column, filter?.value, orderBy?.column, limit, enabled]);

  return { data, isLoading, error, refetch: fetchData };
}
```

### Pattern 4: Mutation Hooks

```typescript
// hooks/useMutation.ts
import { useState } from 'react';
import { supabase } from '@/lib/supabase';

interface UseMutationOptions<T, V> {
  table: string;
  onSuccess?: (data: T) => void;
  onError?: (error: Error) => void;
}

interface UseMutationResult<T, V> {
  mutate: (variables: V) => Promise<T | null>;
  isLoading: boolean;
  error: Error | null;
  data: T | null;
}

export function useInsert<T, V extends Partial<T>>({
  table,
  onSuccess,
  onError,
}: UseMutationOptions<T, V>): UseMutationResult<T, V> {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const [data, setData] = useState<T | null>(null);

  const mutate = async (variables: V): Promise<T | null> => {
    setIsLoading(true);
    setError(null);

    try {
      const { data: result, error: mutationError } = await supabase
        .from(table)
        .insert(variables)
        .select()
        .single();

      if (mutationError) throw mutationError;

      setData(result as T);
      onSuccess?.(result as T);
      return result as T;
    } catch (err) {
      const error = err as Error;
      setError(error);
      onError?.(error);
      return null;
    } finally {
      setIsLoading(false);
    }
  };

  return { mutate, isLoading, error, data };
}

export function useUpdate<T, V extends Partial<T>>({
  table,
  onSuccess,
  onError,
}: UseMutationOptions<T, V>) {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const [data, setData] = useState<T | null>(null);

  const mutate = async (id: string, variables: V): Promise<T | null> => {
    setIsLoading(true);
    setError(null);

    try {
      const { data: result, error: mutationError } = await supabase
        .from(table)
        .update(variables)
        .eq('id', id)
        .select()
        .single();

      if (mutationError) throw mutationError;

      setData(result as T);
      onSuccess?.(result as T);
      return result as T;
    } catch (err) {
      const error = err as Error;
      setError(error);
      onError?.(error);
      return null;
    } finally {
      setIsLoading(false);
    }
  };

  return { mutate, isLoading, error, data };
}

export function useDelete<T>({ table, onSuccess, onError }: UseMutationOptions<T, { id: string }>) {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);

  const mutate = async (id: string): Promise<boolean> => {
    setIsLoading(true);
    setError(null);

    try {
      const { error: mutationError } = await supabase.from(table).delete().eq('id', id);

      if (mutationError) throw mutationError;

      onSuccess?.({} as T);
      return true;
    } catch (err) {
      const error = err as Error;
      setError(error);
      onError?.(error);
      return false;
    } finally {
      setIsLoading(false);
    }
  };

  return { mutate, isLoading, error };
}
```

## 8.3 Shared Type Definitions

```typescript
// types/database.types.ts
export interface Database {
  public: {
    Tables: {
      users: {
        Row: {
          id: string;
          email: string;
          full_name: string | null;
          avatar_url: string | null;
          created_at: string;
          updated_at: string;
        };
        Insert: {
          id?: string;
          email: string;
          full_name?: string | null;
          avatar_url?: string | null;
          created_at?: string;
          updated_at?: string;
        };
        Update: {
          id?: string;
          email?: string;
          full_name?: string | null;
          avatar_url?: string | null;
          updated_at?: string;
        };
      };
      projects: {
        Row: {
          id: string;
          name: string;
          description: string | null;
          owner_id: string;
          status: 'draft' | 'active' | 'archived';
          created_at: string;
          updated_at: string;
        };
        Insert: {
          id?: string;
          name: string;
          description?: string | null;
          owner_id: string;
          status?: 'draft' | 'active' | 'archived';
          created_at?: string;
          updated_at?: string;
        };
        Update: {
          name?: string;
          description?: string | null;
          status?: 'draft' | 'active' | 'archived';
          updated_at?: string;
        };
      };
      tasks: {
        Row: {
          id: string;
          title: string;
          description: string | null;
          project_id: string;
          assignee_id: string | null;
          status: 'todo' | 'in_progress' | 'review' | 'done';
          priority: 'low' | 'medium' | 'high';
          due_date: string | null;
          created_at: string;
          updated_at: string;
        };
        Insert: {
          id?: string;
          title: string;
          description?: string | null;
          project_id: string;
          assignee_id?: string | null;
          status?: 'todo' | 'in_progress' | 'review' | 'done';
          priority?: 'low' | 'medium' | 'high';
          due_date?: string | null;
          created_at?: string;
          updated_at?: string;
        };
        Update: {
          title?: string;
          description?: string | null;
          assignee_id?: string | null;
          status?: 'todo' | 'in_progress' | 'review' | 'done';
          priority?: 'low' | 'medium' | 'high';
          due_date?: string | null;
          updated_at?: string;
        };
      };
    };
  };
}

// Convenience types
export type User = Database['public']['Tables']['users']['Row'];
export type Project = Database['public']['Tables']['projects']['Row'];
export type Task = Database['public']['Tables']['tasks']['Row'];

export type UserInsert = Database['public']['Tables']['users']['Insert'];
export type ProjectInsert = Database['public']['Tables']['projects']['Insert'];
export type TaskInsert = Database['public']['Tables']['tasks']['Insert'];

export type UserUpdate = Database['public']['Tables']['users']['Update'];
export type ProjectUpdate = Database['public']['Tables']['projects']['Update'];
export type TaskUpdate = Database['public']['Tables']['tasks']['Update'];
```

## 8.4 Fullstack Template Prompt

````markdown
# LOVABLE PROMPT: Fullstack Template with Supabase

Create a fullstack application template with the following specifications:

## Overview

Build a complete fullstack template that includes authentication, database operations, and a polished UI using the Glassmorphism design engine.

## Technical Stack

- Frontend: React 18 + TypeScript + Vite
- Styling: Tailwind CSS + Framer Motion
- Backend: Supabase (PostgreSQL + Auth + Storage)
- State: React Query (TanStack Query)
- Forms: React Hook Form + Zod validation

## Database Schema

### Users Table

```sql
CREATE TABLE users (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  email TEXT UNIQUE NOT NULL,
  full_name TEXT,
  avatar_url TEXT,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);
```
````

### Projects Table

```sql
CREATE TABLE projects (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  name TEXT NOT NULL,
  description TEXT,
  owner_id UUID REFERENCES users(id) ON DELETE CASCADE,
  status TEXT DEFAULT 'draft' CHECK (status IN ('draft', 'active', 'archived')),
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);
```

### Tasks Table

```sql
CREATE TABLE tasks (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  title TEXT NOT NULL,
  description TEXT,
  project_id UUID REFERENCES projects(id) ON DELETE CASCADE,
  assignee_id UUID REFERENCES users(id) ON DELETE SET NULL,
  status TEXT DEFAULT 'todo' CHECK (status IN ('todo', 'in_progress', 'review', 'done')),
  priority TEXT DEFAULT 'medium' CHECK (priority IN ('low', 'medium', 'high')),
  due_date TIMESTAMPTZ,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);
```

## Features to Implement

### 1. Authentication

- Sign up with email/password
- Sign in with email/password
- Sign out
- Password reset flow
- Protected routes
- Auth state persistence

### 2. User Profile

- View profile
- Edit profile (name, avatar)
- Upload avatar to Supabase Storage

### 3. Projects CRUD

- List all projects
- Create new project
- View project details
- Edit project
- Delete project
- Filter by status

### 4. Tasks CRUD

- List tasks by project
- Create new task
- Edit task (inline or modal)
- Delete task
- Drag-and-drop status change
- Filter by status/priority
- Assign to user

### 5. Real-time Updates

- Subscribe to task changes
- Live collaboration indicators

## UI Components (Glassmorphism Style)

### Layout

- Glass sidebar navigation
- Glass header with user menu
- Main content area with glass cards

### Components

- Glass Card
- Glass Button (primary, secondary, ghost)
- Glass Input
- Glass Select
- Glass Modal
- Glass Toast notifications
- Glass Avatar
- Glass Badge
- Glass Progress bar

### Pages

1. **Auth Pages**
   - /login
   - /register
   - /forgot-password
   - /reset-password

2. **Dashboard**
   - / (redirect to /dashboard)
   - /dashboard (project overview)

3. **Projects**
   - /projects (list)
   - /projects/:id (detail with tasks)
   - /projects/new (create)

4. **Profile**
   - /profile (view/edit)
   - /settings (app settings)

## File Structure

```
src/
├── components/
│   ├── ui/
│   │   ├── Button.tsx
│   │   ├── Card.tsx
│   │   ├── Input.tsx
│   │   ├── Modal.tsx
│   │   └── ...
│   ├── layout/
│   │   ├── Sidebar.tsx
│   │   ├── Header.tsx
│   │   └── Layout.tsx
│   ├── auth/
│   │   ├── LoginForm.tsx
│   │   ├── RegisterForm.tsx
│   │   └── ProtectedRoute.tsx
│   ├── projects/
│   │   ├── ProjectCard.tsx
│   │   ├── ProjectForm.tsx
│   │   └── ProjectList.tsx
│   └── tasks/
│       ├── TaskCard.tsx
│       ├── TaskForm.tsx
│       ├── TaskList.tsx
│       └── KanbanBoard.tsx
├── hooks/
│   ├── useAuth.ts
│   ├── useProjects.ts
│   ├── useTasks.ts
│   └── useRealtime.ts
├── lib/
│   ├── supabase.ts
│   └── utils.ts
├── pages/
│   ├── auth/
│   ├── dashboard/
│   ├── projects/
│   └── profile/
├── types/
│   └── database.types.ts
└── App.tsx
```

## Environment Variables

```
VITE_SUPABASE_URL=your_supabase_url
VITE_SUPABASE_ANON_KEY=your_supabase_anon_key
```

## Styling Requirements

- Dark gradient background (#0F0F23 to #2A1B4D)
- Glass cards with backdrop-blur-xl
- Purple accent (#A855F7) for primary actions
- Cyan accent (#4CC9F0) for info/links
- Smooth animations with Framer Motion
- Responsive design (mobile-first)

## DO NOT INCLUDE

- Actual Supabase credentials (use placeholders)
- Payment processing
- Email sending (just UI)
- Third-party analytics
- Cookie consent

## Deliverables

1. Complete React application
2. All components with TypeScript types
3. Supabase client configuration
4. Database type definitions
5. Custom hooks for data fetching
6. Protected route implementation
7. Responsive layouts
8. Loading and error states

```

## 8.5 Unification Folder Structure

```

unified-templates/
├── README.md
├── package.json
├── tsconfig.json
├── vite.config.ts
├── tailwind.config.js
├── .env.example
│
├── src/
│ ├── main.tsx
│ ├── App.tsx
│ ├── index.css
│ │
│ ├── components/
│ │ ├── ui/ # Shared UI components
│ │ │ ├── Button/
│ │ │ │ ├── Button.tsx
│ │ │ │ ├── Button.test.tsx
│ │ │ │ └── index.ts
│ │ │ ├── Card/
│ │ │ ├── Input/
│ │ │ ├── Modal/
│ │ │ ├── Toast/
│ │ │ └── index.ts
│ │ │
│ │ ├── layout/ # Layout components
│ │ │ ├── Header.tsx
│ │ │ ├── Sidebar.tsx
│ │ │ ├── Footer.tsx
│ │ │ └── Layout.tsx
│ │ │
│ │ ├── auth/ # Auth components
│ │ │ ├── LoginForm.tsx
│ │ │ ├── RegisterForm.tsx
│ │ │ ├── ProtectedRoute.tsx
│ │ │ └── AuthProvider.tsx
│ │ │
│ │ └── features/ # Feature-specific components
│ │ ├── dashboard/
│ │ ├── projects/
│ │ └── tasks/
│ │
│ ├── hooks/ # Custom hooks
│ │ ├── useAuth.ts
│ │ ├── useQuery.ts
│ │ ├── useMutation.ts
│ │ ├── useRealtime.ts
│ │ └── index.ts
│ │
│ ├── lib/ # Library configurations
│ │ ├── supabase.ts
│ │ ├── queryClient.ts
│ │ └── utils.ts
│ │
│ ├── pages/ # Page components
│ │ ├── auth/
│ │ │ ├── Login.tsx
│ │ │ ├── Register.tsx
│ │ │ └── ForgotPassword.tsx
│ │ ├── dashboard/
│ │ │ └── Dashboard.tsx
│ │ ├── projects/
│ │ │ ├── ProjectList.tsx
│ │ │ └── ProjectDetail.tsx
│ │ └── profile/
│ │ └── Profile.tsx
│ │
│ ├── styles/ # Design system styles
│ │ ├── tokens.css # CSS custom properties
│ │ ├── engines/ # Engine-specific styles
│ │ │ ├── glassmorphism.css
│ │ │ ├── neumorphism.css
│ │ │ ├── brutalist.css
│ │ │ ├── cyberpunk.css
│ │ │ └── pastel.css
│ │ └── utilities.css
│ │
│ ├── types/ # TypeScript types
│ │ ├── database.types.ts
│ │ ├── api.types.ts
│ │ └── index.ts
│ │
│ └── constants/ # Constants and config
│ ├── routes.ts
│ ├── api.ts
│ └── config.ts
│
├── public/
│ ├── favicon.ico
│ └── assets/
│
├── supabase/ # Supabase configuration
│ ├── migrations/
│ │ └── 001_initial_schema.sql
│ ├── seed.sql
│ └── config.toml
│
└── docs/
├── SETUP.md
├── COMPONENTS.md
├── API.md
└── DEPLOYMENT.md

````

---

# 9. IMPLEMENTATION ROADMAP

## 9.1 Phase 1: Foundation (Week 1)

| Day | Task | Credits | Output |
|-----|------|---------|--------|
| 1 | Generate Master Template Gallery | 15-20 | Base gallery with 5 engines |
| 2 | Download and organize files | 0 | Local project setup |
| 3 | Apply design tokens | 0 | Unified styling |
| 4 | Test all components | 0 | QA complete |
| 5 | Document components | 0 | Storybook/docs |

## 9.2 Phase 2: Engine Templates (Week 2)

| Day | Task | Credits | Output |
|-----|------|---------|--------|
| 1 | Cyberpunk Dashboard | 10-12 | HUD dashboard |
| 2 | Glass Weather Dashboard | 8-10 | Weather app |
| 3 | Brutalist Task Manager | 6-8 | Kanban board |
| 4 | Neu Settings Panel | 6-8 | Settings UI |
| 5 | Pastel File Manager | 6-8 | File browser |

## 9.3 Phase 3: Additional Templates (Week 3)

| Day | Task | Credits | Output |
|-----|------|---------|--------|
| 1 | Glass Music Player | 6-8 | Audio player |
| 2 | Cyberpunk Chat | 8-10 | Chat interface |
| 3 | Neu Recipe App | 6-8 | Recipe UI |
| 4 | Brutalist Travel | 6-8 | Travel planner |
| 5 | Pastel Calendar | 6-8 | Calendar app |

## 9.4 Phase 4: Fullstack Integration (Week 4)

| Day | Task | Credits | Output |
|-----|------|---------|--------|
| 1 | Fullstack Template | 15-20 | Complete app |
| 2 | Supabase setup | 0 | Database ready |
| 3 | Auth implementation | 0 | Auth working |
| 4 | CRUD operations | 0 | Data flow complete |
| 5 | Testing & polish | 0 | Production ready |

## 9.5 Credit Summary

| Phase | Templates | Credits Used | Running Total |
|-------|-----------|--------------|---------------|
| Phase 1 | 1 | 15-20 | 15-20 |
| Phase 2 | 5 | 36-46 | 51-66 |
| Phase 3 | 5 | 32-42 | 83-108 |
| Phase 4 | 1 | 15-20 | 98-128 |
| **Buffer** | - | ~72-102 | **200** |

---

# 10. TECHNICAL SPECIFICATIONS

## 10.1 Core Dependencies

```json
{
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "react-router-dom": "^6.20.0",
    "@supabase/supabase-js": "^2.38.0",
    "@tanstack/react-query": "^5.8.0",
    "framer-motion": "^10.16.0",
    "@react-three/fiber": "^8.15.0",
    "@react-three/drei": "^9.88.0",
    "three": "^0.158.0",
    "lucide-react": "^0.294.0",
    "react-hook-form": "^7.48.0",
    "@hookform/resolvers": "^3.3.0",
    "zod": "^3.22.0",
    "date-fns": "^2.30.0",
    "clsx": "^2.0.0",
    "tailwind-merge": "^2.0.0"
  },
  "devDependencies": {
    "typescript": "^5.3.0",
    "vite": "^5.0.0",
    "@vitejs/plugin-react": "^4.2.0",
    "tailwindcss": "^3.3.0",
    "autoprefixer": "^10.4.0",
    "postcss": "^8.4.0",
    "@types/react": "^18.2.0",
    "@types/react-dom": "^18.2.0",
    "@types/three": "^0.158.0",
    "eslint": "^8.54.0",
    "prettier": "^3.1.0"
  }
}
````

## 10.2 Tailwind Configuration

```javascript
// tailwind.config.js
/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        // Alawein Design System
        quantum: {
          purple: '#A855F7',
          pink: '#EC4899',
          cyan: '#4CC9F0',
        },
        void: {
          start: '#0F0F23',
          mid: '#1A1B3D',
          end: '#2A1B4D',
        },
        surface: {
          card: 'rgba(255, 255, 255, 0.02)',
          border: 'rgba(255, 255, 255, 0.05)',
          glass: 'rgba(255, 255, 255, 0.03)',
        },
      },
      fontFamily: {
        display: ['SF Mono', 'Monaco', 'Inconsolata', 'Fira Code', 'monospace'],
        body: ['Inter var', '-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'sans-serif'],
        mono: ['JetBrains Mono', 'Fira Code', 'SF Mono', 'monospace'],
      },
      boxShadow: {
        'glow-purple': '0 0 20px rgba(168, 85, 247, 0.3)',
        'glow-cyan': '0 0 20px rgba(76, 201, 240, 0.3)',
        'glow-pink': '0 0 20px rgba(236, 72, 153, 0.3)',
        'neu-raised': '8px 8px 16px #c5ccd6, -8px -8px 16px #ffffff',
        'neu-pressed': 'inset 4px 4px 8px #c5ccd6, inset -4px -4px 8px #ffffff',
        brutal: '4px 4px 0px 0px rgba(0,0,0,1)',
      },
      backdropBlur: {
        glass: '40px',
      },
      animation: {
        float: 'float 6s ease-in-out infinite',
        glow: 'glow 2s ease-in-out infinite alternate',
        glitch: 'glitch 3s infinite',
        orbit: 'orbit 12s linear infinite',
      },
      keyframes: {
        float: {
          '0%, 100%': { transform: 'translateY(0)' },
          '50%': { transform: 'translateY(-10px)' },
        },
        glow: {
          '0%': { boxShadow: '0 0 20px rgba(168, 85, 247, 0.3)' },
          '100%': { boxShadow: '0 0 30px rgba(168, 85, 247, 0.6)' },
        },
        glitch: {
          '0%, 90%, 100%': { transform: 'translate(0)' },
          '92%': { transform: 'translate(-2px, 1px)' },
          '94%': { transform: 'translate(2px, -1px)' },
          '96%': { transform: 'translate(-1px, 2px)' },
          '98%': { transform: 'translate(1px, -2px)' },
        },
        orbit: {
          '0%': { transform: 'rotate(0deg)' },
          '100%': { transform: 'rotate(360deg)' },
        },
      },
    },
  },
  plugins: [],
};
```

## 10.3 Vite Configuration

```typescript
// vite.config.ts
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
      '@components': path.resolve(__dirname, './src/components'),
      '@hooks': path.resolve(__dirname, './src/hooks'),
      '@lib': path.resolve(__dirname, './src/lib'),
      '@pages': path.resolve(__dirname, './src/pages'),
      '@types': path.resolve(__dirname, './src/types'),
      '@styles': path.resolve(__dirname, './src/styles'),
    },
  },
  build: {
    rollupOptions: {
      output: {
        manualChunks: {
          'react-vendor': ['react', 'react-dom', 'react-router-dom'],
          'three-vendor': ['three', '@react-three/fiber', '@react-three/drei'],
          'ui-vendor': ['framer-motion', 'lucide-react'],
        },
      },
    },
  },
});
```

---

# 11. POST-PROCESSING CHECKLIST

## 11.1 Immediate Post-Download Tasks

```markdown
## After Downloading Each Template

### 1. Project Setup

- [ ] Create new folder in `unified-templates/templates/[template-name]/`
- [ ] Initialize git repository
- [ ] Add `.gitignore` for node_modules, build artifacts
- [ ] Create `README.md` with template documentation

### 2. Code Quality

- [ ] Run Prettier to format all files
- [ ] Run ESLint and fix any issues
- [ ] Add TypeScript strict mode if not present
- [ ] Remove any unused dependencies

### 3. Design Token Integration

- [ ] Replace hardcoded colors with CSS custom properties
- [ ] Update to use Alawein Design System tokens
- [ ] Verify typography matches design system
- [ ] Check spacing consistency

### 4. Accessibility Audit

- [ ] Run axe-core accessibility check
- [ ] Verify color contrast ratios (4.5:1 minimum)
- [ ] Add missing ARIA labels
- [ ] Test keyboard navigation
- [ ] Add skip links if needed
- [ ] Verify focus indicators are visible

### 5. Performance Optimization

- [ ] Optimize images (WebP format, proper sizing)
- [ ] Add lazy loading for images
- [ ] Minimize CSS (remove unused styles)
- [ ] Add preload for critical fonts
- [ ] Verify Lighthouse score > 90

### 6. Responsive Testing

- [ ] Test at 320px (mobile)
- [ ] Test at 768px (tablet)
- [ ] Test at 1024px (laptop)
- [ ] Test at 1440px (desktop)
- [ ] Test at 1920px (large desktop)

### 7. Animation Enhancement

- [ ] Add `prefers-reduced-motion` support
- [ ] Verify animations use GPU-accelerated properties
- [ ] Add Framer Motion for complex animations
- [ ] Implement scroll-triggered animations

### 8. Component Extraction

- [ ] Identify reusable components
- [ ] Extract to `shared/components/`
- [ ] Add prop types and documentation
- [ ] Create Storybook stories (optional)
```

## 11.2 Deployment Checklist

```markdown
## Deployment Preparation

### 1. Build Configuration

- [ ] Verify Vite build succeeds
- [ ] Check bundle size (< 200KB initial JS)
- [ ] Verify all assets are included
- [ ] Test production build locally

### 2. SEO & Meta Tags

- [ ] Add page title
- [ ] Add meta description
- [ ] Add Open Graph tags
- [ ] Add Twitter Card tags
- [ ] Add favicon set
- [ ] Add robots.txt
- [ ] Add sitemap.xml (if multi-page)

### 3. Environment Variables

- [ ] Create .env.example with all required vars
- [ ] Document each variable's purpose
- [ ] Set up production environment

### 4. Hosting Configuration

- [ ] Create Vercel/Netlify config
- [ ] Set up custom domain (if needed)
- [ ] Configure redirects
- [ ] Set up headers (security, caching)

### 5. Final Verification

- [ ] Test all links
- [ ] Test all forms
- [ ] Test on real mobile device
- [ ] Test in multiple browsers
- [ ] Verify no console errors
- [ ] Check network requests
```

---

# 12. APPENDICES

## Appendix A: CSS Design Tokens File

```css
/* src/styles/tokens.css */

:root {
  /* === COLORS === */

  /* Primary Palette */
  --color-quantum-purple: #a855f7;
  --color-plasma-pink: #ec4899;
  --color-electron-cyan: #4cc9f0;

  /* Backgrounds */
  --color-void-start: #0f0f23;
  --color-void-mid: #1a1b3d;
  --color-void-end: #2a1b4d;

  /* Surfaces */
  --color-card-dark: rgba(255, 255, 255, 0.02);
  --color-card-border: rgba(255, 255, 255, 0.05);
  --color-glass: rgba(255, 255, 255, 0.03);

  /* Text */
  --color-text-primary: #ffffff;
  --color-text-secondary: #a0a0c0;
  --color-text-muted: #808090;

  /* Status */
  --color-success: #10b981;
  --color-warning: #f59e0b;
  --color-error: #ef4444;
  --color-info: #4cc9f0;

  /* Gradients */
  --gradient-text: linear-gradient(90deg, #a855f7 0%, #ec4899 50%, #4cc9f0 100%);
  --gradient-bg: linear-gradient(135deg, #0f0f23 0%, #1a1b3d 50%, #2a1b4d 100%);
  --gradient-button: linear-gradient(135deg, #a855f7 0%, #ec4899 100%);

  /* === TYPOGRAPHY === */

  --font-display: 'SF Mono', 'Monaco', 'Inconsolata', 'Fira Code', monospace;
  --font-body: 'Inter var', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  --font-mono: 'JetBrains Mono', 'Fira Code', 'SF Mono', monospace;

  /* Type Scale */
  --text-xs: 0.75rem;
  --text-sm: 0.875rem;
  --text-base: 1rem;
  --text-lg: 1.125rem;
  --text-xl: 1.25rem;
  --text-2xl: 1.5rem;
  --text-3xl: 1.875rem;
  --text-4xl: 2.25rem;
  --text-5xl: 3rem;

  /* === SPACING === */

  --space-1: 0.25rem;
  --space-2: 0.5rem;
  --space-3: 0.75rem;
  --space-4: 1rem;
  --space-6: 1.5rem;
  --space-8: 2rem;
  --space-12: 3rem;
  --space-16: 4rem;
  --space-24: 6rem;

  /* === EFFECTS === */

  /* Shadows */
  --shadow-sm: 0 1px 2px 0 rgb(0 0 0 / 0.05);
  --shadow-md: 0 4px 6px -1px rgb(0 0 0 / 0.1);
  --shadow-lg: 0 10px 15px -3px rgb(0 0 0 / 0.1);
  --shadow-glow-purple: 0 0 20px rgba(168, 85, 247, 0.3);
  --shadow-glow-cyan: 0 0 20px rgba(76, 201, 240, 0.3);

  /* Border Radius */
  --radius-sm: 0.125rem;
  --radius-md: 0.375rem;
  --radius-lg: 0.5rem;
  --radius-xl: 0.75rem;
  --radius-2xl: 1rem;
  --radius-full: 9999px;

  /* Blur */
  --blur-sm: blur(4px);
  --blur-md: blur(12px);
  --blur-glass: blur(40px) saturate(180%);

  /* === ANIMATION === */

  --duration-fast: 150ms;
  --duration-normal: 300ms;
  --duration-slow: 500ms;

  --ease-default: cubic-bezier(0.4, 0, 0.2, 1);
  --ease-in: cubic-bezier(0.4, 0, 1, 1);
  --ease-out: cubic-bezier(0, 0, 0.2, 1);
  --ease-spring: cubic-bezier(0.175, 0.885, 0.32, 1.275);
  --ease-expo-out: cubic-bezier(0.19, 1, 0.22, 1);
}

/* === UTILITY CLASSES === */

.gradient-text {
  background: var(--gradient-text);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}

.glass-panel {
  background: var(--color-glass);
  backdrop-filter: var(--blur-glass);
  border: 1px solid var(--color-card-border);
}

.glow-purple {
  box-shadow: var(--shadow-glow-purple);
}

.glow-cyan {
  box-shadow: var(--shadow-glow-cyan);
}

/* Reduced Motion */
@media (prefers-reduced-motion: reduce) {
  *,
  *::before,
  *::after {
    animation-duration: 0.01ms !important;
    animation-iteration-count: 1 !important;
    transition-duration: 0.01ms !important;
  }
}
```

## Appendix B: Quick Reference Card

```markdown
## Lovable Prompt Best Practices

### DO ✅

- Be extremely specific about colors (use hex codes)
- Specify exact fonts by name
- Describe layout with column counts
- List all sections in order
- Specify animation types and durations
- Mention accessibility requirements
- Request specific tech stack

### DON'T ❌

- Use vague terms like "modern" or "clean" alone
- Assume Lovable knows your brand
- Request features Lovable can't generate (backend logic)
- Ask for too many pages in one prompt
- Forget to specify dark/light mode preference
- Skip responsive requirements

## Credit Optimization Tips

1. **Batch Similar Templates**: Generate variations in sequence
2. **Start Simple**: Get base layout first, then iterate
3. **Use Specific Prompts**: Vague prompts waste credits on revisions
4. **Download Immediately**: Don't lose work to session timeouts
5. **Document What Works**: Note successful prompt patterns
6. **Iterate Locally**: Do refinements in your IDE, not Lovable
```

## Appendix C: Emergency Prompt Template

```markdown
# QUICK LOVABLE PROMPT TEMPLATE

Create a [TYPE] with:

## Colors

- Background: [HEX]
- Text: [HEX]
- Accent: [HEX]

## Typography

- Headings: [FONT]
- Body: [FONT]

## Layout

1. [SECTION 1]
2. [SECTION 2]
3. [SECTION 3]

## Features

- [FEATURE 1]
- [FEATURE 2]

## Tech Stack

- React + TypeScript
- Tailwind CSS

## DO NOT INCLUDE

- [ANTI-PATTERN 1]
- [ANTI-PATTERN 2]
```

---

# Document Metadata

| Field                | Value                    |
| -------------------- | ------------------------ |
| Version              | 2.0                      |
| Created              | December 4, 2024         |
| Updated              | December 5, 2024         |
| Author               | Frontend Specialist Mode |
| Owner                | Meshal Alawein           |
| Status               | Complete                 |
| Credits Budget       | 200                      |
| Templates Documented | 43+                      |
| Design Engines       | 5                        |

---

_End of Document_
