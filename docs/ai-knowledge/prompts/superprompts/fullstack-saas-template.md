---
name: 'Lovable Template Generation Superprompt'
version: '1.0'
category: 'project'
tags: ['lovable', 'templates', 'ui', 'ux', 'design-system', 'frontend']
created: '2024-12-04'
author: 'Meshal Alawein'
credits_budget: 200
---

# Lovable Template Generation Strategy

## Executive Summary

This document provides a comprehensive strategy for maximizing 200 Lovable credits to generate a library of deployment-ready UI/UX templates. All templates are designed to align with the Alawein Design System - a sophisticated, physics-inspired aesthetic featuring quantum visualizations, GPU-accelerated animations, and scientific elegance.

---

## Part 1: Master Design Token Reference

### 1.1 Color System

```yaml
# ALAWEIN DESIGN SYSTEM - COLOR TOKENS
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
    card-gradient: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)'
    header-gradient: 'linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)'
```

### 1.2 Typography System

```yaml
# TYPOGRAPHY TOKENS
typography:
  # Font Families
  fonts:
    display: "'SF Mono', 'Monaco', 'Inconsolata', 'Fira Code', monospace"
    body: "'Inter var', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"
    mono: "'JetBrains Mono', 'Fira Code', 'SF Mono', monospace"
    serif: "'Instrument Serif', 'Playfair Display', Georgia, serif"

  # Type Scale (1.25 ratio - Major Third)
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

  # Font Weights
  weights:
    light: 300
    normal: 400
    medium: 500
    semibold: 600
    bold: 700

  # Letter Spacing
  tracking:
    tight: '-0.02em'
    normal: '0'
    wide: '0.025em'
    wider: '0.05em'

  # Line Heights
  leading:
    tight: 1.25
    normal: 1.5
    relaxed: 1.75
```

### 1.3 Spacing System

```yaml
# SPACING TOKENS (Golden Ratio: 1.618)
spacing:
  0: '0'
  1: '0.25rem' # 4px
  2: '0.5rem' # 8px
  3: '0.75rem' # 12px
  4: '1rem' # 16px
  5: '1.25rem' # 20px
  6: '1.5rem' # 24px
  8: '2rem' # 32px
  10: '2.5rem' # 40px
  12: '3rem' # 48px
  16: '4rem' # 64px
  20: '5rem' # 80px
  24: '6rem' # 96px
  32: '8rem' # 128px
```

### 1.4 Animation System

```yaml
# ANIMATION TOKENS
animation:
  # Durations
  duration:
    instant: '0ms'
    fast: '150ms'
    normal: '300ms'
    slow: '500ms'
    slower: '700ms'
    slowest: '1000ms'

  # Easing Functions
  easing:
    default: 'cubic-bezier(0.4, 0, 0.2, 1)'
    in: 'cubic-bezier(0.4, 0, 1, 1)'
    out: 'cubic-bezier(0, 0, 0.2, 1)'
    in-out: 'cubic-bezier(0.4, 0, 0.2, 1)'
    spring: 'cubic-bezier(0.175, 0.885, 0.32, 1.275)'
    expo-out: 'cubic-bezier(0.19, 1, 0.22, 1)'

  # Orbital Animation Speeds
  orbital:
    fast: '10s'
    normal: '12s'
    slow: '15s'
```

### 1.5 Effects System

```yaml
# EFFECTS TOKENS
effects:
  # Shadows
  shadows:
    sm: '0 1px 2px 0 rgb(0 0 0 / 0.05)'
    default: '0 1px 3px 0 rgb(0 0 0 / 0.1)'
    md: '0 4px 6px -1px rgb(0 0 0 / 0.1)'
    lg: '0 10px 15px -3px rgb(0 0 0 / 0.1)'
    xl: '0 20px 25px -5px rgb(0 0 0 / 0.1)'
    glow-purple: '0 0 20px rgba(168, 85, 247, 0.3)'
    glow-cyan: '0 0 20px rgba(76, 201, 240, 0.3)'
    glow-pink: '0 0 20px rgba(236, 72, 153, 0.3)'

  # Border Radius
  radius:
    none: '0'
    sm: '0.125rem'
    default: '0.25rem'
    md: '0.375rem'
    lg: '0.5rem'
    xl: '0.75rem'
    2xl: '1rem'
    full: '9999px'

  # Backdrop Blur
  blur:
    sm: 'blur(4px)'
    default: 'blur(8px)'
    md: 'blur(12px)'
    lg: 'blur(16px)'
    xl: 'blur(24px)'
    glass: 'blur(40px) saturate(180%)'
```

---

## Part 2: Template Folder Structure

```
lovable-templates/
├── README.md                          # Template library documentation
├── design-tokens/
│   ├── tokens.css                     # CSS custom properties
│   ├── tokens.json                    # JSON format for tools
│   └── tailwind.config.js             # Tailwind configuration
│
├── 01-scientific-research/            # Scientific/Research Templates
│   ├── quantum-lab/                   # Physics research portal
│   │   ├── index.html
│   │   ├── styles.css
│   │   └── README.md
│   ├── optimization-dashboard/        # Optilibria-style
│   ├── research-paper-viewer/         # Academic paper showcase
│   └── simulation-interface/          # Interactive simulations
│
├── 02-dashboards-analytics/           # Dashboard Templates
│   ├── metrics-command-center/        # ORCHEX-style dashboard
│   ├── performance-analytics/         # REPZCoach-style
│   ├── real-time-monitor/             # Live data visualization
│   └── admin-panel/                   # Admin interface
│
├── 03-documentation/                  # Documentation Templates
│   ├── api-docs/                      # API documentation
│   ├── project-wiki/                  # Project documentation
│   └── changelog-viewer/              # Release notes
│
├── 04-portfolio-personal/             # Portfolio Templates
│   ├── developer-profile/             # Personal branding
│   ├── project-showcase/              # Project gallery
│   └── resume-cv/                     # Interactive resume
│
├── 05-landing-pages/                  # Landing Page Templates
│   ├── open-source-project/           # GitHub project landing
│   ├── saas-product/                  # Product launch
│   ├── event-conference/              # Event promotion
│   └── waitlist-signup/               # Coming soon page
│
├── 06-interactive-webgl/              # Interactive/WebGL Templates
│   ├── physics-playground/            # MeatheadPhysicist-style
│   ├── 3d-visualization/              # Three.js showcase
│   └── particle-system/               # Particle effects demo
│
├── 07-web-applications/               # Web App Templates
│   ├── task-manager/                  # Productivity app
│   ├── code-editor/                   # Monaco-style editor
│   └── settings-panel/                # App settings UI
│
├── 08-components/                     # Reusable Components
│   ├── navigation/                    # Nav components
│   ├── cards/                         # Card variations
│   ├── forms/                         # Form elements
│   ├── modals/                        # Modal dialogs
│   └── charts/                        # Data visualization
│
└── shared/                            # Shared Assets
    ├── icons/                         # Custom icon set
    ├── animations/                    # CSS/JS animations
    └── utilities/                     # Helper functions
```

---

## Part 3: Category-Specific Lovable Prompts

### Credit Allocation Strategy

| Category             | Templates | Credits Each | Total Credits |
| -------------------- | --------- | ------------ | ------------- |
| Scientific/Research  | 4         | 8-10         | 35            |
| Dashboards/Analytics | 4         | 8-10         | 35            |
| Documentation        | 3         | 5-7          | 18            |
| Portfolio/Personal   | 3         | 6-8          | 21            |
| Landing Pages        | 4         | 6-8          | 28            |
| Interactive/WebGL    | 3         | 10-12        | 33            |
| Web Applications     | 3         | 8-10         | 27            |
| **Buffer**           | -         | -            | 3             |
| **TOTAL**            | 24        | -            | **200**       |

---

### 3.1 Scientific/Research Templates

#### Template 1: Quantum Lab Portal

```markdown
# LOVABLE PROMPT: Quantum Lab Portal

Create a sophisticated scientific research portal with the following specifications:

## Visual Design

- Dark theme with deep space background (#0F0F23 to #2A1B4D gradient)
- Accent colors: quantum purple (#A855F7), plasma pink (#EC4899), electron cyan (#4CC9F0)
- Monospace typography (JetBrains Mono or Fira Code) for all headings
- Inter or system sans-serif for body text
- Subtle grid pattern overlay (10% opacity)

## Layout Structure

1. **Header**: Fixed navigation with glass morphism effect (backdrop-filter: blur(40px))
   - Logo with gradient text effect
   - Navigation: Research | Publications | Team | Contact
   - GitHub star button

2. **Hero Section**:
   - Large gradient text headline: "Quantum Materials Research"
   - Animated orbital particles (3 concentric circles with rotating dots)
   - Mathematical symbols floating in background (∇²ψ, ∂H/∂t, λ·∇f)
   - CTA buttons: "Explore Research" and "View Publications"

3. **Research Areas Grid** (3 columns):
   - Glass morphism cards with hover glow effect
   - Icon + Title + Description
   - Topics: Quantum Computing, Materials Science, Optimization

4. **Latest Publications Section**:
   - Timeline-style layout
   - Paper cards with: Title, Authors, Journal, Date, Abstract preview
   - "Read More" links

5. **Team Section**:
   - Circular profile images with gradient borders
   - Name, Role, Research Focus
   - Social links (GitHub, Google Scholar, LinkedIn)

6. **Footer**:
   - Dark glass panel
   - Quick links, Contact info, Social icons
   - "Built with physics in mind" tagline

## Animations

- Smooth scroll behavior
- Cards lift on hover (translateY(-4px))
- Gradient text shimmer on hero
- Orbital particles rotate continuously
- Staggered fade-in on scroll

## Technical Requirements

- React + TypeScript
- Tailwind CSS
- Framer Motion for animations
- Fully responsive (mobile-first)
- WCAG 2.1 AA accessible
- Dark mode only (no light mode toggle needed)

## DO NOT INCLUDE

- Generic stock photos
- Fake statistics or testimonials
- Cookie consent banners
- Chat widgets
- Newsletter popups
```

#### Template 2: Optimization Dashboard (Optilibria-style)

```markdown
# LOVABLE PROMPT: Optimization Algorithm Dashboard

Create a technical dashboard for visualizing optimization algorithms:

## Visual Design

- Dark theme: #0F0F23 background with subtle purple tint
- Accent: Purple (#A855F7) for primary actions, Cyan (#4CC9F0) for data
- Monospace font for all numbers and code
- Card-based layout with glass morphism

## Layout Structure

1. **Top Bar**:
   - Logo: "OPTILIBRIA" in gradient text
   - Algorithm selector dropdown
   - Run/Pause/Reset controls
   - Settings gear icon

2. **Main Dashboard Grid** (12-column):
   - Left sidebar (3 cols): Parameter controls
     - Sliders for: Learning Rate, Iterations, Tolerance
     - Checkboxes for: Verbose, GPU Acceleration
     - "Run Optimization" button with gradient
   - Center (6 cols): Visualization area
     - Large canvas for convergence plot
     - Real-time loss curve (animated line chart)
     - Gradient descent visualization
   - Right sidebar (3 cols): Results panel
     - Current iteration counter
     - Best solution found
     - Convergence status indicator
     - Export results button

3. **Bottom Panel**:
   - Console/log output (monospace, scrollable)
   - Performance metrics: Time elapsed, Memory usage, GPU utilization

## Interactive Elements

- Draggable parameter sliders with real-time preview
- Zoomable/pannable visualization canvas
- Hover tooltips on data points
- Click to inspect specific iterations

## Code Display

- Syntax-highlighted code block showing current algorithm
- Copy button with success feedback
- Language: Python with JAX/NumPy style

## Animations

- Smooth transitions between algorithm states
- Pulsing glow on active elements
- Progress bar with gradient fill
- Number counters animate on change

## Technical Requirements

- React + TypeScript
- Tailwind CSS + CSS Modules
- Chart.js or Recharts for visualizations
- Responsive down to tablet (1024px minimum)
```

#### Template 3: Research Paper Viewer

```markdown
# LOVABLE PROMPT: Academic Paper Showcase

Create an elegant research paper presentation interface:

## Visual Design

- Dark theme with reading-optimized contrast
- Serif font (Instrument Serif) for paper titles
- Sans-serif (Inter) for metadata and UI
- Monospace for equations and code
- Accent: Soft purple (#A855F7) for links and highlights

## Layout Structure

1. **Paper Header**:
   - Large title with gradient underline
   - Authors with affiliation tooltips
   - Publication venue and date
   - Citation count badge
   - Download PDF / BibTeX buttons

2. **Abstract Panel**:
   - Highlighted box with left border accent
   - Expandable/collapsible

3. **Table of Contents** (sticky sidebar):
   - Auto-generated from headings
   - Current section highlighted
   - Smooth scroll on click

4. **Main Content Area**:
   - Clean typography with proper line height (1.75)
   - LaTeX-rendered equations (KaTeX style)
   - Figure captions with numbering
   - Code blocks with syntax highlighting
   - Inline citations with hover preview

5. **Figures Gallery**:
   - Lightbox for full-size viewing
   - Figure numbering and captions
   - Download individual figures

6. **References Section**:
   - Numbered list with hover cards
   - Links to DOI/arXiv
   - Copy citation button

## Features

- Reading progress indicator (top bar)
- Estimated reading time
- Share buttons (Twitter, LinkedIn, Email)
- Related papers suggestions

## Technical Requirements

- React + TypeScript
- Tailwind CSS
- KaTeX for math rendering
- Prism.js for code highlighting
```

#### Template 4: Simulation Interface

```markdown
# LOVABLE PROMPT: Physics Simulation Interface

Create an interactive physics simulation control panel:

## Visual Design

- Dark theme with neon accents
- Colors: Deep purple background, cyan (#4CC9F0) for active states
- Futuristic/sci-fi aesthetic
- Glowing borders and elements

## Layout Structure

1. **Control Panel Header**:
   - Simulation title with status indicator (Running/Paused/Complete)
   - Time step counter
   - FPS display

2. **3D Viewport Placeholder**:
   - Large central area (will integrate Three.js/WebGL)
   - Orbit controls hint overlay
   - Axis indicator in corner
   - Grid toggle button

3. **Parameter Panel** (collapsible sidebar):
   - Grouped controls:
     - Physics: Gravity, Friction, Restitution
     - Particles: Count, Mass, Radius
     - Forces: Magnitude, Direction
   - Preset buttons: "Zero-G", "Earth", "Moon"
   - Reset to defaults button

4. **Timeline Control**:
   - Play/Pause/Step buttons
   - Speed slider (0.1x to 10x)
   - Timeline scrubber
   - Keyframe markers

5. **Data Output Panel**:
   - Real-time graphs (energy, momentum)
   - Numerical readouts
   - Export data button (CSV/JSON)

## Animations

- Smooth parameter transitions
- Pulsing status indicators
- Glowing active controls
- Particle effects on interactions

## Technical Requirements

- React + TypeScript
- Tailwind CSS
- Placeholder for Three.js integration
- WebSocket-ready for real-time data
```

---

### 3.2 Dashboard/Analytics Templates

#### Template 5: Metrics Command Center (ORCHEX-style)

```markdown
# LOVABLE PROMPT: Code Analysis Command Center

Create a comprehensive metrics dashboard for code analysis:

## Visual Design

- Dark theme with gradient background (#667eea to #764ba2)
- White container cards with subtle shadows
- Accent colors for status: Green (good), Yellow (warning), Red (critical)
- Clean, data-focused typography

## Layout Structure

1. **Header**:
   - "ORCHEX Command Center" title with rocket emoji
   - Timestamp of last analysis
   - Refresh button

2. **Summary Cards Row** (4 cards):
   - Files Analyzed (number with icon)
   - Total Lines (formatted with commas)
   - Complexity Score (color-coded)
   - Maintainability Index (progress ring)

3. **Charts Section** (2-column grid):
   - Left: Radar chart (Complexity vs Chaos vs Maintainability)
   - Right: Bar chart (Analysis time per repository)

4. **Repository Cards Grid**:
   - Card per repository with:
     - Repository name
     - Key metrics list
     - Status indicators
     - "View Details" link

5. **Performance Table**:
   - Sortable columns: Repository, Analysis Time, Status
   - Status badges with checkmarks
   - Hover row highlighting

6. **Footer**:
   - Generation timestamp
   - "ORCHEX Demo Environment" badge

## Data Visualization

- Chart.js for all charts
- Animated number counters
- Color-coded thresholds
- Tooltips on hover

## Technical Requirements

- React + TypeScript
- Tailwind CSS
- Chart.js integration
- Responsive grid layout
```

#### Template 6: Performance Analytics (REPZCoach-style)

```markdown
# LOVABLE PROMPT: Athletic Performance Dashboard

Create a sports/fitness performance analytics dashboard:

## Visual Design

- Dark theme with energetic accents
- Primary: Electric blue (#4CC9F0)
- Secondary: Vibrant purple (#A855F7)
- High contrast for readability during workouts

## Layout Structure

1. **Athlete Header**:
   - Profile photo (circular with gradient border)
   - Name and sport
   - Current training phase badge
   - Quick stats: Sessions this week, PR count

2. **Performance Overview** (card grid):
   - Strength metrics card
   - Endurance metrics card
   - Recovery score card
   - Weekly volume card

3. **Progress Charts**:
   - Line chart: Weight progression over time
   - Bar chart: Volume by muscle group
   - Radar chart: Strength balance

4. **Recent Workouts List**:
   - Date, workout name, duration
   - Key lifts with weights
   - RPE rating
   - Notes preview

5. **Goals Section**:
   - Progress bars toward goals
   - Target vs Current comparison
   - Days remaining

6. **Recommendations Panel**:
   - AI-generated suggestions
   - Recovery recommendations
   - Next workout preview

## Animations

- Smooth chart transitions
- Progress bar fills
- Card hover effects
- Pull-to-refresh gesture support

## Technical Requirements

- React + TypeScript
- Tailwind CSS
- Recharts for visualizations
- Mobile-first responsive
```

#### Template 7: Real-Time Monitor

```markdown
# LOVABLE PROMPT: Real-Time System Monitor

Create a live system monitoring dashboard:

## Visual Design

- Dark theme optimized for always-on displays
- Green (#10B981) for healthy, Yellow (#F59E0B) for warning, Red (#EF4444) for critical
- Minimal chrome, maximum data density
- Monospace numbers for precision

## Layout Structure

1. **Status Bar**:
   - System name
   - Connection status indicator (pulsing dot)
   - Last update timestamp
   - Alert count badge

2. **Vital Signs Grid** (2x3):
   - CPU Usage (gauge + sparkline)
   - Memory Usage (gauge + sparkline)
   - Disk I/O (gauge + sparkline)
   - Network Traffic (gauge + sparkline)
   - Active Connections (number)
   - Uptime (formatted duration)

3. **Live Charts Panel**:
   - Streaming line chart (last 60 seconds)
   - Multiple metrics overlay
   - Auto-scaling Y-axis

4. **Alerts Feed**:
   - Scrolling list of recent alerts
   - Severity icons
   - Timestamp
   - Acknowledge button

5. **Process Table**:
   - Top processes by resource usage
   - Sortable columns
   - Kill process action (with confirmation)

## Real-Time Features

- WebSocket connection indicator
- Auto-reconnect logic
- Data buffering for smooth charts
- Configurable refresh rate

## Technical Requirements

- React + TypeScript
- Tailwind CSS
- Lightweight charting (uPlot or similar)
- WebSocket-ready architecture
```

#### Template 8: Admin Panel

```markdown
# LOVABLE PROMPT: Modern Admin Panel

Create a clean, functional admin interface:

## Visual Design

- Dark sidebar, light content area option
- Or full dark mode
- Purple accent (#A855F7) for primary actions
- Clean, professional aesthetic

## Layout Structure

1. **Sidebar Navigation**:
   - Logo at top
   - Icon + label nav items
   - Collapsible to icons only
   - User profile at bottom
   - Sections: Dashboard, Users, Content, Settings, Logs

2. **Top Bar**:
   - Breadcrumb navigation
   - Search bar (command palette style)
   - Notifications bell
   - User dropdown

3. **Dashboard View**:
   - Stats cards row
   - Recent activity feed
   - Quick actions grid

4. **Data Table View**:
   - Search and filter bar
   - Sortable columns
   - Bulk actions
   - Pagination
   - Row actions (Edit, Delete)

5. **Form View**:
   - Clean form layout
   - Validation states
   - Save/Cancel buttons
   - Unsaved changes warning

## Features

- Keyboard shortcuts (Cmd+K for search)
- Toast notifications
- Confirmation modals
- Loading states

## Technical Requirements

- React + TypeScript
- Tailwind CSS
- React Table for data grids
- React Hook Form for forms
```

---

### 3.3 Documentation Templates

#### Template 9: API Documentation

```markdown
# LOVABLE PROMPT: API Documentation Portal

Create a developer-friendly API documentation site:

## Visual Design

- Dark theme with syntax highlighting
- Purple accent for links and highlights
- Monospace for all code
- Clean, scannable layout

## Layout Structure

1. **Header**:
   - API name and version badge
   - Search bar
   - Authentication status
   - Theme toggle (dark/light)

2. **Sidebar Navigation**:
   - Collapsible sections:
     - Getting Started
     - Authentication
     - Endpoints (grouped by resource)
     - Errors
     - Rate Limits
   - Search within docs

3. **Endpoint Documentation**:
   - Method badge (GET/POST/PUT/DELETE) color-coded
   - Endpoint path with copy button
   - Description
   - Parameters table
   - Request example (with language tabs)
   - Response example
   - Try it out button

4. **Code Examples Panel**:
   - Language selector (cURL, Python, JavaScript, Go)
   - Syntax highlighted code
   - Copy button
   - Run in terminal hint

5. **Response Schema**:
   - Expandable JSON schema
   - Type annotations
   - Required field indicators

## Features

- Deep linking to sections
- Code copy with feedback
- Responsive sidebar (drawer on mobile)
- Keyboard navigation

## Technical Requirements

- React + TypeScript
- Tailwind CSS
- Prism.js for syntax highlighting
- Markdown rendering
```

#### Template 10: Project Wiki

```markdown
# LOVABLE PROMPT: Project Documentation Wiki

Create a comprehensive project documentation site:

## Visual Design

- Dark theme with excellent readability
- Purple accent for navigation highlights
- Serif headings, sans-serif body
- Wide content area for documentation

## Layout Structure

1. **Header**:
   - Project logo and name
   - Version selector dropdown
   - GitHub link
   - Search

2. **Sidebar**:
   - Tree-style navigation
   - Expandable sections
   - Current page highlighted
   - "Edit this page" link

3. **Content Area**:
   - Breadcrumb trail
   - Page title with anchor link
   - Table of contents (right sidebar on desktop)
   - Rich content:
     - Headings with anchor links
     - Code blocks with copy
     - Callout boxes (info, warning, tip)
     - Tables
     - Images with captions

4. **Page Footer**:
   - Last updated date
   - Contributors
   - Previous/Next navigation
   - "Was this helpful?" feedback

## Features

- Full-text search
- Keyboard shortcuts
- Print-friendly styles
- Offline support (PWA)

## Technical Requirements

- React + TypeScript
- Tailwind CSS
- MDX support
- Algolia-style search UI
```

#### Template 11: Changelog Viewer

```markdown
# LOVABLE PROMPT: Release Changelog Interface

Create an elegant changelog/release notes viewer:

## Visual Design

- Dark theme
- Timeline-style layout
- Version badges with semantic versioning colors
- Clean typography

## Layout Structure

1. **Header**:
   - Product name
   - "What's New" title
   - Subscribe to updates button
   - RSS feed link

2. **Filter Bar**:
   - Version range selector
   - Category filters (Features, Fixes, Breaking)
   - Search releases

3. **Timeline View**:
   - Version number with date
   - Release type badge (Major/Minor/Patch)
   - Grouped changes:
     - ✨ New Features
     - 🐛 Bug Fixes
     - 💥 Breaking Changes
     - 📚 Documentation
   - Expandable details
   - Link to full release notes

4. **Individual Release Card**:
   - Version header with copy button
   - Release date
   - Contributor avatars
   - Change list with icons
   - "View on GitHub" link

## Features

- Infinite scroll or pagination
- Deep linking to versions
- Compare versions feature
- Email subscription modal

## Technical Requirements

- React + TypeScript
- Tailwind CSS
- Markdown rendering
- RSS feed generation
```

---

### 3.4 Portfolio/Personal Templates

#### Template 12: Developer Profile

```markdown
# LOVABLE PROMPT: Developer Portfolio Profile

Create a stunning developer portfolio page:

## Visual Design

- Dark theme with quantum-physics aesthetic
- Gradient background (#0F0F23 to #2A1B4D)
- Animated orbital particles
- Gradient text for name
- Monospace accents

## Layout Structure

1. **Hero Section**:
   - Large name with gradient text effect
   - Title: "Computational Physicist" or similar
   - Tagline with typing animation
   - Location badge with pin icon
   - Social links (GitHub, LinkedIn, Twitter, Google Scholar)
   - Mathematical symbols floating (∇²ψ, ∂H/∂t)

2. **About Section**:
   - Brief bio (2-3 paragraphs)
   - Current focus areas
   - Profile photo with gradient border (optional)

3. **Skills/Expertise**:
   - Categorized skill tags
   - Categories: Languages, Frameworks, Tools, Domains
   - Hover effects on tags

4. **Featured Projects**:
   - 3-4 highlighted projects
   - Project cards with:
     - Screenshot/preview
     - Name and description
     - Tech stack tags
     - GitHub stars count
     - Live demo link

5. **Experience Timeline**:
   - Vertical timeline
   - Company, role, dates
   - Key achievements
   - Tech used

6. **Publications/Research** (optional):
   - Paper titles with links
   - Citation counts
   - Conference/journal names

7. **Contact Section**:
   - Email with copy button
   - Calendar booking link
   - Contact form (optional)

## Animations

- Smooth scroll
- Fade-in on scroll
- Hover effects on all interactive elements
- Orbital particle animation in hero
- Typing effect for tagline

## Technical Requirements

- React + TypeScript
- Tailwind CSS
- Framer Motion
- Fully responsive
```

#### Template 13: Project Showcase

```markdown
# LOVABLE PROMPT: Project Gallery Showcase

Create a portfolio project gallery:

## Visual Design

- Dark theme
- Card-based layout
- Hover animations
- Filter animations

## Layout Structure

1. **Header**:
   - "Projects" title
   - Filter buttons (All, Web, Mobile, AI/ML, Open Source)
   - View toggle (Grid/List)
   - Sort dropdown

2. **Project Grid**:
   - Masonry or uniform grid
   - Project cards with:
     - Cover image/screenshot
     - Project name
     - Short description
     - Tech stack icons
     - GitHub/Live links
     - Star count badge

3. **Project Detail Modal**:
   - Full project description
   - Image gallery/carousel
   - Features list
   - Technical details
   - Links to repo and demo
   - Related projects

4. **Stats Section**:
   - Total projects
   - Total GitHub stars
   - Technologies used count

## Animations

- Filter transitions (FLIP animation)
- Card hover lift
- Modal open/close
- Image lazy loading with fade

## Technical Requirements

- React + TypeScript
- Tailwind CSS
- Framer Motion for animations
- Image optimization
```

#### Template 14: Interactive Resume

```markdown
# LOVABLE PROMPT: Interactive Resume/CV

Create a modern interactive resume:

## Visual Design

- Dark theme with print-friendly option
- Clean, professional layout
- Accent color for highlights
- Excellent typography

## Layout Structure

1. **Header**:
   - Name (large)
   - Title/Role
   - Contact info row (Email, Phone, Location, LinkedIn, GitHub)
   - Download PDF button

2. **Summary Section**:
   - Professional summary (3-4 sentences)
   - Key highlights/achievements

3. **Experience Section**:
   - Company logo (small)
   - Company name, Location
   - Role, Dates
   - Bullet points of achievements
   - Tech tags

4. **Education Section**:
   - Institution, Degree
   - Dates, GPA (optional)
   - Relevant coursework

5. **Skills Section**:
   - Categorized skill bars or tags
   - Proficiency indicators

6. **Projects Section**:
   - Mini project cards
   - Links to full details

7. **Publications/Certifications**:
   - List with links

## Features

- Print stylesheet
- PDF export
- Section navigation
- Expandable details

## Technical Requirements

- React + TypeScript
- Tailwind CSS
- Print CSS
- PDF generation ready
```

---

### 3.5 Landing Page Templates

#### Template 15: Open Source Project Landing

````markdown
# LOVABLE PROMPT: Open Source Project Landing Page

Create a landing page for an open source project:

## Visual Design

- Dark theme with gradient accents
- Hero with code/terminal aesthetic
- Clean, developer-focused design
- No marketing fluff

## Layout Structure

1. **Navigation**:
   - Logo
   - Links: Docs, Examples, GitHub, Community
   - GitHub stars badge (live count)
   - "Get Started" button

2. **Hero Section**:
   - Project name with gradient text
   - One-line description
   - Install command with copy button:
     ```
     npm install project-name
     ```
   - Primary CTA: "Read the Docs"
   - Secondary CTA: "View on GitHub"

3. **Features Grid** (3 columns):
   - Icon + Title + Description
   - No fake statistics
   - Honest feature descriptions

4. **Code Example Section**:
   - Tabbed code examples
   - Syntax highlighted
   - "Try in Browser" button (if applicable)

5. **Quick Start**:
   - Step-by-step installation
   - Minimal working example
   - Link to full documentation

6. **Community Section**:
   - Discord/Slack link
   - GitHub Discussions link
   - Contributing guide link
   - Code of Conduct link

7. **Footer**:
   - License badge
   - Version number
   - Links: Docs, GitHub, Twitter
   - "Made with ❤️ by [Name]"

## DO NOT INCLUDE

- Pricing sections
- Testimonials
- "Trusted by X companies"
- Newsletter signup popups
- Cookie banners

## Technical Requirements

- React + TypeScript
- Tailwind CSS
- Responsive
- Fast loading (no heavy images)
````

#### Template 16: SaaS Product Landing

```markdown
# LOVABLE PROMPT: SaaS Product Landing Page

Create a modern SaaS product landing page:

## Visual Design

- Dark theme with vibrant accents
- Gradient CTAs
- Glass morphism cards
- Professional but not corporate

## Layout Structure

1. **Navigation**:
   - Logo
   - Product, Features, Pricing, Docs
   - Sign In / Get Started buttons

2. **Hero**:
   - Bold headline (benefit-focused)
   - Subheadline explaining the product
   - Email signup or CTA button
   - Product screenshot/mockup

3. **Social Proof** (subtle):
   - "Used by developers at..." with logos
   - Or GitHub stars if open source
   - Keep minimal and honest

4. **Features Section**:
   - 3-4 key features
   - Icon + Title + Description
   - Optional: small illustration

5. **How It Works**:
   - 3-step process
   - Numbered steps with icons
   - Brief descriptions

6. **Pricing** (if applicable):
   - 2-3 tiers
   - Clear feature comparison
   - Highlighted recommended plan
   - FAQ below pricing

7. **CTA Section**:
   - Final call to action
   - Repeat main CTA button

8. **Footer**:
   - Product links
   - Company links
   - Legal links
   - Social icons

## Technical Requirements

- React + TypeScript
- Tailwind CSS
- Smooth scroll
- Responsive
```

#### Template 17: Event/Conference Landing

```markdown
# LOVABLE PROMPT: Tech Event Landing Page

Create a landing page for a tech conference/event:

## Visual Design

- Dark theme with energetic accents
- Bold typography
- Countdown timer
- Speaker photos

## Layout Structure

1. **Hero**:
   - Event name (large, gradient)
   - Date and location
   - Countdown timer
   - "Register Now" CTA
   - Virtual/In-person badge

2. **About Section**:
   - Event description
   - Key topics/themes
   - Who should attend

3. **Speakers Grid**:
   - Speaker photos (circular)
   - Name, Title, Company
   - Topic they're presenting
   - Social links

4. **Schedule**:
   - Day tabs (if multi-day)
   - Timeline view
   - Session cards with:
     - Time
     - Title
     - Speaker
     - Track/Room

5. **Venue/Virtual Info**:
   - Location map (if in-person)
   - Virtual platform info
   - Access instructions

6. **Sponsors** (if applicable):
   - Tiered sponsor logos
   - "Become a Sponsor" link

7. **Registration CTA**:
   - Ticket types
   - Pricing
   - Register button

8. **Footer**:
   - Contact info
   - Social links
   - Code of Conduct link

## Technical Requirements

- React + TypeScript
- Tailwind CSS
- Countdown timer component
- Schedule filtering
```

#### Template 18: Waitlist/Coming Soon

```markdown
# LOVABLE PROMPT: Coming Soon / Waitlist Page

Create a minimal coming soon page:

## Visual Design

- Dark theme with subtle animation
- Centered layout
- Focus on email capture
- Elegant and mysterious

## Layout Structure

1. **Centered Content**:
   - Logo or product name
   - "Coming Soon" or launch date
   - One-line teaser description
   - Email signup form:
     - Email input
     - "Notify Me" button
   - Success message after signup

2. **Background**:
   - Subtle animated gradient
   - Or particle effect
   - Or geometric pattern

3. **Social Links**:
   - Twitter, GitHub, etc.
   - "Follow for updates"

4. **Footer** (minimal):
   - Copyright
   - Privacy link

## Features

- Email validation
- Success/error states
- Animated background
- Mobile responsive

## Technical Requirements

- React + TypeScript
- Tailwind CSS
- Form handling
- Minimal dependencies
```

---

### 3.6 Interactive/WebGL Templates

#### Template 19: Physics Playground (MeatheadPhysicist-style)

```markdown
# LOVABLE PROMPT: Interactive Physics Playground

Create an educational physics simulation interface:

## Visual Design

- Dark theme with neon accents
- Futuristic/sci-fi aesthetic
- Glowing elements
- Grid background

## Layout Structure

1. **Header Bar**:
   - "Physics Playground" title
   - Simulation selector dropdown
   - Reset button
   - Settings gear

2. **Main Canvas Area**:
   - Large WebGL/Canvas placeholder
   - Fullscreen toggle
   - Zoom controls
   - Grid toggle

3. **Control Panel** (sidebar):
   - Simulation parameters
   - Sliders with real-time preview
   - Preset buttons
   - "Run Simulation" button

4. **Info Panel**:
   - Current simulation explanation
   - Key equations displayed
   - Variable definitions

5. **Data Output**:
   - Real-time measurements
   - Mini charts
   - Export data button

## Placeholder Elements

- Canvas area ready for Three.js/WebGL
- Placeholder for physics engine integration
- Mock data for demonstration

## Technical Requirements

- React + TypeScript
- Tailwind CSS
- Canvas/WebGL placeholder
- Responsive layout
```

#### Template 20: 3D Visualization Showcase

```markdown
# LOVABLE PROMPT: 3D Data Visualization Interface

Create a 3D data visualization dashboard:

## Visual Design

- Dark theme
- Cyan and purple accents
- Futuristic UI elements
- Glass morphism panels

## Layout Structure

1. **Viewport**:
   - Large 3D canvas (Three.js placeholder)
   - Camera controls overlay
   - Axis helper
   - Background: subtle grid or stars

2. **Data Panel** (left sidebar):
   - Dataset selector
   - Variable mapping dropdowns
   - Color scale selector
   - Filter controls

3. **Visualization Options** (right sidebar):
   - Chart type selector (scatter, surface, volume)
   - Point size slider
   - Opacity slider
   - Animation toggle

4. **Bottom Bar**:
   - Timeline scrubber (for animated data)
   - Play/pause controls
   - Export screenshot button

## Technical Requirements

- React + TypeScript
- Tailwind CSS
- Three.js placeholder structure
- Responsive panels
```

#### Template 21: Particle System Demo

```markdown
# LOVABLE PROMPT: Particle Effects Showcase

Create a particle system demonstration page:

## Visual Design

- Full dark background
- Colorful particles
- Minimal UI, maximum visual impact
- Floating control panel

## Layout Structure

1. **Full-Screen Canvas**:
   - Particle system visualization
   - Mouse interaction area

2. **Floating Control Panel** (draggable):
   - Particle count slider
   - Color picker
   - Speed control
   - Gravity toggle
   - Mouse attraction toggle
   - Preset effects dropdown

3. **Info Overlay** (toggleable):
   - FPS counter
   - Particle count
   - Performance metrics

4. **Bottom Controls**:
   - Fullscreen toggle
   - Screenshot button
   - Share button

## Technical Requirements

- React + TypeScript
- Tailwind CSS
- Canvas API or Three.js placeholder
- RequestAnimationFrame loop structure
```

---

### 3.7 Web Application Templates

#### Template 22: Task Manager App

```markdown
# LOVABLE PROMPT: Modern Task Manager Application

Create a productivity task management app:

## Visual Design

- Dark theme with purple accents
- Clean, minimal interface
- Focus on content, not chrome
- Smooth animations

## Layout Structure

1. **Sidebar**:
   - User avatar and name
   - Navigation:
     - Inbox
     - Today
     - Upcoming
     - Projects (expandable)
     - Labels
   - "Add Project" button
   - Settings at bottom

2. **Main Content**:
   - View title and count
   - Add task input (always visible)
   - Task list:
     - Checkbox
     - Task title
     - Due date badge
     - Priority indicator
     - Project tag
     - Hover actions (edit, delete)

3. **Task Detail Panel** (slide-in):
   - Task title (editable)
   - Description (rich text)
   - Due date picker
   - Priority selector
   - Labels
   - Subtasks
   - Comments
   - Activity log

4. **Quick Add Modal**:
   - Keyboard shortcut triggered
   - Natural language input
   - Smart parsing preview

## Features

- Drag and drop reordering
- Keyboard shortcuts
- Quick add with natural language
- Filters and search

## Technical Requirements

- React + TypeScript
- Tailwind CSS
- Drag and drop library
- Date picker component
```

#### Template 23: Code Editor Interface

```markdown
# LOVABLE PROMPT: Web-Based Code Editor

Create a Monaco-style code editor interface:

## Visual Design

- Dark theme (VS Code inspired)
- Syntax highlighting
- Monospace typography throughout
- Minimal distractions

## Layout Structure

1. **Title Bar**:
   - File name with unsaved indicator
   - Language selector
   - Theme toggle
   - Settings

2. **Toolbar**:
   - File operations (New, Open, Save)
   - Edit operations (Undo, Redo, Find)
   - Run button
   - Format button

3. **Sidebar** (collapsible):
   - File explorer tree
   - Search panel
   - Extensions panel

4. **Editor Area**:
   - Line numbers
   - Code content area
   - Minimap (optional)
   - Breadcrumb path

5. **Bottom Panel** (collapsible):
   - Terminal tab
   - Output tab
   - Problems tab

6. **Status Bar**:
   - Line/column position
   - Language mode
   - Encoding
   - Indentation

## Features

- Syntax highlighting (placeholder)
- Line numbers
- Code folding indicators
- Multiple tabs

## Technical Requirements

- React + TypeScript
- Tailwind CSS
- Monaco Editor placeholder
- Resizable panels
```

#### Template 24: Settings Panel

```markdown
# LOVABLE PROMPT: Application Settings Interface

Create a comprehensive settings panel:

## Visual Design

- Dark theme
- Clean, organized layout
- Clear section hierarchy
- Accessible form controls

## Layout Structure

1. **Settings Navigation** (sidebar):
   - Search settings
   - Categories:
     - General
     - Appearance
     - Editor
     - Keyboard Shortcuts
     - Privacy
     - Account
     - About

2. **Settings Content**:
   - Category title
   - Setting groups with headers
   - Individual settings:
     - Label and description
     - Control (toggle, dropdown, input, slider)
     - Reset to default link

3. **Setting Types**:
   - Toggle switches
   - Dropdown selects
   - Text inputs
   - Number inputs with steppers
   - Color pickers
   - Keyboard shortcut recorder
   - File/folder pickers

4. **Actions Bar**:
   - Save changes button
   - Reset all button
   - Import/Export settings

## Features

- Search/filter settings
- Keyboard navigation
- Unsaved changes warning
- Setting descriptions

## Technical Requirements

- React + TypeScript
- Tailwind CSS
- Form state management
- Accessible controls
```

---

## Part 4: Post-Processing Checklist

### 4.1 Immediate Post-Download Tasks

```markdown
## After Downloading Each Template

### 1. Project Setup

- [ ] Create new folder in `lovable-templates/[category]/[template-name]/`
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

### 4.2 Design System Alignment

````markdown
## Design System Integration Steps

### Color Updates

Replace Lovable defaults with Alawein tokens:

```css
/* Before (Lovable default) */
background: #1a1a2e;
color: #ffffff;
accent: #4361ee;

/* After (Alawein Design System) */
background: var(--color-void-start);
color: var(--color-text-primary);
accent: var(--color-quantum-purple);
```
````

### Typography Updates

```css
/* Before */
font-family: 'Inter', sans-serif;

/* After */
font-family: var(--font-body);
/* For headings */
font-family: var(--font-display);
/* For code */
font-family: var(--font-mono);
```

### Animation Updates

```css
/* Before */
transition: all 0.3s ease;

/* After */
transition:
  transform var(--duration-normal) var(--ease-expo-out),
  opacity var(--duration-normal) var(--ease-expo-out);
```

### Shadow Updates

```css
/* Before */
box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);

/* After */
box-shadow: var(--shadow-md);
/* Or for glow effects */
box-shadow: var(--shadow-glow-purple);
```

````

### 4.3 Custom Enhancement Additions

```markdown
## Alawein Signature Enhancements

### 1. Orbital Particle Animation
Add to hero sections:
```tsx
// components/OrbitalParticles.tsx
const OrbitalParticles = () => {
  return (
    <div className="orbital-container">
      <div className="orbit orbit-1">
        <div className="particle particle-cyan" />
      </div>
      <div className="orbit orbit-2">
        <div className="particle particle-purple" />
      </div>
      <div className="orbit orbit-3">
        <div className="particle particle-pink" />
      </div>
    </div>
  );
};
````

### 2. Gradient Text Effect

```css
.gradient-text {
  background: var(--gradient-text);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}
```

### 3. Glass Morphism Cards

```css
.glass-card {
  background: var(--color-glass);
  backdrop-filter: var(--blur-glass);
  border: 1px solid var(--color-card-border);
  border-radius: var(--radius-xl);
}
```

### 4. Mathematical Symbol Decorations

Add floating math symbols to scientific templates:

```tsx
const MathDecorations = () => (
  <div className="math-decorations">
    <span className="math-symbol" style={{ top: '20%', left: '5%' }}>
      ∇²ψ
    </span>
    <span className="math-symbol" style={{ top: '80%', left: '10%' }}>
      ∂H/∂t
    </span>
    <span className="math-symbol" style={{ top: '15%', right: '8%' }}>
      λ·∇f
    </span>
    <span className="math-symbol" style={{ top: '75%', right: '5%' }}>
      min(E)
    </span>
  </div>
);
```

### 5. Glow Effects on Hover

```css
.glow-on-hover {
  transition: box-shadow var(--duration-normal) var(--ease-out);
}

.glow-on-hover:hover {
  box-shadow: var(--shadow-glow-purple);
}
```

````

### 4.4 Deployment Preparation

```markdown
## Deployment Checklist

### 1. Build Configuration
- [ ] Verify Vite/Next.js build succeeds
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

### 3. Analytics Setup (Optional)
- [ ] Add Plausible/Fathom analytics
- [ ] Configure privacy-respecting tracking
- [ ] Set up goal tracking

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
````

---

## Part 5: Quick Reference Card

### Lovable Prompt Best Practices

```markdown
## DO ✅

- Be extremely specific about colors (use hex codes)
- Specify exact fonts by name
- Describe layout with column counts
- List all sections in order
- Specify animation types and durations
- Mention accessibility requirements
- Request specific tech stack

## DON'T ❌

- Use vague terms like "modern" or "clean" alone
- Assume Lovable knows your brand
- Request features Lovable can't generate (backend logic)
- Ask for too many pages in one prompt
- Forget to specify dark/light mode preference
- Skip responsive requirements
```

### Credit Optimization Tips

```markdown
## Maximize Your 200 Credits

1. **Batch Similar Templates**: Generate variations in sequence
2. **Start Simple**: Get base layout first, then iterate
3. **Use Specific Prompts**: Vague prompts waste credits on revisions
4. **Download Immediately**: Don't lose work to session timeouts
5. **Document What Works**: Note successful prompt patterns
6. **Iterate Locally**: Do refinements in your IDE, not Lovable
```

### Emergency Prompt Template

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

## Appendix: Design Token CSS File

```css
/* lovable-templates/design-tokens/tokens.css */

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

---

_Document Version: 1.0_
_Created: December 4, 2024_
_Author: Frontend Specialist Mode_
_For: Meshal Alawein - Alawein Design System_
