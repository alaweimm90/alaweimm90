# Storybook - Component Documentation & Development Guide

Live It Iconic uses Storybook as the central hub for component documentation and development. This comprehensive guide covers setup, usage, best practices, and deployment.

## What is Storybook?

Storybook is an isolated development environment for UI components. It allows you to:
- Develop components in isolation
- Document component usage and variations
- Test component states visually
- Generate accessibility reports
- Maintain a living component library
- Test interactions and animations

## Table of Contents

1. [Getting Started](#getting-started)
2. [Project Structure](#project-structure)
3. [Writing Stories](#writing-stories)
4. [Story Examples](#story-examples)
5. [Design System Documentation](#design-system-documentation)
6. [Testing in Storybook](#testing-in-storybook)
7. [Accessibility Testing](#accessibility-testing)
8. [Chromatic Integration](#chromatic-integration)
9. [Best Practices](#best-practices)
10. [Troubleshooting](#troubleshooting)

## Getting Started

### Installation

Storybook has already been installed with all necessary addons:

```bash
npm install
```

### Running Storybook

Start Storybook in development mode:

```bash
npm run storybook
```

Storybook will open at `http://localhost:6006`

### Building Storybook

Create a static build for deployment:

```bash
npm run storybook:build
```

This generates the `storybook-static` directory.

### Deploying Storybook

Deploy the built Storybook:

```bash
npm run storybook:deploy
```

## Project Structure

```
live-it-iconic/
├── .storybook/
│   ├── main.ts          # Storybook configuration
│   └── preview.ts       # Global preview settings
│
├── src/
│   ├── components/
│   │   ├── ui/
│   │   │   ├── button.tsx
│   │   │   ├── button.stories.tsx    # Button stories
│   │   │   ├── card.tsx
│   │   │   ├── card.stories.tsx      # Card stories
│   │   │   └── ...
│   │   ├── Hero.tsx
│   │   ├── Hero.stories.tsx          # Hero stories
│   │   ├── CartDrawer.tsx
│   │   ├── CartDrawer.stories.tsx    # CartDrawer stories
│   │   └── ...
│   │
│   └── stories/
│       ├── DesignSystem.mdx          # Design system intro
│       ├── Colors.stories.tsx        # Color palette showcase
│       └── Typography.stories.tsx    # Typography system
│
└── docs/
    └── STORYBOOK.md                  # This file
```

## Writing Stories

### Story File Naming Convention

Stories should follow this naming pattern:
- `ComponentName.stories.tsx` for TSX stories
- `ComponentName.mdx` for MDX documentation

### Basic Story Structure

```tsx
import type { Meta, StoryObj } from '@storybook/react';
import { Button } from './button';

const meta: Meta<typeof Button> = {
  title: 'UI/Button',
  component: Button,
  tags: ['autodocs'],
  argTypes: {
    variant: {
      control: 'select',
      options: ['primary', 'secondary', 'ghost'],
    },
  },
};

export default meta;
type Story = StoryObj<typeof Button>;

export const Primary: Story = {
  args: {
    variant: 'primary',
    children: 'Click Me',
  },
};
```

### Key Concepts

**Meta (Configuration)**
- `title`: Story location in Storybook UI (e.g., 'UI/Button')
- `component`: React component being documented
- `tags`: Array of tags ('autodocs' enables automatic documentation)
- `argTypes`: Control definitions for component props
- `parameters`: Story-specific settings (backgrounds, layout, etc.)
- `decorators`: Wrapper components for all stories

**Stories**
- Named exports that define component states
- Use `args` to pass props to component
- Use `render` for custom layouts

**Args**
- Component props passed to the component
- Can be controlled via Storybook Controls panel

## Story Examples

### Button Component

```tsx
export const Primary: Story = {
  args: {
    variant: 'primary',
    size: 'lg',
    children: 'Shop Now',
  },
};

export const Disabled: Story = {
  args: {
    variant: 'primary',
    disabled: true,
    children: 'Disabled',
  },
};

export const AllVariants: Story = {
  render: () => (
    <div className="flex gap-4">
      <Button variant="primary">Primary</Button>
      <Button variant="secondary">Secondary</Button>
      <Button variant="ghost">Ghost</Button>
    </div>
  ),
};
```

### ProductCard with State

```tsx
import { useState } from 'react';

export const Default: Story = {
  render: () => {
    const [isFavorited, setIsFavorited] = useState(false);
    
    return (
      <ProductCard
        product={mockProduct}
        isFavorited={isFavorited}
        onToggleFavorite={() => setIsFavorited(!isFavorited)}
      />
    );
  },
};
```

### Interactive Components

```tsx
export const Interactive: Story = {
  render: function Render(args) {
    const [isOpen, setIsOpen] = useState(false);
    return (
      <CartDrawer 
        isOpen={isOpen}
        onClose={() => setIsOpen(false)}
        {...args}
      />
    );
  },
};
```

## Design System Documentation

### MDX Documents

Create `.mdx` files for comprehensive documentation:

```mdx
import { Meta } from '@storybook/blocks';

<Meta title="Design System/Colors" />

# Color System

Our palette includes...

## Primary Colors
...
```

### Color Palette Story

Located in `src/stories/Colors.stories.tsx`:

```tsx
export const Primary = () => (
  <div className="grid grid-cols-3 gap-4">
    <ColorSwatch 
      name="Gold" 
      hex="#C1A060"
      description="Primary accent color"
    />
    ...
  </div>
);
```

### Typography Story

Located in `src/stories/Typography.stories.tsx`:

```tsx
export const Headings = () => (
  <div className="space-y-4">
    <h1 className="text-5xl font-display">Display Large</h1>
    <h2 className="text-4xl font-display">Display Medium</h2>
    <h3 className="text-3xl font-display">Display Small</h3>
  </div>
);
```

## Testing in Storybook

### Interaction Testing

Use the Interactions addon to test user interactions:

```tsx
import { expect, within, userEvent } from '@storybook/test';

export const WithInteractions: Story = {
  play: async ({ canvasElement }) => {
    const canvas = within(canvasElement);
    const button = canvas.getByRole('button', { name: /click/i });
    
    await userEvent.click(button);
    await expect(button).toHaveAttribute('aria-pressed', 'true');
  },
};
```

### Visual Regression Testing

Chromatic handles visual regression testing automatically. On every commit:
1. Storybook builds
2. Stories are captured
3. Compared against baseline
4. Flagged if changes detected

## Accessibility Testing

### a11y Addon

The Accessibility addon automatically checks stories for:
- Color contrast
- Missing alt text
- Missing labels
- ARIA issues
- Keyboard navigation

To run accessibility checks:
1. Open a story in Storybook
2. Click the "Accessibility" tab
3. Review violations and warnings

### Manual Accessibility Testing

- Test keyboard navigation (Tab, Enter, Escape)
- Test with screen readers (NVDA, JAWS, VoiceOver)
- Verify focus indicators are visible
- Check color contrast ratios

## Chromatic Integration

### Setup

1. Create a Chromatic account at chromatic.com
2. Add `CHROMATIC_PROJECT_TOKEN` to GitHub secrets
3. The workflow runs automatically on push/PR

### Workflow

The `.github/workflows/chromatic.yml` file:

```yaml
- Triggers on push to main/develop and pull requests
- Builds Storybook
- Publishes to Chromatic
- Flags UI changes
- Prevents regressions
```

### Reviewing Changes

1. Chromatic provides a URL in the PR
2. Review UI changes visually
3. Accept or deny changes
4. Auto-accept baseline changes

### Configuration

`.github/workflows/chromatic.yml` settings:

```yaml
onlyChanged: true         # Only test changed stories
autoAcceptChanges: main   # Auto-accept on main branch
buildScriptName: storybook:build
```

## Best Practices

### 1. Story Organization

- Group stories by component type
- Use descriptive titles: `'UI/Button'`, `'Components/ProductCard'`
- Keep stories small and focused

### 2. Args and Controls

```tsx
argTypes: {
  variant: {
    control: 'select',
    options: ['primary', 'secondary', 'ghost'],
    description: 'Visual style variant',
  },
  size: {
    control: 'select',
    options: ['sm', 'md', 'lg'],
  },
  disabled: {
    control: 'boolean',
  },
  onClick: { action: 'clicked' },
}
```

### 3. Decorators for Layout

```tsx
const meta: Meta<typeof Button> = {
  decorators: [
    (Story) => (
      <div className="flex items-center justify-center min-h-screen">
        <Story />
      </div>
    ),
  ],
};
```

### 4. Document Variations

- Create separate stories for each significant state
- Use descriptive names: `Disabled`, `Loading`, `Error`, etc.

### 5. Design System Stories

- Group design tokens by category
- Show all variations in one place
- Include contrast ratios for colors
- Document usage guidelines

### 6. Accessibility

```tsx
const meta: Meta<typeof Component> = {
  tags: ['autodocs'],
  parameters: {
    a11y: {
      config: {
        rules: [
          {
            id: 'color-contrast',
            enabled: true,
          },
        ],
      },
    },
  },
};
```

## Troubleshooting

### Storybook Won't Start

```bash
# Clear cache and reinstall
rm -rf node_modules .storybook-cache
npm install
npm run storybook
```

### Stories Not Found

- Check `.storybook/main.ts` for correct pattern
- Ensure story files match pattern: `*.stories.tsx`
- Check for typos in story names

### Styling Issues

- Verify Tailwind CSS is imported in `.storybook/preview.ts`
- Check that CSS modules are working
- Review Vite configuration for asset handling

### Chromatic Issues

- Verify `CHROMATIC_PROJECT_TOKEN` is set in GitHub secrets
- Check workflow logs for errors
- Ensure `storybook:build` script exists

### Import Errors

- Check import paths are absolute (use `@/` alias)
- Verify Vite alias configuration
- Check for circular dependencies

## Adding New Stories

### Checklist

- [ ] Create `ComponentName.stories.tsx` next to component
- [ ] Add to `.storybook/main.ts` if needed
- [ ] Define Meta configuration
- [ ] Create at least 3 stories (default, variant, interactive)
- [ ] Add accessibility tags
- [ ] Document argTypes
- [ ] Test all stories visually
- [ ] Commit to git

### Template

```tsx
import type { Meta, StoryObj } from '@storybook/react';
import { YourComponent } from './YourComponent';

const meta: Meta<typeof YourComponent> = {
  title: 'Category/ComponentName',
  component: YourComponent,
  tags: ['autodocs'],
  argTypes: {
    // Define prop controls
  },
  parameters: {
    // Story-specific settings
  },
};

export default meta;
type Story = StoryObj<typeof YourComponent>;

export const Default: Story = {
  args: {
    // Default props
  },
};

export const Variant: Story = {
  args: {
    // Variant props
  },
};

export const Interactive: Story = {
  render: () => {
    // Interactive implementation
  },
};
```

## Resources

- [Storybook Documentation](https://storybook.js.org/docs/react/get-started/introduction)
- [Storybook Best Practices](https://storybook.js.org/docs/react/writing-stories/introduction)
- [Chromatic Documentation](https://www.chromatic.com/docs/)
- [Accessibility Testing](https://www.w3.org/WAI/test-evaluate/)
- [Live It Iconic Design System](/design-system)

## Support

For questions or issues:
1. Check Storybook docs
2. Review existing stories
3. Check GitHub issues
4. Ask in team discussions
