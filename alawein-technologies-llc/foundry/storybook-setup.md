# Storybook Setup Guide

## Complete guide for setting up and using Storybook with the Foundry Design System

---

## Table of Contents

1. [Installation](#installation)
2. [Configuration](#configuration)
3. [Writing Stories](#writing-stories)
4. [Addons](#addons)
5. [Running Storybook](#running-storybook)
6. [Deployment](#deployment)
7. [Best Practices](#best-practices)

---

## Installation

### Quick Setup

```bash
# Initialize Storybook in your project
npx storybook@latest init

# Install additional dependencies
npm install --save-dev @storybook/addon-essentials @storybook/addon-a11y @storybook/addon-docs
```

### Manual Installation

```bash
# Install Storybook for React
npm install --save-dev @storybook/react @storybook/builder-vite

# Install essential addons
npm install --save-dev \
  @storybook/addon-essentials \
  @storybook/addon-interactions \
  @storybook/addon-links \
  @storybook/addon-a11y \
  @storybook/addon-viewport \
  @storybook/addon-docs \
  @storybook/addon-controls

# Install testing utilities
npm install --save-dev \
  @storybook/testing-library \
  @storybook/jest \
  @storybook/addon-coverage
```

---

## Configuration

### Main Configuration (.storybook/main.ts)

```typescript
import type { StorybookConfig } from '@storybook/react-vite';

const config: StorybookConfig = {
  stories: [
    '../components/**/*.stories.@(js|jsx|ts|tsx|mdx)',
    '../stories/**/*.stories.@(js|jsx|ts|tsx|mdx)',
  ],
  addons: [
    '@storybook/addon-essentials',
    '@storybook/addon-a11y',
    '@storybook/addon-interactions',
    '@storybook/addon-links',
    '@storybook/addon-viewport',
    '@storybook/addon-docs',
    {
      name: '@storybook/addon-styling',
      options: {
        postCss: true,
      },
    },
  ],
  framework: {
    name: '@storybook/react-vite',
    options: {},
  },
  core: {
    builder: '@storybook/builder-vite',
  },
  features: {
    storyStoreV7: true,
  },
  viteFinal: async (config) => {
    // Add any Vite configuration here
    return config;
  },
};

export default config;
```

### Preview Configuration (.storybook/preview.tsx)

```tsx
import React from 'react';
import type { Preview } from '@storybook/react';
import { ToastProvider } from '../components/Toast';
import { themeManager } from '../theme';
import '../styles/globals.css'; // Your global styles

const preview: Preview = {
  parameters: {
    actions: { argTypesRegex: '^on[A-Z].*' },
    controls: {
      matchers: {
        color: /(background|color)$/i,
        date: /Date$/,
      },
    },
    viewport: {
      viewports: {
        mobile: {
          name: 'Mobile',
          styles: {
            width: '375px',
            height: '667px',
          },
        },
        tablet: {
          name: 'Tablet',
          styles: {
            width: '768px',
            height: '1024px',
          },
        },
        desktop: {
          name: 'Desktop',
          styles: {
            width: '1440px',
            height: '900px',
          },
        },
      },
    },
    backgrounds: {
      default: 'light',
      values: [
        {
          name: 'light',
          value: '#ffffff',
        },
        {
          name: 'dark',
          value: '#0f172a',
        },
        {
          name: 'gray',
          value: '#f3f4f6',
        },
      ],
    },
    docs: {
      theme: {
        base: 'light',
        brandTitle: 'Foundry Design System',
        brandUrl: 'https://Foundry.dev',
      },
    },
  },
  decorators: [
    (Story, context) => {
      // Apply dark mode based on background
      React.useEffect(() => {
        const isDark = context.globals.backgrounds?.value === '#0f172a';
        if (isDark) {
          document.documentElement.classList.add('dark');
        } else {
          document.documentElement.classList.remove('dark');
        }
      }, [context.globals.backgrounds]);

      return (
        <ToastProvider position="top-right">
          <div className="min-h-screen p-8">
            <Story />
          </div>
        </ToastProvider>
      );
    },
  ],
  globalTypes: {
    theme: {
      description: 'Global theme for components',
      defaultValue: 'light',
      toolbar: {
        title: 'Theme',
        icon: 'paintbrush',
        items: ['light', 'dark', 'system'],
        dynamicTitle: true,
      },
    },
  },
};

export default preview;
```

### Manager Configuration (.storybook/manager.ts)

```typescript
import { addons } from '@storybook/addons';
import { themes } from '@storybook/theming';

addons.setConfig({
  theme: {
    ...themes.light,
    brandTitle: 'Foundry Design System',
    brandUrl: 'https://Foundry.dev',
    brandTarget: '_self',
  },
  sidebar: {
    showRoots: true,
    collapsedRoots: ['other'],
  },
  toolbar: {
    title: { hidden: false },
    zoom: { hidden: false },
    eject: { hidden: false },
    copy: { hidden: false },
    fullscreen: { hidden: false },
  },
});
```

---

## Writing Stories

### Basic Story Structure

```tsx
// Button.stories.tsx
import type { Meta, StoryObj } from '@storybook/react';
import { Button } from './Button';

const meta: Meta<typeof Button> = {
  title: 'Components/Button',
  component: Button,
  parameters: {
    layout: 'centered',
    docs: {
      description: {
        component: 'A flexible button component with multiple variants and states.',
      },
    },
  },
  tags: ['autodocs'],
  argTypes: {
    variant: {
      control: 'select',
      options: ['primary', 'secondary', 'success', 'danger', 'warning', 'ghost', 'outline', 'link'],
      description: 'The visual style variant of the button',
    },
    size: {
      control: 'radio',
      options: ['xs', 'sm', 'md', 'lg', 'xl'],
      description: 'The size of the button',
    },
    disabled: {
      control: 'boolean',
      description: 'Whether the button is disabled',
    },
    loading: {
      control: 'boolean',
      description: 'Whether the button is in loading state',
    },
    fullWidth: {
      control: 'boolean',
      description: 'Whether the button should take full width',
    },
    onClick: { action: 'clicked' },
  },
};

export default meta;
type Story = StoryObj<typeof meta>;

// Default story
export const Default: Story = {
  args: {
    children: 'Button',
    variant: 'primary',
    size: 'md',
  },
};

// Variants
export const AllVariants: Story = {
  render: () => (
    <div className="flex flex-wrap gap-4">
      <Button variant="primary">Primary</Button>
      <Button variant="secondary">Secondary</Button>
      <Button variant="success">Success</Button>
      <Button variant="danger">Danger</Button>
      <Button variant="warning">Warning</Button>
      <Button variant="ghost">Ghost</Button>
      <Button variant="outline">Outline</Button>
      <Button variant="link">Link</Button>
    </div>
  ),
};

// Sizes
export const AllSizes: Story = {
  render: () => (
    <div className="flex items-center gap-4">
      <Button size="xs">Extra Small</Button>
      <Button size="sm">Small</Button>
      <Button size="md">Medium</Button>
      <Button size="lg">Large</Button>
      <Button size="xl">Extra Large</Button>
    </div>
  ),
};

// With Icons
export const WithIcons: Story = {
  render: () => (
    <div className="flex gap-4">
      <Button leftIcon={<DownloadIcon />}>Download</Button>
      <Button rightIcon={<ArrowRightIcon />}>Continue</Button>
      <Button leftIcon={<SaveIcon />} rightIcon={<CheckIcon />}>
        Save & Continue
      </Button>
    </div>
  ),
};

// States
export const States: Story = {
  render: () => (
    <div className="flex gap-4">
      <Button>Normal</Button>
      <Button disabled>Disabled</Button>
      <Button loading>Loading</Button>
      <Button loading loadingText="Processing...">
        Custom Loading
      </Button>
    </div>
  ),
};

// Interactive story with controls
export const Playground: Story = {
  args: {
    children: 'Click me',
    variant: 'primary',
    size: 'md',
    disabled: false,
    loading: false,
    fullWidth: false,
  },
};
```

### Complex Component Stories

```tsx
// Table.stories.tsx
import type { Meta, StoryObj } from '@storybook/react';
import { Table } from './Table';
import { Badge } from './Badge';

const meta: Meta<typeof Table> = {
  title: 'Components/Table',
  component: Table,
  parameters: {
    layout: 'padded',
  },
  tags: ['autodocs'],
};

export default meta;
type Story = StoryObj<typeof meta>;

// Sample data
const generateData = (count: number) => {
  return Array.from({ length: count }, (_, i) => ({
    id: i + 1,
    name: `User ${i + 1}`,
    email: `user${i + 1}@example.com`,
    role: ['Admin', 'User', 'Guest'][i % 3],
    status: ['Active', 'Inactive', 'Pending'][i % 3],
    joinDate: new Date(2024, 0, i + 1).toLocaleDateString(),
  }));
};

const columns = [
  { key: 'id', label: 'ID', sortable: true },
  { key: 'name', label: 'Name', sortable: true },
  { key: 'email', label: 'Email' },
  { key: 'role', label: 'Role', sortable: true },
  {
    key: 'status',
    label: 'Status',
    render: (value: string) => (
      <Badge
        variant={
          value === 'Active' ? 'success' :
          value === 'Inactive' ? 'danger' :
          'warning'
        }
      >
        {value}
      </Badge>
    ),
  },
  { key: 'joinDate', label: 'Join Date', sortable: true },
];

export const Default: Story = {
  args: {
    columns,
    data: generateData(10),
  },
};

export const WithPagination: Story = {
  args: {
    columns,
    data: generateData(100),
    paginated: true,
    pageSize: 10,
    pageSizeOptions: [5, 10, 25, 50],
  },
};

export const WithSelection: Story = {
  args: {
    columns,
    data: generateData(10),
    selectable: true,
  },
};

export const Interactive: Story = {
  args: {
    columns,
    data: generateData(50),
    striped: true,
    hoverable: true,
    sortable: true,
    filterable: true,
    paginated: true,
    selectable: true,
    exportable: true,
  },
};

export const Loading: Story = {
  args: {
    columns,
    data: [],
    loading: true,
  },
};

export const Empty: Story = {
  args: {
    columns,
    data: [],
    emptyMessage: 'No data available',
  },
};
```

### MDX Documentation

```mdx
// Button.stories.mdx
import { Meta, Story, Canvas, ArgsTable } from '@storybook/addon-docs';
import { Button } from './Button';

<Meta title="Components/Button/Documentation" component={Button} />

# Button Component

The Button component is a fundamental UI element that triggers actions when clicked.

## Usage

```tsx
import { Button } from '@Foundry/design-system';

function App() {
  return (
    <Button variant="primary" onClick={() => console.log('Clicked')}>
      Click me
    </Button>
  );
}
```

## Variants

<Canvas>
  <Story name="Variants">
    <div className="flex gap-4">
      <Button variant="primary">Primary</Button>
      <Button variant="secondary">Secondary</Button>
      <Button variant="success">Success</Button>
      <Button variant="danger">Danger</Button>
    </div>
  </Story>
</Canvas>

## Props

<ArgsTable of={Button} />

## Accessibility

- All buttons have proper ARIA labels
- Keyboard navigation support (Tab, Enter, Space)
- Focus indicators for keyboard users
- Disabled state properly communicated

## Best Practices

1. Use descriptive button text
2. Choose appropriate variants for context
3. Provide loading states for async actions
4. Always include onClick handlers
```

---

## Addons

### Essential Addons Configuration

```typescript
// .storybook/main.ts addons section
addons: [
  // Controls - Auto-generate controls based on args
  '@storybook/addon-controls',

  // Actions - Log actions in the panel
  '@storybook/addon-actions',

  // Viewport - Preview in different screen sizes
  '@storybook/addon-viewport',

  // Backgrounds - Switch between backgrounds
  '@storybook/addon-backgrounds',

  // Docs - Auto-generate documentation
  '@storybook/addon-docs',

  // A11y - Accessibility testing
  '@storybook/addon-a11y',

  // Measure - Measure distances
  '@storybook/addon-measure',

  // Outline - Show component boundaries
  '@storybook/addon-outline',

  // Interactions - Test user interactions
  '@storybook/addon-interactions',

  // Coverage - Code coverage reports
  '@storybook/addon-coverage',
],
```

### Custom Addon Example

```typescript
// .storybook/theme-addon.ts
import { makeDecorator } from '@storybook/addons';

export const withTheme = makeDecorator({
  name: 'withTheme',
  parameterName: 'theme',
  wrapper: (storyFn, context, { parameters }) => {
    const theme = parameters || 'light';

    React.useEffect(() => {
      document.documentElement.className = theme;
    }, [theme]);

    return storyFn(context);
  },
});
```

---

## Running Storybook

### Development

```bash
# Start Storybook dev server
npm run storybook

# Default runs on http://localhost:6006
```

### Build for Production

```bash
# Build static Storybook
npm run build-storybook

# Output in storybook-static/
```

### Testing

```bash
# Run interaction tests
npm run test-storybook

# With coverage
npm run test-storybook --coverage

# Watch mode
npm run test-storybook --watch
```

### Scripts (package.json)

```json
{
  "scripts": {
    "storybook": "storybook dev -p 6006",
    "build-storybook": "storybook build",
    "test-storybook": "test-storybook",
    "chromatic": "chromatic --project-token=<token>",
    "storybook:clean": "rm -rf storybook-static",
    "storybook:serve": "npx http-server storybook-static"
  }
}
```

---

## Deployment

### Vercel Deployment

```json
// vercel.json
{
  "buildCommand": "npm run build-storybook",
  "outputDirectory": "storybook-static",
  "framework": null
}
```

### Netlify Deployment

```toml
# netlify.toml
[build]
  command = "npm run build-storybook"
  publish = "storybook-static"

[[headers]]
  for = "/*"
  [headers.values]
    X-Frame-Options = "SAMEORIGIN"
    X-Content-Type-Options = "nosniff"
```

### GitHub Pages

```yaml
# .github/workflows/storybook.yml
name: Deploy Storybook

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Setup Node
        uses: actions/setup-node@v3
        with:
          node-version: '18'

      - name: Install dependencies
        run: npm ci

      - name: Build Storybook
        run: npm run build-storybook

      - name: Deploy to GitHub Pages
        uses: peaceiris/actions-gh-pages@v3
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: ./storybook-static
```

### Docker Deployment

```dockerfile
# Dockerfile
FROM node:18-alpine AS builder

WORKDIR /app
COPY package*.json ./
RUN npm ci

COPY . .
RUN npm run build-storybook

FROM nginx:alpine
COPY --from=builder /app/storybook-static /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

---

## Best Practices

### Story Organization

```
stories/
├── Introduction.stories.mdx
├── Colors.stories.mdx
├── Typography.stories.mdx
├── Spacing.stories.mdx
└── Icons.stories.mdx

components/
├── Button/
│   ├── Button.tsx
│   ├── Button.stories.tsx
│   └── Button.test.tsx
├── Input/
│   ├── Input.tsx
│   ├── Input.stories.tsx
│   └── Input.test.tsx
└── ...
```

### Naming Conventions

```tsx
// Component stories file
ComponentName.stories.tsx

// Story names
export const Default: Story = {};
export const WithProps: Story = {};
export const AllVariants: Story = {};
export const Playground: Story = {};

// Category structure
title: 'Category/Subcategory/ComponentName'
```

### Story Templates

```tsx
// Create a template for similar stories
const Template: StoryFn<typeof Button> = (args) => <Button {...args} />;

export const Primary = Template.bind({});
Primary.args = {
  variant: 'primary',
  children: 'Primary Button',
};

export const Secondary = Template.bind({});
Secondary.args = {
  variant: 'secondary',
  children: 'Secondary Button',
};
```

### Testing in Stories

```tsx
// Add interaction tests
export const ClickTest: Story = {
  args: {
    children: 'Click me',
  },
  play: async ({ canvasElement }) => {
    const canvas = within(canvasElement);
    const button = canvas.getByRole('button');

    await userEvent.click(button);
    await expect(button).toHaveFocus();
  },
};
```

### Performance Tips

1. **Lazy Load Stories**: Use dynamic imports for large components
2. **Optimize Assets**: Compress images and use appropriate formats
3. **Code Splitting**: Split stories by category
4. **Use Production Build**: Always deploy production builds
5. **CDN for Assets**: Host static assets on CDN

### Accessibility Testing

```tsx
// In preview.tsx
export const parameters = {
  a11y: {
    element: '#storybook-root',
    config: {
      rules: [
        {
          id: 'color-contrast',
          enabled: true,
        },
      ],
    },
    options: {
      checks: { 'color-contrast': { options: { noScroll: true } } },
      restoreScroll: true,
    },
  },
};
```

---

## Troubleshooting

### Common Issues

**Issue**: Styles not loading
```typescript
// In preview.tsx
import '../src/styles/globals.css';
import 'tailwindcss/tailwind.css';
```

**Issue**: Dark mode not working
```typescript
// Add to decorator
useEffect(() => {
  const isDark = context.globals.theme === 'dark';
  document.documentElement.classList.toggle('dark', isDark);
}, [context.globals.theme]);
```

**Issue**: Build errors
```bash
# Clear cache and rebuild
rm -rf node_modules storybook-static
npm install
npm run build-storybook
```

**Issue**: Slow performance
```typescript
// Optimize stories
features: {
  buildStoriesJson: true,
  storyStoreV7: true,
}
```

---

## Resources

- [Storybook Documentation](https://storybook.js.org/docs/react/get-started/introduction)
- [Storybook Addons](https://storybook.js.org/addons)
- [Component Story Format](https://storybook.js.org/docs/react/api/csf)
- [Chromatic](https://www.chromatic.com/)
- [Storybook Design System](https://storybook.js.org/design-system)

---

*Happy Storybook Development! 🎨*