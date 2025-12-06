# Development Guide

## Getting Started

### Prerequisites

- Node.js 20+ 
- npm or yarn
- Git

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd qmlab

# Install dependencies
npm install

# Start development server
npm run dev
```

## Available Scripts

```bash
# Development
npm run dev          # Start dev server (http://localhost:5173)
npm run build        # Production build
npm run preview      # Preview production build

# Code Quality
npm run lint         # Run ESLint
npm run lint:fix     # Fix ESLint issues
npm run format       # Format with Prettier
npm run typecheck    # TypeScript type checking
npm run style:lint   # Lint CSS files

# Testing
npm test             # Run unit tests
npm run test:watch   # Run tests in watch mode
npm run test:a11y    # Run accessibility tests
npm run test:e2e     # Run end-to-end tests
```

## Code Style

### JavaScript/TypeScript

- **Quotes**: Single quotes
- **Semicolons**: Required
- **Line width**: 100 characters
- **Trailing commas**: Always
- **Indentation**: 2 spaces

### File Naming

- **Components**: PascalCase (e.g., `ButtonPrimary.tsx`)
- **Hooks**: camelCase with `use` prefix (e.g., `use-mobile.tsx`)
- **Utils**: kebab-case (e.g., `format-date.ts`)
- **Tests**: Same as file + `.test.ts(x)` (e.g., `ButtonPrimary.test.tsx`)
- **Types**: PascalCase (e.g., `UserProfile.ts`)

### Import Order

1. React and external libraries
2. Internal modules (by depth)
3. Components
4. Hooks
5. Utils
6. Types
7. Styles

```typescript
// Good
import React from 'react';
import { useQuery } from '@tanstack/react-query';

import { Button } from '@/components/ui/button';
import { useMobile } from '@/hooks/use-mobile';
import { cn } from '@/lib/utils';
import type { UserProfile } from '@/types';

import './styles.css';
```

### Component Structure

```typescript
// 1. Imports
import React from 'react';
import { cn } from '@/lib/utils';

// 2. Types
interface ComponentProps {
  title: string;
  // ...
}

// 3. Component
export function Component({ title }: ComponentProps) {
  // 3a. Hooks
  const [state, setState] = React.useState();
  
  // 3b. Handlers
  const handleClick = () => {
    // ...
  };
  
  // 3c. Render
  return (
    <div className={cn('...')}>
      {title}
    </div>
  );
}
```

## Git Workflow

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```bash
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Code style (formatting, etc.)
- `refactor`: Code refactoring
- `perf`: Performance improvement
- `test`: Adding/updating tests
- `build`: Build system changes
- `ci`: CI configuration
- `chore`: Other changes
- `a11y`: Accessibility improvements

**Examples:**
```bash
feat(circuit): Add quantum gate drag-and-drop
fix(bloch): Correct sphere rotation calculation
docs(readme): Update installation instructions
a11y(button): Improve keyboard navigation
```

### Branch Naming

```bash
<type>/<description>

# Examples
feat/quantum-gate-library
fix/bloch-sphere-rendering
docs/api-documentation
refactor/component-structure
```

### Pull Requests

1. Create a feature branch
2. Make changes with conventional commits
3. Run quality checks: `npm run lint && npm run typecheck && npm test`
4. Push and create PR
5. Wait for CI checks and review
6. Squash and merge

## Testing

### Unit Tests

```typescript
import { render, screen } from '@testing-library/react';
import { Button } from './Button';

describe('Button', () => {
  it('renders with text', () => {
    render(<Button>Click me</Button>);
    expect(screen.getByText('Click me')).toBeInTheDocument();
  });
});
```

### Accessibility Tests

```typescript
import { axe, toHaveNoViolations } from 'jest-axe';
expect.extend(toHaveNoViolations);

it('should have no accessibility violations', async () => {
  const { container } = render(<Button>Click me</Button>);
  const results = await axe(container);
  expect(results).toHaveNoViolations();
});
```

### E2E Tests (Playwright)

```typescript
import { test, expect } from '@playwright/test';

test('quantum circuit creation', async ({ page }) => {
  await page.goto('/');
  await page.click('[data-testid="new-circuit"]');
  await expect(page.locator('.circuit-canvas')).toBeVisible();
});
```

## Debugging

### React DevTools

Install [React Developer Tools](https://react.dev/learn/react-developer-tools) for component inspection.

### Logger

Use centralized logger instead of `console.log`:

```typescript
import { logger } from '@/lib/logger';

logger.info('User action', { userId: 123 });
logger.error('Failed to load', { error });
logger.debug('State update', { state });
```

### Performance

- Use React DevTools Profiler
- Check Lighthouse scores
- Monitor Core Web Vitals
- Use `useCallback` and `useMemo` judiciously

## Troubleshooting

### Port already in use

```bash
# Kill process on port 5173
npx kill-port 5173

# Or use different port
npm run dev -- --port 3000
```

### Type errors after dependency update

```bash
# Clear TypeScript cache
rm -rf node_modules/.cache

# Reinstall dependencies
rm -rf node_modules
npm install
```

### Build fails

```bash
# Clean build artifacts
rm -rf dist

# Rebuild
npm run build
```

## Best Practices

1. **Write tests** for new features and bug fixes
2. **Use TypeScript** strictly (avoid `any`)
3. **Follow accessibility** guidelines (WCAG 2.1 AA)
4. **Optimize performance** (lazy loading, code splitting)
5. **Document complex logic** with comments
6. **Keep components small** and focused
7. **Use semantic HTML** elements
8. **Validate user input** on client and server
9. **Handle errors gracefully** with boundaries
10. **Review your own PR** before requesting review

## Resources

- [React Documentation](https://react.dev/)
- [TypeScript Handbook](https://www.typescriptlang.org/docs/)
- [Tailwind CSS](https://tailwindcss.com/)
- [Radix UI](https://www.radix-ui.com/)
- [React Query](https://tanstack.com/query/latest)
- [WCAG Guidelines](https://www.w3.org/WAI/WCAG21/quickref/)
