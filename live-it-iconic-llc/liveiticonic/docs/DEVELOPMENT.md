# Development Guide

Comprehensive guide for developing LiveItIconic - the AI-powered product launch platform and e-commerce application.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Environment](#development-environment)
- [Project Structure](#project-structure)
- [Code Standards](#code-standards)
- [Working with Features](#working-with-features)
- [Testing](#testing)
- [Debugging](#debugging)
- [Common Tasks](#common-tasks)
- [Troubleshooting](#troubleshooting)

## Getting Started

### Prerequisites

- **Node.js**: v18+ (https://nodejs.org)
- **npm**: v9+ (comes with Node.js)
- **Git**: Latest version
- **Code Editor**: VS Code recommended

### Initial Setup

```bash
# Clone repository
git clone https://github.com/yourusername/live-it-iconic.git
cd live-it-iconic

# Install dependencies
npm install

# Create environment file
cp .env.example .env.local

# Start development server
npm run dev
```

### Environment Variables

Create `.env.local` in project root:

```env
# Supabase Configuration
VITE_SUPABASE_URL=your_supabase_url
VITE_SUPABASE_ANON_KEY=your_supabase_key

# Stripe Configuration
VITE_STRIPE_PUBLIC_KEY=your_stripe_key

# API Configuration
VITE_API_URL=http://localhost:3000/api

# Feature Flags
VITE_ENABLE_AI_AGENTS=true
VITE_ENABLE_PWA=true

# Analytics
VITE_ANALYTICS_ID=your_analytics_id
```

## Development Environment

### Available Scripts

```bash
# Start development server
npm run dev

# Build for production
npm run build

# Run tests
npm run test

# Run tests with coverage
npm run test:coverage

# Lint code
npm run lint

# Format code with Prettier
npm run format

# Type check
npm run type-check
```

### Recommended VS Code Extensions

- **ES7+ React/Redux/React-Native snippets** - dsznajder.es7-react-js-snippets
- **Prettier** - esbenp.prettier-vscode
- **ESLint** - dbaeumer.vscode-eslint
- **TypeScript Vue Plugin** - Vue.volar
- **Tailwind CSS IntelliSense** - bradlc.vscode-tailwindcss
- **Thunder Client** - rangav.vscode-thunder-client

### VS Code Settings

Add to `.vscode/settings.json`:

```json
{
  "editor.defaultFormatter": "esbenp.prettier-vscode",
  "editor.formatOnSave": true,
  "editor.codeActionsOnSave": {
    "source.fixAll.eslint": true
  },
  "[typescript]": {
    "editor.defaultFormatter": "esbenp.prettier-vscode"
  }
}
```

## Project Structure

```
live-it-iconic-e3e1196b/
├── src/
│   ├── components/          # React components
│   │   ├── admin/          # Admin interface
│   │   ├── checkout/       # Checkout flow
│   │   ├── product/        # Product display
│   │   ├── ui/             # Reusable UI components
│   │   └── ...
│   ├── contexts/           # React Context providers
│   │   ├── AuthContext.tsx
│   │   └── CartContext.tsx
│   ├── hooks/              # Custom React hooks
│   │   ├── useLocalStorage.ts
│   │   ├── use-cart.ts
│   │   └── ...
│   ├── services/           # API services
│   │   ├── authService.ts
│   │   ├── productService.ts
│   │   ├── orderService.ts
│   │   └── ...
│   ├── pages/              # Page components
│   │   ├── HomePage.tsx
│   │   ├── ProductPage.tsx
│   │   ├── CheckoutPage.tsx
│   │   └── ...
│   ├── launch-platform/    # AI agent system
│   │   ├── agents/        # Agent implementations
│   │   ├── core/          # Core orchestration
│   │   ├── types/         # Type definitions
│   │   └── ...
│   ├── lib/               # Utilities and helpers
│   │   ├── utils.ts       # Common utilities
│   │   └── supabase.ts    # Supabase client
│   ├── types/             # TypeScript types
│   │   ├── product.ts
│   │   ├── order.ts
│   │   ├── cart.ts
│   │   └── ...
│   ├── styles/            # Global styles
│   ├── App.tsx            # Root component
│   └── main.tsx           # Entry point
├── docs/                  # Documentation
│   ├── API.md
│   ├── ARCHITECTURE.md
│   └── ...
├── public/               # Static assets
├── .env.example         # Environment template
├── package.json         # Dependencies
├── tsconfig.json        # TypeScript config
├── vite.config.ts       # Vite config
└── README.md            # Project README
```

## Code Standards

### TypeScript

- Use strict mode: `"strict": true`
- Always type function parameters and returns
- Use interfaces for object types
- Avoid `any` type - use `unknown` if necessary
- Use enums for constants

```typescript
// Good
interface User {
  id: string;
  email: string;
  name?: string;
}

function getUser(id: string): Promise<User> {
  // Implementation
}

// Avoid
function getUser(id: any): any {
  // Implementation
}
```

### Component Structure

```typescript
/**
 * ComponentName
 *
 * Brief description of what component does
 * @component
 * @example
 * const { user } = await fetchUser();
 * return <ComponentName user={user} />
 */

import React from 'react';
import { ComponentProps } from './types';

interface Props {
  title: string;
  onSubmit: (data: FormData) => void;
  isLoading?: boolean;
}

export const ComponentName: React.FC<Props> = ({
  title,
  onSubmit,
  isLoading = false
}) => {
  const [state, setState] = React.useState('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSubmit({ /* data */ });
  };

  return (
    <div>
      <h1>{title}</h1>
      <form onSubmit={handleSubmit}>
        {/* Form content */}
      </form>
    </div>
  );
};

ComponentName.displayName = 'ComponentName';
```

### File Naming

- **Components**: PascalCase (`ProductCard.tsx`)
- **Hooks**: camelCase with `use` prefix (`useLocalStorage.ts`)
- **Services**: camelCase with suffix (`authService.ts`)
- **Types**: PascalCase (`Product.ts`)
- **Constants**: UPPER_SNAKE_CASE (`API_ENDPOINTS.ts`)
- **Utilities**: camelCase (`formatPrice.ts`)

### Imports Organization

```typescript
// 1. React and external libraries
import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';

// 2. Internal components
import { Button } from '@/components/ui/Button';
import { Header } from '@/components/layout/Header';

// 3. Hooks and contexts
import { useAuth } from '@/contexts/AuthContext';
import { useLocalStorage } from '@/hooks/useLocalStorage';

// 4. Services
import { authService } from '@/services/authService';

// 5. Types and utilities
import { User } from '@/types/user';
import { formatPrice } from '@/lib/utils';

// 6. Styles
import styles from './Component.module.css';
```

### Naming Conventions

```typescript
// Variables and functions: camelCase
const userName = 'John';
const getUserById = (id: string) => { /* ... */ };

// Classes and interfaces: PascalCase
class UserService { /* ... */ }
interface UserProfile { /* ... */ }

// Constants: UPPER_SNAKE_CASE
const MAX_RETRIES = 3;
const API_TIMEOUT = 5000;

// React components: PascalCase
const UserProfile = () => <div />;

// Event handlers: on + PascalCase
const handleSubmit = () => { /* ... */ };
const onUserChange = () => { /* ... */ };
```

## Working with Features

### Feature Branches

```bash
# Create feature branch
git checkout -b feature/user-authentication

# Make changes and commit
git add .
git commit -m "feat(auth): implement user authentication"

# Push to remote
git push -u origin feature/user-authentication

# Create Pull Request on GitHub
```

### Commit Message Format

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types**:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Code style (formatting, missing semicolons, etc)
- `refactor`: Code refactoring
- `perf`: Performance improvement
- `test`: Test addition/modification
- `chore`: Build process, dependencies

**Examples**:
```
feat(auth): add email verification
fix(cart): resolve quantity update issue
docs(api): update endpoint documentation
style(components): format button styles
refactor(services): simplify error handling
test(hooks): add useLocalStorage tests
```

### Working with Agents

See [Agent Documentation](../src/launch-platform/agents/README.md) for detailed agent development guide.

### Working with Services

See [Services Documentation](../src/services/README.md) for service patterns and best practices.

### Working with Hooks

See [Hooks Documentation](../src/hooks/README.md) for custom hook development.

## Testing

### Test Structure

Tests should be placed alongside the code they test:

```
src/
├── components/
│   ├── Button.tsx
│   ├── Button.test.tsx      # Component test
│   └── Button.stories.tsx   # Storybook story
├── hooks/
│   ├── useLocalStorage.ts
│   └── useLocalStorage.test.ts
├── services/
│   ├── authService.ts
│   └── authService.test.ts
```

### Writing Tests

```typescript
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { Button } from './Button';

describe('Button Component', () => {
  it('should render button with text', () => {
    render(<Button>Click me</Button>);
    expect(screen.getByRole('button', { name: /click me/i })).toBeInTheDocument();
  });

  it('should call onClick handler when clicked', async () => {
    const handleClick = jest.fn();
    render(<Button onClick={handleClick}>Click</Button>);

    await userEvent.click(screen.getByRole('button'));
    expect(handleClick).toHaveBeenCalledTimes(1);
  });

  it('should be disabled when disabled prop is true', () => {
    render(<Button disabled>Click</Button>);
    expect(screen.getByRole('button')).toBeDisabled();
  });
});
```

### Running Tests

```bash
# Run all tests
npm run test

# Run specific test file
npm run test -- Button.test.ts

# Run tests in watch mode
npm run test -- --watch

# Generate coverage report
npm run test:coverage
```

### Coverage Targets

- **Statements**: 80%+
- **Branches**: 75%+
- **Functions**: 80%+
- **Lines**: 80%+

## Debugging

### Browser DevTools

1. Open Chrome DevTools: `F12` or `Ctrl+Shift+I`
2. Go to **Sources** tab
3. Set breakpoints in code
4. Step through execution

### VS Code Debugging

Add `.vscode/launch.json`:

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "type": "chrome",
      "request": "launch",
      "name": "Launch Chrome",
      "url": "http://localhost:5173",
      "webRoot": "${workspaceFolder}",
      "sourceMapPathOverride": {
        "webpack://*": "${webRoot}/*"
      }
    }
  ]
}
```

### Console Logging

```typescript
// Use descriptive labels
console.log('[Component] Rendering with props:', props);
console.error('[Service] Failed to fetch user:', error);

// Avoid
console.log('data');
console.log(error);
```

### React Developer Tools

- Install Chrome extension: [React Developer Tools](https://chrome.google.com/webstore)
- Inspect components, props, and state
- Track component renders

## Common Tasks

### Add New Component

```bash
# Create component directory
mkdir src/components/NewComponent

# Create component file
cat > src/components/NewComponent/NewComponent.tsx << 'EOF'
/**
 * NewComponent
 *
 * @component
 */

import React from 'react';

interface Props {
  title: string;
}

export const NewComponent: React.FC<Props> = ({ title }) => {
  return <div>{title}</div>;
};

NewComponent.displayName = 'NewComponent';
EOF

# Create component test
cat > src/components/NewComponent/NewComponent.test.tsx << 'EOF'
import { render, screen } from '@testing-library/react';
import { NewComponent } from './NewComponent';

describe('NewComponent', () => {
  it('should render with title', () => {
    render(<NewComponent title="Test" />);
    expect(screen.getByText('Test')).toBeInTheDocument();
  });
});
EOF

# Create index file for easier imports
cat > src/components/NewComponent/index.ts << 'EOF'
export { NewComponent } from './NewComponent';
EOF
```

### Add New Hook

```bash
# Create hook file
cat > src/hooks/useNewHook.ts << 'EOF'
/**
 * useNewHook
 *
 * Description of what the hook does
 *
 * @template T - Type of value
 * @param {string} key - Hook parameter description
 * @returns {Object} Object containing hook values and functions
 *
 * @example
 * const value = useNewHook('key');
 */

import { useState } from 'react';

export function useNewHook(key: string) {
  const [value, setValue] = useState('');

  return { value, setValue };
}
EOF

# Create test
cat > src/hooks/useNewHook.test.ts << 'EOF'
import { renderHook, act } from '@testing-library/react';
import { useNewHook } from './useNewHook';

describe('useNewHook', () => {
  it('should initialize with default value', () => {
    const { result } = renderHook(() => useNewHook('test'));
    expect(result.current.value).toBe('');
  });
});
EOF
```

### Add New Service

```bash
# Create service file
cat > src/services/newService.ts << 'EOF'
/**
 * New Service
 *
 * Description of service purpose
 *
 * @module newService
 */

interface NewParams {
  id: string;
  name: string;
}

/**
 * New Service
 *
 * @object newService
 */
export const newService = {
  /**
   * Service method description
   *
   * @async
   * @param {NewParams} params - Method parameters
   * @returns {Promise<object>} Result data
   */
  async method(params: NewParams): Promise<any> {
    try {
      const response = await fetch('/api/endpoint', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(params)
      });

      if (!response.ok) throw new Error('Request failed');

      return await response.json();
    } catch (error) {
      console.error('Service method failed:', error);
      throw error;
    }
  }
};
EOF

# Create test
cat > src/services/newService.test.ts << 'EOF'
import { newService } from './newService';

describe('newService', () => {
  it('should call API endpoint', async () => {
    const result = await newService.method({ id: '1', name: 'Test' });
    expect(result).toBeDefined();
  });
});
EOF
```

### Database Queries

Use Supabase client directly in services:

```typescript
import { supabaseClient } from '@/lib/supabase';

export const productService = {
  async getProduct(id: string) {
    const { data, error } = await supabaseClient
      .from('products')
      .select('*')
      .eq('id', id)
      .single();

    if (error) throw error;
    return data;
  }
};
```

## Troubleshooting

### Port Already in Use

```bash
# Kill process on port 5173
kill -9 $(lsof -t -i:5173)

# Or use different port
npm run dev -- --port 3000
```

### Module Not Found

```bash
# Clear node_modules and reinstall
rm -rf node_modules package-lock.json
npm install

# Clear Vite cache
rm -rf .vite
```

### Type Errors

```bash
# Run type check
npm run type-check

# Fix type issues in TypeScript strict mode
# Look for red squiggly lines in editor
```

### Build Failures

```bash
# Check for unused imports and variables
npm run lint

# Fix auto-fixable issues
npm run lint -- --fix

# Type check before building
npm run type-check && npm run build
```

### Network Issues

```bash
# Check environment variables
cat .env.local

# Verify API endpoints are accessible
# Check browser Network tab for failed requests
# Check server logs for errors

# Clear browser cache
# Open DevTools > Storage > Clear Site Data
```

## Performance Tips

### Code Splitting

```typescript
// Use dynamic imports for large components
const AdminDashboard = lazy(() => import('@/pages/AdminDashboard'));

<Suspense fallback={<LoadingSpinner />}>
  <AdminDashboard />
</Suspense>
```

### Memoization

```typescript
import { useMemo, useCallback } from 'react';

export const Component = ({ users, onSelect }) => {
  // Memoize expensive calculations
  const sortedUsers = useMemo(
    () => users.sort((a, b) => a.name.localeCompare(b.name)),
    [users]
  );

  // Memoize callbacks to prevent unnecessary re-renders
  const handleSelect = useCallback(
    (user) => onSelect(user),
    [onSelect]
  );

  return <UserList users={sortedUsers} onSelect={handleSelect} />;
};
```

### Image Optimization

```typescript
// Use next-gen formats
<picture>
  <source srcSet="image.webp" type="image/webp" />
  <source srcSet="image.jpg" type="image/jpeg" />
  <img src="image.jpg" alt="Description" loading="lazy" />
</picture>
```

## Related Documentation

- [Architecture Guide](./ARCHITECTURE.md)
- [API Documentation](./API.md)
- [Agent Documentation](../src/launch-platform/agents/README.md)
- [Services Documentation](../src/services/README.md)
- [Component Documentation](../src/components/README.md)
- [TypeScript Handbook](https://www.typescriptlang.org/docs/)
- [React Documentation](https://react.dev)
- [Vite Documentation](https://vitejs.dev)

---

For questions or issues, refer to the troubleshooting section or open a GitHub issue.
