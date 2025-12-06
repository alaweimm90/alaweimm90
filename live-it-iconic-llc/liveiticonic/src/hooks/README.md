# Custom Hooks Documentation

Comprehensive documentation for all custom React hooks used throughout the LiveItIconic application.

## Overview

Custom hooks provide reusable logic for state management, side effects, and component utilities. All hooks follow React's hooks rules and are fully typed with TypeScript.

### Available Hooks

```
Custom Hooks
├── useLocalStorage - Persist state to localStorage
├── useCart - Shopping cart state management
├── useAuth - Authentication state and user info
├── useAnalytics - Track events and user behavior
├── useScrollAnimation - Trigger animations on scroll
├── useCanvasAnimation - Handle canvas drawing and animations
├── usePWA - Progressive Web App functionality
├── use-form-field - Form field state management
├── use-sidebar - Sidebar toggle and state
├── use-toast - Toast notifications
├── use-mobile - Mobile responsive detection
└── use-form-field - Form field state management
```

## Core Hooks

### useLocalStorage

**Purpose**: Persist component state to localStorage with automatic serialization

**Location**: `/src/hooks/useLocalStorage.ts`

**Signature**:
```typescript
function useLocalStorage<T>(
  key: string,
  initialValue: T
): [T, (value: T | ((val: T) => T)) => void]
```

**Features**:
- Automatic JSON serialization/deserialization
- Error handling for quota exceeded
- Updater function support (like useState)
- Type-safe with TypeScript generics

**Example Usage**:
```typescript
import { useLocalStorage } from '@/hooks/useLocalStorage';

// Basic usage
const [theme, setTheme] = useLocalStorage('theme', 'dark');

// Update value
setTheme('light');

// With complex objects
interface User {
  id: string;
  email: string;
  preferences: { language: string };
}

const [user, setUser] = useLocalStorage<User>('currentUser', {
  id: '1',
  email: 'user@example.com',
  preferences: { language: 'en' }
});

// Using updater function
setUser(prev => ({
  ...prev,
  preferences: { language: 'fr' }
}));
```

**Browser Compatibility**:
- Works in all modern browsers
- Gracefully handles disabled/unavailable localStorage
- Logs errors but doesn't throw

**Use Cases**:
- User preferences and settings
- Theme selection
- Cart state persistence
- Recently viewed items
- Draft content

---

### useCart

**Purpose**: Manage shopping cart state and operations

**Location**: `/src/hooks/use-cart.ts`

**Signature**:
```typescript
function useCart(): {
  items: CartItem[];
  addItem: (product: Product, quantity: number) => void;
  removeItem: (productId: string) => void;
  updateQuantity: (productId: string, quantity: number) => void;
  clear: () => void;
  total: number;
}
```

**Features**:
- Add/remove/update items
- Calculate totals
- Persist to localStorage
- Discount calculation
- Stock validation

**Example Usage**:
```typescript
import { useCart } from '@/hooks/use-cart';

function ProductCard({ product }) {
  const { addItem, items } = useCart();

  const inCart = items.some(item => item.id === product.id);

  return (
    <button
      onClick={() => addItem(product, 1)}
      disabled={inCart}
    >
      {inCart ? 'In Cart' : 'Add to Cart'}
    </button>
  );
}

// In checkout page
function CheckoutPage() {
  const { items, total, clear } = useCart();

  return (
    <div>
      <h2>Order Total: ${(total / 100).toFixed(2)}</h2>
      <button onClick={() => placeOrder().then(() => clear())}>
        Place Order
      </button>
    </div>
  );
}
```

**Best Practices**:
- Clear cart after successful order
- Validate quantity before adding
- Show stock warnings
- Persist cart for logged-out users

---

### useAuth

**Purpose**: Manage authentication state and user information

**Location**: `/src/hooks/useAuth.ts` (if exists) or use AuthContext

**Signature**:
```typescript
function useAuth(): {
  user: User | null;
  isLoading: boolean;
  isAuthenticated: boolean;
  signIn: (email: string, password: string) => Promise<void>;
  signOut: () => Promise<void>;
  error: Error | null;
}
```

**Features**:
- Get current user
- Sign in/out
- Loading states
- Error handling
- Auto-persistence

**Example Usage**:
```typescript
import { useAuth } from '@/hooks/useAuth';

function AuthGuard({ children }) {
  const { user, isLoading } = useAuth();

  if (isLoading) return <LoadingSpinner />;

  return user ? children : <Redirect to="/login" />;
}

function LoginForm() {
  const { signIn, error, isLoading } = useAuth();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      await signIn(email, password);
      // Redirect happens automatically
    } catch (err) {
      // Error shown in UI
    }
  };

  return (
    <form onSubmit={handleSubmit}>
      {error && <ErrorAlert>{error.message}</ErrorAlert>}
      <input
        type="email"
        value={email}
        onChange={(e) => setEmail(e.target.value)}
      />
      <input
        type="password"
        value={password}
        onChange={(e) => setPassword(e.target.value)}
      />
      <button type="submit" disabled={isLoading}>
        {isLoading ? 'Signing in...' : 'Sign In'}
      </button>
    </form>
  );
}
```

---

### useAnalytics

**Purpose**: Track user events and behavior for analytics

**Location**: `/src/hooks/useAnalytics.ts`

**Signature**:
```typescript
function useAnalytics(): {
  trackEvent: (name: string, properties?: Record<string, any>) => void;
  trackPageView: (pageName: string) => void;
  setUserId: (userId: string) => void;
}
```

**Features**:
- Event tracking
- Page view tracking
- User identification
- Custom properties
- Error handling

**Example Usage**:
```typescript
import { useAnalytics } from '@/hooks/useAnalytics';

function ProductPage({ product }) {
  const { trackEvent, trackPageView } = useAnalytics();

  useEffect(() => {
    trackPageView(`/products/${product.id}`);
    trackEvent('product_viewed', {
      productId: product.id,
      productName: product.name,
      price: product.price
    });
  }, [product.id]);

  const handleAddToCart = () => {
    trackEvent('add_to_cart', {
      productId: product.id,
      quantity: 1
    });
    addToCart(product);
  };

  return (
    <button onClick={handleAddToCart}>
      Add to Cart
    </button>
  );
}
```

**Common Events**:
- `page_view` - Page navigation
- `product_viewed` - Product detail page
- `add_to_cart` - Item added to cart
- `checkout_started` - Checkout initiated
- `purchase_completed` - Order placed
- `search` - Search query
- `error` - Error occurred

---

### useScrollAnimation

**Purpose**: Trigger animations when elements scroll into view

**Location**: `/src/hooks/useScrollAnimation.ts`

**Signature**:
```typescript
function useScrollAnimation(ref: React.RefObject<HTMLElement>): {
  isVisible: boolean;
  animate: boolean;
}
```

**Features**:
- Intersection Observer API
- Automatic trigger on scroll
- Performance optimized
- Reusable animations

**Example Usage**:
```typescript
import { useScrollAnimation } from '@/hooks/useScrollAnimation';

function FeatureSection() {
  const ref = useRef<HTMLDivElement>(null);
  const { isVisible, animate } = useScrollAnimation(ref);

  return (
    <div
      ref={ref}
      className={`transition-all duration-500 ${
        animate ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-10'
      }`}
    >
      <h2>Animated Section</h2>
      <p>This animates when scrolled into view</p>
    </div>
  );
}
```

**CSS Integration**:
```css
.fade-in {
  transition: opacity 0.6s ease-in-out, transform 0.6s ease-in-out;
}

.fade-in.active {
  opacity: 1;
  transform: translateY(0);
}

.fade-in:not(.active) {
  opacity: 0;
  transform: translateY(20px);
}
```

---

### useCanvasAnimation

**Purpose**: Handle canvas-based drawing and animations

**Location**: `/src/hooks/useCanvasAnimation.ts`

**Signature**:
```typescript
function useCanvasAnimation(canvasRef: React.RefObject<HTMLCanvasElement>): {
  draw: (callback: (ctx: CanvasRenderingContext2D) => void) => void;
  animate: (callback: (ctx: CanvasRenderingContext2D, progress: number) => void) => void;
  clear: () => void;
}
```

**Features**:
- Canvas context management
- Animation frame handling
- Clear and redraw
- Performance optimization

**Example Usage**:
```typescript
import { useCanvasAnimation } from '@/hooks/useCanvasAnimation';

function AnimatedCanvas() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const { animate, clear } = useCanvasAnimation(canvasRef);

  useEffect(() => {
    animate((ctx, progress) => {
      clear();
      ctx.fillStyle = 'rgba(0, 0, 0, 0.1)';
      ctx.fillRect(0, 0, ctx.canvas.width, ctx.canvas.height);

      ctx.fillStyle = 'blue';
      ctx.beginPath();
      ctx.arc(
        ctx.canvas.width / 2 + Math.cos(progress * Math.PI * 2) * 50,
        ctx.canvas.height / 2 + Math.sin(progress * Math.PI * 2) * 50,
        10,
        0,
        Math.PI * 2
      );
      ctx.fill();
    });
  }, [animate, clear]);

  return <canvas ref={canvasRef} width={800} height={600} />;
}
```

---

### usePWA

**Purpose**: Implement Progressive Web App features

**Location**: `/src/hooks/usePWA.ts`

**Signature**:
```typescript
function usePWA(): {
  isInstallable: boolean;
  isPWA: boolean;
  install: () => Promise<void>;
  deferredPrompt: any;
}
```

**Features**:
- Installation prompts
- Offline support
- App shortcuts
- Notification permissions

**Example Usage**:
```typescript
import { usePWA } from '@/hooks/usePWA';

function InstallPrompt() {
  const { isInstallable, install } = usePWA();

  if (!isInstallable) return null;

  return (
    <button onClick={install} className="install-button">
      Install App
    </button>
  );
}
```

---

### use-form-field

**Purpose**: Manage individual form field state and validation

**Location**: `/src/hooks/use-form-field.ts`

**Features**:
- Field state management
- Validation
- Error messages
- Touch tracking

**Example Usage**:
```typescript
import { useFormField } from '@/hooks/use-form-field';

function SignUpForm() {
  const email = useFormField('', (value) => {
    if (!value.includes('@')) return 'Invalid email';
  });
  const password = useFormField('', (value) => {
    if (value.length < 8) return 'Min 8 characters';
  });

  return (
    <form>
      <div>
        <input
          {...email.bind}
          placeholder="Email"
          onBlur={email.markTouched}
        />
        {email.touched && email.error && (
          <span className="error">{email.error}</span>
        )}
      </div>
      <div>
        <input
          {...password.bind}
          type="password"
          placeholder="Password"
          onBlur={password.markTouched}
        />
        {password.touched && password.error && (
          <span className="error">{password.error}</span>
        )}
      </div>
    </form>
  );
}
```

---

### use-sidebar

**Purpose**: Manage sidebar open/close state

**Location**: `/src/hooks/use-sidebar.ts`

**Features**:
- Toggle sidebar
- Close on navigation
- Mobile responsive

**Example Usage**:
```typescript
import { useSidebar } from '@/hooks/use-sidebar';

function Layout() {
  const { isOpen, toggle, close } = useSidebar();

  return (
    <>
      <button onClick={toggle}>Menu</button>
      <aside className={isOpen ? 'open' : 'closed'}>
        <nav>
          <Link to="/" onClick={close}>Home</Link>
          <Link to="/products" onClick={close}>Products</Link>
        </nav>
      </aside>
    </>
  );
}
```

---

### use-toast

**Purpose**: Display toast notifications

**Location**: `/src/hooks/use-toast.ts`

**Features**:
- Toast queue management
- Auto-dismiss
- Different types (success, error, info, warning)
- Stack multiple toasts

**Example Usage**:
```typescript
import { useToast } from '@/hooks/use-toast';

function CheckoutForm() {
  const toast = useToast();

  const handleSubmit = async () => {
    try {
      await submitOrder();
      toast.success('Order placed successfully!');
    } catch (error) {
      toast.error('Failed to place order');
    }
  };

  return <button onClick={handleSubmit}>Place Order</button>;
}
```

---

### use-mobile

**Purpose**: Detect mobile device and responsive breakpoints

**Location**: `/src/hooks/use-mobile.tsx`

**Signature**:
```typescript
function useMediaQuery(query: string): boolean
function useMobile(): boolean
function useTablet(): boolean
function useDesktop(): boolean
```

**Features**:
- Media query matching
- Device type detection
- Breakpoint helpers
- Real-time updates

**Example Usage**:
```typescript
import { useMobile, useTablet } from '@/hooks/use-mobile';

function ResponsiveComponent() {
  const isMobile = useMobile();
  const isTablet = useTablet();

  return (
    <>
      {isMobile && <MobileLayout />}
      {isTablet && !isMobile && <TabletLayout />}
      {!isMobile && !isTablet && <DesktopLayout />}
    </>
  );
}
```

**Breakpoints**:
- Mobile: < 640px
- Tablet: 640px - 1024px
- Desktop: > 1024px

---

## Hooks Best Practices

### 1. Follow Rules of Hooks

```typescript
// Good - unconditional hook calls
function Component() {
  const [state, setState] = useLocalStorage('key', {});
  const { isOpen, toggle } = useSidebar();
  return <div>{state}</div>;
}

// Bad - conditional hook calls
function Component({ shouldUseHook }) {
  if (shouldUseHook) {
    const [state, setState] = useLocalStorage('key', {}); // Wrong!
  }
  return <div />;
}
```

### 2. Use Appropriate Cleanup

```typescript
// Good - cleanup in useEffect
useEffect(() => {
  const handler = () => { /* ... */ };
  window.addEventListener('resize', handler);

  return () => window.removeEventListener('resize', handler);
}, []);

// Bad - no cleanup
useEffect(() => {
  window.addEventListener('resize', () => { /* ... */ });
}, []);
```

### 3. Proper Dependency Arrays

```typescript
// Good - include all dependencies
useEffect(() => {
  const handler = () => { /* uses userId */ };
  window.addEventListener('resize', handler);
  return () => window.removeEventListener('resize', handler);
}, [userId]);

// Bad - missing dependency
useEffect(() => {
  const handler = () => { /* uses userId */ };
  window.addEventListener('resize', handler);
  return () => window.removeEventListener('resize', handler);
}, []); // Error: userId should be in dependency array
```

### 4. Type Safety with TypeScript

```typescript
// Good - explicit types
const [user, setUser] = useLocalStorage<User>('user', defaultUser);
const { items, addItem } = useCart();

// Less safe - inferred types
const [data, setData] = useLocalStorage('key', { id: 1 });
```

### 5. Combine Hooks for Complex Logic

```typescript
// Good - compose hooks
function useProduct(productId: string) {
  const [product, setProduct] = useState<Product | null>(null);
  const { trackEvent } = useAnalytics();
  const { addItem } = useCart();

  useEffect(() => {
    loadProduct(productId).then(setProduct);
    trackEvent('product_viewed', { productId });
  }, [productId]);

  return {
    product,
    addToCart: () => {
      addItem(product, 1);
      trackEvent('add_to_cart', { productId });
    }
  };
}
```

## Testing Hooks

### Using React Testing Library

```typescript
import { renderHook, act } from '@testing-library/react';
import { useLocalStorage } from '@/hooks/useLocalStorage';

test('useLocalStorage persists value', () => {
  const { result } = renderHook(() =>
    useLocalStorage('test-key', 'initial')
  );

  act(() => {
    result.current[1]('updated');
  });

  expect(result.current[0]).toBe('updated');
  expect(localStorage.getItem('test-key')).toBe('"updated"');
});
```

## Related Documentation

- [React Hooks Documentation](https://react.dev/reference/react/hooks)
- [Rules of Hooks](https://react.dev/reference/rules/rules-of-hooks)
- [Building Custom Hooks](https://react.dev/learn/reusing-logic-with-custom-hooks)
- [Component Documentation](../components/README.md)

---

For detailed implementation details, see individual hook files in this directory.
