# Context API Documentation

Comprehensive documentation for React Context providers used for global state management in the LiveItIconic application.

## Overview

Contexts provide a way to pass data through the component tree without having to pass props down manually at every level. The application uses Context API for managing:

- Authentication state and user information
- Shopping cart data and operations
- Global notifications and alerts
- Notifications and toast messages

### Available Contexts

```
Contexts
├── AuthContext - User authentication and profile
└── CartContext - Shopping cart state management
```

## AuthContext

**Purpose**: Manage user authentication state, login/logout, and user profile information globally

**Location**: `/src/contexts/AuthContext.tsx`

**Provides**:
```typescript
interface AuthContextType {
  user: User | null;
  isLoading: boolean;
  isAuthenticated: boolean;
  signIn: (email: string, password: string) => Promise<void>;
  signUp: (email: string, password: string, name?: string) => Promise<void>;
  signOut: () => Promise<void>;
  error: Error | null;
  clearError: () => void;
}
```

### Features

- User authentication state
- Sign in/up/out operations
- User profile information
- Loading and error states
- Automatic persistence to localStorage
- Protected routes support

### Usage

**Wrap your app with AuthProvider**:
```typescript
// src/App.tsx
import { AuthProvider } from '@/contexts/AuthContext';

function App() {
  return (
    <AuthProvider>
      <Router>
        <Routes>
          {/* Your routes */}
        </Routes>
      </Router>
    </AuthProvider>
  );
}

export default App;
```

**Access auth state in components**:
```typescript
import { useAuth } from '@/contexts/AuthContext';

function Profile() {
  const { user, signOut, isLoading } = useAuth();

  if (isLoading) return <LoadingSpinner />;

  return (
    <div>
      <h1>Welcome, {user?.name || user?.email}</h1>
      <button onClick={signOut}>Sign Out</button>
    </div>
  );
}
```

### Use Cases

**Protected Routes**:
```typescript
function ProtectedRoute({ children }: { children: React.ReactNode }) {
  const { isAuthenticated, isLoading } = useAuth();

  if (isLoading) return <LoadingSpinner />;

  return isAuthenticated ? (
    children
  ) : (
    <Navigate to="/login" replace />
  );
}
```

**User Profile Display**:
```typescript
function Header() {
  const { user, isAuthenticated, signOut } = useAuth();

  return (
    <header>
      {isAuthenticated && user ? (
        <>
          <span>Welcome, {user.name}</span>
          <button onClick={signOut}>Sign Out</button>
        </>
      ) : (
        <Link to="/login">Sign In</Link>
      )}
    </header>
  );
}
```

**Sign Up Flow**:
```typescript
function SignUpForm() {
  const { signUp, isLoading, error } = useAuth();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [name, setName] = useState('');
  const navigate = useNavigate();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      await signUp(email, password, name);
      navigate('/dashboard');
    } catch (err) {
      // Error handled in context
    }
  };

  return (
    <form onSubmit={handleSubmit}>
      {error && <ErrorAlert>{error.message}</ErrorAlert>}
      <input
        value={email}
        onChange={(e) => setEmail(e.target.value)}
        placeholder="Email"
        type="email"
      />
      <input
        value={name}
        onChange={(e) => setName(e.target.value)}
        placeholder="Full Name"
      />
      <input
        value={password}
        onChange={(e) => setPassword(e.target.value)}
        placeholder="Password"
        type="password"
      />
      <button type="submit" disabled={isLoading}>
        {isLoading ? 'Creating Account...' : 'Sign Up'}
      </button>
    </form>
  );
}
```

**Auto-Login on App Load**:
```typescript
function AppInitializer() {
  const { user, isLoading } = useAuth();

  useEffect(() => {
    // User is automatically loaded from localStorage
    // during context initialization
    if (!isLoading && user) {
      console.log('Logged in as:', user.email);
    }
  }, [isLoading, user]);

  if (isLoading) return <LoadingSplash />;

  return <App />;
}
```

---

## CartContext

**Purpose**: Manage shopping cart state, items, and operations globally

**Location**: `/src/contexts/CartContext.tsx`

**Provides**:
```typescript
interface CartContextType {
  items: CartItem[];
  itemCount: number;
  subtotal: number;
  tax: number;
  shipping: number;
  total: number;
  addItem: (product: Product, quantity: number) => void;
  removeItem: (productId: string) => void;
  updateQuantity: (productId: string, quantity: number) => void;
  clearCart: () => void;
  applyDiscount: (code: string) => Promise<void>;
  discountCode: string | null;
  discountAmount: number;
}
```

### Features

- Add/remove/update cart items
- Calculate subtotal, tax, shipping, total
- Apply discount codes
- Persist cart to localStorage
- Real-time price calculations
- Stock validation

### Usage

**Wrap your app with CartProvider**:
```typescript
// src/App.tsx
import { CartProvider } from '@/contexts/CartContext';

function App() {
  return (
    <CartProvider>
      <Router>
        <Routes>
          {/* Your routes */}
        </Routes>
      </Router>
    </CartProvider>
  );
}
```

**Access cart state in components**:
```typescript
import { useCart } from '@/contexts/CartContext';

function ShoppingCart() {
  const { items, total, removeItem, updateQuantity } = useCart();

  return (
    <div>
      <h1>Shopping Cart ({items.length} items)</h1>
      {items.map((item) => (
        <div key={item.id} className="cart-item">
          <img src={item.product.image} alt={item.product.name} />
          <div>
            <h3>{item.product.name}</h3>
            <p>${(item.product.price / 100).toFixed(2)}</p>
            <input
              type="number"
              value={item.quantity}
              onChange={(e) =>
                updateQuantity(item.id, parseInt(e.target.value))
              }
              min="1"
            />
          </div>
          <button onClick={() => removeItem(item.id)}>Remove</button>
        </div>
      ))}
      <h2>Total: ${(total / 100).toFixed(2)}</h2>
    </div>
  );
}
```

### Use Cases

**Product Card Add to Cart**:
```typescript
function ProductCard({ product }: { product: Product }) {
  const { addItem, items } = useCart();
  const [quantity, setQuantity] = useState(1);
  const inCart = items.some((item) => item.product.id === product.id);

  const handleAddToCart = () => {
    addItem(product, quantity);
    // Show toast notification
  };

  return (
    <div className="product-card">
      <img src={product.image} alt={product.name} />
      <h3>{product.name}</h3>
      <p>${(product.price / 100).toFixed(2)}</p>
      <input
        type="number"
        value={quantity}
        onChange={(e) => setQuantity(parseInt(e.target.value))}
        min="1"
        max={product.inventory.quantity}
      />
      <button
        onClick={handleAddToCart}
        disabled={inCart || product.inventory.quantity === 0}
      >
        {inCart ? 'In Cart' : 'Add to Cart'}
      </button>
    </div>
  );
}
```

**Checkout Summary**:
```typescript
function CheckoutSummary() {
  const { items, subtotal, tax, shipping, total, discountAmount, discountCode } =
    useCart();

  return (
    <div className="checkout-summary">
      <h3>Order Summary</h3>
      <div className="summary-row">
        <span>Subtotal ({items.length} items)</span>
        <span>${(subtotal / 100).toFixed(2)}</span>
      </div>
      {discountCode && (
        <div className="summary-row discount">
          <span>Discount ({discountCode})</span>
          <span>-${(discountAmount / 100).toFixed(2)}</span>
        </div>
      )}
      <div className="summary-row">
        <span>Shipping</span>
        <span>${(shipping / 100).toFixed(2)}</span>
      </div>
      <div className="summary-row">
        <span>Tax</span>
        <span>${(tax / 100).toFixed(2)}</span>
      </div>
      <div className="summary-row total">
        <span>Total</span>
        <span>${(total / 100).toFixed(2)}</span>
      </div>
    </div>
  );
}
```

**Apply Discount Code**:
```typescript
function DiscountCodeForm() {
  const { applyDiscount, discountCode } = useCart();
  const [code, setCode] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleApply = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError('');

    try {
      await applyDiscount(code);
      setCode('');
      // Show success message
    } catch (err) {
      setError('Invalid discount code');
    } finally {
      setLoading(false);
    }
  };

  return (
    <form onSubmit={handleApply}>
      {discountCode && <p>Discount applied: {discountCode}</p>}
      {error && <p className="error">{error}</p>}
      <input
        type="text"
        value={code}
        onChange={(e) => setCode(e.target.value)}
        placeholder="Discount code"
        disabled={!!discountCode || loading}
      />
      <button type="submit" disabled={loading || !!discountCode}>
        {loading ? 'Applying...' : 'Apply Code'}
      </button>
    </form>
  );
}
```

**Mini Cart Display**:
```typescript
function MiniCart() {
  const { itemCount, total } = useCart();
  const navigate = useNavigate();

  return (
    <div className="mini-cart">
      <button
        onClick={() => navigate('/cart')}
        className="cart-button"
      >
        <ShoppingCartIcon />
        {itemCount > 0 && <span className="badge">{itemCount}</span>}
      </button>
      {itemCount > 0 && (
        <p className="cart-total">${(total / 100).toFixed(2)}</p>
      )}
    </div>
  );
}
```

---

## Context Best Practices

### 1. Provider Setup

```typescript
// Good - single provider wrapping app
function App() {
  return (
    <AuthProvider>
      <CartProvider>
        <Router>
          {/* App content */}
        </Router>
      </CartProvider>
    </AuthProvider>
  );
}

// Also good - separate provider component
function AppProviders({ children }: { children: React.ReactNode }) {
  return (
    <AuthProvider>
      <CartProvider>
        {children}
      </CartProvider>
    </AuthProvider>
  );
}

function App() {
  return (
    <AppProviders>
      <Router>
        {/* App content */}
      </Router>
    </AppProviders>
  );
}
```

### 2. Hook Usage

```typescript
// Good - use custom hook
import { useAuth } from '@/contexts/AuthContext';

function Profile() {
  const { user } = useAuth();
  return <div>{user?.name}</div>;
}

// Avoid - accessing context directly
import { AuthContext } from '@/contexts/AuthContext';

function Profile() {
  const context = useContext(AuthContext);
  if (!context) throw new Error('Must be inside provider');
  return <div>{context.user?.name}</div>;
}
```

### 3. Avoid Unnecessary Re-renders

```typescript
// Good - memoize context value
const AuthProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [user, setUser] = useState<User | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  const value = useMemo(
    () => ({ user, isLoading, signIn, signOut }),
    [user, isLoading]
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
};

// Bad - creates new object every render
const value = { user, isLoading, signIn, signOut };
return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
```

### 4. Error Boundaries

```typescript
function AuthProvider({ children }: { children: React.ReactNode }) {
  return (
    <ErrorBoundary fallback={<AuthError />}>
      <AuthProviderContent>{children}</AuthProviderContent>
    </ErrorBoundary>
  );
}
```

### 5. Type Safety

```typescript
// Good - custom hook with error handling
export function useAuth() {
  const context = useContext(AuthContext);

  if (!context) {
    throw new Error('useAuth must be used within AuthProvider');
  }

  return context;
}

// Usage - TypeScript ensures provider is present
function Component() {
  const { user } = useAuth(); // Type-safe
}
```

---

## Common Patterns

### Protected Routes

```typescript
function ProtectedRoute({ children }: { children: React.ReactNode }) {
  const { isAuthenticated, isLoading } = useAuth();
  const navigate = useNavigate();

  useEffect(() => {
    if (!isLoading && !isAuthenticated) {
      navigate('/login', { replace: true });
    }
  }, [isAuthenticated, isLoading, navigate]);

  if (isLoading) return <LoadingSpinner />;
  if (!isAuthenticated) return null;

  return children;
}
```

### Redirect on Login

```typescript
function LoginPage() {
  const { signIn, isLoading } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();

  const handleLogin = async (email: string, password: string) => {
    await signIn(email, password);
    const from = (location.state as any)?.from?.pathname || '/dashboard';
    navigate(from);
  };

  return <LoginForm onSubmit={handleLogin} isLoading={isLoading} />;
}
```

### Conditional Rendering

```typescript
function Navigation() {
  const { isAuthenticated, user } = useAuth();
  const { itemCount } = useCart();

  return (
    <nav>
      {isAuthenticated ? (
        <>
          <Link to="/profile">{user?.name}</Link>
          <Link to="/cart">Cart ({itemCount})</Link>
          <LogoutButton />
        </>
      ) : (
        <>
          <Link to="/login">Login</Link>
          <Link to="/signup">Sign Up</Link>
        </>
      )}
    </nav>
  );
}
```

---

## Related Documentation

- [React Context Documentation](https://react.dev/reference/react/useContext)
- [Authentication Guide](../docs/ARCHITECTURE.md#authentication)
- [State Management](../docs/ARCHITECTURE.md#state-management)
- [Services Layer](./services/README.md)

---

For implementation details, see individual context files in this directory.
