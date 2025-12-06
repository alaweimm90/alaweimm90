# Live It Iconic - i18n Integration Guide

## Installation Complete

The i18n system is fully installed and configured. All you need to do now is integrate it into your components.

## Automatic Setup

The following have been automatically configured:

1. **i18n Configuration** - Already initialized in `src/App.tsx`
2. **CurrencyProvider** - Wraps the entire app
3. **Language Detection** - Automatically detects browser locale
4. **localStorage** - Saves language and currency preferences

## Quick Integration Examples

### 1. Add Selectors to Header/Navigation

```tsx
// In your Header or Navigation component
import { LanguageSelector } from '@/components/LanguageSelector';
import { CurrencySelector } from '@/components/CurrencySelector';

export function Navigation() {
  return (
    <nav className="flex items-center justify-between">
      <div>Logo</div>
      <div className="flex items-center gap-4">
        <LanguageSelector />
        <CurrencySelector />
      </div>
    </nav>
  );
}
```

### 2. Translate Hero Section

```tsx
// In components/Hero.tsx
import { useTranslation } from 'react-i18next';
import { useI18nCurrency } from '@/hooks/useI18nCurrency';

export function Hero() {
  const { t } = useTranslation();
  const { format } = useI18nCurrency();

  return (
    <section>
      <h1>{t('hero.title')}</h1>
      <p>{t('hero.tagline')}</p>
      <p>{t('hero.subtitle')}</p>
      <button>{t('hero.cta')}</button>
    </section>
  );
}
```

### 3. Translate Product Card

```tsx
// In components/ProductCard.tsx
import { useTranslation } from 'react-i18next';
import { useI18nCurrency } from '@/hooks/useI18nCurrency';

export function ProductCard({ product }) {
  const { t } = useTranslation();
  const { format } = useI18nCurrency();

  return (
    <div>
      <img src={product.image} alt={product.name} />
      <h3>{product.name}</h3>
      <p>{format(product.price)}</p>
      <p>{product.inStock ? t('product.inStock') : t('product.outOfStock')}</p>
      <button>{t('product.addToCart')}</button>
    </div>
  );
}
```

### 4. Translate Shopping Cart

```tsx
// In components/Cart.tsx
import { useTranslation } from 'react-i18next';
import { useI18nCurrency } from '@/hooks/useI18nCurrency';
import { useCart } from '@/contexts/CartContext'; // Existing hook

export function Cart() {
  const { t } = useTranslation();
  const { format } = useI18nCurrency();
  const { items, total } = useCart();

  return (
    <div>
      <h1>{t('cart.title')}</h1>

      {items.length === 0 ? (
        <p>{t('cart.empty')}</p>
      ) : (
        <>
          {items.map(item => (
            <div key={item.id}>
              <span>{item.name}</span>
              <span>{format(item.price)}</span>
              <button onClick={() => removeItem(item.id)}>
                {t('cart.removeItem')}
              </button>
            </div>
          ))}

          <div>
            <p>{t('cart.subtotal')}: {format(total)}</p>
            <p>{t('cart.shipping')}: {format(10)}</p>
            <p>{t('cart.total')}: {format(total + 10)}</p>
          </div>

          <button>{t('cart.checkout')}</button>
        </>
      )}
    </div>
  );
}
```

### 5. Translate Checkout

```tsx
// In components/CheckoutForm.tsx
import { useTranslation } from 'react-i18next';
import { useI18nCurrency } from '@/hooks/useI18nCurrency';

export function CheckoutForm() {
  const { t } = useTranslation();
  const { format } = useI18nCurrency();

  return (
    <form>
      <h1>{t('checkout.title')}</h1>

      <fieldset>
        <legend>{t('checkout.shippingInfo')}</legend>
        <input placeholder={t('checkout.firstName')} />
        <input placeholder={t('checkout.lastName')} />
        <input placeholder={t('checkout.email')} />
        <input placeholder={t('checkout.address')} />
      </fieldset>

      <fieldset>
        <legend>{t('checkout.paymentInfo')}</legend>
        {/* Payment fields */}
      </fieldset>

      <div>
        <h2>{t('checkout.orderSummary')}</h2>
        <p>{t('cart.subtotal')}: {format(99.99)}</p>
        <p>{t('cart.shipping')}: {format(10)}</p>
        <p>{t('cart.total')}: {format(109.99)}</p>
      </div>

      <button type="submit">{t('checkout.placeOrder')}</button>
    </form>
  );
}
```

### 6. Translate Form Labels

```tsx
// In components/LoginForm.tsx
import { useTranslation } from 'react-i18next';
import { useForm } from 'react-hook-form';

export function LoginForm() {
  const { t } = useTranslation();
  const { register, handleSubmit, formState: { errors } } = useForm();

  return (
    <form onSubmit={handleSubmit(onSubmit)}>
      <label>{t('auth.email')}</label>
      <input {...register('email')} />
      {errors.email && <span>{t('errors.invalidEmail')}</span>}

      <label>{t('auth.password')}</label>
      <input type="password" {...register('password')} />
      {errors.password && <span>{t('errors.passwordTooShort')}</span>}

      <button type="submit">{t('auth.login')}</button>
      <p>{t('auth.signupNow')}</p>
    </form>
  );
}
```

### 7. Use in Admin Dashboard

```tsx
// In pages/admin/Dashboard.tsx
import { useTranslation } from 'react-i18next';
import { useI18nCurrency } from '@/hooks/useI18nCurrency';

export function AdminDashboard() {
  const { t } = useTranslation();
  const { format } = useI18nCurrency();

  return (
    <div>
      <h1>Admin {t('navigation.home')}</h1>
      <p>{t('common.loading')}</p>
      <p>Revenue: {format(50000)}</p>
    </div>
  );
}
```

### 8. Error Messages

```tsx
// Anywhere in your app
import { useTranslation } from 'react-i18next';
import { toast } from 'sonner';

export function MyComponent() {
  const { t } = useTranslation();

  const handleError = (error) => {
    toast.error(t('errors.serverError'));
  };

  const handleSuccess = () => {
    toast.success(t('messages.orderPlaced'));
  };

  return <div>...</div>;
}
```

## Common Import Patterns

### For Translations Only
```tsx
import { useTranslation } from 'react-i18next';

const { t, i18n } = useTranslation();
```

### For Currency Only
```tsx
import { useCurrency } from '@/contexts/CurrencyContext';
import { formatCurrency } from '@/lib/currency';

const { currency } = useCurrency();
```

### For Both (Recommended)
```tsx
import { useI18nCurrency } from '@/hooks/useI18nCurrency';

const { format, getLanguage, getCurrency } = useI18nCurrency();
```

## Testing i18n in Development

### Check Current Language
Open browser console:
```javascript
// Get current language
console.log(i18next.language);

// Get all resources
console.log(i18next.getResourceBundle(i18next.language, 'translation'));

// Test translation
console.log(i18next.t('hero.title'));
```

### Check Current Currency
```javascript
// Get current currency from localStorage
localStorage.getItem('preferred-currency');

// Get exchange rates
localStorage.getItem('exchange-rates');
```

### Switch Language Programmatically
```javascript
// In development console
i18next.changeLanguage('es');
i18next.changeLanguage('fr');
i18next.changeLanguage('en');
```

## Translation Best Practices

### 1. Always Use Translation Keys
```tsx
// GOOD
<p>{t('product.price')}</p>

// BAD
<p>Price</p>
```

### 2. Format Prices with useI18nCurrency
```tsx
// GOOD
const { format } = useI18nCurrency();
<p>{format(99.99)}</p>

// BAD
<p>${99.99}</p>
```

### 3. Group Related Translations
```tsx
// Translation file
{
  "checkout": {
    "title": "Checkout",
    "placeOrder": "Place Order"
  }
}

// Component
<h1>{t('checkout.title')}</h1>
<button>{t('checkout.placeOrder')}</button>
```

### 4. Use Constants for Translation Keys
```tsx
// constants.ts
export const TRANSLATION_KEYS = {
  HERO: {
    TITLE: 'hero.title',
    TAGLINE: 'hero.tagline',
  }
};

// Component
<h1>{t(TRANSLATION_KEYS.HERO.TITLE)}</h1>
```

### 5. Handle Missing Keys
i18next shows the key path if translation is missing (development) and falls back to the key in production.

## File Locations Reference

### Core Configuration
- `/src/i18n/config.ts` - i18n setup
- `/src/i18n/locales/` - Translation files

### Utilities & Context
- `/src/lib/currency.ts` - Currency functions
- `/src/contexts/CurrencyContext.tsx` - Currency provider

### Components
- `/src/components/LanguageSelector.tsx` - Language switcher
- `/src/components/CurrencySelector.tsx` - Currency switcher
- `/src/components/I18nExample.tsx` - Example component

### Hooks
- `/src/hooks/useI18nCurrency.ts` - Combined hook

### Main App
- `/src/App.tsx` - Updated with CurrencyProvider

## Documentation Files

### Quick Reference (Start Here)
- `/I18N_QUICK_REFERENCE.md` - Copy-paste examples

### Complete Guide
- `/I18N_DOCUMENTATION.md` - Full documentation

### Implementation Details
- `/I18N_IMPLEMENTATION_SUMMARY.md` - What was implemented

### Checklist
- `/I18N_INTEGRATION_CHECKLIST.md` - Verification checklist

### This File
- `/I18N_INTEGRATION_GUIDE.md` - Integration examples

## Adding More Translations

When you add new features:

1. **Add to English first** (`en.json`):
```json
{
  "myFeature": {
    "title": "My Feature Title",
    "description": "Feature description"
  }
}
```

2. **Add to French** (`fr.json`):
```json
{
  "myFeature": {
    "title": "Titre de ma fonction",
    "description": "Description de la fonction"
  }
}
```

3. **Add to Spanish** (`es.json`):
```json
{
  "myFeature": {
    "title": "Título de mi función",
    "description": "Descripción de la función"
  }
}
```

4. **Use in component**:
```tsx
<h1>{t('myFeature.title')}</h1>
<p>{t('myFeature.description')}</p>
```

## Common Integration Tasks

### Task 1: Add Selectors to Navigation
See example #1 above

### Task 2: Internationalize Existing Component
1. Import: `import { useI18nCurrency } from '@/hooks/useI18nCurrency';`
2. Hook: `const { t, format } = useI18nCurrency();`
3. Replace hardcoded strings with `t('key.path')`
4. Replace hardcoded prices with `format(amount)`

### Task 3: Add New Language (e.g., German)
1. Create `/src/i18n/locales/de.json`
2. Update `/src/i18n/config.ts` - add import and resource
3. Update `/src/components/LanguageSelector.tsx` - add language option

### Task 4: Add New Currency (e.g., CHF)
1. Update `/src/lib/currency.ts` - add to currencies object
2. Update `/src/contexts/CurrencyContext.tsx` - add rates
3. Automatically available in CurrencySelector

## Troubleshooting

### Translations not appearing?
1. Check i18n config is imported: `import '@/i18n/config'` in App.tsx
2. Verify translation key exists in JSON file
3. Check browser console for errors
4. Clear cache: `localStorage.clear()`

### Currency not updating?
1. Verify CurrencyProvider wraps app in App.tsx
2. Check useCurrency hook is called correctly
3. Verify currency code is valid (USD, EUR, etc.)

### How to debug?
```tsx
// In any component
const { t, i18n } = useTranslation();
const { currency } = useCurrency();

console.log('Current language:', i18n.language);
console.log('Current currency:', currency);
console.log('Translation test:', t('hero.title'));
```

## Next Actions

1. **Review Examples** - Check `/src/components/I18nExample.tsx`
2. **Update Navigation** - Add `LanguageSelector` component
3. **Update Header** - Add `CurrencySelector` component
4. **Translate Pages** - Use `useI18nCurrency` in your pages
5. **Test Switching** - Verify language and currency change

## Support Resources

- **Examples**: `/src/components/I18nExample.tsx`
- **Quick Ref**: `/I18N_QUICK_REFERENCE.md`
- **Full Docs**: `/I18N_DOCUMENTATION.md`
- **i18next**: https://www.i18next.com/
- **react-i18next**: https://react.i18next.com/
