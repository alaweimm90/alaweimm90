# Live It Iconic - i18n Quick Reference

## Quick Start

### 1. Use Translations
```tsx
import { useTranslation } from 'react-i18next';

export function Component() {
  const { t, i18n } = useTranslation();

  return (
    <>
      <h1>{t('hero.title')}</h1>
      <button onClick={() => i18n.changeLanguage('es')}>
        Spanish
      </button>
    </>
  );
}
```

### 2. Format Currency
```tsx
import { useCurrency } from '@/contexts/CurrencyContext';
import { formatCurrency } from '@/lib/currency';

export function PriceDisplay() {
  const { currency } = useCurrency();

  return <p>Price: {formatCurrency(99.99, currency)}</p>;
}
```

### 3. Use Combined Hook
```tsx
import { useI18nCurrency } from '@/hooks/useI18nCurrency';

export function CheckoutCart() {
  const { format } = useI18nCurrency();

  return (
    <div>
      <p>Total: {format(149.99)}</p>
    </div>
  );
}
```

### 4. Add Selectors to UI
```tsx
import { LanguageSelector } from '@/components/LanguageSelector';
import { CurrencySelector } from '@/components/CurrencySelector';

export function Header() {
  return (
    <header>
      <LanguageSelector />
      <CurrencySelector />
    </header>
  );
}
```

## Common Translation Keys

### Navigation
```typescript
t('navigation.home')      // "Home" / "Accueil" / "Inicio"
t('navigation.shop')      // "Shop" / "Boutique" / "Tienda"
t('navigation.cart')      // "Cart" / "Panier" / "Carrito"
```

### Product
```typescript
t('product.addToCart')    // "Add to Cart" / "Ajouter au panier"
t('product.price')        // "Price" / "Prix" / "Precio"
t('product.inStock')      // "In Stock" / "En stock"
```

### Cart/Checkout
```typescript
t('cart.total')           // "Total"
t('cart.checkout')        // "Checkout" / "Passer la commande"
t('checkout.placeOrder')  // "Place Order" / "Passer la commande"
```

### Messages
```typescript
t('messages.addedToCart')       // "Added to cart successfully"
t('messages.orderPlaced')       // "Order placed successfully"
t('errors.invalidEmail')        // "Please enter a valid email"
```

## Currency Formatting

### Basic Formatting
```typescript
import { formatCurrency } from '@/lib/currency';

formatCurrency(99.99, 'USD')    // "$99.99"
formatCurrency(99.99, 'EUR')    // "€99,99" (locale-aware)
formatCurrency(99.99, 'GBP')    // "£99.99"
```

### Convert & Format
```typescript
const { convertAndFormat } = useI18nCurrency();

convertAndFormat(100, 'USD', 'EUR')  // Converts USD→EUR and formats
convertAndFormat(100, 'USD')         // Converts to current currency
```

### Get Exchange Rate
```typescript
const { getRate } = useI18nCurrency();

getRate('USD', 'EUR')  // 0.92
```

## Supported Currencies

```
USD - $ (US Dollar)
EUR - € (Euro)
GBP - £ (British Pound)
CAD - CA$ (Canadian Dollar)
JPY - ¥ (Japanese Yen)
AUD - A$ (Australian Dollar)
```

## Supported Languages

```
en - English (🇺🇸)
fr - Français (🇫🇷)
es - Español (🇪🇸)
```

## File Locations

```
/src/i18n/config.ts           - i18n setup
/src/i18n/locales/en.json     - English translations
/src/i18n/locales/fr.json     - French translations
/src/i18n/locales/es.json     - Spanish translations
/src/lib/currency.ts          - Currency utilities
/src/contexts/CurrencyContext.tsx - Currency provider
/src/components/LanguageSelector.tsx - Language switcher
/src/components/CurrencySelector.tsx - Currency switcher
/src/hooks/useI18nCurrency.ts - Combined hook
```

## Adding a New Translation Key

1. Add to `en.json`:
```json
{
  "mySection": {
    "myKey": "My Text Here"
  }
}
```

2. Add to `fr.json`:
```json
{
  "mySection": {
    "myKey": "Mon Texte Ici"
  }
}
```

3. Add to `es.json`:
```json
{
  "mySection": {
    "myKey": "Mi Texto Aquí"
  }
}
```

4. Use in component:
```tsx
const { t } = useTranslation();
t('mySection.myKey')
```

## Adding a New Language

1. Create `/src/i18n/locales/de.json` (German example)
2. Update `/src/i18n/config.ts`:
   ```typescript
   import deTranslations from './locales/de.json';

   resources: {
     // ... existing
     de: { translation: deTranslations },
   }
   ```
3. Update `/src/components/LanguageSelector.tsx`:
   ```typescript
   { code: 'de', name: 'Deutsch', flag: '🇩🇪' }
   ```

## Adding a New Currency

1. Update `/src/lib/currency.ts`:
   ```typescript
   export const currencies = {
     // ... existing
     CHF: { symbol: 'CHF', code: 'CHF', locale: 'fr-CH' },
   };
   ```

2. Add exchange rates to `/src/contexts/CurrencyContext.tsx`:
   ```typescript
   'USD_CHF': 0.89,
   'CHF_USD': 1.124,
   // ... add all pairs
   ```

## Hooks Reference

### useTranslation()
```typescript
const { t, i18n } = useTranslation();

t('key.path')                    // Get translation
i18n.language                    // Current language
i18n.changeLanguage('es')        // Change language
i18n.options.fallbackLng         // Fallback language
```

### useCurrency()
```typescript
const {
  currency,           // Current currency code
  setCurrency,        // Set currency function
  rates,              // Exchange rates object
  isLoading,          // Loading state
  error,              // Error message
  conversionRate      // Get rate function
} = useCurrency();
```

### useI18nCurrency()
```typescript
const {
  format,             // Format in current currency
  formatIn,           // Format in specific currency
  convertAndFormat,   // Convert then format
  getRate,            // Get exchange rate
  getLanguage,        // Get current language
  getCurrency,        // Get current currency
  currentLanguage,    // Current language value
  currentCurrency     // Current currency value
} = useI18nCurrency();
```

## Common Patterns

### Product List with Prices
```tsx
export function ProductList({ products }) {
  const { format } = useI18nCurrency();
  const { t } = useTranslation();

  return (
    <ul>
      {products.map(product => (
        <li key={product.id}>
          <h3>{product.name}</h3>
          <p>{format(product.price)}</p>
          <button>{t('product.addToCart')}</button>
        </li>
      ))}
    </ul>
  );
}
```

### Checkout Summary
```tsx
export function OrderSummary({ items, shipping, tax }) {
  const { format } = useI18nCurrency();
  const { t } = useTranslation();

  const subtotal = items.reduce((sum, i) => sum + i.price, 0);
  const total = subtotal + shipping + tax;

  return (
    <div>
      <h2>{t('checkout.orderSummary')}</h2>
      <p>{t('cart.subtotal')}: {format(subtotal)}</p>
      <p>{t('cart.shipping')}: {format(shipping)}</p>
      <p>{t('cart.total')}: {format(total)}</p>
      <button>{t('checkout.placeOrder')}</button>
    </div>
  );
}
```

### Multi-Currency Price
```tsx
export function PriceComparison({ priceUSD }) {
  const { convertAndFormat } = useI18nCurrency();

  return (
    <div>
      <p>USD: {convertAndFormat(priceUSD, 'USD', 'USD')}</p>
      <p>EUR: {convertAndFormat(priceUSD, 'USD', 'EUR')}</p>
      <p>GBP: {convertAndFormat(priceUSD, 'USD', 'GBP')}</p>
    </div>
  );
}
```

## Debugging

### Check Current Language
```tsx
const { i18n } = useTranslation();
console.log(i18n.language);          // Current language
console.log(i18n.languages);         // All available languages
console.log(i18n.resolvedLanguage);  // Resolved language
```

### Check Current Currency
```tsx
const { currency, rates } = useCurrency();
console.log(currency);               // Current currency
console.log(Object.keys(rates));     // Available rates
```

### Check Translation
```tsx
const { t } = useTranslation();
console.log(t('missing.key'));       // Shows key path if not found
```

## Performance Tips

1. **Cache translations**: useTranslation() is memoized
2. **Cache currency**: useCurrency() is optimized
3. **Use useI18nCurrency()** for combined operations
4. **Avoid recreating contexts** - use memoization
5. **Lazy load pages** - already done in App.tsx

## Troubleshooting

### Translations not showing
- Check if `import '@/i18n/config'` is in App.tsx
- Verify translation key exists in JSON file
- Check browser console for errors

### Currency not updating
- Ensure `CurrencyProvider` wraps app in App.tsx
- Check localStorage: `localStorage.getItem('preferred-currency')`
- Verify currency code is valid

### Wrong locale formatting
- Check currency config in `/src/lib/currency.ts`
- Verify locale string is correct (e.g., 'en-US')
- Test with `formatCurrency()` directly

## More Information

- Full Documentation: `/I18N_DOCUMENTATION.md`
- Implementation Summary: `/I18N_IMPLEMENTATION_SUMMARY.md`
- Example Component: `/src/components/I18nExample.tsx`
