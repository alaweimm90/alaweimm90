# Dependency Analysis Report

Generated: 2025-08-05T00:08:30.060Z

## Summary
- Total Dependencies: 84
- Heavy Dependencies: 46
- Moderate Dependencies: 2
- Light Dependencies: 36

## Heavy Dependencies (Bundle Impact: High)
- **@axe-core/react** (^4.10.2)
- **@radix-ui/react-accordion** (^1.2.0)
- **@radix-ui/react-alert-dialog** (^1.1.1)
- **@radix-ui/react-aspect-ratio** (^1.1.0)
- **@radix-ui/react-avatar** (^1.1.0)
- **@radix-ui/react-checkbox** (^1.1.1)
- **@radix-ui/react-collapsible** (^1.1.0)
- **@radix-ui/react-context-menu** (^2.2.1)
- **@radix-ui/react-dialog** (^1.1.2)
- **@radix-ui/react-dropdown-menu** (^2.1.1)
- **@radix-ui/react-hover-card** (^1.1.1)
- **@radix-ui/react-label** (^2.1.0)
- **@radix-ui/react-menubar** (^1.1.1)
- **@radix-ui/react-navigation-menu** (^1.2.0)
- **@radix-ui/react-popover** (^1.1.1)
- **@radix-ui/react-progress** (^1.1.0)
- **@radix-ui/react-radio-group** (^1.2.0)
- **@radix-ui/react-scroll-area** (^1.1.0)
- **@radix-ui/react-select** (^2.1.1)
- **@radix-ui/react-separator** (^1.1.0)
- **@radix-ui/react-slider** (^1.2.0)
- **@radix-ui/react-slot** (^1.1.0)
- **@radix-ui/react-switch** (^1.1.0)
- **@radix-ui/react-tabs** (^1.1.0)
- **@radix-ui/react-toast** (^1.2.1)
- **@radix-ui/react-toggle** (^1.1.0)
- **@radix-ui/react-toggle-group** (^1.1.0)
- **@radix-ui/react-tooltip** (^1.1.4)
- **@sentry/react** (^10.0.0)
- **@stripe/react-stripe-js** (^3.8.0)
- **@stripe/stripe-js** (^7.6.1)
- **@supabase/supabase-js** (^2.52.1)
- **@tanstack/react-query** (^5.56.2)
- **@testing-library/react** (^16.3.0)
- **@xyflow/react** (^12.8.2)
- **embla-carousel-react** (^8.3.0)
- **framer-motion** (^12.23.11)
- **lucide-react** (^0.462.0)
- **react** (^18.3.1)
- **react-day-picker** (^8.10.1)
- **react-dom** (^18.3.1)
- **react-hook-form** (^7.53.0)
- **react-i18next** (^15.6.1)
- **react-resizable-panels** (^2.1.3)
- **react-router-dom** (^6.26.2)
- **recharts** (^3.1.0)

## Optimization Recommendations
### 1. Code Splitting Opportunities
Consider lazy loading these heavy components:
- Recharts components (charts)
- Framer Motion animations
- Radix UI components not used on initial load

### 2. Bundle Size Optimizations
- Use tree shaking for Radix UI components
- Consider lighter alternatives for date manipulation
- Optimize Lucide React icon imports
- Use dynamic imports for rarely used components

### 3. Performance Recommendations
- Implement React.memo for expensive components
- Use useMemo for complex calculations
- Lazy load non-critical routes
- Optimize image loading with intersection observer

