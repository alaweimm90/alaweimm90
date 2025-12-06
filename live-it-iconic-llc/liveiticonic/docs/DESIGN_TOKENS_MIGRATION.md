# Design Tokens Migration Guide

## Overview

This guide helps you migrate from the legacy theme system (`src/theme/tokens.ts`) to the new modular Design Tokens system (`src/design-tokens/`).

**Good News**: You can migrate gradually! Both systems work in parallel during the transition.

---

## Why Migrate?

### New System Benefits

| Feature | Legacy | New |
|---------|--------|-----|
| **Modularity** | Monolithic file | 8 focused modules |
| **Type Safety** | Basic types | Full TypeScript support |
| **Tree Shaking** | Limited | Excellent |
| **Cross-Platform** | Web-only | Web + Mobile exports |
| **Organization** | Hard to find tokens | Clear categorization |
| **Documentation** | Inline comments | Comprehensive docs |
| **Utility Functions** | Limited | Rich helpers |
| **CSS Variables** | Manual | Auto-generated |
| **Maintainability** | Difficult | Easy |

---

## Migration Strategies

### Strategy 1: Gradual Migration (Recommended)

Migrate component by component, file by file. Keep both systems running.

**Timeline**: 2-4 weeks for large projects

**Steps**:
1. Import new tokens alongside legacy
2. Replace one component at a time
3. Verify functionality
4. Remove legacy imports once complete
5. Delete legacy file when fully migrated

### Strategy 2: File-by-File Migration

Migrate entire feature areas (buttons, cards, forms, etc.) together.

**Timeline**: 1-2 weeks

**Steps**:
1. Group components by feature
2. Migrate each feature area
3. Test entire feature
4. Move to next feature

### Strategy 3: Big Bang Migration

Migrate everything at once. Fastest but highest risk.

**Timeline**: 1-2 days

**Steps**:
1. Replace all imports simultaneously
2. Run full test suite
3. Deploy
4. Monitor for issues

**Recommendation**: Use for small projects only.

---

## Step-by-Step Migration Guide

### Step 1: Audit Current Usage

Before migrating, understand how tokens are currently used.

```bash
# Find all legacy token imports
grep -r "from '@/theme/tokens'" src/

# Find all legacy color usage
grep -r "tokens.colors" src/

# Find all legacy spacing usage
grep -r "tokens.spacing" src/
```

### Step 2: Update Component Imports

#### Before (Legacy)
```typescript
import { tokens } from '@/theme/tokens';

const MyComponent = () => {
  const color = tokens.colors.gold;
  const spacing = tokens.spacing[4];
  const radius = tokens.radius.xl;
  // ...
};
```

#### After (New)
```typescript
import { colors, spacing, borders } from '@/design-tokens';

const MyComponent = () => {
  const color = colors.gold[300];
  const spacing = spacing[4];
  const radius = borders.radii.xl;
  // ...
};
```

### Step 3: Update Color References

Colors now have shade variants. Update references:

#### Before (Legacy)
```typescript
const buttonColor = tokens.colors.gold;
const bgColor = tokens.colors.bg;
const textColor = tokens.colors.cloud;
```

#### After (New)
```typescript
const buttonColor = colors.gold[300];      // Primary gold
const bgColor = colors.charcoal[300];      // Primary background
const textColor = colors.cloud[200];       // Primary cloud
```

### Step 4: Update Spacing References

Spacing keys remain the same, but now imported from dedicated module:

#### Before (Legacy)
```typescript
const padding = `${tokens.spacing[4]} ${tokens.spacing[6]}`;
```

#### After (New)
```typescript
const padding = `${spacing[4]} ${spacing[6]}`;
// or use helper functions
import { createPaddingObject } from '@/design-tokens/spacing';
const padding = createPaddingObject(4, 6);
```

### Step 5: Update Motion References

Motion structure simplified with better organization:

#### Before (Legacy)
```typescript
const duration = tokens.motion.duration.enter;  // 280ms
const easing = tokens.motion.ease;              // Brand easing string
```

#### After (New)
```typescript
import { motion } from '@/design-tokens';
const duration = motion.durations.slow;         // '240ms'
const easing = motion.easings.brand;            // 'cubic-bezier(...)'
```

### Step 6: Update Typography References

Typography access is more explicit:

#### Before (Legacy)
```typescript
const fontFamily = tokens.typography.fonts.display;
const fontSize = tokens.typography.sizes.base;
const fontWeight = tokens.typography.weights.bold;
```

#### After (New)
```typescript
import { typography } from '@/design-tokens';
const fontFamily = typography.fontFamilies.display;
const fontSize = typography.fontSizes.base;
const fontWeight = typography.fontWeights.bold;

// Or use presets
import { typographyPresets } from '@/design-tokens/typography';
const bodyStyle = typographyPresets.bodyRegular;
```

### Step 7: Update Tailwind Integration

Tailwind is already configured! Just update class names if needed.

#### Before
```jsx
<button className="bg-lii-gold text-lii-charcoal">Button</button>
```

#### After
```jsx
<button className="bg-lii-gold-300 text-lii-charcoal-300">Button</button>
```

---

## Common Migration Patterns

### Pattern 1: Color Updates

```typescript
// Legacy token colors
const legacyColorMap = {
  gold: colors.gold[300],
  charcoal: colors.charcoal[300],
  cloud: colors.cloud[200],
  bg: colors.charcoal[300],
  ink: colors.charcoal[400],
  ash: colors.ash,
  signalRed: colors.error[500],
};

// Use in components
const bgColor = legacyColorMap.gold;
```

### Pattern 2: Styled Components Migration

#### Before
```typescript
import styled from 'styled-components';
import { tokens } from '@/theme/tokens';

const Button = styled.button`
  background-color: ${tokens.colors.gold};
  color: ${tokens.colors.charcoal};
  padding: ${tokens.spacing[4]}px ${tokens.spacing[6]}px;
`;
```

#### After
```typescript
import styled from 'styled-components';
import { colors, spacing } from '@/design-tokens';

const Button = styled.button`
  background-color: ${colors.gold[300]};
  color: ${colors.charcoal[300]};
  padding: ${spacing[4]} ${spacing[6]};
`;
```

### Pattern 3: CSS-in-JS Migration

#### Before
```typescript
import { tokens } from '@/theme/tokens';

const styles = {
  button: {
    backgroundColor: tokens.colors.gold,
    padding: tokens.spacing[4],
    fontSize: tokens.typography.sizes.base,
  },
};
```

#### After
```typescript
import { colors, spacing, typography } from '@/design-tokens';

const styles = {
  button: {
    backgroundColor: colors.gold[300],
    padding: spacing[4],
    fontSize: typography.fontSizes.base,
  },
};
```

### Pattern 4: Responsive Style Migration

#### Before
```typescript
const styles = {
  fontSize: tokens.typography.sizes.base,
  [`@media (min-width: ${tokens.breakpoints.md})`]: {
    fontSize: tokens.typography.sizes.lg,
  },
};
```

#### After
```typescript
import { typography, breakpoints, mediaQueries } from '@/design-tokens';

const styles = {
  fontSize: typography.fontSizes.base,
  [`@media ${mediaQueries.minMd}`]: {
    fontSize: typography.fontSizes.lg,
  },
};
```

---

## Automated Migration Tools

### Find and Replace Patterns

Use your IDE's find-and-replace with regex:

```
# Find legacy color imports
Find: tokens\.colors\.([a-zA-Z]+)
Replace: colors.$1[300]  (manual verification needed)

# Find legacy spacing
Find: tokens\.spacing\[(\d+)\]
Replace: spacing[$1]

# Find legacy radius
Find: tokens\.radius\.([a-zA-Z]+)
Replace: borders.radii.$1

# Find legacy typography
Find: tokens\.typography\.fonts\.([a-zA-Z]+)
Replace: typography.fontFamilies.$1

# Find legacy font sizes
Find: tokens\.typography\.sizes\.([a-zA-Z]+)
Replace: typography.fontSizes.$1
```

### Codemod Script

Create a migration codemod for large codebases:

```typescript
// scripts/migrate-tokens.js
const fs = require('fs');
const path = require('path');

const replacements = [
  {
    pattern: /from ['"]@\/theme\/tokens['"]/g,
    replacement: `from '@/design-tokens'`,
  },
  {
    pattern: /tokens\.colors\.gold(?!\[)/g,
    replacement: 'colors.gold[300]',
  },
  {
    pattern: /tokens\.colors\.charcoal(?!\[)/g,
    replacement: 'colors.charcoal[300]',
  },
  {
    pattern: /tokens\.colors\.cloud(?!\[)/g,
    replacement: 'colors.cloud[200]',
  },
  {
    pattern: /tokens\.spacing\[(\d+)\]/g,
    replacement: 'spacing[$1]',
  },
  {
    pattern: /tokens\.radius\./g,
    replacement: 'borders.radii.',
  },
];

function migrateFile(filePath) {
  let content = fs.readFileSync(filePath, 'utf-8');

  replacements.forEach(({ pattern, replacement }) => {
    content = content.replace(pattern, replacement);
  });

  fs.writeFileSync(filePath, content, 'utf-8');
  console.log(`Migrated: ${filePath}`);
}

// Usage
const srcDir = './src';
const walk = (dir) => {
  fs.readdirSync(dir).forEach((file) => {
    const filePath = path.join(dir, file);
    if (fs.statSync(filePath).isDirectory()) {
      walk(filePath);
    } else if (filePath.endsWith('.tsx') || filePath.endsWith('.ts')) {
      migrateFile(filePath);
    }
  });
};

walk(srcDir);
```

---

## Validation Checklist

Before considering migration complete:

### Code Quality
- [ ] No legacy token imports remain
- [ ] All TypeScript errors resolved
- [ ] ESLint checks pass
- [ ] No console warnings

### Functionality
- [ ] All components render correctly
- [ ] Styling applied as expected
- [ ] Responsive design works
- [ ] Dark mode (if applicable) works
- [ ] All animations smooth

### Performance
- [ ] Bundle size same or reduced
- [ ] No performance regressions
- [ ] CSS variables loaded
- [ ] Tailwind classes working

### Testing
- [ ] Unit tests pass
- [ ] Component tests pass
- [ ] Visual regression tests pass
- [ ] E2E tests pass

---

## Troubleshooting Common Issues

### Issue 1: Color Shades Missing

**Problem**: Token reference doesn't include shade value
```typescript
// WRONG
const color = colors.gold;  // Returns object, not string

// RIGHT
const color = colors.gold[300];  // Returns hex value
```

**Solution**: Always include shade number for color tokens.

### Issue 2: Import Path Errors

**Problem**: Import fails with "module not found"
```typescript
// WRONG
import { colors } from './design-tokens';

// RIGHT
import { colors } from '@/design-tokens';
```

**Solution**: Use absolute imports with `@/` alias.

### Issue 3: Tailwind Classes Not Working

**Problem**: Tailwind classes don't apply
```html
<!-- WRONG -->
<div class="text-gold">Text</div>

<!-- RIGHT -->
<div class="text-lii-gold-300">Text</div>
```

**Solution**: Check `tailwind.config.ts` is properly configured.

### Issue 4: TypeScript Errors with Generics

**Problem**: Type inference issues with token helpers
```typescript
// WRONG
const size: SpacingKey = 'invalid';

// RIGHT
const size: SpacingKey = 4;
```

**Solution**: Use proper type imports and verify values.

### Issue 5: CSS Variables Not Available

**Problem**: CSS custom properties undefined
```css
/* Missing */
.element {
  color: var(--lii-color-gold-300);
}
```

**Solution**: Run CSS generator or check head for `<style>` tag.

---

## Rollback Plan

If issues arise, rollback is simple:

```bash
# Revert changes to specific files
git checkout src/path/to/component.tsx

# Or revert entire migration
git revert <commit-hash>

# Keep legacy tokens available
# Both systems work in parallel
```

---

## Performance Considerations

### Bundle Size Impact

- **Before**: 1 large token file (11KB)
- **After**: Multiple small modules (~10KB total, better tree-shakeable)
- **CSS Variables**: Auto-generated, no impact

### Tree Shaking

New modular system enables better tree shaking:

```typescript
// Only imported modules are included
import { colors } from '@/design-tokens';
// No spacing, motion, typography in bundle
```

### Runtime Performance

- **Identical**: Direct value access (no runtime cost)
- **Identical**: CSS variables (native browser feature)
- **Better**: Smaller component bundle sizes

---

## Post-Migration Steps

### 1. Clean Up Legacy Files

Once migration complete, remove legacy tokens:

```bash
# Verify no more legacy imports
grep -r "from '@/theme/tokens'" src/ || echo "✓ No legacy imports found"

# Remove legacy file
rm src/theme/tokens.ts

# Update theme folder if empty
rm -rf src/theme/
```

### 2. Update Documentation

- [ ] Update README.md with new import paths
- [ ] Update component documentation
- [ ] Update team design system guide
- [ ] Update onboarding documentation

### 3. Team Communication

- [ ] Notify team of migration completion
- [ ] Update development guidelines
- [ ] Share documentation links
- [ ] Conduct brief knowledge-sharing session

### 4. Version Control

```bash
# Create migration commit
git commit -m "chore: migrate legacy tokens to modular design-tokens

- Move from src/theme/tokens to src/design-tokens
- Update all imports across codebase
- Maintain backward compatibility via tailwind config
- No visual or functional changes"
```

---

## Timeline Example: Medium Project (50 components)

### Week 1: Preparation
- Day 1-2: Audit current usage
- Day 3: Create migration plan & documentation
- Day 4-5: Set up migration scripts

### Week 2: Execution
- Day 1-3: Migrate 60% of components (30 components)
- Day 4-5: Migrate remaining components, test thoroughly

### Week 3: Validation & Cleanup
- Day 1-2: Complete testing & bug fixes
- Day 3: Remove legacy files
- Day 4-5: Documentation & team knowledge transfer

---

## Getting Help

### Documentation
- Design tokens guide: `/docs/DESIGN_TOKENS.md`
- Token reference: View `/src/design-tokens/` directly
- Tailwind config: `/tailwind.config.ts`

### Common Questions
- **Q**: Can I use both systems together?
  - **A**: Yes, migration is gradual by design.

- **Q**: Will this affect production?
  - **A**: No, both systems are identical in output.

- **Q**: How do I add new tokens?
  - **A**: Add to respective file in `/src/design-tokens/`.

- **Q**: What about dark mode?
  - **A**: Use CSS variables with `prefers-color-scheme` media query.

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | Nov 2024 | Initial release |

---

**Need Help?** Contact your design system maintainer or refer to the comprehensive documentation in `/docs/DESIGN_TOKENS.md`.
