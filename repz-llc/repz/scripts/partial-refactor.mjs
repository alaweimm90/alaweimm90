#!/usr/bin/env node

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const rootDir = path.resolve(__dirname, '..');
const srcDir = path.join(rootDir, 'src');

console.log('🔧 Partial source refactoring to feature-first...\n');

const stats = {
  moved: 0,
  created: 0,
  errors: 0
};

// Function to move file safely
function moveFile(source, destination) {
  try {
    const destDir = path.dirname(destination);
    if (!fs.existsSync(destDir)) {
      fs.mkdirSync(destDir, { recursive: true });
    }
    
    if (fs.existsSync(source)) {
      fs.renameSync(source, destination);
      console.log(`✅ Moved: ${path.relative(srcDir, source)} → ${path.relative(srcDir, destination)}`);
      stats.moved++;
      return true;
    }
    return false;
  } catch (error) {
    console.error(`❌ Error moving ${path.relative(srcDir, source)}:`, error.message);
    stats.errors++;
    return false;
  }
}

// Move some clear UI components to atomic design structure
console.log('📦 Moving UI components to atomic design...\n');

const uiMoves = [
  // Atoms (basic components)
  { from: 'components/ui/button.tsx', to: 'ui/atoms/Button.tsx' },
  { from: 'components/ui/input.tsx', to: 'ui/atoms/Input.tsx' },
  { from: 'components/ui/label.tsx', to: 'ui/atoms/Label.tsx' },
  { from: 'components/ui/badge.tsx', to: 'ui/atoms/Badge.tsx' },
  { from: 'components/ui/avatar.tsx', to: 'ui/atoms/Avatar.tsx' },
  { from: 'components/ui/separator.tsx', to: 'ui/atoms/Separator.tsx' },
  { from: 'components/ui/skeleton.tsx', to: 'ui/atoms/Skeleton.tsx' },
  { from: 'components/ui/switch.tsx', to: 'ui/atoms/Switch.tsx' },
  { from: 'components/ui/checkbox.tsx', to: 'ui/atoms/Checkbox.tsx' },
  
  // Molecules (component combinations)
  { from: 'components/ui/card.tsx', to: 'ui/molecules/Card.tsx' },
  { from: 'components/ui/alert.tsx', to: 'ui/molecules/Alert.tsx' },
  { from: 'components/ui/dialog.tsx', to: 'ui/molecules/Dialog.tsx' },
  { from: 'components/ui/dropdown-menu.tsx', to: 'ui/molecules/DropdownMenu.tsx' },
  { from: 'components/ui/select.tsx', to: 'ui/molecules/Select.tsx' },
  { from: 'components/ui/tabs.tsx', to: 'ui/molecules/Tabs.tsx' },
  { from: 'components/ui/accordion.tsx', to: 'ui/molecules/Accordion.tsx' },
  { from: 'components/ui/toast.tsx', to: 'ui/molecules/Toast.tsx' },
  { from: 'components/ui/toaster.tsx', to: 'ui/molecules/Toaster.tsx' },
  { from: 'components/ui/use-toast.ts', to: 'ui/molecules/useToast.ts' },

  // Organisms (complex components)
  { from: 'components/Navigation.tsx', to: 'ui/organisms/Navigation.tsx' },
  { from: 'components/RepzLogo.tsx', to: 'ui/organisms/RepzLogo.tsx' }
];

uiMoves.forEach(({ from, to }) => {
  const sourcePath = path.join(srcDir, from);
  const destPath = path.join(srcDir, to);
  moveFile(sourcePath, destPath);
});

// Move some feature components
console.log('\n🎯 Moving feature components...\n');

const featureMoves = [
  // Authentication
  { from: 'components/auth/LoginForm.tsx', to: 'features/auth/components/LoginForm.tsx' },
  { from: 'components/auth/SignupForm.tsx', to: 'features/auth/components/SignupForm.tsx' },
  { from: 'components/auth/AuthLayout.tsx', to: 'features/auth/components/AuthLayout.tsx' },
  { from: 'components/auth/OnboardingLayout.tsx', to: 'features/auth/components/OnboardingLayout.tsx' },
  
  // Pricing
  { from: 'components/pricing/PricingCard.tsx', to: 'features/pricing/components/PricingCard.tsx' },
  { from: 'components/pricing/ElegantPricing.tsx', to: 'features/pricing/components/ElegantPricing.tsx' },
  { from: 'components/pricing/TierComparison.tsx', to: 'features/pricing/components/TierComparison.tsx' },
  
  // Dashboard
  { from: 'components/dashboard/ClientDashboard.tsx', to: 'features/dashboard/components/ClientDashboard.tsx' },
  { from: 'components/dashboard/CoachDashboard.tsx', to: 'features/dashboard/components/CoachDashboard.tsx' },
  { from: 'components/dashboard/AdminDashboard.tsx', to: 'features/dashboard/components/AdminDashboard.tsx' },
  
  // Analytics
  { from: 'components/analytics/AnalyticsDashboard.tsx', to: 'features/analytics/components/AnalyticsDashboard.tsx' },
  { from: 'components/analytics/AnalyticsHub.tsx', to: 'features/analytics/components/AnalyticsHub.tsx' },
  
  // AI
  { from: 'components/ai/AIHub.tsx', to: 'features/ai/components/AIHub.tsx' },
  { from: 'components/ai/AICoachingPanel.tsx', to: 'features/ai/components/AICoachingPanel.tsx' }
];

featureMoves.forEach(({ from, to }) => {
  const sourcePath = path.join(srcDir, from);
  const destPath = path.join(srcDir, to);
  moveFile(sourcePath, destPath);
});

// Move some hooks
console.log('\n🪝 Moving hooks to features...\n');

const hookMoves = [
  { from: 'hooks/useAuth.tsx', to: 'features/auth/hooks/useAuth.tsx' },
  { from: 'hooks/useClientData.tsx', to: 'features/dashboard/hooks/useClientData.tsx' },
  { from: 'hooks/useAnalytics.tsx', to: 'features/analytics/hooks/useAnalytics.tsx' },
  { from: 'hooks/usePricing.tsx', to: 'features/pricing/hooks/usePricing.tsx' }
];

hookMoves.forEach(({ from, to }) => {
  const sourcePath = path.join(srcDir, from);
  const destPath = path.join(srcDir, to);
  moveFile(sourcePath, destPath);
});

// Create barrel exports for moved components
console.log('\n📄 Creating barrel exports...\n');

const barrelExports = {
  'ui/atoms/index.ts': `// Atomic UI components
export { Button } from './Button';
export { Input } from './Input';
export { Label } from './Label';
export { Badge } from './Badge';
export { Avatar } from './Avatar';
export { Separator } from './Separator';
export { Skeleton } from './Skeleton';
export { Switch } from './Switch';
export { Checkbox } from './Checkbox';
`,
  
  'ui/molecules/index.ts': `// Molecular UI components
export { Card } from './Card';
export { Alert } from './Alert';
export { Dialog } from './Dialog';
export { DropdownMenu } from './DropdownMenu';
export { Select } from './Select';
export { Tabs } from './Tabs';
export { Accordion } from './Accordion';
export { Toast } from './Toast';
export { Toaster } from './Toaster';
export { useToast } from './useToast';
`,

  'ui/organisms/index.ts': `// Organism UI components
export { Navigation } from './Navigation';
export { RepzLogo } from './RepzLogo';
`,

  'features/auth/index.ts': `// Auth feature exports
export * from './components';
export * from './hooks';
`,

  'features/pricing/index.ts': `// Pricing feature exports
export * from './components';
export * from './hooks';
`,

  'features/dashboard/index.ts': `// Dashboard feature exports
export * from './components';
export * from './hooks';
`,

  'features/analytics/index.ts': `// Analytics feature exports
export * from './components';
export * from './hooks';
`,

  'features/ai/index.ts': `// AI feature exports
export * from './components';
`
};

Object.entries(barrelExports).forEach(([filePath, content]) => {
  const fullPath = path.join(srcDir, filePath);
  const dir = path.dirname(fullPath);
  
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true });
  }
  
  if (!fs.existsSync(fullPath)) {
    fs.writeFileSync(fullPath, content);
    console.log(`✅ Created: ${filePath}`);
    stats.created++;
  }
});

// Update main features index
const mainFeaturesIndex = `// Feature modules
export * as auth from './auth';
export * as pricing from './pricing';
export * as dashboard from './dashboard';
export * as analytics from './analytics';
export * as ai from './ai';
`;

fs.writeFileSync(path.join(srcDir, 'features/index.ts'), mainFeaturesIndex);
stats.created++;

// Summary
console.log('\n📊 Partial Refactoring Summary:');
console.log('─'.repeat(50));
console.log(`Files moved: ${stats.moved}`);
console.log(`Files created: ${stats.created}`);
console.log(`Errors: ${stats.errors}`);
console.log('─'.repeat(50));

console.log('\n✅ Partial source refactoring complete!');
console.log('\n📁 New structure preview:');
console.log(`
src/
├── features/
│   ├── auth/
│   │   ├── components/
│   │   │   ├── LoginForm.tsx
│   │   │   ├── SignupForm.tsx
│   │   │   └── ...
│   │   ├── hooks/
│   │   │   └── useAuth.tsx
│   │   └── index.ts
│   ├── pricing/
│   ├── dashboard/
│   ├── analytics/
│   └── ai/
├── ui/
│   ├── atoms/
│   │   ├── Button.tsx
│   │   ├── Input.tsx
│   │   └── ...
│   ├── molecules/
│   │   ├── Card.tsx
│   │   ├── Dialog.tsx
│   │   └── ...
│   └── organisms/
│       ├── Navigation.tsx
│       └── RepzLogo.tsx
└── components/ (remaining)
`);

process.exit(0);