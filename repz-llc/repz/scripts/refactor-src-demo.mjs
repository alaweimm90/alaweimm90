#!/usr/bin/env node

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const rootDir = path.resolve(__dirname, '..');
const srcDir = path.join(rootDir, 'src');

console.log('🔧 Demo: Refactoring src/ to feature-first structure...\n');

// Create feature directories
const features = [
  'auth', 'dashboard', 'pricing', 'profile', 'workout', 
  'nutrition', 'analytics', 'intake', 'communication',
  'ai', 'mobile', 'testing', 'landing', 'admin', 'shared'
];

console.log('📁 Creating feature structure...\n');

const featuresDir = path.join(srcDir, 'features');
if (!fs.existsSync(featuresDir)) {
  fs.mkdirSync(featuresDir, { recursive: true });
}

features.forEach(feature => {
  const featureDir = path.join(featuresDir, feature);
  if (!fs.existsSync(featureDir)) {
    fs.mkdirSync(featureDir, { recursive: true });
    console.log(`✅ Created: features/${feature}/`);
    
    // Create subdirectories
    ['components', 'hooks', 'utils', 'types', 'api', 'pages'].forEach(subdir => {
      fs.mkdirSync(path.join(featureDir, subdir), { recursive: true });
    });
  }
});

// Create UI atomic design structure
console.log('\n📁 Creating atomic design structure...\n');

const uiDir = path.join(srcDir, 'ui');
['atoms', 'molecules', 'organisms', 'templates', 'theme'].forEach(category => {
  const categoryDir = path.join(uiDir, category);
  if (!fs.existsSync(categoryDir)) {
    fs.mkdirSync(categoryDir, { recursive: true });
    console.log(`✅ Created: ui/${category}/`);
  }
});

// Show example file mappings
console.log('\n📋 Example file mappings:');
console.log('─'.repeat(50));

const exampleMappings = [
  { from: 'components/auth/LoginForm.tsx', to: 'features/auth/components/LoginForm.tsx' },
  { from: 'components/ui/Button.tsx', to: 'ui/atoms/Button.tsx' },
  { from: 'components/dashboard/ClientDashboard.tsx', to: 'features/dashboard/components/ClientDashboard.tsx' },
  { from: 'hooks/useAuth.tsx', to: 'features/auth/hooks/useAuth.tsx' },
  { from: 'pages/Pricing.tsx', to: 'features/pricing/pages/Pricing.tsx' },
  { from: 'components/ui/Card.tsx', to: 'ui/molecules/Card.tsx' },
  { from: 'components/Navigation.tsx', to: 'ui/organisms/Navigation.tsx' }
];

exampleMappings.forEach(({ from, to }) => {
  console.log(`${from} → ${to}`);
});

console.log('─'.repeat(50));

// Create example index files
const authIndexContent = `// Auth feature exports
export * from './components';
export * from './hooks';
export * from './utils';
export * from './types';
`;

fs.writeFileSync(path.join(featuresDir, 'auth/index.ts'), authIndexContent);

const featuresIndexContent = features
  .map(feature => `export * as ${feature} from './${feature}';`)
  .join('\n') + '\n';

fs.writeFileSync(path.join(featuresDir, 'index.ts'), featuresIndexContent);

// Summary
console.log('\n✅ Demo refactoring structure created!');
console.log('\nNew structure:');
console.log(`
src/
├── features/
│   ├── auth/
│   │   ├── components/
│   │   ├── hooks/
│   │   ├── utils/
│   │   ├── types/
│   │   ├── api/
│   │   └── pages/
│   ├── dashboard/
│   ├── pricing/
│   └── ... (12 more features)
├── ui/
│   ├── atoms/
│   ├── molecules/
│   ├── organisms/
│   ├── templates/
│   └── theme/
├── App.tsx
├── main.tsx
└── index.css
`);

console.log('\n💡 To complete the refactoring, run the full refactor-src.mjs script');

process.exit(0);