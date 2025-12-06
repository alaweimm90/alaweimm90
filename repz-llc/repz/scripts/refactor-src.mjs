#!/usr/bin/env node

import fs from 'fs';
import path from 'path';
import { execSync } from 'child_process';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const rootDir = path.resolve(__dirname, '..');
const srcDir = path.join(rootDir, 'src');

console.log('🔧 Refactoring src/ to feature-first structure...\n');

// Define feature categories based on analysis
const featureMap = {
  auth: [
    'Login', 'Signup', 'Register', 'Auth', 'ProtectedRoute', 'Session',
    'Password', 'Forgot', 'Reset', 'Verification', 'OnboardingLayout'
  ],
  dashboard: [
    'Dashboard', 'AdminDashboard', 'ClientDashboard', 'CoachDashboard',
    'DashboardLayout', 'Overview', 'Stats', 'Metrics'
  ],
  pricing: [
    'Pricing', 'Price', 'Tier', 'Subscription', 'Billing', 'Payment',
    'Checkout', 'Plan', 'Package', 'Upgrade'
  ],
  profile: [
    'Profile', 'UserProfile', 'ClientProfile', 'CoachProfile', 'Settings',
    'Account', 'PersonalInfo', 'Preferences'
  ],
  workout: [
    'Workout', 'Exercise', 'Training', 'Program', 'Routine', 'Fitness',
    'Activity', 'Movement', 'Rep', 'Set'
  ],
  nutrition: [
    'Nutrition', 'Meal', 'Diet', 'Food', 'Calorie', 'Macro', 'Recipe',
    'NutritionPlan', 'MealPlan'
  ],
  analytics: [
    'Analytics', 'Chart', 'Graph', 'Report', 'Insight', 'Metric',
    'Progress', 'Tracking', 'Performance'
  ],
  intake: [
    'Intake', 'Onboarding', 'Welcome', 'GettingStarted', 'Setup',
    'InitialAssessment', 'Questionnaire'
  ],
  communication: [
    'Message', 'Chat', 'Notification', 'Email', 'SMS', 'Alert',
    'Communication', 'Contact'
  ],
  ai: [
    'AI', 'ML', 'Prediction', 'Recommendation', 'Smart', 'Auto',
    'Intelligent', 'Suggest'
  ],
  mobile: [
    'Mobile', 'Responsive', 'Touch', 'Gesture', 'PWA', 'Offline',
    'Native', 'Capacitor'
  ],
  testing: [
    'Test', 'Testing', 'Debug', 'Development', 'Demo', 'Sandbox',
    'Mock', 'Stub'
  ],
  landing: [
    'Landing', 'Home', 'Hero', 'Feature', 'CTA', 'About', 'Contact',
    'Privacy', 'Terms', 'RepzHome'
  ],
  admin: [
    'Admin', 'Management', 'System', 'Config', 'Control', 'Moderate',
    'SuperUser', 'BackOffice'
  ]
};

// Atomic design categories
const atomicMap = {
  atoms: [
    'Button', 'Input', 'Label', 'Icon', 'Badge', 'Avatar', 'Spinner',
    'Toggle', 'Checkbox', 'Radio', 'Link', 'Text', 'Heading'
  ],
  molecules: [
    'Card', 'Form', 'Modal', 'Dropdown', 'Menu', 'Tabs', 'Accordion',
    'Alert', 'Toast', 'Tooltip', 'Popover', 'Dialog'
  ],
  organisms: [
    'Header', 'Footer', 'Navigation', 'Sidebar', 'Layout', 'Section',
    'List', 'Table', 'Grid', 'Carousel', 'Timeline'
  ],
  templates: [
    'PageTemplate', 'DashboardTemplate', 'FormTemplate', 'ListTemplate',
    'DetailTemplate', 'EmptyState'
  ]
};

// Create feature directories
console.log('📁 Creating feature directories...\n');

const featuresDir = path.join(srcDir, 'features');
if (!fs.existsSync(featuresDir)) {
  fs.mkdirSync(featuresDir, { recursive: true });
}

Object.keys(featureMap).forEach(feature => {
  const featureDir = path.join(featuresDir, feature);
  if (!fs.existsSync(featureDir)) {
    fs.mkdirSync(featureDir, { recursive: true });
    console.log(`📁 Created: features/${feature}/`);
  }
  
  // Create subdirectories
  ['components', 'hooks', 'utils', 'types', 'api'].forEach(subdir => {
    const subdirPath = path.join(featureDir, subdir);
    if (!fs.existsSync(subdirPath)) {
      fs.mkdirSync(subdirPath, { recursive: true });
    }
  });
});

// Create UI directories for atomic design
console.log('\n📁 Creating atomic design structure...\n');

const uiDir = path.join(srcDir, 'ui');
if (!fs.existsSync(uiDir)) {
  fs.mkdirSync(uiDir, { recursive: true });
}

['atoms', 'molecules', 'organisms', 'templates', 'theme'].forEach(category => {
  const categoryDir = path.join(uiDir, category);
  if (!fs.existsSync(categoryDir)) {
    fs.mkdirSync(categoryDir, { recursive: true });
    console.log(`📁 Created: ui/${category}/`);
  }
});

// Statistics
const stats = {
  moved: 0,
  created: 0,
  errors: 0
};

// Function to determine feature from file/component name
function getFeatureFromName(name) {
  for (const [feature, keywords] of Object.entries(featureMap)) {
    if (keywords.some(keyword => name.includes(keyword))) {
      return feature;
    }
  }
  return 'shared'; // Default fallback
}

// Function to determine atomic category
function getAtomicCategory(name) {
  for (const [category, keywords] of Object.entries(atomicMap)) {
    if (keywords.some(keyword => name.includes(keyword))) {
      return category;
    }
  }
  return null;
}

// Function to move file
function moveFileToFeature(source, destination) {
  try {
    const destDir = path.dirname(destination);
    if (!fs.existsSync(destDir)) {
      fs.mkdirSync(destDir, { recursive: true });
    }
    
    // Try git mv first
    try {
      execSync(`git mv "${source}" "${destination}"`, { 
        stdio: 'pipe',
        cwd: rootDir 
      });
    } catch {
      // Fallback to regular move
      fs.renameSync(source, destination);
    }
    
    console.log(`✅ Moved: ${path.relative(srcDir, source)} → ${path.relative(srcDir, destination)}`);
    stats.moved++;
    return true;
  } catch (error) {
    console.error(`❌ Error moving ${path.relative(srcDir, source)}:`, error.message);
    stats.errors++;
    return false;
  }
}

// Process components directory
console.log('\n🔍 Processing components directory...\n');

const componentsDir = path.join(srcDir, 'components');
if (fs.existsSync(componentsDir)) {
  // Recursively process all component files
  function processDirectory(dir) {
    const items = fs.readdirSync(dir);
    
    items.forEach(item => {
      const itemPath = path.join(dir, item);
      const stat = fs.statSync(itemPath);
      
      if (stat.isDirectory()) {
        // Check if it's a UI component directory
        const atomicCategory = getAtomicCategory(item);
        if (atomicCategory && dir === componentsDir) {
          // This is a UI category directory, move its contents
          const uiCategoryDir = path.join(uiDir, atomicCategory);
          const subItems = fs.readdirSync(itemPath);
          
          subItems.forEach(subItem => {
            const subItemPath = path.join(itemPath, subItem);
            const destPath = path.join(uiCategoryDir, subItem);
            moveFileToFeature(subItemPath, destPath);
          });
          
          // Remove empty directory
          try {
            fs.rmdirSync(itemPath);
          } catch {}
        } else {
          // Check if it's a feature directory
          const feature = getFeatureFromName(item);
          if (feature !== 'shared' && dir === componentsDir) {
            // Move entire directory to feature
            const destPath = path.join(featuresDir, feature, 'components', item);
            moveFileToFeature(itemPath, destPath);
          } else {
            // Process subdirectory
            processDirectory(itemPath);
          }
        }
      } else if (stat.isFile() && (item.endsWith('.tsx') || item.endsWith('.ts'))) {
        // Determine where this file should go
        const baseName = path.basename(item, path.extname(item));
        
        // Check if it's an atomic UI component
        const atomicCategory = getAtomicCategory(baseName);
        if (atomicCategory) {
          const destPath = path.join(uiDir, atomicCategory, item);
          moveFileToFeature(itemPath, destPath);
        } else {
          // Determine feature
          const feature = getFeatureFromName(baseName);
          const destPath = path.join(featuresDir, feature, 'components', item);
          moveFileToFeature(itemPath, destPath);
        }
      }
    });
  }
  
  processDirectory(componentsDir);
}

// Process hooks directory
console.log('\n🔍 Processing hooks directory...\n');

const hooksDir = path.join(srcDir, 'hooks');
if (fs.existsSync(hooksDir)) {
  const hooks = fs.readdirSync(hooksDir);
  
  hooks.forEach(hook => {
    const hookPath = path.join(hooksDir, hook);
    if (fs.statSync(hookPath).isFile()) {
      const feature = getFeatureFromName(hook);
      const destPath = path.join(featuresDir, feature, 'hooks', hook);
      moveFileToFeature(hookPath, destPath);
    }
  });
}

// Process utils directory
console.log('\n🔍 Processing utils directory...\n');

const utilsDir = path.join(srcDir, 'utils');
if (fs.existsSync(utilsDir)) {
  const utils = fs.readdirSync(utilsDir);
  
  utils.forEach(util => {
    const utilPath = path.join(utilsDir, util);
    if (fs.statSync(utilPath).isFile()) {
      const feature = getFeatureFromName(util);
      const destPath = path.join(featuresDir, feature, 'utils', util);
      moveFileToFeature(utilPath, destPath);
    }
  });
}

// Process pages directory
console.log('\n🔍 Processing pages directory...\n');

const pagesDir = path.join(srcDir, 'pages');
if (fs.existsSync(pagesDir)) {
  function processPages(dir) {
    const items = fs.readdirSync(dir);
    
    items.forEach(item => {
      const itemPath = path.join(dir, item);
      const stat = fs.statSync(itemPath);
      
      if (stat.isDirectory()) {
        processPages(itemPath);
      } else if (stat.isFile() && (item.endsWith('.tsx') || item.endsWith('.ts'))) {
        const feature = getFeatureFromName(item);
        const relativePath = path.relative(pagesDir, itemPath);
        const destPath = path.join(featuresDir, feature, 'pages', relativePath);
        moveFileToFeature(itemPath, destPath);
      }
    });
  }
  
  processPages(pagesDir);
}

// Move theme-related files to ui/theme
console.log('\n🔍 Moving theme files...\n');

const themeFiles = ['theme.ts', 'colors.ts', 'global.css', 'tokens.json'];
themeFiles.forEach(file => {
  const possiblePaths = [
    path.join(srcDir, file),
    path.join(srcDir, 'styles', file),
    path.join(srcDir, 'lib', file),
    path.join(rootDir, file)
  ];
  
  for (const filePath of possiblePaths) {
    if (fs.existsSync(filePath)) {
      const destPath = path.join(uiDir, 'theme', file);
      moveFileToFeature(filePath, destPath);
      break;
    }
  }
});

// Create barrel exports for features
console.log('\n📄 Creating barrel exports...\n');

Object.keys(featureMap).forEach(feature => {
  const indexPath = path.join(featuresDir, feature, 'index.ts');
  if (!fs.existsSync(indexPath)) {
    const indexContent = `// ${feature} feature exports
export * from './components';
export * from './hooks';
export * from './utils';
export * from './types';
`;
    fs.writeFileSync(indexPath, indexContent);
    stats.created++;
  }
});

// Create main features index
const featuresIndexPath = path.join(featuresDir, 'index.ts');
const featuresIndexContent = Object.keys(featureMap)
  .map(feature => `export * as ${feature} from './${feature}';`)
  .join('\n');

fs.writeFileSync(featuresIndexPath, featuresIndexContent + '\n');
stats.created++;

// Create UI index files
['atoms', 'molecules', 'organisms', 'templates'].forEach(category => {
  const indexPath = path.join(uiDir, category, 'index.ts');
  if (!fs.existsSync(indexPath)) {
    fs.writeFileSync(indexPath, `// ${category} exports\n`);
    stats.created++;
  }
});

// Clean up empty directories
console.log('\n🧹 Cleaning up empty directories...\n');

function removeEmptyDirs(dir) {
  if (!fs.existsSync(dir)) return;
  
  let files = fs.readdirSync(dir);
  if (files.length > 0) {
    files.forEach(file => {
      const fullPath = path.join(dir, file);
      if (fs.statSync(fullPath).isDirectory()) {
        removeEmptyDirs(fullPath);
      }
    });
    files = fs.readdirSync(dir);
  }
  
  if (files.length === 0 && dir !== srcDir) {
    fs.rmdirSync(dir);
    console.log(`🗑️  Removed empty directory: ${path.relative(srcDir, dir)}`);
  }
}

removeEmptyDirs(path.join(srcDir, 'components'));
removeEmptyDirs(path.join(srcDir, 'pages'));
removeEmptyDirs(path.join(srcDir, 'hooks'));
removeEmptyDirs(path.join(srcDir, 'utils'));

// Summary
console.log('\n📊 Source Refactoring Summary:');
console.log('─'.repeat(50));
console.log(`Files moved: ${stats.moved}`);
console.log(`Files created: ${stats.created}`);
console.log(`Errors: ${stats.errors}`);
console.log('─'.repeat(50));
console.log('\n✅ Source code has been refactored to feature-first structure!');

// Document the new structure
const newStructure = `
src/
├── features/
│   ├── auth/
│   │   ├── components/
│   │   ├── hooks/
│   │   ├── utils/
│   │   ├── types/
│   │   └── api/
│   ├── dashboard/
│   ├── pricing/
│   ├── profile/
│   ├── workout/
│   ├── nutrition/
│   ├── analytics/
│   ├── intake/
│   ├── communication/
│   ├── ai/
│   ├── mobile/
│   ├── testing/
│   ├── landing/
│   ├── admin/
│   └── shared/
├── ui/
│   ├── atoms/
│   ├── molecules/
│   ├── organisms/
│   ├── templates/
│   └── theme/
├── App.tsx
├── main.tsx
└── index.css
`;

console.log('\n📁 New structure:', newStructure);

process.exit(0);