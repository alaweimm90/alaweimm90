import { createRoot } from 'react-dom/client'
import App from './App.tsx'
import './index.css'
import './styles/brand-effects.css'

import { pwaManager } from './lib/pwa-utils'
import { onCLS, onFID, onLCP } from 'web-vitals'
import { AccessibilityTester } from './lib/accessibility-testing'

// Register service worker
pwaManager.registerServiceWorker();

// Run accessibility tests in development
if (import.meta.env.DEV) {
  setTimeout(async () => {
    const issues = await AccessibilityTester.runBasicTests();
    AccessibilityTester.logResults(issues);
  }, 2000);
}

// Minimal Web Vitals logging – replace console.log with analytics sink if desired
onCLS(console.log);
onFID(console.log);
onLCP(console.log);

createRoot(document.getElementById("root")!).render(<App />);
