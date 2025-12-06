import { createRoot } from 'react-dom/client';
import App from './App.tsx';
import './index.css';
import './styles/brand.css';
import { initWebVitals } from './lib/webVitals.ts';

// Initialize Web Vitals monitoring as early as possible
// This helps track LCP, INP, CLS, and other performance metrics
if (import.meta.env.PROD) {
  initWebVitals();
}

// Register service worker for caching and performance
if ('serviceWorker' in navigator && import.meta.env.PROD) {
  window.addEventListener('load', () => {
    navigator.serviceWorker
      .register('/sw.js')
      .then(() => {
        // Service worker registered successfully
      })
      .catch(() => {
        // Service worker registration failed
      });
  });
}

createRoot(document.getElementById('root')!).render(<App />);
