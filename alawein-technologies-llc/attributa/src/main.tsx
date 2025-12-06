import { createRoot } from 'react-dom/client'
import { BrowserRouter } from 'react-router-dom'
import App from './App.tsx'
import './index.css'
import { trackWebVitals } from './lib/vitals'

const rootElement = document.getElementById("root");
if (!rootElement) {
  console.error("Failed to find root element");
  document.body.innerHTML = '<div style="text-align: center; padding: 2rem;"><h1>Failed to load application</h1><p>Please refresh the page or clear your browser cache.</p></div>';
} else {
  createRoot(rootElement).render(
    <BrowserRouter
      future={{
        v7_startTransition: true,
        v7_relativeSplatPath: true,
      }}
    >
      <App />
    </BrowserRouter>
  );
}

// Start tracking web vitals
trackWebVitals();

// Register service worker for caching
if ('serviceWorker' in navigator && window.location.hostname !== 'localhost') {
  window.addEventListener('load', () => {
    navigator.serviceWorker.register('/sw.js')
      .then((registration) => {
        console.log('Service Worker registered successfully:', registration.scope);
      })
      .catch((registrationError) => {
        console.warn('Service Worker registration failed (this is normal in development):', registrationError);
      });
  });
}
