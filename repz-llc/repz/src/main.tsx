import { createRoot } from 'react-dom/client'
import App from './App.tsx'
import './index.css'
import { initializeMonitoring } from './utils/monitoring'

// Initialize monitoring before app starts
initializeMonitoring();

createRoot(document.getElementById("root")!).render(<App />);
