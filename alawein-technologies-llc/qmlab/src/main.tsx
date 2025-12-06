import { createRoot } from 'react-dom/client'
import { AppQuantum } from './AppQuantum.tsx'
import './index.css'
import './styles/professional.css'
import './styles/quantum-aesthetic.css'

// Import accessibility validator in development
if (import.meta.env.DEV) {
  import('./utils/accessibility-validator').then(({ accessibilityValidator }) => {
    // Make validator available globally in dev mode
    (window as any).a11yValidator = accessibilityValidator;
    (window as any).runA11yAudit = async () => {
      const issues = await accessibilityValidator.runAudit();
      console.log(accessibilityValidator.generateReport());
      return issues;
    };
    console.log('🔍 Accessibility validator loaded. Run window.runA11yAudit() to test.');
  });
}

createRoot(document.getElementById("root")!).render(<AppQuantum />);
