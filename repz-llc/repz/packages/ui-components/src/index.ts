/**
 * REPZ UI Components Library
 * Enterprise-grade React component library built with atomic design principles
 * 
 * @version 1.0.0
 * @author REPZ Team
 * @license MIT
 */

// Export all atoms
export * from './atoms';

// Export utilities and types
export * from './lib/utils';
export * from './types';

// Export design tokens (for JavaScript consumers)
export { designTokens } from './tokens';

// CSS imports for bundlers that support it
import './styles/globals.css';
