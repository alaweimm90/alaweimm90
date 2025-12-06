import { useEffect, useState } from 'react';
import { isFeatureEnabled } from '@/config/feature-flags';
import { cn } from '@/lib/utils';

interface SemanticThemeConfig {
  useSemanticTokens: boolean;
  useModernAnimations: boolean;
  useOptimizedStyles: boolean;
}

export const useSemanticTheme = (): SemanticThemeConfig => {
  const [config, setConfig] = useState<SemanticThemeConfig>({
    useSemanticTokens: false,
    useModernAnimations: false,
    useOptimizedStyles: false,
  });

  useEffect(() => {
    setConfig({
      useSemanticTokens: isFeatureEnabled('USE_SEMANTIC_TOKENS'),
      useModernAnimations: isFeatureEnabled('USE_MODERN_ANIMATIONS'),
      useOptimizedStyles: isFeatureEnabled('USE_OPTIMIZED_STYLES'),
    });
  }, []);

  return config;
};

/**
 * Utility function to conditionally apply semantic theme classes
 * Helps with gradual migration from hardcoded colors to semantic tokens
 */
export const useThemeClasses = () => {
  const theme = useSemanticTheme();

  return {
    // Background utilities
    bgPrimary: theme.useSemanticTokens ? 'bg-brand-primary' : 'bg-orange-500',
    bgSecondary: theme.useSemanticTokens ? 'bg-brand-secondary' : 'bg-blue-500',
    bgAccent: theme.useSemanticTokens ? 'bg-brand-accent' : 'bg-amber-500',
    bgCard: theme.useSemanticTokens ? 'card-elegant' : 'bg-gray-800',
    bgGlass: theme.useSemanticTokens ? 'glass-medium' : 'bg-gray-800/80 backdrop-blur-sm',

    // Text utilities
    textPrimary: theme.useSemanticTokens ? 'text-brand-primary' : 'text-orange-500',
    textSecondary: theme.useSemanticTokens ? 'text-brand-secondary' : 'text-blue-500',
    textAccent: theme.useSemanticTokens ? 'text-brand-accent' : 'text-amber-500',
    textHeading: theme.useSemanticTokens ? 'text-heading' : 'text-white',
    textBody: theme.useSemanticTokens ? 'text-body' : 'text-gray-100',
    textMuted: theme.useSemanticTokens ? 'text-muted' : 'text-gray-400',

    // Button utilities
    btnPrimary: theme.useSemanticTokens ? 'btn-brand' : 'bg-orange-500 text-white hover:bg-orange-600',
    btnSecondary: theme.useSemanticTokens ? 'btn-secondary' : 'bg-blue-500 text-white hover:bg-blue-600',
    btnAccent: theme.useSemanticTokens ? 'btn-accent' : 'bg-amber-500 text-black hover:bg-amber-600',

    // Animation utilities
    animateFloat: theme.useModernAnimations ? 'animate-elegant-float' : '',
    animatePulse: theme.useModernAnimations ? 'animate-sophisticated-pulse' : 'animate-pulse',
    animateGlow: theme.useModernAnimations ? 'animate-premium-glow' : '',
    animateSlide: theme.useModernAnimations ? 'animate-elegant-slide' : 'animate-slide-in-left',

    // Transition utilities
    transitionSmooth: theme.useOptimizedStyles ? 'transition-smooth ease-elegant' : 'transition-all duration-300',
    transitionElegant: theme.useOptimizedStyles ? 'transition-elegant ease-elegant' : 'transition-all duration-500',

    // Shadow utilities
    shadowElegant: theme.useSemanticTokens ? 'shadow-elegant' : 'shadow-2xl',
    shadowFloating: theme.useSemanticTokens ? 'shadow-floating' : 'shadow-xl',
    shadowSubtle: theme.useSemanticTokens ? 'shadow-subtle' : 'shadow-md',

    // Status utilities
    statusSuccess: theme.useSemanticTokens ? 'status-success' : 'bg-green-500 text-white',
    statusWarning: theme.useSemanticTokens ? 'status-warning' : 'bg-yellow-500 text-black',
    statusError: theme.useSemanticTokens ? 'status-error' : 'bg-red-500 text-white',
    statusInfo: theme.useSemanticTokens ? 'status-info' : 'bg-blue-500 text-white',
  };
};

/**
 * Higher-order function to create theme-aware className builders
 */
export const createThemeClasses = (baseClasses: string, semanticClasses?: string) => {
  const theme = useSemanticTheme();
  return cn(
    baseClasses,
    theme.useSemanticTokens && semanticClasses ? semanticClasses : ''
  );
};

/**
 * Utility to get CSS custom property values for use in inline styles
 */
export const getSemanticTokenValue = (tokenName: string): string => {
  if (typeof window === 'undefined') return '';
  
  const rootStyles = getComputedStyle(document.documentElement);
  return rootStyles.getPropertyValue(`--${tokenName}`).trim();
};

/**
 * CSS-in-JS style generator using semantic tokens
 */
export const useSemanticStyles = () => {
  const theme = useSemanticTheme();
  
  if (!theme.useSemanticTokens) return {};
  
  return {
    brandPrimary: {
      backgroundColor: `hsl(${getSemanticTokenValue('repz-orange')})`,
      color: 'white',
    },
    brandSecondary: {
      backgroundColor: `hsl(${getSemanticTokenValue('blue-steel')})`,
      color: 'white',
    },
    brandAccent: {
      backgroundColor: `hsl(${getSemanticTokenValue('amber-elegant')})`,
      color: `hsl(${getSemanticTokenValue('charcoal-premium')})`,
    },
    cardGlass: {
      background: getSemanticTokenValue('surface-glass'),
      backdropFilter: 'blur(16px)',
      WebkitBackdropFilter: 'blur(16px)',
      border: `1px solid hsla(${getSemanticTokenValue('silver-muted')}, 0.3)`,
    },
    shadowElegant: {
      boxShadow: getSemanticTokenValue('shadow-elegant'),
    },
    glowOrange: {
      boxShadow: getSemanticTokenValue('glow-orange'),
    },
  };
};

/**
 * Migration helper - logs usage of old classes for tracking
 */
export const useMigrationLogger = (componentName: string, oldClasses: string[]) => {
  const theme = useSemanticTheme();
  
  useEffect(() => {
    if (theme.useSemanticTokens && oldClasses.length > 0) {
      console.info(`🎨 [Theme Migration] ${componentName} is using legacy classes:`, oldClasses);
      console.info('Consider migrating to semantic tokens for better consistency');
    }
  }, [componentName, oldClasses, theme.useSemanticTokens]);
};

export default useSemanticTheme;