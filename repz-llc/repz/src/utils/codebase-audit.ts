/**
 * AUTOMATED CODEBASE AUDIT UTILITIES
 * Tools for detecting dead code, unused components, and architectural issues
 */

import { ROUTES } from '@/constants/routes';

/**
 * Page/Component Usage Tracker
 */
export interface ComponentUsage {
  path: string;
  name: string;
  referencedBy: string[];
  isOrphaned: boolean;
  lastModified?: Date;
}

export interface RouteUsage {
  route: string;
  isValid: boolean;
  isDeprecated: boolean;
  referencedBy: string[];
  redirectsTo?: string;
}

/**
 * Audit configuration
 */
export const AUDIT_CONFIG = {
  // Files to exclude from dead code analysis
  excludePatterns: [
    '**/node_modules/**',
    '**/*.test.tsx',
    '**/*.test.ts',
    '**/dist/**',
    '**/build/**'
  ],
  
  // Required components that should always exist
  coreComponents: [
    'Button',
    'Card', 
    'Modal',
    'Input',
    'Header',
    'Footer'
  ],
  
  // Pages that should always be accessible
  corePages: [
    ROUTES.HOME,
    ROUTES.DASHBOARD,
    ROUTES.LOGIN,
    ROUTES.SIGNUP
  ]
} as const;

/**
 * Simulate dead code detection (would use static analysis in real implementation)
 */
export class CodebaseAuditor {
  private componentUsage: Map<string, ComponentUsage> = new Map();
  private routeUsage: Map<string, RouteUsage> = new Map();
  
  /**
   * Analyze component usage across the codebase
   */
  analyzeComponentUsage(codebaseFiles: string[]): ComponentUsage[] {
    // In a real implementation, this would:
    // 1. Parse all TSX/TS files using AST
    // 2. Track import statements and component usage
    // 3. Identify orphaned components
    
    const mockResults: ComponentUsage[] = [
      {
        path: 'src/components/deprecated/OldPricingCard.tsx',
        name: 'OldPricingCard', 
        referencedBy: [],
        isOrphaned: true
      },
      {
        path: 'src/pages/OldPlansPage.tsx',
        name: 'OldPlansPage',
        referencedBy: [],
        isOrphaned: true
      },
      {
        path: 'src/components/pricing/CentralizedTierCard.tsx',
        name: 'CentralizedTierCard',
        referencedBy: ['MonthlyCoachingPrices.tsx', 'InPersonTraining.tsx'],
        isOrphaned: false
      }
    ];
    
    return mockResults;
  }
  
  /**
   * Analyze route usage and identify broken links
   */
  analyzeRouteUsage(codebaseFiles: string[]): RouteUsage[] {
    const results: RouteUsage[] = [];
    
    // Check all valid routes
    Object.values(ROUTES).forEach(route => {
      results.push({
        route,
        isValid: true,
        isDeprecated: false,
        referencedBy: [], // Would be populated by AST analysis
      });
    });
    
    // Check deprecated routes
    const deprecatedRoutes = [
      '/plans', '/prices', '/repz-home', '/intake', '/calendar', '/workouts/today', '/marketplace'
    ];
    
    deprecatedRoutes.forEach(route => {
      results.push({
        route,
        isValid: false,
        isDeprecated: true,
        referencedBy: [], // Would show where these are still referenced
        redirectsTo: this.getRedirectTarget(route)
      });
    });
    
    return results;
  }
  
  /**
   * Generate audit report
   */
  generateAuditReport(): AuditReport {
    const componentIssues = this.analyzeComponentUsage([]);
    const routeIssues = this.analyzeRouteUsage([]);
    
    return {
      timestamp: new Date(),
      orphanedComponents: componentIssues.filter(c => c.isOrphaned),
      deprecatedRoutes: routeIssues.filter(r => r.isDeprecated),
      brokenLinks: routeIssues.filter(r => !r.isValid && !r.isDeprecated),
      recommendations: this.generateRecommendations(componentIssues, routeIssues)
    };
  }
  
  private getRedirectTarget(deprecatedRoute: string): string {
    // Map deprecated routes to their redirects
    const redirectMap: Record<string, string> = {
      '/plans': ROUTES.HOME,
      '/prices': ROUTES.HOME,
      '/repz-home': ROUTES.HOME,
      '/monthly-coaching-prices': ROUTES.MONTHLY_COACHING,
      '/in-person-training-prices': ROUTES.IN_PERSON
    };
    
    return redirectMap[deprecatedRoute] || ROUTES.HOME;
  }
  
  private generateRecommendations(
    componentIssues: ComponentUsage[], 
    routeIssues: RouteUsage[]
  ): string[] {
    const recommendations: string[] = [];
    
    const orphanedCount = componentIssues.filter(c => c.isOrphaned).length;
    if (orphanedCount > 0) {
      recommendations.push(`Remove ${orphanedCount} orphaned components`);
    }
    
    const deprecatedCount = routeIssues.filter(r => r.isDeprecated).length;
    if (deprecatedCount > 0) {
      recommendations.push(`Clean up ${deprecatedCount} deprecated routes`);
    }
    
    recommendations.push('Ensure all navigation uses centralized ROUTES constants');
    recommendations.push('Verify all components use the centralized design system');
    
    return recommendations;
  }
}

export interface AuditReport {
  timestamp: Date;
  orphanedComponents: ComponentUsage[];
  deprecatedRoutes: RouteUsage[];
  brokenLinks: RouteUsage[];
  recommendations: string[];
}

/**
 * Helper to validate navigation consistency
 */
export const validateNavigation = (navigationItems: Array<{ label?: string; path: string }>): ValidationResult[] => {
  const results: ValidationResult[] = [];
  
  navigationItems.forEach(item => {
    if (!Object.values(ROUTES).includes(item.path)) {
      results.push({
        type: 'error',
        message: `Navigation item "${item.label}" links to invalid route: ${item.path}`,
        suggestion: 'Use a route from ROUTES constant or update the route definition'
      });
    }
  });
  
  return results;
};

export interface ValidationResult {
  type: 'error' | 'warning' | 'info';
  message: string;
  suggestion?: string;
}

/**
 * Theme consistency checker
 */
export const checkThemeConsistency = (componentFiles: string[]): ValidationResult[] => {
  const results: ValidationResult[] = [];
  
  // In real implementation, would check for:
  // - Hardcoded colors instead of CSS variables
  // - Direct Tailwind classes instead of design tokens
  // - Inconsistent spacing/typography
  
  // Mock results for demonstration
  results.push({
    type: 'warning',
    message: 'Found hardcoded color values in 3 components',
    suggestion: 'Use design system color tokens instead'
  });
  
  return results;
};

/**
 * Export singleton auditor instance
 */
export const auditor = new CodebaseAuditor();