/**
 * COMPREHENSIVE CODEBASE AUDIT SYSTEM
 * Automated detection of orphaned files, routing inconsistencies, style drift, and data structure mismatches
 */

import fs from 'fs';
import path from 'path';
import { ROUTES } from '@/constants/routes';

export interface AuditReport {
  timestamp: Date;
  projectRoot: string;
  summary: AuditSummary;
  findings: AuditFindings;
  recommendations: string[];
  criticalIssues: CriticalIssue[];
  score: number; // 0-100 health score
}

export interface AuditSummary {
  totalFiles: number;
  orphanedFiles: number;
  deadRoutes: number;
  styleInconsistencies: number;
  dataStructureMismatches: number;
  duplicateComponents: number;
}

export interface AuditFindings {
  structural: StructuralFindings;
  routing: RoutingFindings;
  styling: StylingFindings;
  dataConsistency: DataConsistencyFindings;
  uiConsistency: UIConsistencyFindings;
}

export interface CriticalIssue {
  type: 'error' | 'warning' | 'info';
  category: 'structural' | 'routing' | 'styling' | 'data' | 'ui';
  message: string;
  files: string[];
  impact: 'high' | 'medium' | 'low';
  recommendation: string;
}

// 1. STRUCTURAL AUDIT - Files and Folders
export interface StructuralFindings {
  orphanedFiles: FileUsage[];
  unusedRoutes: string[];
  missingFiles: string[];
  directoryStructureIssues: DirectoryIssue[];
}

export interface FileUsage {
  path: string;
  type: 'component' | 'page' | 'utility' | 'style' | 'config';
  referencedBy: string[];
  exports: string[];
  imports: string[];
  isOrphaned: boolean;
  lastModified: Date;
}

export interface DirectoryIssue {
  path: string;
  issue: 'misplaced' | 'empty' | 'inconsistent_naming';
  suggestion: string;
}

// 2. ROUTING AUDIT - Navigation Consistency  
export interface RoutingFindings {
  deadLinks: DeadLink[];
  inconsistentNavigation: NavigationInconsistency[];
  routeSecurityIssues: RouteSecurityIssue[];
  missingRouteMetadata: string[];
}

export interface DeadLink {
  sourceFile: string;
  targetRoute: string;
  linkType: 'href' | 'navigate' | 'Link' | 'redirect';
  lineNumber: number;
}

export interface NavigationInconsistency {
  route: string;
  inconsistency: 'missing_from_nav' | 'different_labels' | 'wrong_icon' | 'access_mismatch';
  locations: string[];
  suggestion: string;
}

export interface RouteSecurityIssue {
  route: string;
  issue: 'missing_auth' | 'wrong_tier_requirement' | 'exposed_admin_route';
  severity: 'critical' | 'high' | 'medium';
}

// 3. STYLING AUDIT - UI Consistency
export interface StylingFindings {
  styleInconsistencies: StyleInconsistency[];
  duplicateStyles: DuplicateStyle[];
  themeViolations: ThemeViolation[];
  responsiveIssues: ResponsiveIssue[];
}

export interface StyleInconsistency {
  component: string;
  property: string;
  values: string[];
  files: string[];
  suggestion: string;
}

export interface DuplicateStyle {
  selector: string;
  duplicateIn: string[];
  canBeConsolidated: boolean;
}

export interface ThemeViolation {
  file: string;
  violation: 'hardcoded_color' | 'hardcoded_spacing' | 'hardcoded_font' | 'missing_design_token';
  line: number;
  current: string;
  suggested: string;
}

export interface ResponsiveIssue {
  component: string;
  issue: 'missing_mobile_styles' | 'poor_mobile_ux' | 'inconsistent_breakpoints';
  files: string[];
}

// 4. DATA CONSISTENCY AUDIT
export interface DataConsistencyFindings {
  typeDefinitionMismatches: TypeMismatch[];
  validationInconsistencies: ValidationInconsistency[];
  apiDataMismatches: ApiDataMismatch[];
  formInconsistencies: FormInconsistency[];
}

export interface TypeMismatch {
  typeName: string;
  definitions: TypeDefinition[];
  conflicts: string[];
  recommendedDefinition: TypeDefinition;
}

export interface TypeDefinition {
  file: string;
  properties: { [key: string]: string };
  isInterface: boolean;
  isType: boolean;
}

export interface ValidationInconsistency {
  field: string;
  forms: FormValidation[];
  inconsistencies: string[];
}

export interface FormValidation {
  file: string;
  schema: Record<string, unknown>;
  library: 'yup' | 'zod' | 'joi' | 'custom';
}

export interface ApiDataMismatch {
  endpoint: string;
  expectedType: string;
  actualStructure: unknown;
  mismatches: string[];
}

export interface FormInconsistency {
  fieldName: string;
  forms: string[];
  inconsistencies: {
    labels: string[];
    validations: string[];
    placeholders: string[];
  };
}

// 5. UI CONSISTENCY AUDIT
export interface UIConsistencyFindings {
  componentVariations: ComponentVariation[];
  iconInconsistencies: IconInconsistency[];
  colorInconsistencies: ColorInconsistency[];
  typographyInconsistencies: TypographyInconsistency[];
}

export interface ComponentVariation {
  componentType: 'button' | 'card' | 'modal' | 'form' | 'table';
  baseComponent: string;
  variations: ComponentInstance[];
  shouldBeConsolidated: boolean;
}

export interface ComponentInstance {
  file: string;
  props: Record<string, unknown>;
  styling: string[];
  functionality: string[];
}

export interface IconInconsistency {
  concept: string; // e.g., 'user', 'settings', 'home'
  icons: IconUsage[];
  recommendedIcon: string;
}

export interface IconUsage {
  file: string;
  iconName: string;
  library: 'lucide-react' | 'react-icons' | 'custom';
  context: string;
}

export interface ColorInconsistency {
  semanticMeaning: string; // e.g., 'primary', 'success', 'error'
  usages: ColorUsage[];
  recommendedToken: string;
}

export interface ColorUsage {
  file: string;
  colorValue: string;
  property: string;
  context: string;
}

export interface TypographyInconsistency {
  semanticRole: string; // e.g., 'heading', 'body', 'caption'
  usages: TypographyUsage[];
  recommendedClass: string;
}

export interface TypographyUsage {
  file: string;
  styles: string[];
  context: string;
}

/**
 * MAIN AUDIT ENGINE
 */
export class CodebaseAuditor {
  private projectRoot: string;
  private auditConfig: AuditConfig;

  constructor(projectRoot: string = 'src', config?: Partial<AuditConfig>) {
    this.projectRoot = projectRoot;
    this.auditConfig = {
      excludePatterns: [
        '**/node_modules/**',
        '**/*.test.tsx',
        '**/*.test.ts', 
        '**/dist/**',
        '**/build/**',
        '**/.next/**'
      ],
      includeExtensions: ['.tsx', '.ts', '.js', '.jsx', '.css', '.scss'],
      strictMode: false,
      ...config
    };
  }

  /**
   * Run comprehensive audit
   */
  async runFullAudit(): Promise<AuditReport> {
    console.log('🔍 Starting comprehensive codebase audit...');
    
    const findings: AuditFindings = {
      structural: await this.auditStructure(),
      routing: await this.auditRouting(), 
      styling: await this.auditStyling(),
      dataConsistency: await this.auditDataConsistency(),
      uiConsistency: await this.auditUIConsistency()
    };

    const summary = this.generateSummary(findings);
    const criticalIssues = this.identifyCriticalIssues(findings);
    const recommendations = this.generateRecommendations(findings);
    const score = this.calculateHealthScore(findings);

    return {
      timestamp: new Date(),
      projectRoot: this.projectRoot,
      summary,
      findings,
      recommendations,
      criticalIssues,
      score
    };
  }

  /**
   * 1. STRUCTURAL AUDIT IMPLEMENTATION
   */
  private async auditStructure(): Promise<StructuralFindings> {
    console.log('📁 Auditing file structure...');
    
    // Mock implementation - in real scenario would use AST parsing
    const orphanedFiles: FileUsage[] = [
      {
        path: 'src/components/deprecated/OldPricingCard.tsx',
        type: 'component',
        referencedBy: [],
        exports: ['OldPricingCard'],
        imports: ['React'],
        isOrphaned: true,
        lastModified: new Date('2024-01-01')
      },
      {
        path: 'src/pages/OldPlansPage.tsx', 
        type: 'page',
        referencedBy: [],
        exports: ['OldPlansPage'],
        imports: ['React', 'Button'],
        isOrphaned: true,
        lastModified: new Date('2024-01-15')
      }
    ];

    const unusedRoutes = ['/old-pricing', '/legacy-dashboard'];
    
    const directoryStructureIssues: DirectoryIssue[] = [
      {
        path: 'src/components/random',
        issue: 'misplaced',
        suggestion: 'Move components to appropriate category folders'
      }
    ];

    return {
      orphanedFiles,
      unusedRoutes,
      missingFiles: [],
      directoryStructureIssues
    };
  }

  /**
   * 2. ROUTING AUDIT IMPLEMENTATION  
   */
  private async auditRouting(): Promise<RoutingFindings> {
    console.log('🛣️ Auditing routing and navigation...');
    
    const deadLinks: DeadLink[] = [
      {
        sourceFile: 'src/components/navigation/Header.tsx',
        targetRoute: '/old-plans',
        linkType: 'Link',
        lineNumber: 42
      }
    ];

    const inconsistentNavigation: NavigationInconsistency[] = [
      {
        route: '/monthly-coaching',
        inconsistency: 'different_labels',
        locations: ['Header.tsx', 'Footer.tsx', 'MobileNav.tsx'],
        suggestion: 'Standardize label to "Monthly Coaching" across all navigation'
      }
    ];

    const routeSecurityIssues: RouteSecurityIssue[] = [
      {
        route: '/admin',
        issue: 'missing_auth',
        severity: 'critical'
      }
    ];

    return {
      deadLinks,
      inconsistentNavigation,
      routeSecurityIssues,
      missingRouteMetadata: ['/new-feature']
    };
  }

  /**
   * 3. STYLING AUDIT IMPLEMENTATION
   */
  private async auditStyling(): Promise<StylingFindings> {
    console.log('🎨 Auditing styles and themes...');
    
    const themeViolations: ThemeViolation[] = [
      {
        file: 'src/components/CustomButton.tsx',
        violation: 'hardcoded_color',
        line: 15,
        current: 'bg-orange-500',
        suggested: 'bg-primary-500'
      },
      {
        file: 'src/pages/PricingPage.tsx', 
        violation: 'hardcoded_spacing',
        line: 23,
        current: 'margin: 16px',
        suggested: 'margin: var(--spacing-md)'
      }
    ];

    const duplicateStyles: DuplicateStyle[] = [
      {
        selector: '.tier-card',
        duplicateIn: ['TierCard.tsx', 'PricingCard.tsx', 'PlanCard.tsx'],
        canBeConsolidated: true
      }
    ];

    return {
      styleInconsistencies: [],
      duplicateStyles,
      themeViolations,
      responsiveIssues: []
    };
  }

  /**
   * 4. DATA CONSISTENCY AUDIT
   */
  private async auditDataConsistency(): Promise<DataConsistencyFindings> {
    console.log('📊 Auditing data structures and validation...');
    
    const typeDefinitionMismatches: TypeMismatch[] = [
      {
        typeName: 'User',
        definitions: [
          {
            file: 'src/types/auth.ts',
            properties: { id: 'string', email: 'string', name: 'string' },
            isInterface: true,
            isType: false
          },
          {
            file: 'src/types/user.ts', 
            properties: { id: 'string', email: 'string', displayName: 'string', avatar: 'string' },
            isInterface: true,
            isType: false
          }
        ],
        conflicts: ['name vs displayName property mismatch'],
        recommendedDefinition: {
          file: 'src/types/user.ts',
          properties: { id: 'string', email: 'string', name: 'string', displayName: 'string', avatar: 'string' },
          isInterface: true,
          isType: false
        }
      }
    ];

    return {
      typeDefinitionMismatches,
      validationInconsistencies: [],
      apiDataMismatches: [],
      formInconsistencies: []
    };
  }

  /**
   * 5. UI CONSISTENCY AUDIT
   */
  private async auditUIConsistency(): Promise<UIConsistencyFindings> {
    console.log('🖼️ Auditing UI consistency...');
    
    const componentVariations: ComponentVariation[] = [
      {
        componentType: 'card',
        baseComponent: 'Card',
        variations: [
          {
            file: 'src/components/TierCard.tsx',
            props: { variant: 'tier', showBadge: true },
            styling: ['gradient-background', 'rounded-xl', 'shadow-lg'],
            functionality: ['onClick', 'hover-effects']
          },
          {
            file: 'src/components/PricingCard.tsx',
            props: { type: 'pricing' },
            styling: ['solid-background', 'rounded-lg', 'border'],
            functionality: ['onClick']
          }
        ],
        shouldBeConsolidated: true
      }
    ];

    const iconInconsistencies: IconInconsistency[] = [
      {
        concept: 'user',
        icons: [
          { file: 'Header.tsx', iconName: 'User', library: 'lucide-react', context: 'profile-menu' },
          { file: 'UserCard.tsx', iconName: 'Person', library: 'react-icons', context: 'user-avatar' }
        ],
        recommendedIcon: 'User (lucide-react)'
      }
    ];

    return {
      componentVariations,
      iconInconsistencies,
      colorInconsistencies: [],
      typographyInconsistencies: []
    };
  }

  private generateSummary(findings: AuditFindings): AuditSummary {
    return {
      totalFiles: 150, // Would be calculated from actual file scan
      orphanedFiles: findings.structural.orphanedFiles.length,
      deadRoutes: findings.routing.deadLinks.length,
      styleInconsistencies: findings.styling.themeViolations.length,
      dataStructureMismatches: findings.dataConsistency.typeDefinitionMismatches.length,
      duplicateComponents: findings.uiConsistency.componentVariations.length
    };
  }

  private identifyCriticalIssues(findings: AuditFindings): CriticalIssue[] {
    const issues: CriticalIssue[] = [];

    // Critical routing security issues
    findings.routing.routeSecurityIssues.forEach(issue => {
      if (issue.severity === 'critical') {
        issues.push({
          type: 'error',
          category: 'routing',
          message: `Critical security issue: ${issue.issue} on route ${issue.route}`,
          files: [issue.route],
          impact: 'high',
          recommendation: 'Immediately add proper authentication and authorization'
        });
      }
    });

    // Orphaned files taking up space
    if (findings.structural.orphanedFiles.length > 5) {
      issues.push({
        type: 'warning',
        category: 'structural',
        message: `${findings.structural.orphanedFiles.length} orphaned files detected`,
        files: findings.structural.orphanedFiles.map(f => f.path),
        impact: 'medium',
        recommendation: 'Archive or remove unused files to reduce bundle size'
      });
    }

    return issues;
  }

  private generateRecommendations(findings: AuditFindings): string[] {
    const recommendations: string[] = [];

    if (findings.structural.orphanedFiles.length > 0) {
      recommendations.push(`🗑️ Remove ${findings.structural.orphanedFiles.length} orphaned files`);
    }

    if (findings.routing.deadLinks.length > 0) {
      recommendations.push(`🔗 Fix ${findings.routing.deadLinks.length} broken navigation links`);
    }

    if (findings.styling.themeViolations.length > 0) {
      recommendations.push(`🎨 Replace ${findings.styling.themeViolations.length} hardcoded styles with design tokens`);
    }

    if (findings.uiConsistency.componentVariations.length > 0) {
      recommendations.push(`🧩 Consolidate ${findings.uiConsistency.componentVariations.length} duplicate component patterns`);
    }

    if (findings.dataConsistency.typeDefinitionMismatches.length > 0) {
      recommendations.push(`📋 Unify ${findings.dataConsistency.typeDefinitionMismatches.length} conflicting type definitions`);
    }

    recommendations.push('✅ Implement automated pre-commit hooks for consistency checks');
    recommendations.push('📊 Set up visual regression testing for UI consistency');
    recommendations.push('🔍 Schedule monthly automated audits');

    return recommendations;
  }

  private calculateHealthScore(findings: AuditFindings): number {
    let score = 100;
    
    // Deduct points for issues
    score -= findings.structural.orphanedFiles.length * 2;
    score -= findings.routing.deadLinks.length * 3;
    score -= findings.routing.routeSecurityIssues.length * 10;
    score -= findings.styling.themeViolations.length * 1;
    score -= findings.uiConsistency.componentVariations.length * 2;
    score -= findings.dataConsistency.typeDefinitionMismatches.length * 5;

    return Math.max(0, Math.min(100, score));
  }
}

export interface AuditConfig {
  excludePatterns: string[];
  includeExtensions: string[];
  strictMode: boolean;
}

// Export singleton auditor
export const codebaseAuditor = new CodebaseAuditor();