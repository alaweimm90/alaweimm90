import { useState, useEffect, useCallback } from 'react';
import { accessibilityTestRunner, type AccessibilityReport } from '@/lib/accessibility-audit';

interface AccessibilityAuditState {
  isRunning: boolean;
  report: AccessibilityReport | null;
  error: string | null;
  lastRun: number | null;
  summary: {
    totalAudits: number;
    averageScore: number;
    complianceRate: number;
    commonViolations: Array<{ type: string; count: number }>;
  };
}

interface AccessibilityAuditActions {
  runAudit: () => Promise<void>;
  exportReport: (format: 'json' | 'csv') => string;
  clearHistory: () => void;
  getViolationsByType: (type: string) => any[];
  getComplianceStatus: () => {
    wcagA: boolean;
    wcagAA: boolean;
    wcagAAA: boolean;
    score: number;
  };
}

export const useAccessibilityAudit = (): AccessibilityAuditState & AccessibilityAuditActions => {
  const [state, setState] = useState<AccessibilityAuditState>({
    isRunning: false,
    report: null,
    error: null,
    lastRun: null,
    summary: {
      totalAudits: 0,
      averageScore: 0,
      complianceRate: 0,
      commonViolations: []
    }
  });

  // Update summary when reports change
  const updateSummary = useCallback(() => {
    const summary = accessibilityTestRunner.getSummary();
    setState(prev => ({ ...prev, summary }));
  }, []);

  // Run accessibility audit
  const runAudit = useCallback(async () => {
    setState(prev => ({ ...prev, isRunning: true, error: null }));

    try {
      const report = await accessibilityTestRunner.runAudit();
      setState(prev => ({
        ...prev,
        isRunning: false,
        report,
        lastRun: Date.now(),
        error: null
      }));
      updateSummary();
    } catch (error) {
      setState(prev => ({
        ...prev,
        isRunning: false,
        error: error instanceof Error ? error.message : 'Unknown error occurred',
      }));
    }
  }, [updateSummary]);

  // Export report in different formats
  const exportReport = useCallback((format: 'json' | 'csv'): string => {
    const report = state.report;
    if (!report) return '';

    if (format === 'json') {
      return accessibilityTestRunner.exportReport(report);
    } else {
      // Convert to CSV format
      const headers = [
        'Violation ID',
        'Type',
        'Severity',
        'WCAG Level',
        'WCAG Criterion',
        'Element',
        'Description',
        'Recommendation'
      ];

      const rows = report.violations.map(violation => [
        violation.id,
        violation.type,
        violation.severity,
        violation.wcagLevel,
        violation.wcagCriterion,
        violation.element,
        `"${violation.description.replace(/"/g, '""')}"`,
        `"${violation.recommendation.replace(/"/g, '""')}"`
      ]);

      return [headers.join(','), ...rows.map(row => row.join(','))].join('\n');
    }
  }, [state.report]);

  // Clear audit history
  const clearHistory = useCallback(() => {
    // Note: This would require extending AccessibilityTestRunner with a clear method
    setState(prev => ({
      ...prev,
      report: null,
      lastRun: null,
      summary: {
        totalAudits: 0,
        averageScore: 0,
        complianceRate: 0,
        commonViolations: []
      }
    }));
  }, []);

  // Get violations by type
  const getViolationsByType = useCallback((type: string) => {
    if (!state.report) return [];
    return state.report.violations.filter(violation => violation.type === type);
  }, [state.report]);

  // Get compliance status
  const getComplianceStatus = useCallback(() => {
    if (!state.report) {
      return {
        wcagA: false,
        wcagAA: false,
        wcagAAA: false,
        score: 0
      };
    }

    return {
      ...state.report.compliance,
      score: state.report.score
    };
  }, [state.report]);

  // Load initial state on mount
  useEffect(() => {
    const latestReport = accessibilityTestRunner.getLatestReport();
    if (latestReport) {
      setState(prev => ({
        ...prev,
        report: latestReport,
        lastRun: latestReport.timestamp
      }));
    }
    updateSummary();
  }, [updateSummary]);

  // Auto-run audit in development mode
  useEffect(() => {
    if (process.env.NODE_ENV === 'development') {
      const autoAuditTimer = setTimeout(() => {
        if (!state.report) {
          runAudit();
        }
      }, 3000); // Auto-run after 3 seconds in development

      return () => clearTimeout(autoAuditTimer);
    }
  }, [state.report, runAudit]);

  return {
    ...state,
    runAudit,
    exportReport,
    clearHistory,
    getViolationsByType,
    getComplianceStatus
  };
};

// Hook for real-time accessibility monitoring
export const useAccessibilityMonitor = () => {
  const [violations, setViolations] = useState<any[]>([]);
  const [isMonitoring, setIsMonitoring] = useState(false);

  const startMonitoring = useCallback(() => {
    setIsMonitoring(true);
    
    // Set up MutationObserver to detect DOM changes
    const observer = new MutationObserver((mutations) => {
      let hasSignificantChange = false;
      
      mutations.forEach(mutation => {
        if (mutation.type === 'childList' && mutation.addedNodes.length > 0) {
          mutation.addedNodes.forEach(node => {
            if (node.nodeType === Node.ELEMENT_NODE) {
              const element = node as Element;
              // Check for interactive elements
              if (element.matches('button, a, input, select, textarea, [role="button"], [tabindex]')) {
                hasSignificantChange = true;
              }
            }
          });
        }
      });

      // Re-run audit if significant changes detected
      if (hasSignificantChange) {
        setTimeout(() => {
          accessibilityTestRunner.runAudit().then(report => {
            setViolations(report.violations);
          });
        }, 1000); // Debounce
      }
    });

    observer.observe(document.body, {
      childList: true,
      subtree: true,
      attributes: true,
      attributeFilter: ['aria-label', 'aria-labelledby', 'aria-describedby', 'tabindex', 'role']
    });

    return () => {
      observer.disconnect();
      setIsMonitoring(false);
    };
  }, []);

  const stopMonitoring = useCallback(() => {
    setIsMonitoring(false);
  }, []);

  return {
    violations,
    isMonitoring,
    startMonitoring,
    stopMonitoring
  };
};

// Hook for specific WCAG compliance checking
export const useWCAGCompliance = () => {
  const [complianceStatus, setComplianceStatus] = useState({
    colorContrast: { passed: 0, failed: 0, status: 'unknown' as 'passed' | 'failed' | 'unknown' },
    touchTargets: { passed: 0, failed: 0, status: 'unknown' as 'passed' | 'failed' | 'unknown' },
    keyboardNav: { passed: 0, failed: 0, status: 'unknown' as 'passed' | 'failed' | 'unknown' },
    ariaLabels: { passed: 0, failed: 0, status: 'unknown' as 'passed' | 'failed' | 'unknown' },
    headingStructure: { status: 'unknown' as 'passed' | 'failed' | 'unknown' },
    landmarks: { status: 'unknown' as 'passed' | 'failed' | 'unknown' }
  });

  const checkCompliance = useCallback(async () => {
    const report = await accessibilityTestRunner.runAudit();
    
    // Analyze violations by type
    const violationsByType = report.violations.reduce((acc, violation) => {
      if (!acc[violation.type]) acc[violation.type] = [];
      acc[violation.type].push(violation);
      return acc;
    }, {} as Record<string, any[]>);

    setComplianceStatus({
      colorContrast: {
        passed: violationsByType['color-contrast'] ? 0 : 1,
        failed: violationsByType['color-contrast']?.length || 0,
        status: violationsByType['color-contrast'] ? 'failed' : 'passed'
      },
      touchTargets: {
        passed: violationsByType['touch-targets'] ? 0 : 1,
        failed: violationsByType['touch-targets']?.length || 0,
        status: violationsByType['touch-targets'] ? 'failed' : 'passed'
      },
      keyboardNav: {
        passed: violationsByType['keyboard-navigation'] ? 0 : 1,
        failed: violationsByType['keyboard-navigation']?.length || 0,
        status: violationsByType['keyboard-navigation'] ? 'failed' : 'passed'
      },
      ariaLabels: {
        passed: violationsByType['aria-labels'] ? 0 : 1,
        failed: violationsByType['aria-labels']?.length || 0,
        status: violationsByType['aria-labels'] ? 'failed' : 'passed'
      },
      headingStructure: {
        status: report.violations.some(v => v.description.includes('heading')) ? 'failed' : 'passed'
      },
      landmarks: {
        status: report.violations.some(v => v.description.includes('landmark')) ? 'failed' : 'passed'
      }
    });

    return complianceStatus;
  }, [complianceStatus]);

  return {
    complianceStatus,
    checkCompliance
  };
};