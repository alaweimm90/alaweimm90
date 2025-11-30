// ATLAS Enterprise - Security Types

export interface SecurityScanResult {
  target: string;
  scanType: 'full' | 'quick' | 'compliance';
  timestamp: string;
  summary: {
    totalIssues: number;
    criticalIssues: number;
    highIssues: number;
    mediumIssues: number;
    lowIssues: number;
    compliance: number; // 0-100
  };
  vulnerabilities: SecurityVulnerability[];
  recommendations: SecurityRecommendation[];
}

export interface SecurityVulnerability {
  id: string;
  title: string;
  description: string;
  severity: 'critical' | 'high' | 'medium' | 'low' | 'info';
  cwe?: string; // Common Weakness Enumeration
  cvss?: number; // CVSS score
  location?: CodeLocation;
  evidence: string;
  remediation: string;
  references: string[];
}

export interface SecurityRecommendation {
  id: string;
  title: string;
  description: string;
  priority: 'high' | 'medium' | 'low';
  category: 'authentication' | 'authorization' | 'encryption' | 'input-validation' | 'configuration';
  implementation: string;
}

export interface ComplianceCheckResult {
  framework: ComplianceFramework;
  timestamp: string;
  overallCompliance: number; // 0-100
  requirements: ComplianceRequirement[];
  violations: ComplianceViolation[];
  recommendations: string[];
}

export interface ComplianceRequirement {
  id: string;
  title: string;
  description: string;
  status: 'compliant' | 'non-compliant' | 'not-applicable';
  evidence?: string;
  remediation?: string;
}

export interface ComplianceViolation {
  requirementId: string;
  severity: 'critical' | 'high' | 'medium' | 'low';
  description: string;
  impact: string;
  remediation: string;
}

export type ComplianceFramework = 'soc2' | 'gdpr' | 'hipaa' | 'pci-dss' | 'iso27001';

export interface AuditEvent {
  id: string;
  timestamp: string;
  action: string;
  actor?: string;
  target?: string;
  details: Record<string, any>;
  ipAddress?: string;
  userAgent?: string;
  success: boolean;
  errorMessage?: string;
}

export interface AccessControlPolicy {
  id: string;
  name: string;
  description: string;
  rules: AccessControlRule[];
  createdAt: string;
  updatedAt: string;
}

export interface AccessControlRule {
  id: string;
  resource: string;
  action: string;
  conditions: AccessCondition[];
  effect: 'allow' | 'deny';
}

export interface AccessCondition {
  type: 'user' | 'role' | 'group' | 'time' | 'ip' | 'device';
  operator: 'equals' | 'contains' | 'in' | 'regex' | 'time-range';
  value: any;
}

export interface SecurityIncident {
  id: string;
  type: IncidentType;
  severity: 'critical' | 'high' | 'medium' | 'low';
  title: string;
  description: string;
  detectedAt: string;
  status: 'open' | 'investigating' | 'resolved' | 'closed';
  affectedResources: string[];
  indicators: SecurityIndicator[];
  response: IncidentResponse;
}

export type IncidentType =
  | 'unauthorized-access'
  | 'data-breach'
  | 'malware'
  | 'denial-of-service'
  | 'insider-threat'
  | 'configuration-error'
  | 'third-party-compromise';

export interface SecurityIndicator {
  type: 'ip' | 'user' | 'file' | 'network' | 'behavior';
  value: string;
  confidence: number;
  source: string;
}

export interface IncidentResponse {
  actions: ResponseAction[];
  assignedTo?: string;
  timeline: ResponseEvent[];
  resolution?: string;
}

export interface ResponseAction {
  id: string;
  type: 'isolate' | 'block' | 'alert' | 'investigate' | 'remediate';
  description: string;
  executedAt?: string;
  status: 'pending' | 'executed' | 'failed';
}

export interface ResponseEvent {
  timestamp: string;
  event: string;
  actor?: string;
  details?: string;
}

export interface CodeLocation {
  file: string;
  line: number;
  column?: number;
  function?: string;
  class?: string;
}

export interface SecurityConfig {
  scan: {
    enabled: boolean;
    schedule: string;
    types: string[];
    exclusions: string[];
  };
  compliance: {
    frameworks: ComplianceFramework[];
    autoCheck: boolean;
    reportSchedule: string;
  };
  audit: {
    enabled: boolean;
    retentionDays: number;
    logLevel: 'debug' | 'info' | 'warn' | 'error';
  };
  accessControl: {
    enabled: boolean;
    defaultPolicy: 'allow' | 'deny';
    mfaRequired: boolean;
  };
}</content>
</edit_file>