import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { AlertTriangle, Shield, Code, ExternalLink } from 'lucide-react';
import { Report } from '@/types';

interface CodeSecurityTabProps {
  report: Report;
}

const CodeSecurityTab = ({ report }: CodeSecurityTabProps) => {
  const findings = report.codeFindings || [];
  const totalFindings = findings.length;
  const highSeverity = findings.filter(f => f.severity === 'HIGH').length;
  const mediumSeverity = findings.filter(f => f.severity === 'MEDIUM').length;
  const lowSeverity = findings.filter(f => f.severity === 'LOW').length;
  
  // Calculate findings per KLOC (mock - assume 1000 lines total for demo)
  const totalLines = 1000;
  const findingsPerKloc = (totalFindings / totalLines) * 1000;
  
  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'HIGH': return 'bg-error/10 text-error border-error/20';
      case 'MEDIUM': return 'bg-warning/10 text-warning border-warning/20';
      case 'LOW': return 'bg-info/10 text-info border-info/20';
      default: return 'bg-muted/10 text-muted-foreground border-muted/20';
    }
  };

  const getCWELink = (cwe: string) => `https://cwe.mitre.org/data/definitions/${cwe.replace('CWE-', '')}.html`;

  if (findings.length === 0) {
    return (
      <Card>
        <CardContent className="flex items-center justify-center py-12">
          <div className="text-center space-y-2">
            <Shield className="h-12 w-12 text-success mx-auto" />
            <h3 className="text-lg font-semibold">No security issues found</h3>
            <p className="text-muted-foreground">
              The code analysis did not detect any security vulnerabilities.
            </p>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="space-y-6">
      {/* Summary */}
      <Card>
        <CardHeader>
          <CardTitle>Security Analysis Summary</CardTitle>
          <CardDescription>
            Automated vulnerability detection using Semgrep/Bandit rules
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid md:grid-cols-4 gap-6">
            <div className="text-center space-y-2">
              <div className="text-3xl font-bold text-primary">
                {totalFindings}
              </div>
              <div className="space-y-1">
                <p className="text-sm font-medium">Total Findings</p>
                <p className="text-xs text-muted-foreground">Security issues</p>
              </div>
            </div>
            
            <div className="text-center space-y-2">
              <div className="text-3xl font-bold text-error">
                {highSeverity}
              </div>
              <div className="space-y-1">
                <p className="text-sm font-medium">High Severity</p>
                <p className="text-xs text-muted-foreground">Critical fixes needed</p>
              </div>
            </div>
            
            <div className="text-center space-y-2">
              <div className="text-3xl font-bold text-warning">
                {mediumSeverity}
              </div>
              <div className="space-y-1">
                <p className="text-sm font-medium">Medium Severity</p>
                <p className="text-xs text-muted-foreground">Important to fix</p>
              </div>
            </div>
            
            <div className="text-center space-y-2">
              <div className="text-3xl font-bold text-info">
                {lowSeverity}
              </div>
              <div className="space-y-1">
                <p className="text-sm font-medium">Low Severity</p>
                <p className="text-xs text-muted-foreground">Best practices</p>
              </div>
            </div>
          </div>
          
          <div className="mt-6 space-y-2">
            <div className="flex justify-between text-sm">
              <span>Findings per KLOC</span>
              <span className="font-medium">{findingsPerKloc.toFixed(1)}</span>
            </div>
            <div className="w-full bg-muted rounded-full h-2">
              <div 
                className={`h-2 rounded-full ${findingsPerKloc > 10 ? 'bg-error' : findingsPerKloc > 5 ? 'bg-warning' : 'bg-success'}`}
                style={{ width: `${Math.min(findingsPerKloc / 20 * 100, 100)}%` }}
              />
            </div>
            <p className="text-xs text-muted-foreground">
              Industry average: 5-15 findings per KLOC
            </p>
          </div>
        </CardContent>
      </Card>

      {/* Findings by File */}
      <Card>
        <CardHeader>
          <CardTitle>Security Findings</CardTitle>
          <CardDescription>
            Detailed vulnerability analysis grouped by file
          </CardDescription>
        </CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>File & Line</TableHead>
                <TableHead>Rule</TableHead>
                <TableHead>CWE</TableHead>
                <TableHead>Severity</TableHead>
                <TableHead>Code Snippet</TableHead>
                <TableHead className="w-[120px]">Actions</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {findings.map((finding, index) => (
                <TableRow key={index}>
                  <TableCell>
                    <div className="space-y-1">
                      <div className="flex items-center space-x-2">
                        <Code className="h-4 w-4 text-muted-foreground" />
                        <span className="font-medium text-sm">{finding.path}</span>
                      </div>
                      <Badge variant="outline" className="text-xs">
                        Line {finding.line}
                      </Badge>
                    </div>
                  </TableCell>
                  
                  <TableCell>
                    <span className="font-mono text-sm">{finding.rule}</span>
                  </TableCell>
                  
                  <TableCell>
                    <div className="flex items-center space-x-2">
                      <Badge variant="outline" className="font-mono text-xs">
                        {finding.cwe}
                      </Badge>
                      <Button variant="ghost" size="sm" className="h-6 w-6 p-0" asChild>
                        <a href={getCWELink(finding.cwe)} target="_blank" rel="noopener noreferrer">
                          <ExternalLink className="h-3 w-3" />
                        </a>
                      </Button>
                    </div>
                  </TableCell>
                  
                  <TableCell>
                    <Badge className={getSeverityColor(finding.severity)}>
                      {finding.severity}
                    </Badge>
                  </TableCell>
                  
                  <TableCell>
                    <code className="text-xs bg-muted px-2 py-1 rounded block max-w-xs overflow-hidden text-ellipsis whitespace-nowrap">
                      {finding.snippet}
                    </code>
                  </TableCell>
                  
                  <TableCell>
                    <div className="space-y-1">
                      <Button variant="outline" size="sm" disabled>
                        Fix
                      </Button>
                      <p className="text-xs text-muted-foreground">Coming soon</p>
                    </div>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </CardContent>
      </Card>

      {/* Security Recommendations */}
      <Card>
        <CardHeader>
          <CardTitle>Security Recommendations</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {highSeverity > 0 && (
              <div className="p-4 bg-error/10 border border-error/20 rounded-lg">
                <div className="flex items-start space-x-2">
                  <AlertTriangle className="h-5 w-5 text-error mt-0.5" />
                  <div className="space-y-1">
                    <p className="font-medium text-error">Critical Security Issues</p>
                    <p className="text-sm text-error/80">
                      {highSeverity} high-severity vulnerabilities detected. These should be addressed immediately 
                      as they may allow attackers to compromise the system.
                    </p>
                  </div>
                </div>
              </div>
            )}
            
            <div className="grid md:grid-cols-2 gap-4 text-sm">
              <div className="space-y-2">
                <h4 className="font-medium">Immediate Actions</h4>
                <ul className="space-y-1 text-muted-foreground">
                  <li>• Fix all HIGH severity issues first</li>
                  <li>• Review input validation and sanitization</li>
                  <li>• Implement proper authentication controls</li>
                  <li>• Use parameterized queries for database access</li>
                </ul>
              </div>
              
              <div className="space-y-2">
                <h4 className="font-medium">AI Generation Indicators</h4>
                <ul className="space-y-1 text-muted-foreground">
                  <li>• High density of common vulnerabilities</li>
                  <li>• Patterns typical of code generation models</li>
                  <li>• Missing security best practices</li>
                  <li>• Inconsistent security implementations</li>
                </ul>
              </div>
            </div>
            
            <div className="pt-4 border-t">
              <Button disabled className="w-full">
                Generate Safe Patch (Roadmap Feature)
              </Button>
              <p className="text-xs text-muted-foreground mt-2 text-center">
                Automated security patches will be available in a future update
              </p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default CodeSecurityTab;