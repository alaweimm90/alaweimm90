import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { ExternalLink, CheckCircle, XCircle, AlertTriangle } from 'lucide-react';
import { Report } from '@/types';

interface CitationsTabProps {
  report: Report;
}

const CitationsTab = ({ report }: CitationsTabProps) => {
  const citations = report.citations || [];
  const validCitations = citations.filter(c => c.resolves);
  const invalidCitations = citations.filter(c => !c.resolves);
  const validityRate = citations.length > 0 ? validCitations.length / citations.length : 0;

  if (citations.length === 0) {
    return (
      <Card>
        <CardContent className="flex items-center justify-center py-12">
          <div className="text-center space-y-2">
            <AlertTriangle className="h-12 w-12 text-muted-foreground mx-auto" />
            <h3 className="text-lg font-semibold">No citations found</h3>
            <p className="text-muted-foreground">
              This document does not contain detectable citations or references.
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
          <CardTitle>Citation Validation Summary</CardTitle>
          <CardDescription>
            Verification of references against external databases
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid md:grid-cols-3 gap-6">
            <div className="text-center space-y-2">
              <div className="text-3xl font-bold text-primary">
                {citations.length}
              </div>
              <div className="space-y-1">
                <p className="text-sm font-medium">Total References</p>
                <p className="text-xs text-muted-foreground">Found in document</p>
              </div>
            </div>
            
            <div className="text-center space-y-2">
              <div className="text-3xl font-bold text-success">
                {validCitations.length}
              </div>
              <div className="space-y-1">
                <p className="text-sm font-medium">Valid Citations</p>
                <p className="text-xs text-muted-foreground">Resolve correctly</p>
              </div>
            </div>
            
            <div className="text-center space-y-2">
              <div className="text-3xl font-bold text-error">
                {invalidCitations.length}
              </div>
              <div className="space-y-1">
                <p className="text-sm font-medium">Invalid Citations</p>
                <p className="text-xs text-muted-foreground">Non-resolving</p>
              </div>
            </div>
          </div>
          
          <div className="mt-6 space-y-2">
            <div className="flex justify-between text-sm">
              <span>Reference Validity Rate</span>
              <span className="font-medium">{(validityRate * 100).toFixed(1)}%</span>
            </div>
            <div className="w-full bg-muted rounded-full h-2">
              <div 
                className={`h-2 rounded-full ${validityRate >= 0.7 ? 'bg-success' : validityRate >= 0.4 ? 'bg-warning' : 'bg-error'}`}
                style={{ width: `${validityRate * 100}%` }}
              />
            </div>
            <p className="text-xs text-muted-foreground">
              Academic texts typically have 80-95% valid citations
            </p>
          </div>
        </CardContent>
      </Card>

      {/* Detailed Citation Analysis */}
      <Card>
        <CardHeader>
          <CardTitle>Citation Details</CardTitle>
          <CardDescription>
            Individual citation verification and suggested replacements
          </CardDescription>
        </CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead className="w-[40px]">Status</TableHead>
                <TableHead>Raw Citation</TableHead>
                <TableHead>Suggestions</TableHead>
                <TableHead className="w-[120px]">Actions</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {citations.map((citation, index) => (
                <TableRow key={index}>
                  <TableCell>
                    {citation.resolves ? (
                      <CheckCircle className="h-4 w-4 text-success" />
                    ) : (
                      <XCircle className="h-4 w-4 text-error" />
                    )}
                  </TableCell>
                  
                  <TableCell>
                    <div className="space-y-1">
                      <code className="text-sm bg-muted px-2 py-1 rounded">
                        {citation.raw}
                      </code>
                      <div className="flex space-x-2">
                        <Badge variant={citation.resolves ? "default" : "destructive"}>
                          {citation.resolves ? "Valid" : "Invalid"}
                        </Badge>
                      </div>
                    </div>
                  </TableCell>
                  
                  <TableCell>
                    {citation.suggestions && citation.suggestions.length > 0 ? (
                      <div className="space-y-2">
                        {citation.suggestions.map((suggestion, suggestionIndex) => (
                          <div key={suggestionIndex} className="p-2 bg-muted/50 rounded text-sm">
                            <div className="font-medium">{suggestion.title}</div>
                            {suggestion.doi && (
                              <div className="text-xs text-muted-foreground font-mono">
                                {suggestion.doi}
                              </div>
                            )}
                            <div className="flex items-center justify-between mt-1">
                              <Badge variant="outline" className="text-xs">
                                {(suggestion.confidence * 100).toFixed(0)}% match
                              </Badge>
                              {suggestion.doi && (
                                <Button variant="ghost" size="sm" className="h-6 px-2 text-xs">
                                  <ExternalLink className="h-3 w-3 mr-1" />
                                  View
                                </Button>
                              )}
                            </div>
                          </div>
                        ))}
                      </div>
                    ) : (
                      <span className="text-muted-foreground text-sm">No suggestions available</span>
                    )}
                  </TableCell>
                  
                  <TableCell>
                    <div className="space-y-1">
                      <Button variant="outline" size="sm" disabled>
                        Replace
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

      {/* Recommendations */}
      <Card>
        <CardHeader>
          <CardTitle>Recommendations</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {validityRate < 0.7 && (
              <div className="p-4 bg-warning/10 border border-warning/20 rounded-lg">
                <div className="flex items-start space-x-2">
                  <AlertTriangle className="h-5 w-5 text-warning mt-0.5" />
                  <div className="space-y-1">
                    <p className="font-medium text-warning">Low Citation Validity</p>
                    <p className="text-sm text-warning/80">
                      Only {(validityRate * 100).toFixed(1)}% of citations resolve correctly. 
                      This is unusually low for academic content and may indicate AI generation.
                    </p>
                  </div>
                </div>
              </div>
            )}
            
            <div className="grid md:grid-cols-2 gap-4 text-sm">
              <div className="space-y-2">
                <h4 className="font-medium">Manual Verification Required</h4>
                <ul className="space-y-1 text-muted-foreground">
                  <li>• Verify all citations through original sources</li>
                  <li>• Check publication dates and author names</li>
                  <li>• Confirm DOI links resolve correctly</li>
                </ul>
              </div>
              
              <div className="space-y-2">
                <h4 className="font-medium">AI Generation Indicators</h4>
                <ul className="space-y-1 text-muted-foreground">
                  <li>• Non-existent but plausible-sounding papers</li>
                  <li>• Correct citation format but wrong details</li>
                  <li>• Authors or journals that don't exist</li>
                </ul>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default CitationsTab;