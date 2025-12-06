import { memo, useState, useEffect, useMemo } from 'react';
import { CheckCircle, AlertTriangle, XCircle, ExternalLink, BookOpen, Search, Shield } from 'lucide-react';
import { Badge } from '@/components/ui/badge';

interface Citation {
  id: string;
  text: string;
  url?: string;
  type: 'academic' | 'news' | 'government' | 'organization' | 'book' | 'dataset' | 'preprint';
  credibility: 'high' | 'medium' | 'low' | 'unverified';
  datePublished?: string;
  authors?: string[];
  journalOrSource?: string;
}

interface FactCheck {
  id: string;
  claim: string;
  status: 'verified' | 'disputed' | 'false' | 'unverified' | 'partially-true';
  confidence: number; // 0-100
  sources: Citation[];
  reasoning: string;
  lastUpdated: Date;
}

interface Argument {
  id: string;
  speaker: 'left' | 'right';
  content: string;
  strength: number;
  timestamp: Date;
  type: 'opening' | 'argument' | 'counter' | 'closing';
  citations?: Citation[];
  factChecks?: FactCheck[];
}

interface CitationTrackerProps {
  arguments: Argument[];
  onCitationVerified: (citation: Citation, isValid: boolean) => void;
  onFactCheckComplete: (factCheck: FactCheck) => void;
  academicMode?: boolean;
  realTimeChecking?: boolean;
}

const CitationTrackerComponent = ({
  arguments: argumentList,
  onCitationVerified,
  onFactCheckComplete,
  academicMode = false,
  realTimeChecking = true
}: CitationTrackerProps) => {
  const [citationDatabase, setCitationDatabase] = useState<Citation[]>([]);
  const [factCheckResults, setFactCheckResults] = useState<FactCheck[]>([]);
  const [verificationQueue, setVerificationQueue] = useState<Citation[]>([]);
  const [isProcessing, setIsProcessing] = useState(false);

  // Extract citations from argument text (simulated - in real implementation would use NLP)
  const extractCitations = (argument: Argument): Citation[] => {
    const extractedCitations: Citation[] = [];
    const citationPatterns = [
      /\(([^)]+)\s+\d{4}\)/g, // (Author 2024)
      /https?:\/\/[^\s]+/g,   // URLs
      /DOI:\s*[^\s]+/g,       // DOI references
      /"([^"]+)"\s*-\s*([^,]+)/g, // "Quote" - Source
    ];

    citationPatterns.forEach((pattern, index) => {
      let match;
      while ((match = pattern.exec(argument.content)) !== null) {
        const citationText = match[0];
        
        // Simulate citation analysis
        const citation: Citation = {
          id: `citation-${argument.id}-${index}-${Date.now()}`,
          text: citationText,
          type: detectCitationType(citationText),
          credibility: simulateCredibilityCheck(citationText),
          url: citationText.startsWith('http') ? citationText : undefined,
        };

        extractedCitations.push(citation);
      }
    });

    return extractedCitations;
  };

  const detectCitationType = (citationText: string): Citation['type'] => {
    if (citationText.includes('doi.org') || citationText.includes('arxiv')) return 'academic';
    if (citationText.includes('gov')) return 'government';
    if (citationText.includes('news') || citationText.includes('reuters') || citationText.includes('bbc')) return 'news';
    if (citationText.includes('http')) return 'organization';
    if (citationText.includes('ISBN')) return 'book';
    if (citationText.includes('dataset')) return 'dataset';
    return 'academic';
  };

  const simulateCredibilityCheck = (citationText: string): Citation['credibility'] => {
    // Simulate credibility assessment based on source patterns
    const highCredibilityPatterns = ['nature', 'science', 'arxiv', '.gov', 'who.int', 'unesco'];
    const mediumCredibilityPatterns = ['reuters', 'bbc', 'nyt', '.edu', 'pubmed'];
    const lowCredibilityPatterns = ['blog', 'opinion', 'social'];

    const text = citationText.toLowerCase();
    
    if (highCredibilityPatterns.some(pattern => text.includes(pattern))) return 'high';
    if (mediumCredibilityPatterns.some(pattern => text.includes(pattern))) return 'medium';
    if (lowCredibilityPatterns.some(pattern => text.includes(pattern))) return 'low';
    
    return 'unverified';
  };

  // Simulate fact-checking process
  const performFactCheck = (argument: Argument): FactCheck[] => {
    const factChecks: FactCheck[] = [];
    
    // Extract potential claims (simplified - would use NLP in reality)
    const claimPatterns = [
      /studies show that ([^.]+)/gi,
      /research indicates ([^.]+)/gi,
      /according to ([^,]+), ([^.]+)/gi,
      /(\d+%|\d+\s+percent) of ([^.]+)/gi,
    ];

    claimPatterns.forEach(pattern => {
      let match;
      while ((match = pattern.exec(argument.content)) !== null) {
        const claim = match[0];
        
        const factCheck: FactCheck = {
          id: `fact-check-${argument.id}-${Date.now()}`,
          claim,
          status: simulateFactCheckStatus(),
          confidence: Math.floor(Math.random() * 30) + 70, // 70-100% confidence
          sources: generateMockSources(),
          reasoning: generateFactCheckReasoning(),
          lastUpdated: new Date()
        };
        
        factChecks.push(factCheck);
      }
    });

    return factChecks;
  };

  const simulateFactCheckStatus = (): FactCheck['status'] => {
    const statuses: FactCheck['status'][] = ['verified', 'disputed', 'partially-true', 'unverified'];
    return statuses[Math.floor(Math.random() * statuses.length)];
  };

  const generateMockSources = (): Citation[] => {
    const mockSources = [
      { type: 'academic' as const, credibility: 'high' as const, journalOrSource: 'Nature' },
      { type: 'government' as const, credibility: 'high' as const, journalOrSource: 'CDC' },
      { type: 'news' as const, credibility: 'medium' as const, journalOrSource: 'Reuters' },
      { type: 'organization' as const, credibility: 'medium' as const, journalOrSource: 'WHO' },
    ];

    return mockSources.slice(0, Math.floor(Math.random() * 3) + 1).map((source, index) => ({
      id: `mock-source-${index}`,
      text: `${source.journalOrSource} Research`,
      type: source.type,
      credibility: source.credibility,
      journalOrSource: source.journalOrSource
    }));
  };

  const generateFactCheckReasoning = (): string => {
    const reasonings = [
      'Claim aligns with peer-reviewed research and official statistics',
      'Statement requires additional context and has conflicting evidence',
      'Data partially correct but lacks nuance in presentation',
      'Claim requires verification from primary sources',
    ];
    return reasonings[Math.floor(Math.random() * reasonings.length)];
  };

  // Process new arguments for citations and fact-checks
  useEffect(() => {
    if (!realTimeChecking || argumentList.length === 0) return;

    const latestArgument = argumentList[argumentList.length - 1];
    
    // Extract and verify citations
    const citations = extractCitations(latestArgument);
    setCitationDatabase(prev => [...prev, ...citations]);
    
    // Perform fact-checking
    const factChecks = performFactCheck(latestArgument);
    setFactCheckResults(prev => [...prev, ...factChecks]);
    
    // Notify parent components
    citations.forEach(citation => {
      onCitationVerified(citation, citation.credibility !== 'low');
    });
    
    factChecks.forEach(factCheck => {
      onFactCheckComplete(factCheck);
    });
  }, [argumentList, realTimeChecking, onCitationVerified, onFactCheckComplete]);

  // Calculate citation and fact-check statistics
  const statistics = useMemo(() => {
    const totalCitations = citationDatabase.length;
    const highCredibilityCitations = citationDatabase.filter(c => c.credibility === 'high').length;
    const verifiedFactChecks = factCheckResults.filter(f => f.status === 'verified').length;
    const disputedFactChecks = factCheckResults.filter(f => f.status === 'disputed' || f.status === 'false').length;
    
    return {
      totalCitations,
      highCredibilityCitations,
      verifiedFactChecks,
      disputedFactChecks,
      credibilityScore: totalCitations > 0 ? Math.round((highCredibilityCitations / totalCitations) * 100) : 0,
      factualAccuracy: factCheckResults.length > 0 ? Math.round((verifiedFactChecks / factCheckResults.length) * 100) : 0
    };
  }, [citationDatabase, factCheckResults]);

  const getCredibilityIcon = (credibility: Citation['credibility']) => {
    switch (credibility) {
      case 'high': return <CheckCircle className="h-4 w-4 text-green-500" />;
      case 'medium': return <AlertTriangle className="h-4 w-4 text-yellow-500" />;
      case 'low': return <XCircle className="h-4 w-4 text-red-500" />;
      default: return <Search className="h-4 w-4 text-gray-500" />;
    }
  };

  const getFactCheckIcon = (status: FactCheck['status']) => {
    switch (status) {
      case 'verified': return <CheckCircle className="h-4 w-4 text-green-500" />;
      case 'partially-true': return <AlertTriangle className="h-4 w-4 text-yellow-500" />;
      case 'disputed':
      case 'false': return <XCircle className="h-4 w-4 text-red-500" />;
      default: return <Search className="h-4 w-4 text-gray-500" />;
    }
  };

  if (!academicMode && citationDatabase.length === 0 && factCheckResults.length === 0) {
    return null; // Don't show empty tracker in gaming mode
  }

  return (
    <div className="absolute top-4 left-1/2 transform -translate-x-1/2 z-20">
      <div className="glass-panel p-4 rounded-xl border-2 border-primary/20 min-w-96">
        
        {/* Header */}
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2">
            <Shield className="h-5 w-5 text-primary" />
            <h3 className="heading-refined text-sm">Verification Center</h3>
          </div>
          <div className="flex gap-2">
            <Badge variant="secondary" className="text-xs">
              {statistics.credibilityScore}% Sources
            </Badge>
            <Badge variant="secondary" className="text-xs">
              {statistics.factualAccuracy}% Accuracy
            </Badge>
          </div>
        </div>

        {/* Statistics Dashboard */}
        <div className="grid grid-cols-2 gap-4 mb-4">
          <div className="glass-minimal p-3 rounded-lg">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-sm font-medium">Citations</div>
                <div className="text-2xl font-bold text-primary">{statistics.totalCitations}</div>
              </div>
              <BookOpen className="h-6 w-6 text-muted-foreground" />
            </div>
            <div className="text-xs text-muted-foreground mt-1">
              {statistics.highCredibilityCitations} high credibility
            </div>
          </div>
          
          <div className="glass-minimal p-3 rounded-lg">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-sm font-medium">Fact Checks</div>
                <div className="text-2xl font-bold text-primary">{factCheckResults.length}</div>
              </div>
              <Shield className="h-6 w-6 text-muted-foreground" />
            </div>
            <div className="text-xs text-muted-foreground mt-1">
              {statistics.verifiedFactChecks} verified, {statistics.disputedFactChecks} disputed
            </div>
          </div>
        </div>

        {/* Recent Citations */}
        {citationDatabase.length > 0 && (
          <div className="mb-4">
            <div className="text-sm font-medium mb-2">Recent Citations</div>
            <div className="space-y-2 max-h-32 overflow-y-auto">
              {citationDatabase.slice(-3).map(citation => (
                <div key={citation.id} className="glass-subtle p-2 rounded-lg text-xs">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      {getCredibilityIcon(citation.credibility)}
                      <span className="font-medium capitalize">{citation.type}</span>
                    </div>
                    <Badge variant="outline" className="text-xs">
                      {citation.credibility}
                    </Badge>
                  </div>
                  <div className="text-muted-foreground mt-1 truncate">
                    {citation.text}
                  </div>
                  {citation.url && (
                    <div className="flex items-center gap-1 mt-1">
                      <ExternalLink className="h-3 w-3" />
                      <span className="text-primary">Source available</span>
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Recent Fact Checks */}
        {factCheckResults.length > 0 && (
          <div>
            <div className="text-sm font-medium mb-2">Fact Check Results</div>
            <div className="space-y-2 max-h-32 overflow-y-auto">
              {factCheckResults.slice(-2).map(factCheck => (
                <div key={factCheck.id} className="glass-subtle p-2 rounded-lg text-xs">
                  <div className="flex items-center justify-between mb-1">
                    <div className="flex items-center gap-2">
                      {getFactCheckIcon(factCheck.status)}
                      <span className="font-medium capitalize">{factCheck.status.replace('-', ' ')}</span>
                    </div>
                    <Badge variant="outline" className="text-xs">
                      {factCheck.confidence}% confidence
                    </Badge>
                  </div>
                  <div className="text-muted-foreground truncate mb-1">
                    {factCheck.claim}
                  </div>
                  <div className="text-muted-foreground text-xs">
                    {factCheck.reasoning}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Processing Indicator */}
        {isProcessing && (
          <div className="flex items-center gap-2 mt-3 text-xs text-muted-foreground">
            <div className="w-3 h-3 border-2 border-primary border-t-transparent rounded-full animate-spin" />
            <span>Verifying sources and fact-checking claims...</span>
          </div>
        )}
      </div>
    </div>
  );
};

export const CitationTracker = memo(CitationTrackerComponent);