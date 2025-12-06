import React, { useState } from 'react';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible';
import { ChevronDown, ChevronRight, BookOpen, Calculator, FileText, ExternalLink, Copy, Check } from 'lucide-react';
import { InlineMath, BlockMath } from '@/components/ui/Math';

interface PhysicsTheoryProps {
  module: {
    title: string;
    description: string;
    category: string;
    difficulty: 'Beginner' | 'Intermediate' | 'Advanced' | 'Research';
    equation?: string;
    theory: {
      overview: string;
      mathematics: string[];
      references: string[];
    };
  };
  detailedTheory?: {
    introduction: string;
    fundamentals: Array<{
      title: string;
      content: string;
      equations: string[];
      derivation?: string;
    }>;
    applications: Array<{
      title: string;
      description: string;
      examples: string[];
    }>;
    furtherReading: Array<{
      title: string;
      authors: string;
      journal: string;
      year: number;
      doi?: string;
      url?: string;
    }>;
  };
  className?: string;
}

const difficultyColors = {
  'Beginner': 'bg-green-500/10 text-green-400 border-green-500/20',
  'Intermediate': 'bg-blue-500/10 text-blue-400 border-blue-500/20',
  'Advanced': 'bg-purple-500/10 text-purple-400 border-purple-500/20',
  'Research': 'bg-red-500/10 text-red-400 border-red-500/20',
};

const MathSection: React.FC<{ title: string; children: React.ReactNode; expandable?: boolean }> = ({ 
  title, 
  children, 
  expandable = false 
}) => {
  const [isOpen, setIsOpen] = useState(!expandable);

  if (expandable) {
    return (
      <Collapsible open={isOpen} onOpenChange={setIsOpen}>
        <CollapsibleTrigger asChild>
          <Button variant="ghost" className="w-full justify-between p-0 h-auto">
            <h4 className="text-md font-semibold font-serif text-left">{title}</h4>
            {isOpen ? <ChevronDown className="w-4 h-4" /> : <ChevronRight className="w-4 h-4" />}
          </Button>
        </CollapsibleTrigger>
        <CollapsibleContent className="space-y-3 pt-3">
          {children}
        </CollapsibleContent>
      </Collapsible>
    );
  }

  return (
    <div className="space-y-3">
      <h4 className="text-md font-semibold font-serif">{title}</h4>
      {children}
    </div>
  );
};

const EquationBlock: React.FC<{ equation: string; label?: string }> = ({ equation, label }) => {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    await navigator.clipboard.writeText(equation);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="relative bg-muted/30 rounded-lg p-4 border border-muted font-mono text-center group">
      <Button
        variant="ghost"
        size="icon"
        className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity"
        onClick={handleCopy}
      >
        {copied ? <Check className="w-4 h-4" /> : <Copy className="w-4 h-4" />}
      </Button>
      
      {label && (
        <div className="text-xs text-muted-foreground mb-2 font-sans">{label}</div>
      )}
      
      <div className="overflow-x-auto">
        <BlockMath math={equation} errorColor={'#cc0000'} throwOnError={false} />
      </div>
    </div>
  );
};

const ReferenceItem: React.FC<{ reference: any }> = ({ reference }) => {
  const handleOpenReference = () => {
    if (reference.url) {
      window.open(reference.url, '_blank', 'noopener,noreferrer');
    } else if (reference.doi) {
      window.open(`https://doi.org/${reference.doi}`, '_blank', 'noopener,noreferrer');
    }
  };

  return (
    <Card className="p-4 space-y-2">
      <div className="flex items-start justify-between">
        <div className="space-y-1">
          <h5 className="font-serif font-medium text-sm">{reference.title}</h5>
          <p className="text-xs text-muted-foreground">
            {reference.authors} • <em>{reference.journal}</em> • {reference.year}
          </p>
          {reference.doi && (
            <p className="text-xs font-mono text-muted-foreground">
              DOI: {reference.doi}
            </p>
          )}
        </div>
        
        {(reference.url || reference.doi) && (
          <Button
            variant="ghost"
            size="icon"
            onClick={handleOpenReference}
            className="flex-shrink-0"
          >
            <ExternalLink className="w-4 h-4" />
          </Button>
        )}
      </div>
    </Card>
  );
};

export const PhysicsTheory: React.FC<PhysicsTheoryProps> = ({
  module,
  detailedTheory,
  className = ""
}) => {
  return (
    <div className={`space-y-6 ${className}`}>
      {/* Module Header */}
      <Card className="p-6">
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <Badge variant="secondary" className="text-xs">
              {module.category}
            </Badge>
            <Badge variant="outline" className={difficultyColors[module.difficulty]}>
              {module.difficulty}
            </Badge>
          </div>
          
          <div className="space-y-2">
            <h1 className="text-2xl font-bold font-serif text-foreground">
              {module.title}
            </h1>
            <p className="text-muted-foreground leading-relaxed">
              {module.description}
            </p>
          </div>

          {/* Main Equation */}
          {module.equation && (
            <EquationBlock 
              equation={module.equation}
              label="Key Equation"
            />
          )}
        </div>
      </Card>

      {/* Theory Content */}
      <Tabs defaultValue="overview" className="w-full">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="overview" className="flex items-center gap-2">
            <BookOpen className="w-4 h-4" />
            Overview
          </TabsTrigger>
          <TabsTrigger value="mathematics" className="flex items-center gap-2">
            <Calculator className="w-4 h-4" />
            Mathematics
          </TabsTrigger>
          <TabsTrigger value="applications" className="flex items-center gap-2">
            <FileText className="w-4 h-4" />
            Applications
          </TabsTrigger>
          <TabsTrigger value="references" className="flex items-center gap-2">
            <ExternalLink className="w-4 h-4" />
            References
          </TabsTrigger>
        </TabsList>

        {/* Overview Tab */}
        <TabsContent value="overview" className="space-y-6">
          <Card className="p-6">
            <h3 className="text-lg font-semibold font-serif mb-4">Physics Overview</h3>
            <div className="prose prose-sm max-w-none">
              <p className="text-muted-foreground leading-relaxed">
                {detailedTheory?.introduction || module.theory.overview}
              </p>
            </div>
          </Card>

          {detailedTheory?.fundamentals && (
            <Card className="p-6">
              <h3 className="text-lg font-semibold font-serif mb-4">Fundamental Concepts</h3>
              <div className="space-y-6">
                {detailedTheory.fundamentals.map((concept, index) => (
                  <MathSection 
                    key={index} 
                    title={concept.title}
                    expandable={index > 0}
                  >
                    <p className="text-sm text-muted-foreground leading-relaxed">
                      {concept.content}
                    </p>
                    
                    {concept.equations.map((eq, eqIndex) => (
                      <EquationBlock 
                        key={eqIndex}
                        equation={eq}
                      />
                    ))}
                    
                    {concept.derivation && (
                      <div className="mt-4 p-4 bg-accent/10 rounded-lg">
                        <h5 className="text-sm font-semibold mb-2">Derivation:</h5>
                        <p className="text-xs text-muted-foreground">
                          {concept.derivation}
                        </p>
                      </div>
                    )}
                  </MathSection>
                ))}
              </div>
            </Card>
          )}
        </TabsContent>

        {/* Mathematics Tab */}
        <TabsContent value="mathematics" className="space-y-6">
          <Card className="p-6">
            <h3 className="text-lg font-semibold font-serif mb-4">Mathematical Framework</h3>
            <div className="space-y-4">
              {module.theory.mathematics.map((math, index) => (
                <div key={index} className="p-4 bg-muted/20 rounded-lg">
                  <p className="text-sm font-mono">{math}</p>
                </div>
              ))}
            </div>
          </Card>
        </TabsContent>

        {/* Applications Tab */}
        <TabsContent value="applications" className="space-y-6">
          {detailedTheory?.applications ? (
            detailedTheory.applications.map((app, index) => (
              <Card key={index} className="p-6">
                <h3 className="text-lg font-semibold font-serif mb-3">{app.title}</h3>
                <p className="text-sm text-muted-foreground mb-4 leading-relaxed">
                  {app.description}
                </p>
                
                <div className="space-y-2">
                  <h4 className="text-sm font-semibold">Examples:</h4>
                  <ul className="list-disc list-inside space-y-1">
                    {app.examples.map((example, exIndex) => (
                      <li key={exIndex} className="text-sm text-muted-foreground">
                        {example}
                      </li>
                    ))}
                  </ul>
                </div>
              </Card>
            ))
          ) : (
            <Card className="p-6">
              <h3 className="text-lg font-semibold font-serif mb-4">Applications</h3>
              <p className="text-muted-foreground">
                This module demonstrates fundamental concepts with direct applications 
                in computational physics, materials science, and quantum mechanics research.
              </p>
            </Card>
          )}
        </TabsContent>

        {/* References Tab */}
        <TabsContent value="references" className="space-y-4">
          <div className="space-y-4">
            {detailedTheory?.furtherReading ? (
              detailedTheory.furtherReading.map((ref, index) => (
                <ReferenceItem key={index} reference={ref} />
              ))
            ) : (
              module.theory.references.map((ref, index) => (
                <Card key={index} className="p-4">
                  <p className="text-sm font-serif">{ref}</p>
                </Card>
              ))
            )}
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default PhysicsTheory;