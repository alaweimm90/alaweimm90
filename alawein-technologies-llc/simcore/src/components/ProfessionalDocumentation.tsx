import React, { useState, useMemo } from 'react';
import { BookOpen, Search, ExternalLink, Code, Download, ChevronRight, FileText, Lightbulb } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible';
import { cn } from '@/lib/utils';

interface DocumentationSection {
  id: string;
  title: string;
  category: 'theory' | 'usage' | 'examples' | 'api';
  content: string;
  codeExample?: string;
  references?: string[];
  difficulty: 'beginner' | 'intermediate' | 'advanced';
}

interface ProfessionalDocumentationProps {
  moduleName: string;
  sections?: DocumentationSection[];
  className?: string;
}

const defaultSections: DocumentationSection[] = [
  {
    id: 'overview',
    title: 'Theoretical Overview',
    category: 'theory',
    content: 'This simulation implements advanced quantum mechanical principles using tight-binding models and density functional theory calculations. The visualization provides real-time insights into electronic band structures, density of states, and quantum transport properties.',
    difficulty: 'intermediate',
    references: [
      'Ashcroft & Mermin - Solid State Physics',
      'Kittel - Introduction to Solid State Physics'
    ]
  },
  {
    id: 'parameters',
    title: 'Parameter Controls',
    category: 'usage',
    content: 'Interactive controls allow real-time modification of physical parameters. Each parameter includes tooltips with physical significance and recommended ranges based on experimental data.',
    codeExample: `// Example parameter usage
const parameters = {
  latticeConstant: 2.46, // Angstroms
  hoppingEnergy: 2.8,    // eV
  temperature: 300       // Kelvin
};`,
    difficulty: 'beginner'
  },
  {
    id: 'advanced',
    title: 'Advanced Features',
    category: 'examples',
    content: 'Explore quantum Hall effects, topological phases, and many-body interactions through specialized visualization modes and calculation engines.',
    difficulty: 'advanced',
    references: [
      'Thouless et al. - Quantized Hall Conductance',
      'Haldane - Model for Quantum Hall Effect'
    ]
  }
];

export const ProfessionalDocumentation: React.FC<ProfessionalDocumentationProps> = ({
  moduleName,
  sections = defaultSections,
  className
}) => {
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedCategory, setSelectedCategory] = useState<string>('all');

  const filteredSections = useMemo(() => {
    return sections.filter(section => {
      const matchesSearch = section.title.toLowerCase().includes(searchTerm.toLowerCase()) ||
                           section.content.toLowerCase().includes(searchTerm.toLowerCase());
      const matchesCategory = selectedCategory === 'all' || section.category === selectedCategory;
      return matchesSearch && matchesCategory;
    });
  }, [sections, searchTerm, selectedCategory]);

  const categoryColors = {
    theory: 'bg-blue-500/10 text-blue-700 border-blue-500/20',
    usage: 'bg-green-500/10 text-green-700 border-green-500/20',
    examples: 'bg-purple-500/10 text-purple-700 border-purple-500/20',
    api: 'bg-orange-500/10 text-orange-700 border-orange-500/20'
  };

  const difficultyColors = {
    beginner: 'bg-green-100 text-green-800',
    intermediate: 'bg-yellow-100 text-yellow-800',
    advanced: 'bg-red-100 text-red-800'
  };

  return (
    <Card className={cn("w-full", className)}>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <BookOpen className="w-5 h-5" />
          {moduleName} Documentation
        </CardTitle>
        
        <div className="flex gap-2">
          <div className="relative flex-1">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-muted-foreground" />
            <Input
              placeholder="Search documentation..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="pl-10"
            />
          </div>
          <Button variant="outline" size="sm">
            <Download className="w-4 h-4 mr-2" />
            PDF
          </Button>
        </div>
      </CardHeader>
      
      <CardContent>
        <Tabs value={selectedCategory} onValueChange={setSelectedCategory}>
          <TabsList className="grid grid-cols-5 w-full">
            <TabsTrigger value="all">All</TabsTrigger>
            <TabsTrigger value="theory">Theory</TabsTrigger>
            <TabsTrigger value="usage">Usage</TabsTrigger>
            <TabsTrigger value="examples">Examples</TabsTrigger>
            <TabsTrigger value="api">API</TabsTrigger>
          </TabsList>
          
          <TabsContent value={selectedCategory} className="mt-4">
            <ScrollArea className="h-[600px] pr-4">
              <div className="space-y-4">
                {filteredSections.map((section) => (
                  <DocumentationCard key={section.id} section={section} />
                ))}
                
                {filteredSections.length === 0 && (
                  <div className="text-center py-8 text-muted-foreground">
                    <FileText className="w-12 h-12 mx-auto mb-4 opacity-50" />
                    <p>No documentation found matching your search.</p>
                  </div>
                )}
              </div>
            </ScrollArea>
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
};

const DocumentationCard: React.FC<{ section: DocumentationSection }> = ({ section }) => {
  const [isExpanded, setIsExpanded] = useState(false);

  const categoryColors = {
    theory: 'bg-blue-500/10 text-blue-700 border-blue-500/20',
    usage: 'bg-green-500/10 text-green-700 border-green-500/20',
    examples: 'bg-purple-500/10 text-purple-700 border-purple-500/20',
    api: 'bg-orange-500/10 text-orange-700 border-orange-500/20'
  };

  const difficultyColors = {
    beginner: 'bg-green-100 text-green-800',
    intermediate: 'bg-yellow-100 text-yellow-800',
    advanced: 'bg-red-100 text-red-800'
  };

  return (
    <Card className="border-l-4 border-l-primary/50">
      <Collapsible open={isExpanded} onOpenChange={setIsExpanded}>
        <CollapsibleTrigger asChild>
          <CardHeader className="cursor-pointer hover:bg-muted/50 transition-colors">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <h3 className="font-semibold">{section.title}</h3>
                <div className="flex gap-2">
                  <Badge variant="outline" className={categoryColors[section.category]}>
                    {section.category}
                  </Badge>
                  <Badge variant="secondary" className={difficultyColors[section.difficulty]}>
                    {section.difficulty}
                  </Badge>
                </div>
              </div>
              <ChevronRight className={cn(
                "w-4 h-4 transition-transform",
                isExpanded && "rotate-90"
              )} />
            </div>
          </CardHeader>
        </CollapsibleTrigger>
        
        <CollapsibleContent>
          <CardContent className="pt-0">
            <div className="space-y-4">
              <p className="text-muted-foreground leading-relaxed">
                {section.content}
              </p>
              
              {section.codeExample && (
                <div className="bg-muted rounded-lg p-4">
                  <div className="flex items-center gap-2 mb-2">
                    <Code className="w-4 h-4" />
                    <span className="text-sm font-medium">Code Example</span>
                  </div>
                  <pre className="text-sm overflow-x-auto">
                    <code>{section.codeExample}</code>
                  </pre>
                </div>
              )}
              
              {section.references && section.references.length > 0 && (
                <div>
                  <div className="flex items-center gap-2 mb-2">
                    <ExternalLink className="w-4 h-4" />
                    <span className="text-sm font-medium">References</span>
                  </div>
                  <ul className="space-y-1">
                    {section.references.map((ref, index) => (
                      <li key={index} className="text-sm text-muted-foreground flex items-center gap-2">
                        <div className="w-1 h-1 bg-primary rounded-full" />
                        {ref}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          </CardContent>
        </CollapsibleContent>
      </Collapsible>
    </Card>
  );
};

// Quick Help Tooltip Component
export const QuickHelpTooltip: React.FC<{ topic: string; content: string }> = ({ topic, content }) => {
  return (
    <div className="group relative inline-block">
      <Button variant="ghost" size="sm" className="h-6 w-6 p-0">
        <Lightbulb className="w-3 h-3" />
      </Button>
      <div className="absolute bottom-full left-1/2 transform -translate-x-1/2 mb-2 w-64 p-3 bg-popover border rounded-lg shadow-lg opacity-0 group-hover:opacity-100 transition-opacity z-50">
        <h4 className="font-medium text-sm mb-1">{topic}</h4>
        <p className="text-xs text-muted-foreground">{content}</p>
      </div>
    </div>
  );
};