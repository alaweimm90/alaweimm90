import React from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import { BookOpen, Target, Lightbulb, Info } from 'lucide-react';
import { InlineMath, BlockMath } from '@/components/ui/Math';

interface ModuleDocumentationProps {
  title: string;
  description: string;
  objectives: string[];
  keyEquations: Array<{
    name: string;
    equation: string;
    description: string;
  }>;
  parameters: Array<{
    name: string;
    symbol?: string;
    description: string;
    units?: string;
    typicalRange: string;
    physicalMeaning: string;
  }>;
  applications: Array<{
    field: string;
    examples: string[];
  }>;
  tips: string[];
  className?: string;
}

export const DocumentationPanel: React.FC<ModuleDocumentationProps> = ({
  title,
  description,
  objectives,
  keyEquations,
  parameters,
  applications,
  tips,
  className = ''
}) => {
  return (
    <Card className={`bg-card/50 backdrop-blur-sm border-border/50 ${className}`}>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <BookOpen className="w-5 h-5" />
          {title} Documentation
        </CardTitle>
        <CardDescription>{description}</CardDescription>
      </CardHeader>
      <CardContent>
        <Tabs defaultValue="overview" className="w-full">
          <TabsList className="grid w-full grid-cols-4">
            <TabsTrigger value="overview" className="text-xs min-h-[--touch-target-min]">
              <Target className="w-3 h-3 mr-[--spacing-xs]" />
              Overview
            </TabsTrigger>
            <TabsTrigger value="equations" className="text-xs min-h-[--touch-target-min]">
              Equations
            </TabsTrigger>
            <TabsTrigger value="parameters" className="text-xs min-h-[--touch-target-min]">
              <Info className="w-3 h-3 mr-[--spacing-xs]" />
              Parameters
            </TabsTrigger>
            <TabsTrigger value="applications" className="text-xs min-h-[--touch-target-min]">
              <Lightbulb className="w-3 h-3 mr-[--spacing-xs]" />
              Uses
            </TabsTrigger>
          </TabsList>

          <TabsContent value="overview" className="space-y-[--spacing-lg] mt-[--spacing-lg]">
            <div>
              <h4 className="font-medium text-sm mb-[--spacing-sm] text-foreground">Learning Objectives</h4>
              <div className="space-y-[--spacing-sm]">
                {objectives.map((objective, index) => (
                  <div key={index} className="flex items-start gap-[--spacing-sm]">
                    <div className="w-2 h-2 bg-primary rounded-full mt-2 flex-shrink-0" />
                    <div className="text-sm text-foreground leading-relaxed">{objective}</div>
                  </div>
                ))}
              </div>
            </div>
            
            <Separator />
            
            <div>
              <h4 className="font-medium text-sm mb-[--spacing-sm] text-foreground">Tips for Learning</h4>
              <div className="space-y-[--spacing-sm]">
                {tips.map((tip, index) => (
                  <div key={index} className="flex items-start gap-[--spacing-sm] p-[--spacing-sm] bg-accent/20 rounded border border-accent/30">
                    <Lightbulb className="w-4 h-4 text-accent-foreground mt-0.5 flex-shrink-0" />
                    <div className="text-sm text-foreground leading-relaxed">{tip}</div>
                  </div>
                ))}
              </div>
            </div>
          </TabsContent>

          <TabsContent value="equations" className="space-y-[--spacing-lg] mt-[--spacing-lg]">
            <div className="space-y-[--spacing-lg]">
              {keyEquations.map((eq, index) => (
                <div key={index} className="p-[--spacing-md] bg-muted/30 rounded border border-border/30">
                  <div className="font-medium text-sm mb-[--spacing-sm] text-foreground">{eq.name}</div>
                  <div className="flex flex-col gap-[--spacing-sm]">
                    <div className="bg-background/50 p-[--spacing-md] rounded border overflow-x-auto">
                      <BlockMath math={eq.equation} />
                    </div>
                    <div className="text-sm text-muted-foreground leading-relaxed">{eq.description}</div>
                  </div>
                </div>
              ))}
            </div>
          </TabsContent>

          <TabsContent value="parameters" className="space-y-[--spacing-lg] mt-[--spacing-lg]">
            <div className="space-y-[--spacing-md]">
              {parameters.map((param, index) => (
                <div key={index} className="p-[--spacing-md] bg-muted/30 rounded border border-border/30">
                  <div className="flex items-center gap-[--spacing-sm] mb-[--spacing-sm]">
                    <Badge variant="outline" className="text-xs font-mono">
                      {param.symbol || param.name}
                    </Badge>
                    {param.units && (
                      <Badge variant="secondary" className="text-xs">
                        {param.units}
                      </Badge>
                    )}
                  </div>
                  <div className="space-y-[--spacing-sm]">
                    <div className="font-medium text-sm text-foreground">{param.name}</div>
                    <div className="text-sm text-foreground leading-relaxed">{param.description}</div>
                    <div className="text-xs text-muted-foreground">
                      <span className="font-medium">Range:</span> {param.typicalRange}
                    </div>
                    <div className="text-xs p-[--spacing-sm] bg-accent/10 rounded border border-accent/20">
                      <span className="font-medium">Physical meaning:</span> {param.physicalMeaning}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </TabsContent>

          <TabsContent value="applications" className="space-y-[--spacing-lg] mt-[--spacing-lg]">
            <div className="space-y-[--spacing-lg]">
              {applications.map((app, index) => (
                <div key={index} className="p-[--spacing-md] bg-muted/30 rounded border border-border/30">
                  <div className="font-medium text-sm mb-[--spacing-sm] flex items-center gap-[--spacing-sm]">
                    <Badge variant="outline">{app.field}</Badge>
                  </div>
                  <div className="space-y-[--spacing-xs]">
                    {app.examples.map((example, exIndex) => (
                      <div key={exIndex} className="flex items-start gap-[--spacing-sm]">
                        <div className="w-2 h-2 bg-primary rounded-full mt-2 flex-shrink-0" />
                        <div className="text-sm text-foreground leading-relaxed">{example}</div>
                      </div>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
};