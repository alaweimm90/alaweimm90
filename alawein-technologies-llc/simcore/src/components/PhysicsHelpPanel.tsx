import React from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Badge } from '@/components/ui/badge';
import { BookOpen, Info, Lightbulb, Calculator } from 'lucide-react';
import { InlineMath, BlockMath } from '@/components/ui/Math';

interface PhysicsHelpPanelProps {
  title: string;
  description: string;
  keyEquations: Array<{
    name: string;
    equation: string;
    description: string;
  }>;
  physicalPhenomena: Array<{
    name: string;
    description: string;
    relevance: string;
  }>;
  applications: string[];
  tips?: string[];
  className?: string;
}

export const PhysicsHelpPanel: React.FC<PhysicsHelpPanelProps> = ({
  title,
  description,
  keyEquations,
  physicalPhenomena,
  applications,
  tips = [],
  className = ''
}) => {
  return (
    <Card className={`bg-card/50 backdrop-blur-sm border-border/50 ${className}`}>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <BookOpen className="w-5 h-5" />
          Physics Guide: {title}
        </CardTitle>
        <CardDescription>{description}</CardDescription>
      </CardHeader>
      <CardContent>
        <Tabs defaultValue="equations" className="w-full">
          <TabsList className="grid w-full grid-cols-4">
            <TabsTrigger value="equations" className="text-xs">
              <Calculator className="w-3 h-3 mr-1" />
              Equations
            </TabsTrigger>
            <TabsTrigger value="phenomena" className="text-xs">
              <Info className="w-3 h-3 mr-1" />
              Physics
            </TabsTrigger>
            <TabsTrigger value="applications" className="text-xs">
              <Lightbulb className="w-3 h-3 mr-1" />
              Uses
            </TabsTrigger>
            <TabsTrigger value="tips" className="text-xs">
              Tips
            </TabsTrigger>
          </TabsList>

          <TabsContent value="equations" className="space-y-4 mt-4">
            <div className="space-y-3">
              {keyEquations.map((eq, index) => (
                <div key={index} className="p-3 bg-muted/30 rounded border border-border/30">
                  <div className="font-medium text-sm mb-2">{eq.name}</div>
                  <div className="flex flex-col gap-2">
                    <div className="bg-background/50 p-2 rounded border overflow-x-auto">
                      <BlockMath math={eq.equation} />
                    </div>
                    <div className="text-sm text-muted-foreground">{eq.description}</div>
                  </div>
                </div>
              ))}
            </div>
          </TabsContent>

          <TabsContent value="phenomena" className="space-y-4 mt-4">
            <div className="space-y-3">
              {physicalPhenomena.map((phenomenon, index) => (
                <div key={index} className="p-3 bg-muted/30 rounded border border-border/30">
                  <div className="flex items-center gap-2 mb-2">
                    <Badge variant="outline" className="text-xs">{phenomenon.name}</Badge>
                  </div>
                  <div className="text-sm mb-2">{phenomenon.description}</div>
                  <div className="text-xs text-muted-foreground">
                    <span className="font-medium">Relevance:</span> {phenomenon.relevance}
                  </div>
                </div>
              ))}
            </div>
          </TabsContent>

          <TabsContent value="applications" className="space-y-4 mt-4">
            <div className="space-y-2">
              {applications.map((app, index) => (
                <div key={index} className="flex items-start gap-2 p-2 bg-muted/20 rounded">
                  <div className="w-2 h-2 bg-primary rounded-full mt-2 flex-shrink-0" />
                  <div className="text-sm">{app}</div>
                </div>
              ))}
            </div>
          </TabsContent>

          <TabsContent value="tips" className="space-y-4 mt-4">
            {tips.length > 0 ? (
              <div className="space-y-2">
                {tips.map((tip, index) => (
                  <div key={index} className="flex items-start gap-2 p-3 bg-accent/20 rounded border border-accent/30">
                    <Lightbulb className="w-4 h-4 text-accent-foreground mt-0.5 flex-shrink-0" />
                    <div className="text-sm">{tip}</div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-sm text-muted-foreground text-center py-4">
                No tips available for this module.
              </div>
            )}
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
};