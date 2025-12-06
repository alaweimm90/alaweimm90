import React from 'react';
import { Card } from './card';
import { Badge } from './badge';
import { Progress } from './progress';
import { CheckCircle, AlertCircle, Clock, Zap, BarChart3, Shield } from 'lucide-react';

interface MethodComparisonProps {
  className?: string;
}

export function MethodComparison({ className }: MethodComparisonProps) {
  const methods = [
    {
      name: 'GLTR Analysis',
      description: 'Statistical token likelihood patterns',
      speed: 95,
      accuracy: 78,
      coverage: 85,
      localOnly: true,
      status: 'ready',
      icon: <BarChart3 className="h-4 w-4" />,
      pros: ['Fast processing', 'No API required', 'Well-researched'],
      cons: ['GPT-2 only', 'May miss newer models'],
      bestFor: 'General text analysis'
    },
    {
      name: 'DetectGPT',
      description: 'Probability curvature analysis',
      speed: 45,
      accuracy: 82,
      coverage: 70,
      localOnly: false,
      status: 'api-required',
      icon: <Zap className="h-4 w-4" />,
      pros: ['Model-agnostic', 'Theoretically grounded'],
      cons: ['Slower processing', 'Requires API'],
      bestFor: 'Academic verification'
    },
    {
      name: 'Citation Validation',
      description: 'Reference cross-checking',
      speed: 60,
      accuracy: 95,
      coverage: 30,
      localOnly: false,
      status: 'ready',
      icon: <CheckCircle className="h-4 w-4" />,
      pros: ['Objective results', 'High accuracy'],
      cons: ['Academic only', 'Limited scope'],
      bestFor: 'Research papers'
    },
    {
      name: 'Watermark Detection',
      description: 'Cryptographic markers',
      speed: 90,
      accuracy: 99,
      coverage: 5,
      localOnly: true,
      status: 'experimental',
      icon: <Shield className="h-4 w-4" />,
      pros: ['Extremely accurate', 'Fast processing'],
      cons: ['Rarely present', 'Model dependent'],
      bestFor: 'Known watermarked text'
    }
  ];

  return (
    <div className={`grid md:grid-cols-2 gap-4 ${className}`}>
      {methods.map((method) => (
        <Card key={method.name} className="p-6">
          <div className="flex items-start justify-between mb-4">
            <div className="flex items-center gap-2">
              {method.icon}
              <h3 className="font-semibold">{method.name}</h3>
            </div>
            <Badge 
              variant={
                method.status === 'ready' ? 'default' :
                method.status === 'api-required' ? 'secondary' :
                'outline'
              }
              className="text-xs"
            >
              {method.status === 'ready' ? 'Ready' :
               method.status === 'api-required' ? 'API Required' :
               'Experimental'}
            </Badge>
          </div>

          <p className="text-sm text-muted-foreground mb-4">
            {method.description}
          </p>

          <div className="space-y-3 mb-4">
            <div className="flex items-center justify-between">
              <span className="text-sm">Speed</span>
              <div className="flex items-center gap-2 w-24">
                <Progress value={method.speed} className="h-1" />
                <span className="text-xs">{method.speed}%</span>
              </div>
            </div>
            
            <div className="flex items-center justify-between">
              <span className="text-sm">Accuracy</span>
              <div className="flex items-center gap-2 w-24">
                <Progress value={method.accuracy} className="h-1" />
                <span className="text-xs">{method.accuracy}%</span>
              </div>
            </div>
            
            <div className="flex items-center justify-between">
              <span className="text-sm">Coverage</span>
              <div className="flex items-center gap-2 w-24">
                <Progress value={method.coverage} className="h-1" />
                <span className="text-xs">{method.coverage}%</span>
              </div>
            </div>
          </div>

          <div className="space-y-3 text-xs">
            <div>
              <div className="font-medium text-success mb-1">Strengths</div>
              <ul className="text-muted-foreground space-y-1">
                {method.pros.map((pro, i) => (
                  <li key={i}>• {pro}</li>
                ))}
              </ul>
            </div>
            
            <div>
              <div className="font-medium text-destructive mb-1">Limitations</div>
              <ul className="text-muted-foreground space-y-1">
                {method.cons.map((con, i) => (
                  <li key={i}>• {con}</li>
                ))}
              </ul>
            </div>
          </div>

          <div className="mt-4 pt-3 border-t">
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium">Best for</span>
              <span className="text-sm text-muted-foreground">{method.bestFor}</span>
            </div>
            <div className="flex items-center gap-2 mt-1">
              {method.localOnly ? (
                <Badge variant="outline" className="text-xs">
                  Local Only
                </Badge>
              ) : (
                <Badge variant="secondary" className="text-xs">
                  API Required
                </Badge>
              )}
            </div>
          </div>
        </Card>
      ))}
    </div>
  );
}