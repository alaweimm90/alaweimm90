import React from 'react';
import { Badge } from '@/components/ui/badge';
import { Card } from '@/components/ui/card';
import { CheckCircle, Layers, Palette, Sparkles } from 'lucide-react';

interface StyleSystemStatusProps {
  className?: string;
}

export const StyleSystemStatus: React.FC<StyleSystemStatusProps> = ({ className = '' }) => {
  const completedFeatures = [
    {
      title: 'PhysicsModuleLayout',
      description: 'Unified layout system across all 21+ physics modules',
      status: 'Complete',
      icon: <Layers className="h-4 w-4" />
    },
    {
      title: 'PhysicsModuleHeader', 
      description: 'Consistent headers with domain-specific theming',
      status: 'Complete',
      icon: <Sparkles className="h-4 w-4" />
    },
    {
      title: 'Domain Theme System',
      description: 'Dynamic theming for quantum, statistical, energy, and fields domains',
      status: 'Complete', 
      icon: <Palette className="h-4 w-4" />
    },
    {
      title: 'Responsive Design',
      description: 'Mobile-first responsive layouts with touch optimization',
      status: 'Complete',
      icon: <CheckCircle className="h-4 w-4" />
    }
  ];

  return (
    <Card className={`p-6 ${className}`}>
      <div className="flex items-center gap-2 mb-4">
        <CheckCircle className="h-5 w-5 text-green-500" />
        <h3 className="text-lg font-semibold">Style System Conversion Complete</h3>
      </div>
      
      <p className="text-muted-foreground mb-6">
        All physics modules have been successfully migrated to the unified PhysicsModuleLayout system,
        providing consistent theming, responsive design, and enhanced user experience.
      </p>

      <div className="grid gap-4 sm:grid-cols-2">
        {completedFeatures.map((feature, index) => (
          <div key={index} className="flex items-start gap-3 p-3 rounded-lg bg-muted/50">
            <div className="flex-shrink-0 mt-0.5 text-primary">
              {feature.icon}
            </div>
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2 mb-1">
                <h4 className="font-medium text-sm">{feature.title}</h4>
                <Badge variant="secondary" className="text-xs">
                  {feature.status}
                </Badge>
              </div>
              <p className="text-xs text-muted-foreground leading-relaxed">
                {feature.description}
              </p>
            </div>
          </div>
        ))}
      </div>

      <div className="mt-6 p-4 rounded-lg bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800">
        <div className="flex items-center gap-2 text-green-700 dark:text-green-300">
          <CheckCircle className="h-4 w-4" />
          <span className="font-medium text-sm">Migration Summary</span>
        </div>
        <p className="text-green-600 dark:text-green-400 text-xs mt-1">
          ✅ 21 physics modules converted<br/>
          ✅ 4 utility components updated<br/>
          ✅ Unified design system implemented<br/>
          ✅ Responsive layouts optimized
        </p>
      </div>
    </Card>
  );
};

export default StyleSystemStatus;