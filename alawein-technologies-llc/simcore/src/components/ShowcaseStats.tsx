import React from 'react';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { useDomainTheme } from '@/components/DomainThemeProvider';
import { usePrefersReducedMotion } from '@/lib/accessibility-utils';
import { physicsModules, moduleCategories } from '@/data/modules';

export const ShowcaseStats: React.FC = () => {
  const { currentDomain } = useDomainTheme();
  const prefersReducedMotion = usePrefersReducedMotion();

  const stats = [
    {
      value: physicsModules.length,
      label: 'Total Modules',
      description: 'Interactive simulations'
    },
    {
      value: physicsModules.filter(m => m.isImplemented).length,
      label: 'Ready to Use',
      description: 'Fully functional modules'
    },
    {
      value: moduleCategories.length,
      label: 'Categories',
      description: 'Physics domains covered'
    }
  ];

  return (
    <div className="max-w-6xl mx-auto px-4 md:px-6 mt-12">
      <div className={`grid grid-cols-1 sm:grid-cols-3 gap-4 ${!prefersReducedMotion ? 'animate-fade-in' : ''}`}>
        {stats.map(({ value, label, description }, index) => (
          <Card
            key={label}
            variant="physics"
            domain={currentDomain}
            className={`group relative rounded-2xl overflow-hidden transition-all duration-300 ${
              !prefersReducedMotion ? 'hover:scale-[1.02] hover-scale' : ''
            }`}
          >
            <div
              aria-hidden
              className="pointer-events-none absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-300"
              style={{
                background: `radial-gradient(300px 100px at 50% 0%, hsl(var(--semantic-domain-${currentDomain}) / 0.08), transparent 60%)`
              }}
            />
            <div className="relative p-4 md:p-6 text-center">
              <div 
                className="text-3xl md:text-4xl font-bold mb-2"
                style={{ color: `hsl(var(--semantic-domain-${currentDomain}))` }}
              >
                {value}
              </div>
              <h3 className="text-lg font-semibold text-foreground mb-1">
                {label}
              </h3>
              <p className="text-sm text-muted-foreground">
                {description}
              </p>
              <Badge
                variant="outline"
                className="mt-3 bg-transparent"
                style={{
                  borderColor: `hsl(var(--semantic-domain-${currentDomain}))`,
                  color: `hsl(var(--semantic-domain-${currentDomain}))`,
                }}
              >
                {currentDomain.charAt(0).toUpperCase() + currentDomain.slice(1)}
              </Badge>
            </div>
          </Card>
        ))}
      </div>
    </div>
  );
};