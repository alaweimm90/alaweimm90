import React from 'react';
import { Card, CardContent } from '@/components/ui/card';
import BrandLogo from '@/components/ui/BrandLogo';

import { Button } from '@/components/ui/button';
import { 
  Sparkles, 
  Zap, 
  BookOpen, 
  Users, 
  ArrowRight
} from 'lucide-react';

import { useToast } from '@/hooks/use-toast';

interface WelcomeBannerProps {
  className?: string;
}

export const WelcomeBanner: React.FC<WelcomeBannerProps> = ({ 
  className = '' 
}) => {
  
  const { toast } = useToast();



  return (
    <Card className={`w-full bg-gradient-subtle border-border/50 ${className}`}>
      <CardContent className="p-[--spacing-2xl]">
        <div className="space-y-[--spacing-2xl]">
          {/* Welcome Header */}
          <div className="text-center space-y-[--spacing-lg]">
              <div className="flex items-center justify-center gap-[--spacing-sm] mb-[--spacing-lg]">
                <Sparkles className="w-6 h-6 text-primary animate-pulse" />
                <h2 className="text-3xl font-bold flex items-center gap-2">
                  <span className="text-foreground">Welcome to</span>
                  <BrandLogo inline size="md" withIcon={false} />
                </h2>
                <Sparkles className="w-6 h-6 text-primary animate-pulse" />
              </div>
          </div>



          {/* Call to Action */}
          <div className="text-center pt-[--spacing-xl] border-t border-border/50">
            <div className="space-y-[--spacing-lg]">
              <Button 
                variant="hero" 
                size="lg"
                onClick={() => {
                  const moduleSection = document.getElementById('modules-grid-heading');
                  if (moduleSection) {
                    moduleSection.scrollIntoView({ behavior: 'smooth' });
                  } else {
                    toast({
                      title: "Modules Below",
                      description: "Scroll down to explore all available scientific modules.",
                    });
                  }
                }}
                className="group min-h-[--touch-target-min]"
              >
                Explore All Modules
                <ArrowRight className="w-4 h-4 ml-[--spacing-xs] group-hover:translate-x-1 transition-transform" />
              </Button>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};