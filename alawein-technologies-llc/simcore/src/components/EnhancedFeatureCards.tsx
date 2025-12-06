import React from 'react';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Zap, BookOpen, Atom, Cpu, Layers, BarChart3 } from 'lucide-react';
import { usePrefersReducedMotion } from '@/lib/accessibility-utils';
import { motion } from 'framer-motion';

interface FeatureCardProps {
  icon: React.ComponentType<any>;
  title: string;
  description: string;
  features: string[];
  gradient: string;
  delay?: number;
}

const FeatureCard: React.FC<FeatureCardProps> = ({ 
  icon: Icon, 
  title, 
  description, 
  features, 
  gradient,
  delay = 0 
}) => {
  const prefersReducedMotion = usePrefersReducedMotion();

  return (
    <Card 
      className={`
        relative p-6 space-y-4 overflow-hidden group transition-all duration-500
        hover:shadow-2xl hover:shadow-primary/10 border-border/50
        ${!prefersReducedMotion ? 'hover:scale-105' : ''}
      `}
      style={{
        animationDelay: !prefersReducedMotion ? `${delay}ms` : '0ms'
      }}
    >
      {/* Background gradient overlay */}
      <div className={`absolute inset-0 opacity-0 group-hover:opacity-10 transition-opacity duration-500 ${gradient}`} />
      
      {/* Animated icon */}
      <div className={`
        relative w-16 h-16 rounded-2xl mx-auto flex items-center justify-center
        ${gradient} shadow-lg transition-all duration-500
        ${!prefersReducedMotion ? 'group-hover:scale-110 group-hover:rotate-3' : ''}
      `}>
        <Icon className="w-8 h-8 text-[hsl(var(--semantic-text-primary))]" />
        
        {/* Pulse effect */}
        {!prefersReducedMotion && (
          <div className="absolute inset-0 rounded-2xl animate-ping opacity-30" style={{ background: 'currentColor' }} />
        )}
      </div>
      
      {/* Content */}
      <div className="relative z-10 text-center space-y-3">
        <h3 className="text-xl font-bold text-foreground group-hover:text-primary transition-colors">
          {title}
        </h3>
        <p className="text-muted-foreground leading-relaxed">
          {description}
        </p>
        
        {/* Feature badges */}
        <div className="flex flex-wrap justify-center gap-2 pt-2">
          {features.map((feature, index) => (
            <Badge 
              key={index}
              variant="secondary"
              className={`
                text-xs bg-muted/50 border-0 transition-all duration-300
                ${!prefersReducedMotion ? 'group-hover:bg-primary/10 group-hover:text-primary' : ''}
              `}
            >
              {feature}
            </Badge>
          ))}
        </div>
      </div>
      
      {/* Corner decoration */}
      <div className="absolute top-2 right-2 w-2 h-2 bg-primary/20 rounded-full transition-all duration-500 group-hover:bg-primary/60 group-hover:scale-150" />
    </Card>
  );
};

export const EnhancedFeatureSection: React.FC = () => {
  const prefersReducedMotion = usePrefersReducedMotion();

  const features = [
    {
      icon: Zap,
      title: 'High Performance',
      description: 'WebGPU acceleration and optimized algorithms for real-time scientific simulations with sub-millisecond response times.',
      features: ['WebGPU', 'Real-time', 'Optimized'],
      gradient: 'bg-gradient-to-br from-blue-500 to-cyan-500'
    },
    {
      icon: BookOpen,
      title: 'Theory & Practice',
      description: 'Complete mathematical derivations alongside interactive visualizations, bridging theoretical concepts with hands-on exploration.',
      features: ['LaTeX', 'Interactive', 'Educational'],
      gradient: 'bg-gradient-to-br from-purple-500 to-pink-500'
    },
    {
      icon: Atom,
      title: 'Research-Grade',
      description: 'Export data, reproduce results, and integrate with your computational workflow using industry-standard formats.',
      features: ['Export', 'Reproducible', 'Integration'],
      gradient: 'bg-gradient-to-br from-green-500 to-emerald-500'
    },
    {
      icon: Cpu,
      title: 'Advanced Computing',
      description: 'State-of-the-art numerical methods including finite difference, Monte Carlo, and machine learning techniques.',
      features: ['Numerical', 'Monte Carlo', 'ML'],
      gradient: 'bg-gradient-to-br from-orange-500 to-red-500'
    },
    {
      icon: Layers,
      title: 'Multi-Physics',
      description: 'Coupled simulations spanning quantum mechanics, statistical physics, and materials science in a unified platform.',
      features: ['Quantum', 'Statistical', 'Materials'],
      gradient: 'bg-gradient-to-br from-indigo-500 to-blue-500'
    },
    {
      icon: BarChart3,
      title: 'Data Analytics',
      description: 'Built-in visualization tools and statistical analysis capabilities for comprehensive result interpretation.',
      features: ['Visualization', 'Statistics', 'Analysis'],
      gradient: 'bg-gradient-to-br from-teal-500 to-cyan-500'
    }
  ];

  return (
    <section className="my-32" aria-labelledby="enhanced-features-heading">
      <div className="text-center mb-20">
        <h2 id="enhanced-features-heading" className="text-4xl font-bold mb-6 gradient-text">
          Cutting-Edge Scientific Computing
        </h2>
        <p className="text-xl text-muted-foreground max-w-4xl mx-auto leading-relaxed">
          Built with modern web technologies and computational science expertise to deliver
          professional-grade simulation tools that run directly in your browser.
        </p>
      </div>
      
      <div className={`
        grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8
        ${!prefersReducedMotion ? 'animate-fade-in' : ''}
      `}>
        {features.map((feature, index) => (
          <motion.div
            key={feature.title}
            initial={prefersReducedMotion ? false : { opacity: 0, y: 12, scale: 0.98 }}
            whileInView={prefersReducedMotion ? undefined : { opacity: 1, y: 0, scale: 1 }}
            viewport={{ once: true, amount: 0.2 }}
            whileHover={prefersReducedMotion ? undefined : { scale: 1.02 }}
            transition={{ duration: 0.35, ease: 'easeOut' }}
          >
            <FeatureCard
              {...feature}
              delay={index * 100}
            />
          </motion.div>
        ))}
      </div>
    </section>
  );
};