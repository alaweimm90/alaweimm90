import React, { useMemo } from 'react';
import { EnhancedPhysicsVisualization } from '@/components/EnhancedPhysicsVisualization';
import SectionHeader from '@/components/SectionHeader';
import { useSEO } from '@/hooks/use-seo';

export default function EnhancedVisualization() {
  useSEO({ title: 'Enhanced Physics Visualization – SimCore', description: 'Advanced 3D physics simulations with particles, waves, and quantum effects.' });
  const schemaLD = useMemo(() => ({
    '@context': 'https://schema.org',
    '@type': 'TechArticle',
    headline: 'Enhanced Physics Visualization',
    description: 'Advanced 3D physics simulations with particles, waves, and quantum effects.',
    inLanguage: 'en',
    author: { '@type': 'Organization', name: 'SimCore' },
    publisher: { '@type': 'Organization', name: 'SimCore' },
    keywords: '3D physics simulation, visualization, quantum effects',
    mainEntityOfPage: typeof window !== 'undefined' ? window.location.href : ''
  }), []);
  return (
    <div className="max-w-7xl mx-auto px-4 md:px-6 py-8 md:py-12">
      <script type="application/ld+json" dangerouslySetInnerHTML={{ __html: JSON.stringify(schemaLD) }} />
      <SectionHeader
        as="h1"
        title="Enhanced Physics Visualization"
        subtitle="Advanced 3D physics simulations with particles, waves, and quantum effects."
        variant="field"
        styleType="panel"
        eyebrow="Visualization suite"
      />
      <div className="rounded-3xl bg-gradient-to-b from-background/40 via-card/30 to-background/40 border border-border/30 p-4 md:p-6">
        <EnhancedPhysicsVisualization />
      </div>
    </div>
  );
}