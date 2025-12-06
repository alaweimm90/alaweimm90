import React, { useMemo } from 'react';
import { AdvancedSimulationFeatures } from '@/components/AdvancedSimulationFeatures';
import { useSEO } from '@/hooks/use-seo';

export default function AdvancedSimulation() {
  useSEO({ title: 'Advanced Simulation Features – SimCore', description: 'Browser‑native controls and analysis for research‑grade physics simulations.' });
  const schemaLD = useMemo(() => ({
    '@context': 'https://schema.org',
    '@type': 'TechArticle',
    headline: 'Advanced Simulation Features',
    description: 'Control and analyze advanced physics simulations.',
    inLanguage: 'en',
    author: { '@type': 'Organization', name: 'SimCore' },
    publisher: { '@type': 'Organization', name: 'SimCore' },
    keywords: 'advanced simulation, physics control, analysis',
    mainEntityOfPage: typeof window !== 'undefined' ? window.location.href : ''
  }), []);
  return (
    <div className="container mx-auto p-6">
      <script type="application/ld+json" dangerouslySetInnerHTML={{ __html: JSON.stringify(schemaLD) }} />
      <div className="mb-6">
        <h1 className="text-3xl font-bold">Advanced Simulation Features</h1>
        <p className="text-muted-foreground">
          Comprehensive simulation control and analysis platform
        </p>
      </div>
      
      <AdvancedSimulationFeatures
        simulationType="advanced-physics"
        onParameterChange={(params) => {
          console.log('Parameters updated:', params);
        }}
        onStateChange={(state) => {
          console.log('State updated:', state);
        }}
      />
    </div>
  );
}