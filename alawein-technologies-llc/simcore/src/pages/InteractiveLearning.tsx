import React, { useMemo } from 'react';
import { InteractiveLearningPath } from '@/components/InteractiveLearningPath';
import { useSEO } from '@/hooks/use-seo';

export default function InteractiveLearning() {
  useSEO({ title: 'Interactive Learning – SimCore', description: 'Guided learning paths with simulations and exercises to master scientific concepts.' });
  const schemaLD = useMemo(() => ({
    '@context': 'https://schema.org',
    '@type': 'TechArticle',
    headline: 'Interactive Learning',
    description: 'Guided learning paths with simulations and exercises to master scientific concepts.',
    inLanguage: 'en',
    author: { '@type': 'Organization', name: 'SimCore' },
    publisher: { '@type': 'Organization', name: 'SimCore' },
    keywords: 'interactive learning, physics simulations, guided exercises',
    mainEntityOfPage: typeof window !== 'undefined' ? window.location.href : ''
  }), []);
  return (
    <div className="container mx-auto py-8 px-4">
      <div className="max-w-6xl mx-auto">
        <script type="application/ld+json" dangerouslySetInnerHTML={{ __html: JSON.stringify(schemaLD) }} />
        <div className="mb-8">
          <h1 className="text-3xl font-bold tracking-tight mb-2">Interactive Learning</h1>
          <p className="text-muted-foreground">
            Master scientific concepts through guided, interactive learning paths with simulations and exercises.
          </p>
        </div>
        
        <InteractiveLearningPath />
      </div>
    </div>
  );
}