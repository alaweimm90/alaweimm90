/**
 * WebWorker Demo Page
 * Showcases multi-threaded physics calculations
 */

import React, { useMemo } from 'react';
import { PhysicsModuleLayout } from '@/components/PhysicsModuleLayout';
import { WebWorkerDemo } from '@/components/WebWorkerDemo';
import { useSEO } from '@/hooks/use-seo';

export default function WebWorkerDemoPage() {
  useSEO({ title: 'Web Worker Demo – SimCore', description: 'Multi-threaded physics calculations using Web Workers.' });
  const schemaLD = useMemo(() => ({
    '@context': 'https://schema.org',
    '@type': 'TechArticle',
    headline: 'Web Worker Demo',
    description: 'Multi-threaded physics calculations using Web Workers.',
    inLanguage: 'en',
    author: { '@type': 'Organization', name: 'SimCore' },
    publisher: { '@type': 'Organization', name: 'SimCore' },
    keywords: 'web workers, multithreading, physics calculations',
    mainEntityOfPage: typeof window !== 'undefined' ? window.location.href : ''
  }), []);
  return (
    <PhysicsModuleLayout>
      <script type="application/ld+json" dangerouslySetInnerHTML={{ __html: JSON.stringify(schemaLD) }} />
      <WebWorkerDemo />
    </PhysicsModuleLayout>
  );
}