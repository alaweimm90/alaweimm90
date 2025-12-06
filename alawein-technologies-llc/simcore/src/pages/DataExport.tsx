import React, { useMemo } from 'react';
import { ScientificDataExport } from '@/components/ScientificDataExport';
import SectionHeader from '@/components/SectionHeader';
import { useSEO } from '@/hooks/use-seo';

export default function DataExport() {
  useSEO({ title: 'Data Export & Reporting – SimCore', description: 'Export simulation data and capture publication-quality visualizations.' });
  const schemaLD = useMemo(() => ({
    '@context': 'https://schema.org',
    '@type': 'TechArticle',
    headline: 'Data Export & Reporting',
    description: 'Export simulation data and capture publication-quality visualizations.',
    inLanguage: 'en',
    author: { '@type': 'Organization', name: 'SimCore' },
    publisher: { '@type': 'Organization', name: 'SimCore' },
    keywords: 'scientific data export, visualization, report generation',
    mainEntityOfPage: typeof window !== 'undefined' ? window.location.href : ''
  }), []);
  return (
    <div className="max-w-7xl mx-auto px-4 md:px-6 py-8 md:py-12">
      <script type="application/ld+json" dangerouslySetInnerHTML={{ __html: JSON.stringify(schemaLD) }} />
      <SectionHeader
        as="h1"
        title="Data Export & Reporting"
        subtitle="Export simulation data, capture visualizations, and generate professional scientific reports."
        variant="quantum"
        styleType="panel"
        eyebrow="Tools"
      />
      <div className="card-surface-glass rounded-3xl p-4 md:p-6">
        <ScientificDataExport />
      </div>
    </div>
  );
}