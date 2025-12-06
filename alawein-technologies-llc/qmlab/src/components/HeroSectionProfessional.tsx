import React from 'react';
import { Button } from '@/components/ui/button';
import { ArrowRight, ChevronRight } from 'lucide-react';

export const HeroSectionProfessional: React.FC = () => {
  const scrollToPlayground = () => {
    const mainContent = document.getElementById('main-content');
    if (mainContent) {
      mainContent.scrollIntoView({ behavior: 'smooth' });
    }
  };

  const features = [
    'Drag-and-drop circuit builder',
    'Real-time Bloch sphere visualization',
    'Variational quantum classifier training',
    'Export to Qiskit and Cirq'
  ];

  return (
    <section className="relative min-h-screen flex items-center justify-center bg-gradient-to-b from-slate-950 to-slate-900">
      {/* Subtle grid pattern */}
      <div className="absolute inset-0 bg-[url('/grid.svg')] bg-center opacity-[0.02]" />
      
      <div className="container mx-auto px-6 py-20 relative z-10">
        <div className="max-w-4xl mx-auto text-center">
          {/* Main Content */}
          <div className="mb-8">
            <p className="text-sm uppercase tracking-wider text-slate-500 mb-4">
              Quantum Computing Platform
            </p>
            <h1 className="text-5xl md:text-7xl font-bold text-white mb-6">
              QMLab
            </h1>
            <p className="text-xl md:text-2xl text-slate-400 font-light mb-8 max-w-2xl mx-auto">
              A research-grade platform for quantum machine learning experimentation
            </p>
          </div>

          {/* Feature List */}
          <div className="mb-12">
            <ul className="inline-flex flex-col md:flex-row gap-4 md:gap-8 text-sm text-slate-500">
              {features.map((feature, index) => (
                <li key={index} className="flex items-center gap-2">
                  <ChevronRight className="w-4 h-4 text-slate-600" />
                  <span>{feature}</span>
                </li>
              ))}
            </ul>
          </div>

          {/* CTA Buttons */}
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Button
              size="lg"
              onClick={scrollToPlayground}
              className="h-12 px-8 bg-white text-slate-900 hover:bg-slate-100 font-medium"
            >
              Start Building
              <ArrowRight className="ml-2 w-4 h-4" />
            </Button>
            <Button
              size="lg"
              variant="outline"
              className="h-12 px-8 border-slate-700 text-white hover:bg-slate-800/50"
              onClick={() => {
                const docs = document.getElementById('learning-resources');
                docs?.scrollIntoView({ behavior: 'smooth' });
              }}
            >
              View Documentation
            </Button>
          </div>

          {/* Bottom Stats */}
          <div className="mt-20 pt-12 border-t border-slate-800">
            <div className="grid grid-cols-2 md:grid-cols-4 gap-8 max-w-2xl mx-auto">
              <div>
                <p className="text-2xl font-bold text-white">15+</p>
                <p className="text-xs text-slate-500 uppercase tracking-wider">Quantum Gates</p>
              </div>
              <div>
                <p className="text-2xl font-bold text-white">3D</p>
                <p className="text-xs text-slate-500 uppercase tracking-wider">Visualization</p>
              </div>
              <div>
                <p className="text-2xl font-bold text-white">VQC</p>
                <p className="text-xs text-slate-500 uppercase tracking-wider">Algorithm Support</p>
              </div>
              <div>
                <p className="text-2xl font-bold text-white">100%</p>
                <p className="text-xs text-slate-500 uppercase tracking-wider">Open Source</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};