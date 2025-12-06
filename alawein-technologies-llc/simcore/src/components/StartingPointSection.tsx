import React from 'react';
import { motion } from 'framer-motion';
import { Button } from '@/components/ui/button';
import { Sparkles, Cpu, Layers } from 'lucide-react';
import SectionHeader from '@/components/SectionHeader';
import { IconChip } from '@/components/IconChip';

const container = {
  hidden: { opacity: 0, y: 16 },
  show: {
    opacity: 1,
    y: 0,
    transition: { staggerChildren: 0.08, duration: 0.4, ease: 'easeOut' }
  }
};

const card = {
  hidden: { opacity: 0, y: 12, scale: 0.98 },
  show: { opacity: 1, y: 0, scale: 1, transition: { duration: 0.35 } }
};

export const StartingPointSection: React.FC = () => {
  const handleScrollToModules = () => {
    const el = document.getElementById('modules-section');
    if (el) el.scrollIntoView({ behavior: 'smooth', block: 'start' });
  };

  const handleScrollToDemos = () => {
    const el = document.getElementById('feature-showcase');
    if (el) el.scrollIntoView({ behavior: 'smooth', block: 'start' });
  };

  return (
    <section className="my-24" aria-labelledby="starting-point-heading">
      {/* Divider above */}
      <div className="flex items-center justify-center mb-16">
        <div className="h-px w-full max-w-3xl bg-gradient-to-r from-transparent via-primary/30 to-transparent" />
      </div>

      <div className="max-w-7xl mx-auto px-4 md:px-6">
        <SectionHeader
          id="starting-point-heading"
          title="Choose Your Starting Point"
          subtitle="Pick a path that fits your experience—everything stays sleek, interactive, and scientific"
          variant="quantum"
          styleType="panel"
          eyebrow="Start here"
        />

        <motion.div
          className="grid grid-cols-1 md:grid-cols-3 gap-6 md:gap-8"
          variants={container}
          initial="hidden"
          whileInView="show"
          viewport={{ once: true, amount: 0.2 }}
        >
          {/* Theory-Driven */}
          <motion.div
            variants={card}
            className="group rounded-2xl card-surface-glass card-hover-raise h-full"
          >
            <div className="p-6 md:p-8 h-full flex flex-col justify-between">
              <div className="flex items-center gap-4 mb-4">
                <IconChip>
                  <Layers className="w-6 h-6 text-[hsl(var(--semantic-domain-quantum))]" />
                </IconChip>
                <div>
                  <h3 className="text-xl font-bold text-foreground transition-colors transition-transform duration-300 group-hover:gradient-text-quantum group-hover:-translate-y-0.5">
                    Theory-Driven
                  </h3>
                  <p className="text-sm text-muted-foreground transition-opacity transition-transform duration-300 group-hover:opacity-90 group-hover:-translate-y-0.5">Start from equations and derivations</p>
                </div>
              </div>
              <Button variant="pill" size="lg" className="w-full" onClick={() => window.open('/documentation', '_self')}>
                Read Documentation
              </Button>
            </div>
          </motion.div>

          {/* Interactive Learning */}
          <motion.div
            variants={card}
            className="group rounded-2xl card-surface-glass card-hover-raise h-full"
          >
            <div className="p-6 md:p-8 h-full flex flex-col justify-between">
              <div className="flex items-center gap-4 mb-4">
                <IconChip>
                  <Sparkles className="w-6 h-6 text-[hsl(var(--semantic-domain-quantum))]" />
                </IconChip>
                <div>
                  <h3 className="text-xl font-bold text-foreground transition-colors transition-transform duration-300 group-hover:gradient-text-quantum group-hover:-translate-y-0.5">
                    Interactive Learning
                  </h3>
                  <p className="text-sm text-muted-foreground transition-opacity transition-transform duration-300 group-hover:opacity-90 group-hover:-translate-y-0.5">Hands-on visual exploration</p>
                </div>
              </div>
              <Button variant="pill" size="lg" className="w-full" onClick={handleScrollToModules}>
                Explore Modules
              </Button>
            </div>
          </motion.div>

          {/* ML‑Augmented Discovery */}
          <motion.div
            variants={card}
            className="group rounded-2xl card-surface-glass card-hover-raise h-full"
          >
            <div className="p-6 md:p-8 h-full flex flex-col justify-between">
              <div className="flex items-center gap-4 mb-4">
                <IconChip>
                  <Cpu className="w-6 h-6 text-[hsl(var(--semantic-domain-quantum))]" />
                </IconChip>
                <div>
                  <h3 className="text-xl font-bold text-foreground transition-colors transition-transform duration-300 group-hover:gradient-text-quantum group-hover:-translate-y-0.5">
                    ML‑Augmented Discovery
                  </h3>
                  <p className="text-sm text-muted-foreground transition-opacity transition-transform duration-300 group-hover:opacity-90 group-hover:-translate-y-0.5">PINNs, neural solvers, analytics</p>
                </div>
              </div>
              <Button variant="pill" size="lg" className="w-full" onClick={() => window.open('/ml-showcase', '_self')}>
                Explore ML Tools
              </Button>
            </div>
          </motion.div>
        </motion.div>

        {/* Section divider below */}
        </div>
        <div className="flex items-center justify-center mt-16">
          <div
            className="h-px w-full max-w-3xl bg-gradient-to-r from-transparent via-primary/30 to-transparent"
            role="separator"
            aria-hidden="true"
          />
        </div>
    </section>
  );
};

export default StartingPointSection;
