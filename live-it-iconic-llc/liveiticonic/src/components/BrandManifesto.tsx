import React from 'react';

/**
 * BrandManifesto component displays the brand's core philosophy and values
 *
 * Full-width section with centered title "Confidence, in uniform" and brand mission statement.
 * Features decorative gold divider and responsive typography for different screen sizes.
 *
 * @component
 *
 * @example
 * <BrandManifesto />
 *
 * @remarks
 * - Dark background (lii-ink)
 * - Centered text layout with max-width constraint
 * - Responsive title sizing for mobile/desktop
 * - Gold decorative divider
 * - Font families: display for heading, ui for body text
 */
const BrandManifesto = () => {
  return (
    <section className="relative bg-lii-ink py-20">
      <div className="container mx-auto px-6">
        <div className="max-w-4xl mx-auto text-center">
          <h2 className="text-4xl sm:text-5xl font-display tracking-tight text-lii-cloud leading-tight">
            Confidence, in uniform
          </h2>
          <div className="w-20 h-px bg-lii-gold/50 mx-auto my-8"></div>
          <p className="text-lg sm:text-xl font-ui text-lii-ash leading-relaxed">
            We cut apparel with the discipline of performance machines—clean lines, durable fabrics,
            precise fits. Designed for motion; refined for nights out.
          </p>
        </div>
      </div>
    </section>
  );
};

export default BrandManifesto;
