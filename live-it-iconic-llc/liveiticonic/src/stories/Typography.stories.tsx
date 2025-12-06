import type { Meta } from '@storybook/react-vite';

const meta: Meta = {
  title: 'Design System/Typography',
};

export default meta;

export const Headings = () => (
  <div className="space-y-8">
    <div>
      <h1 className="text-5xl md:text-6xl font-display text-lii-gold mb-2">Display Large (56px)</h1>
      <p className="text-sm text-lii-ash">Playfair Display - Primary headings and hero text</p>
    </div>
    <div>
      <h2 className="text-4xl md:text-5xl font-display text-lii-gold mb-2">Display Medium (48px)</h2>
      <p className="text-sm text-lii-ash">Playfair Display - Section headings</p>
    </div>
    <div>
      <h3 className="text-3xl md:text-4xl font-display text-lii-gold mb-2">Display Small (40px)</h3>
      <p className="text-sm text-lii-ash">Playfair Display - Subsection headings</p>
    </div>
  </div>
);

export const BodyText = () => (
  <div className="space-y-8 max-w-2xl">
    <div>
      <p className="text-lg font-inter text-lii-cloud mb-2">
        Body Large (18px) - Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.
      </p>
      <p className="text-sm text-lii-ash">Inter Variable - Large body text</p>
    </div>
    <div>
      <p className="text-base font-inter text-lii-cloud mb-2">
        Body Medium (16px) - Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.
      </p>
      <p className="text-sm text-lii-ash">Inter Variable - Default body text</p>
    </div>
    <div>
      <p className="text-sm font-inter text-lii-cloud mb-2">
        Body Small (14px) - Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.
      </p>
      <p className="text-sm text-lii-ash">Inter Variable - Small body text</p>
    </div>
    <div>
      <p className="text-xs font-inter text-lii-ash mb-2">
        Caption (12px) - Lorem ipsum dolor sit amet, consectetur adipiscing elit.
      </p>
      <p className="text-sm text-lii-ash">Inter Variable - Captions and small labels</p>
    </div>
  </div>
);

export const FontPairs = () => (
  <div className="space-y-12">
    <div className="border-l-4 border-lii-gold pl-6">
      <h1 className="font-display text-4xl text-lii-gold mb-4">Premium Lifestyle Collection</h1>
      <p className="font-inter text-lii-cloud leading-relaxed">
        Discover our curated selection of luxury apparel designed for the modern luxury automotive enthusiast. 
        Each piece combines premium craftsmanship with contemporary design.
      </p>
    </div>
    <div className="border-l-4 border-lii-ash pl-6">
      <h2 className="font-display text-3xl text-lii-cloud mb-4">About Our Brand</h2>
      <p className="font-inter text-lii-cloud/80 leading-relaxed">
        Live It Iconic represents the pinnacle of luxury lifestyle products. We believe in creating timeless pieces 
        that resonate with discerning collectors and automotive enthusiasts worldwide.
      </p>
    </div>
  </div>
);

export const FontWeights = () => (
  <div className="space-y-6">
    <div>
      <p className="font-display font-light text-2xl text-lii-cloud mb-1">Light (300) - Playfair Display</p>
      <p className="text-sm text-lii-ash">Elegant and sophisticated headings</p>
    </div>
    <div>
      <p className="font-display font-normal text-2xl text-lii-cloud mb-1">Regular (400) - Playfair Display</p>
      <p className="text-sm text-lii-ash">Standard display font weight</p>
    </div>
    <div>
      <p className="font-display font-semibold text-2xl text-lii-gold mb-1">Semibold (600) - Playfair Display</p>
      <p className="text-sm text-lii-ash">Emphasized headings and highlights</p>
    </div>
    <div>
      <p className="font-inter font-light text-lg text-lii-cloud mb-1">Light (300) - Inter Variable</p>
      <p className="text-sm text-lii-ash">Subtle body text</p>
    </div>
    <div>
      <p className="font-inter font-normal text-lg text-lii-cloud mb-1">Regular (400) - Inter Variable</p>
      <p className="text-sm text-lii-ash">Default body text</p>
    </div>
    <div>
      <p className="font-inter font-medium text-lg text-lii-cloud mb-1">Medium (500) - Inter Variable</p>
      <p className="text-sm text-lii-ash">Emphasized body text</p>
    </div>
    <div>
      <p className="font-inter font-semibold text-lg text-lii-gold mb-1">Semibold (600) - Inter Variable</p>
      <p className="text-sm text-lii-ash">Labels and strong emphasis</p>
    </div>
  </div>
);
