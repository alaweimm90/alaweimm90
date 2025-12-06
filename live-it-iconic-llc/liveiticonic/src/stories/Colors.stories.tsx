import type { Meta } from '@storybook/react-vite';

const meta: Meta = {
  title: 'Design System/Colors',
};

export default meta;

interface ColorSwatchProps {
  name: string;
  hex: string;
  textColor: 'light' | 'dark';
  description?: string;
}

function ColorSwatch({ name, hex, textColor, description }: ColorSwatchProps) {
  return (
    <div className="space-y-2">
      <div
        className={"h-32 rounded-lg flex items-center justify-center transition-transform hover:scale-105 cursor-pointer " +
          (textColor === 'light' ? 'text-white' : 'text-black')}
        style={{ backgroundColor: hex }}
        title={hex}
      >
        <span className="font-semibold text-center px-2">{name}</span>
      </div>
      <p className="text-sm font-mono text-lii-cloud/60">{hex}</p>
      {description && <p className="text-xs text-lii-ash">{description}</p>}
    </div>
  );
}

export const Primary = () => (
  <div className="space-y-8">
    <div>
      <h2 className="text-2xl font-display text-lii-gold mb-6">Primary Brand Colors</h2>
      <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-6">
        <ColorSwatch 
          name="Gold" 
          hex="#C1A060" 
          textColor="dark"
          description="Primary accent, CTAs, highlights"
        />
        <ColorSwatch
          name="Charcoal"
          hex="#0B0B0C"
          textColor="light"
          description="Dark backgrounds, text"
        />
        <ColorSwatch
          name="Cloud"
          hex="#E6E9EF"
          textColor="dark"
          description="Light backgrounds, light text"
        />
        <ColorSwatch
          name="Ink"
          hex="#2D3142"
          textColor="light"
          description="Body text, main content"
        />
        <ColorSwatch
          name="Ash"
          hex="#8C93A3"
          textColor="light"
          description="Secondary text, muted"
        />
      </div>
    </div>
  </div>
);

export const Semantic = () => (
  <div className="space-y-8">
    <div>
      <h2 className="text-2xl font-display text-lii-gold mb-6">Semantic Colors</h2>
      <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-6">
        <ColorSwatch
          name="Success"
          hex="#10B981"
          textColor="light"
          description="Positive actions, confirmations"
        />
        <ColorSwatch
          name="Warning"
          hex="#F59E0B"
          textColor="dark"
          description="Caution, alerts"
        />
        <ColorSwatch
          name="Error"
          hex="#EF4444"
          textColor="light"
          description="Destructive actions, errors"
        />
        <ColorSwatch
          name="Info"
          hex="#3B82F6"
          textColor="light"
          description="Information, hints"
        />
      </div>
    </div>
  </div>
);
