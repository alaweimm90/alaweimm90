import React, { useState } from 'react';
import { Card, CardHeader, CardContent, CardFooter } from '@/components/molecules/Card';
import { Button } from '@/components/atoms/Button';
import { Navigation } from '@/components/organisms/Navigation';
import { cn } from '@/lib/utils';
import { Copy, Check, Palette, Type, Layout, Zap, Eye, Code } from 'lucide-react';

/**
 * Design System Documentation Page
 * Interactive documentation for REPZ design system
 * Demonstrates all components and tokens in action
 */

interface TokenDisplayProps {
  category: string;
  tokens: Array<{
    name: string;
    value: string;
    description?: string;
    preview?: React.ReactNode;
  }>;
}

interface ComponentShowcaseProps {
  name: string;
  description: string;
  component: React.ReactNode;
  code: string;
}

const TokenDisplay: React.FC<TokenDisplayProps> = ({ category, tokens }) => {
  const [copiedToken, setCopiedToken] = useState<string | null>(null);

  const copyToClipboard = async (text: string) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopiedToken(text);
      setTimeout(() => setCopiedToken(null), 2000);
    } catch (error) {
      console.error('Failed to copy:', error);
    }
  };

  return (
    <Card variant="default" padding="lg">
      <CardHeader>
        <h3 className="text-xl font-semibold mb-2">{category}</h3>
        <p className="text-sm opacity-80">Click any token to copy</p>
      </CardHeader>
      <CardContent>
        <div className="grid gap-3">
          {tokens.map((token) => (
            <div
              key={token.name}
              className="flex items-center justify-between p-3 rounded-lg bg-[var(--color-surface-overlay)] hover:bg-[var(--color-surface-elevated)] transition-colors cursor-pointer"
              onClick={() => copyToClipboard(token.value)}
            >
              <div className="flex-1">
                <div className="font-mono text-sm text-[var(--color-brand-primary)]">
                  {token.name}
                </div>
                <div className="font-mono text-xs opacity-70">
                  {token.value}
                </div>
                {token.description && (
                  <div className="text-xs opacity-60 mt-1">
                    {token.description}
                  </div>
                )}
              </div>
              {token.preview && (
                <div className="mx-3">
                  {token.preview}
                </div>
              )}
              <div className="text-xs">
                {copiedToken === token.value ? (
                  <Check size={16} className="text-green-500" />
                ) : (
                  <Copy size={16} className="opacity-50" />
                )}
              </div>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
};

const ComponentShowcase: React.FC<ComponentShowcaseProps> = ({ 
  name, 
  description, 
  component, 
  code 
}) => {
  const [showCode, setShowCode] = useState(false);

  return (
    <Card variant="elevated" padding="lg">
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-xl font-semibold">{name}</h3>
            <p className="text-sm opacity-80 mt-1">{description}</p>
          </div>
          <Button
            variant="tertiary"
            size="sm"
            onClick={() => setShowCode(!showCode)}
          >
            <Code size={16} />
            {showCode ? 'Hide Code' : 'Show Code'}
          </Button>
        </div>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          <div className="p-6 bg-[var(--color-surface-base)] rounded-lg border border-[var(--color-surface-overlay)]">
            {component}
          </div>
          {showCode && (
            <div className="bg-[var(--color-surface-elevated)] rounded-lg p-4 overflow-x-auto">
              <pre className="text-sm font-mono">
                <code>{code}</code>
              </pre>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
};

export default function DesignSystemDocs() {
  const [activeSection, setActiveSection] = useState('overview');

  const colorTokens = [
    {
      name: '--color-brand-primary',
      value: '#F15B23',
      description: 'Primary brand color - REPZ Orange',
      preview: <div className="w-6 h-6 rounded bg-[var(--color-brand-primary)]" />
    },
    {
      name: '--color-brand-black',
      value: '#000000',
      description: 'Official REPZ Black',
      preview: <div className="w-6 h-6 rounded bg-[var(--color-brand-black)] border border-white/20" />
    },
    {
      name: '--color-surface-base',
      value: '#141414',
      description: 'Base surface color',
      preview: <div className="w-6 h-6 rounded bg-[var(--color-surface-base)] border border-white/20" />
    },
    {
      name: '--color-surface-elevated',
      value: '#1F1F1F',
      description: 'Elevated surface color',
      preview: <div className="w-6 h-6 rounded bg-[var(--color-surface-elevated)] border border-white/20" />
    },
  ];

  const typographyTokens = [
    {
      name: '--font-size-xs',
      value: 'clamp(0.75rem, 0.71rem + 0.2vw, 0.875rem)',
      description: 'Extra small text (12-14px)',
      preview: <span style={{ fontSize: 'var(--font-size-xs)' }}>Aa</span>
    },
    {
      name: '--font-size-base',
      value: 'clamp(1rem, 0.95rem + 0.25vw, 1.125rem)',
      description: 'Base text size (16-18px)',
      preview: <span style={{ fontSize: 'var(--font-size-base)' }}>Aa</span>
    },
    {
      name: '--font-size-xl',
      value: 'clamp(1.25rem, 1.15rem + 0.5vw, 1.625rem)',
      description: 'Extra large text (20-26px)',
      preview: <span style={{ fontSize: 'var(--font-size-xl)' }}>Aa</span>
    },
    {
      name: '--font-weight-bold',
      value: '700',
      description: 'Bold font weight',
      preview: <span style={{ fontWeight: 'var(--font-weight-bold)' }}>Bold</span>
    },
  ];

  const spacingTokens = [
    {
      name: '--spacing-xs',
      value: '0.25rem',
      description: '4px spacing',
      preview: <div className="w-1 h-6 bg-[var(--color-brand-primary)]" />
    },
    {
      name: '--spacing-sm',
      value: '0.5rem',
      description: '8px spacing',
      preview: <div className="w-2 h-6 bg-[var(--color-brand-primary)]" />
    },
    {
      name: '--spacing-md',
      value: '1rem',
      description: '16px spacing',
      preview: <div className="w-4 h-6 bg-[var(--color-brand-primary)]" />
    },
    {
      name: '--spacing-lg',
      value: '1.5rem',
      description: '24px spacing',
      preview: <div className="w-6 h-6 bg-[var(--color-brand-primary)]" />
    },
  ];

  const sections = [
    { id: 'overview', label: 'Overview', icon: <Eye size={16} /> },
    { id: 'colors', label: 'Colors', icon: <Palette size={16} /> },
    { id: 'typography', label: 'Typography', icon: <Type size={16} /> },
    { id: 'spacing', label: 'Spacing', icon: <Layout size={16} /> },
    { id: 'components', label: 'Components', icon: <Zap size={16} /> },
  ];

  return (
    <div className="min-h-screen bg-[var(--color-surface-base)]">
      <Navigation variant="header" />
      
      <div className="max-w-7xl mx-auto px-6 py-8">
        {/* Header */}
        <div className="text-center mb-12">
          <h1 className="text-5xl font-bold mb-4 bg-gradient-to-r from-[var(--color-brand-primary)] to-[var(--color-brand-primary-light)] bg-clip-text text-transparent">
            REPZ Design System
          </h1>
          <p className="text-xl opacity-80 max-w-2xl mx-auto">
            Enterprise-grade design system built with W3C compliant design tokens, 
            accessibility-first components, and automated quality assurance.
          </p>
        </div>

        {/* Navigation */}
        <div className="flex justify-center mb-8">
          <div className="flex gap-2 p-2 bg-[var(--color-surface-elevated)] rounded-lg">
            {sections.map((section) => (
              <button
                key={section.id}
                onClick={() => setActiveSection(section.id)}
                className={cn(
                  'flex items-center gap-2 px-4 py-2 rounded-md text-sm font-medium transition-all',
                  activeSection === section.id
                    ? 'bg-[var(--color-brand-primary)] text-white'
                    : 'text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)] hover:bg-[var(--color-surface-overlay)]'
                )}
              >
                {section.icon}
                {section.label}
              </button>
            ))}
          </div>
        </div>

        {/* Content */}
        <div className="space-y-8">
          {activeSection === 'overview' && (
            <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
              <Card variant="glass" padding="lg">
                <CardContent>
                  <div className="text-center">
                    <div className="w-12 h-12 bg-[var(--color-brand-primary)] rounded-lg mx-auto mb-4 flex items-center justify-center">
                      <Palette className="text-white" size={24} />
                    </div>
                    <h3 className="text-lg font-semibold mb-2">241 Design Tokens</h3>
                    <p className="text-sm opacity-80">W3C compliant tokens covering colors, typography, spacing, and more</p>
                  </div>
                </CardContent>
              </Card>

              <Card variant="glass" padding="lg">
                <CardContent>
                  <div className="text-center">
                    <div className="w-12 h-12 bg-[var(--color-tier-adaptive)] rounded-lg mx-auto mb-4 flex items-center justify-center">
                      <Zap className="text-white" size={24} />
                    </div>
                    <h3 className="text-lg font-semibold mb-2">Atomic Components</h3>
                    <p className="text-sm opacity-80">Modular components following atomic design principles</p>
                  </div>
                </CardContent>
              </Card>

              <Card variant="glass" padding="lg">
                <CardContent>
                  <div className="text-center">
                    <div className="w-12 h-12 bg-[var(--color-tier-performance)] rounded-lg mx-auto mb-4 flex items-center justify-center">
                      <Eye className="text-white" size={24} />
                    </div>
                    <h3 className="text-lg font-semibold mb-2">Visual Testing</h3>
                    <p className="text-sm opacity-80">Automated visual regression testing with Storybook + Chromatic</p>
                  </div>
                </CardContent>
              </Card>
            </div>
          )}

          {activeSection === 'colors' && (
            <div className="space-y-6">
              <TokenDisplay category="Brand Colors" tokens={colorTokens} />
              <div className="grid md:grid-cols-2 gap-6">
                <Card variant="default" padding="lg">
                  <CardHeader>
                    <h3 className="text-lg font-semibold">Tier Colors</h3>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-3">
                      <div className="flex items-center gap-3">
                        <div className="w-8 h-8 rounded bg-[var(--color-tier-core)]" />
                        <div>
                          <div className="font-medium">Core</div>
                          <div className="text-xs opacity-70">Foundation tier</div>
                        </div>
                      </div>
                      <div className="flex items-center gap-3">
                        <div className="w-8 h-8 rounded bg-[var(--color-tier-adaptive)]" />
                        <div>
                          <div className="font-medium">Adaptive</div>
                          <div className="text-xs opacity-70">Interactive tier</div>
                        </div>
                      </div>
                      <div className="flex items-center gap-3">
                        <div className="w-8 h-8 rounded bg-[var(--color-tier-performance)]" />
                        <div>
                          <div className="font-medium">Performance</div>
                          <div className="text-xs opacity-70">Elite tier</div>
                        </div>
                      </div>
                      <div className="flex items-center gap-3">
                        <div className="w-8 h-8 rounded bg-[var(--color-tier-longevity)]" />
                        <div>
                          <div className="font-medium">Longevity</div>
                          <div className="text-xs opacity-70">Premium tier</div>
                        </div>
                      </div>
                    </div>
                  </CardContent>
                </Card>

                <Card variant="default" padding="lg">
                  <CardHeader>
                    <h3 className="text-lg font-semibold">Usage Guidelines</h3>
                  </CardHeader>
                  <CardContent>
                    <ul className="space-y-2 text-sm">
                      <li>• Use brand primary for call-to-action buttons</li>
                      <li>• Surface colors create visual hierarchy</li>
                      <li>• Tier colors represent subscription levels</li>
                      <li>• All colors meet WCAG 2.2 AA contrast ratios</li>
                    </ul>
                  </CardContent>
                </Card>
              </div>
            </div>
          )}

          {activeSection === 'typography' && (
            <div className="space-y-6">
              <TokenDisplay category="Typography Scale" tokens={typographyTokens} />
              <Card variant="default" padding="lg">
                <CardHeader>
                  <h3 className="text-lg font-semibold">Font Families</h3>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    <div>
                      <div className="font-[var(--font-heading)] text-2xl mb-2">
                        Montserrat - Headings
                      </div>
                      <code className="text-xs">font-family: var(--font-heading)</code>
                    </div>
                    <div>
                      <div className="font-[var(--font-body)] text-lg mb-2">
                        Inter - Body Text
                      </div>
                      <code className="text-xs">font-family: var(--font-body)</code>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>
          )}

          {activeSection === 'spacing' && (
            <div className="space-y-6">
              <TokenDisplay category="Spacing System" tokens={spacingTokens} />
              <Card variant="default" padding="lg">
                <CardHeader>
                  <h3 className="text-lg font-semibold">8px Base Grid</h3>
                </CardHeader>
                <CardContent>
                  <p className="text-sm opacity-80 mb-4">
                    All spacing values are based on an 8px grid system for consistent visual rhythm.
                  </p>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    {['xs', 'sm', 'md', 'lg', 'xl', '2xl', '3xl', '4xl'].map((size) => (
                      <div key={size} className="text-center">
                        <div 
                          className="bg-[var(--color-brand-primary)] mx-auto mb-2"
                          style={{ 
                            width: `var(--spacing-${size})`, 
                            height: `var(--spacing-${size})` 
                          }}
                        />
                        <div className="text-xs font-mono">{size}</div>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            </div>
          )}

          {activeSection === 'components' && (
            <div className="space-y-8">
              <ComponentShowcase
                name="Button Component"
                description="Primary interactive element with multiple variants and states"
                component={
                  <div className="flex flex-wrap gap-4">
                    <Button variant="primary">Primary</Button>
                    <Button variant="secondary">Secondary</Button>
                    <Button variant="tertiary">Tertiary</Button>
                    <Button variant="destructive">Destructive</Button>
                    <Button variant="primary" loading>Loading</Button>
                    <Button variant="primary" disabled>Disabled</Button>
                  </div>
                }
                code={`<Button variant="primary">Primary</Button>
<Button variant="secondary">Secondary</Button>
<Button variant="tertiary">Tertiary</Button>
<Button variant="destructive">Destructive</Button>
<Button variant="primary" loading>Loading</Button>
<Button variant="primary" disabled>Disabled</Button>`}
              />

              <ComponentShowcase
                name="Card Component"
                description="Flexible container component with structured content areas"
                component={
                  <div className="grid md:grid-cols-2 gap-4">
                    <Card variant="default">
                      <CardHeader>
                        <h4 className="font-semibold">Default Card</h4>
                      </CardHeader>
                      <CardContent>
                        <p className="text-sm">Basic card with subtle elevation.</p>
                      </CardContent>
                    </Card>
                    <Card variant="elevated">
                      <CardHeader>
                        <h4 className="font-semibold">Elevated Card</h4>
                      </CardHeader>
                      <CardContent>
                        <p className="text-sm">Enhanced shadow for prominence.</p>
                      </CardContent>
                    </Card>
                  </div>
                }
                code={`<Card variant="default">
  <CardHeader>
    <h4>Card Title</h4>
  </CardHeader>
  <CardContent>
    <p>Card content goes here.</p>
  </CardContent>
</Card>`}
              />
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="mt-16 text-center">
          <Card variant="glass" padding="xl">
            <CardContent>
              <h3 className="text-2xl font-bold mb-4">Ready to Get Started?</h3>
              <p className="text-lg opacity-80 mb-6 max-w-2xl mx-auto">
                Explore our Storybook for interactive component examples, 
                or check out the GitHub repository for implementation details.
              </p>
              <div className="flex flex-wrap justify-center gap-4">
                <Button variant="primary" size="lg">
                  View Storybook
                </Button>
                <Button variant="secondary" size="lg">
                  GitHub Repository
                </Button>
                <Button variant="tertiary" size="lg">
                  Design Tokens
                </Button>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}