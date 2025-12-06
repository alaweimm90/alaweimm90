import React, { useMemo } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { 
  Eye, 
  Keyboard, 
  Mouse, 
  Volume2, 
  Palette, 
  Settings, 
  CheckCircle, 
  Users,
  Zap,
  Shield
} from 'lucide-react';
import { PhysicsModuleLayout } from '@/components/PhysicsModuleLayout';
import { PhysicsModuleHeader } from '@/components/PhysicsModuleHeader';
import AccessibilityEnhancementSystem from '@/components/AccessibilityEnhancementSystem';
import { useSEO } from '@/hooks/use-seo';

export default function AccessibilityFeatures() {
  useSEO({ title: 'Accessibility Features – SimCore', description: 'WCAG 2.1 AA compliant accessible scientific computing tools.' });
  const schemaLD = useMemo(() => ({
    '@context': 'https://schema.org',
    '@type': 'WebPage',
    name: 'Accessibility Features',
    description: 'WCAG 2.1 AA compliant accessible scientific computing tools.',
    inLanguage: 'en',
    isPartOf: { '@type': 'WebSite', name: 'SimCore' },
    mainEntityOfPage: typeof window !== 'undefined' ? window.location.href : ''
  }), []);
  const accessibilityFeatures = [
    {
      icon: Eye,
      title: 'Visual Accessibility',
      description: 'High contrast modes, customizable text sizes, and optimized color schemes',
      features: [
        'Dark/Light theme support',
        'High contrast mode',
        'Scalable typography',
        'Color-blind friendly palettes',
        'Focus indicators'
      ]
    },
    {
      icon: Keyboard,
      title: 'Keyboard Navigation',
      description: 'Full keyboard accessibility for all interactive elements',
      features: [
        'Tab order optimization',
        'Keyboard shortcuts',
        'Skip to content links',
        'Focus management',
        'Escape key handling'
      ]
    },
    {
      icon: Volume2,
      title: 'Screen Reader Support',
      description: 'ARIA labels, semantic HTML, and screen reader optimizations',
      features: [
        'ARIA landmarks',
        'Descriptive labels',
        'Live region updates',
        'Alt text for images',
        'Table headers'
      ]
    },
    {
      icon: Mouse,
      title: 'Motor Accessibility',
      description: 'Touch-friendly interfaces and alternative input methods',
      features: [
        'Large touch targets',
        'Click alternatives',
        'Gesture support',
        'Drag and drop',
        'Voice control ready'
      ]
    }
  ];

  const complianceStandards = [
    { name: 'WCAG 2.1 AA', status: 'Compliant', color: 'bg-green-500/10 text-green-500' },
    { name: 'Section 508', status: 'Compliant', color: 'bg-green-500/10 text-green-500' },
    { name: 'ADA', status: 'Compliant', color: 'bg-green-500/10 text-green-500' },
    { name: 'EN 301 549', status: 'Compliant', color: 'bg-green-500/10 text-green-500' }
  ];

  return (
    <PhysicsModuleLayout>
      <script type="application/ld+json" dangerouslySetInnerHTML={{ __html: JSON.stringify(schemaLD) }} />
      <PhysicsModuleHeader
        title="Accessibility Features"
        description="Comprehensive accessibility tools and inclusive design principles"
        category="Accessibility"
        difficulty="Beginner"
      />

      <div className="container mx-auto px-4 py-8">
        <div className="max-w-6xl mx-auto">
          {/* Overview */}
          <Card className="mb-8 bg-gradient-to-br from-primary/5 to-secondary/5 border-primary/20">
            <CardHeader>
              <CardTitle className="text-2xl flex items-center gap-3">
                <Users className="w-6 h-6 text-primary" />
                Universal Access to Scientific Computing
              </CardTitle>
              <CardDescription className="text-lg">
                SimCore is designed to be accessible to everyone, regardless of abilities or disabilities.
                Our platform adheres to international accessibility standards and best practices.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                <div className="flex items-center gap-2">
                  <Shield className="w-5 h-5 text-primary" />
                  <span className="font-medium">WCAG 2.1 AA</span>
                </div>
                <div className="flex items-center gap-2">
                  <Zap className="w-5 h-5 text-primary" />
                  <span className="font-medium">Real-time Testing</span>
                </div>
                <div className="flex items-center gap-2">
                  <Settings className="w-5 h-5 text-primary" />
                  <span className="font-medium">Customizable</span>
                </div>
                <div className="flex items-center gap-2">
                  <CheckCircle className="w-5 h-5 text-primary" />
                  <span className="font-medium">Continuously Improved</span>
                </div>
              </div>
            </CardContent>
          </Card>

          <Tabs defaultValue="features" className="w-full">
            <TabsList className="grid w-full grid-cols-3">
              <TabsTrigger value="features">Accessibility Features</TabsTrigger>
              <TabsTrigger value="compliance">Standards Compliance</TabsTrigger>
              <TabsTrigger value="testing">Testing Tools</TabsTrigger>
            </TabsList>

            {/* Features Tab */}
            <TabsContent value="features" className="space-y-6 mt-6">
              <div className="grid gap-6">
                {accessibilityFeatures.map((category) => (
                  <Card key={category.title}>
                    <CardHeader>
                      <CardTitle className="flex items-center gap-3">
                        <category.icon className="w-5 h-5 text-primary" />
                        {category.title}
                      </CardTitle>
                      <CardDescription>{category.description}</CardDescription>
                    </CardHeader>
                    <CardContent>
                      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
                        {category.features.map((feature) => (
                          <div key={feature} className="flex items-center gap-2">
                            <CheckCircle className="w-4 h-4 text-green-500 flex-shrink-0" />
                            <span className="text-sm">{feature}</span>
                          </div>
                        ))}
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            </TabsContent>

            {/* Compliance Tab */}
            <TabsContent value="compliance" className="space-y-6 mt-6">
              <Card>
                <CardHeader>
                  <CardTitle>Standards Compliance</CardTitle>
                  <CardDescription>
                    SimCore meets or exceeds international accessibility standards
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {complianceStandards.map((standard) => (
                      <div key={standard.name} className="flex items-center justify-between p-4 border rounded-lg">
                        <span className="font-medium">{standard.name}</span>
                        <Badge className={standard.color}>{standard.status}</Badge>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Implementation Details</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="space-y-3">
                    <h4 className="font-semibold">WCAG 2.1 AA Compliance</h4>
                    <ul className="text-sm text-muted-foreground space-y-1 ml-4">
                      <li>• Perceivable: Content is presentable in multiple ways</li>
                      <li>• Operable: Interface components are usable by everyone</li>
                      <li>• Understandable: Information and UI operation are clear</li>
                      <li>• Robust: Content works with various assistive technologies</li>
                    </ul>
                  </div>
                  
                  <div className="space-y-3">
                    <h4 className="font-semibold">Section 508 Compliance</h4>
                    <ul className="text-sm text-muted-foreground space-y-1 ml-4">
                      <li>• Federal accessibility requirements for government agencies</li>
                      <li>• Keyboard accessibility for all functionality</li>
                      <li>• Alternative formats for multimedia content</li>
                      <li>• Compatible with assistive technologies</li>
                    </ul>
                  </div>
                </CardContent>
              </Card>
            </TabsContent>

            {/* Testing Tab */}
            <TabsContent value="testing" className="space-y-6 mt-6">
              <Card>
                <CardHeader>
                  <CardTitle>Interactive Accessibility Testing</CardTitle>
                  <CardDescription>
                    Use our built-in testing tools to evaluate accessibility features
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <AccessibilityEnhancementSystem />
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>
        </div>
      </div>
    </PhysicsModuleLayout>
  );
}