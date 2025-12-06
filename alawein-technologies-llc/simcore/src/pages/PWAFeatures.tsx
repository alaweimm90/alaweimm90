import React, { useMemo } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Button } from '@/components/ui/button';
import { 
  Smartphone, 
  Download, 
  Wifi, 
  WifiOff, 
  Zap, 
  Shield, 
  CheckCircle, 
  Monitor,
  Battery,
  Bell,
  RefreshCw,
  Globe
} from 'lucide-react';
import { PhysicsModuleLayout } from '@/components/PhysicsModuleLayout';
import { PhysicsModuleHeader } from '@/components/PhysicsModuleHeader';
import PWAEnhancementSystem from '@/components/PWAEnhancementSystem';
import { useSEO } from '@/hooks/use-seo';

export default function PWAFeatures() {
  useSEO({ title: 'PWA Features – SimCore', description: 'Installable, offline-capable scientific computing with native-like performance.' });
  const schemaLD = useMemo(() => ({
    '@context': 'https://schema.org',
    '@type': 'TechArticle',
    headline: 'Progressive Web App Features',
    description: 'Installable, offline-capable scientific computing with native-like performance.',
    inLanguage: 'en',
    author: { '@type': 'Organization', name: 'SimCore' },
    publisher: { '@type': 'Organization', name: 'SimCore' },
    keywords: 'PWA, offline, installable, performance',
    mainEntityOfPage: typeof window !== 'undefined' ? window.location.href : ''
  }), []);
  const pwaFeatures = [
    {
      icon: Download,
      title: 'Installable',
      description: 'Add SimCore to your device home screen for native app-like experience',
      benefits: [
        'No app store required',
        'Direct home screen access',
        'Native app appearance',
        'Automatic updates',
        'Cross-platform compatibility'
      ]
    },
    {
      icon: WifiOff,
      title: 'Offline Capability',
      description: 'Continue working with physics simulations even without internet connection',
      benefits: [
        'Service worker caching',
        'Offline simulation access',
        'Local data persistence',
        'Background synchronization',
        'Seamless online/offline transition'
      ]
    },
    {
      icon: Zap,
      title: 'High Performance',
      description: 'Optimized for speed with advanced caching and resource management',
      benefits: [
        'Instant loading',
        'Efficient caching',
        'WebGPU acceleration',
        'Resource optimization',
        'Smooth animations'
      ]
    },
    {
      icon: Shield,
      title: 'Secure & Reliable',
      description: 'HTTPS-only with service worker security and data protection',
      benefits: [
        'HTTPS enforcement',
        'Secure data transmission',
        'Safe background updates',
        'Privacy protection',
        'Reliable connections'
      ]
    }
  ];

  const deviceFeatures = [
    { name: 'Web App Manifest', status: 'Active', description: 'Enables installation and theming' },
    { name: 'Service Worker', status: 'Active', description: 'Handles offline functionality and caching' },
    { name: 'Push Notifications', status: 'Available', description: 'For updates and reminders' },
    { name: 'Background Sync', status: 'Available', description: 'Sync data when connection resumes' },
    { name: 'WebGPU Support', status: 'Enhanced', description: 'Hardware acceleration when available' },
    { name: 'File System Access', status: 'Available', description: 'Save/load simulation data locally' }
  ];

  const installInstructions = {
    mobile: [
      'Open SimCore in your mobile browser',
      'Tap the "Add to Home Screen" banner',
      'Confirm installation',
      'Launch from your home screen'
    ],
    desktop: [
      'Visit SimCore in Chrome, Edge, or Firefox',
      'Click the install icon in the address bar',
      'Click "Install" in the dialog',
      'Access from your desktop or start menu'
    ]
  };

  return (
    <PhysicsModuleLayout>
      <script type="application/ld+json" dangerouslySetInnerHTML={{ __html: JSON.stringify(schemaLD) }} />
      <PhysicsModuleHeader
        title="Progressive Web App Features"
        description="Native app experience with web technology - installable, offline-capable, and high-performance"
        category="PWA"
        difficulty="Beginner"
      />

      <div className="container mx-auto px-4 py-8">
        <div className="max-w-6xl mx-auto">
          {/* Overview */}
          <Card className="mb-8 bg-gradient-to-br from-primary/5 to-secondary/5 border-primary/20">
            <CardHeader>
              <CardTitle className="text-2xl flex items-center gap-3">
                <Smartphone className="w-6 h-6 text-primary" />
                Native App Experience on Any Device
              </CardTitle>
              <CardDescription className="text-lg">
                SimCore is a Progressive Web App (PWA) that combines the best of web and mobile apps.
                Install it on any device for offline access, push notifications, and native performance.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                <div className="flex items-center gap-2">
                  <Download className="w-5 h-5 text-primary" />
                  <span className="font-medium">Installable</span>
                </div>
                <div className="flex items-center gap-2">
                  <WifiOff className="w-5 h-5 text-primary" />
                  <span className="font-medium">Offline Ready</span>
                </div>
                <div className="flex items-center gap-2">
                  <Zap className="w-5 h-5 text-primary" />
                  <span className="font-medium">Fast Loading</span>
                </div>
                <div className="flex items-center gap-2">
                  <Shield className="w-5 h-5 text-primary" />
                  <span className="font-medium">Secure</span>
                </div>
              </div>
            </CardContent>
          </Card>

          <Tabs defaultValue="features" className="w-full">
            <TabsList className="grid w-full grid-cols-4">
              <TabsTrigger value="features">PWA Features</TabsTrigger>
              <TabsTrigger value="install">Installation</TabsTrigger>
              <TabsTrigger value="capabilities">Device Integration</TabsTrigger>
              <TabsTrigger value="testing">Testing Tools</TabsTrigger>
            </TabsList>

            {/* Features Tab */}
            <TabsContent value="features" className="space-y-6 mt-6">
              <div className="grid gap-6">
                {pwaFeatures.map((feature) => (
                  <Card key={feature.title}>
                    <CardHeader>
                      <CardTitle className="flex items-center gap-3">
                        <feature.icon className="w-5 h-5 text-primary" />
                        {feature.title}
                      </CardTitle>
                      <CardDescription>{feature.description}</CardDescription>
                    </CardHeader>
                    <CardContent>
                      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
                        {feature.benefits.map((benefit) => (
                          <div key={benefit} className="flex items-center gap-2">
                            <CheckCircle className="w-4 h-4 text-green-500 flex-shrink-0" />
                            <span className="text-sm">{benefit}</span>
                          </div>
                        ))}
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            </TabsContent>

            {/* Installation Tab */}
            <TabsContent value="install" className="space-y-6 mt-6">
              <div className="grid gap-6 md:grid-cols-2">
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Smartphone className="w-5 h-5" />
                      Mobile Installation
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-3">
                      {installInstructions.mobile.map((step, index) => (
                        <div key={index} className="flex gap-3">
                          <div className="w-6 h-6 bg-primary text-primary-foreground rounded-full flex items-center justify-center text-sm font-bold flex-shrink-0">
                            {index + 1}
                          </div>
                          <span className="text-sm">{step}</span>
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Monitor className="w-5 h-5" />
                      Desktop Installation
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-3">
                      {installInstructions.desktop.map((step, index) => (
                        <div key={index} className="flex gap-3">
                          <div className="w-6 h-6 bg-primary text-primary-foreground rounded-full flex items-center justify-center text-sm font-bold flex-shrink-0">
                            {index + 1}
                          </div>
                          <span className="text-sm">{step}</span>
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </Card>
              </div>

              <Card>
                <CardHeader>
                  <CardTitle>Installation Benefits</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <div className="flex flex-col items-center text-center p-4 bg-muted/30 rounded-lg">
                      <Battery className="w-8 h-8 text-primary mb-2" />
                      <h4 className="font-semibold mb-1">Better Performance</h4>
                      <p className="text-sm text-muted-foreground">Optimized resource usage and faster startup</p>
                    </div>
                    <div className="flex flex-col items-center text-center p-4 bg-muted/30 rounded-lg">
                      <Bell className="w-8 h-8 text-primary mb-2" />
                      <h4 className="font-semibold mb-1">Push Notifications</h4>
                      <p className="text-sm text-muted-foreground">Stay updated with new features and content</p>
                    </div>
                    <div className="flex flex-col items-center text-center p-4 bg-muted/30 rounded-lg">
                      <RefreshCw className="w-8 h-8 text-primary mb-2" />
                      <h4 className="font-semibold mb-1">Background Sync</h4>
                      <p className="text-sm text-muted-foreground">Automatic updates and data synchronization</p>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </TabsContent>

            {/* Device Integration Tab */}
            <TabsContent value="capabilities" className="space-y-6 mt-6">
              <Card>
                <CardHeader>
                  <CardTitle>Device Integration Features</CardTitle>
                  <CardDescription>
                    Native device capabilities available in SimCore
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-3">
                    {deviceFeatures.map((feature) => (
                      <div key={feature.name} className="flex items-center justify-between p-4 border rounded-lg">
                        <div>
                          <h4 className="font-semibold">{feature.name}</h4>
                          <p className="text-sm text-muted-foreground">{feature.description}</p>
                        </div>
                        <Badge 
                          className={
                            feature.status === 'Active' ? 'bg-green-500/10 text-green-500' :
                            feature.status === 'Enhanced' ? 'bg-blue-500/10 text-blue-500' :
                            'bg-yellow-500/10 text-yellow-500'
                          }
                        >
                          {feature.status}
                        </Badge>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Browser Compatibility</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                    {[
                      { name: 'Chrome', support: 'Full', version: '67+' },
                      { name: 'Firefox', support: 'Full', version: '79+' },
                      { name: 'Safari', support: 'Partial', version: '14+' },
                      { name: 'Edge', support: 'Full', version: '79+' }
                    ].map((browser) => (
                      <div key={browser.name} className="text-center p-4 border rounded-lg">
                        <Globe className="w-8 h-8 mx-auto mb-2 text-primary" />
                        <h4 className="font-semibold">{browser.name}</h4>
                        <p className="text-sm text-muted-foreground">{browser.version}</p>
                        <Badge 
                          className={browser.support === 'Full' ? 'bg-green-500/10 text-green-500' : 'bg-yellow-500/10 text-yellow-500'}
                        >
                          {browser.support}
                        </Badge>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            </TabsContent>

            {/* Testing Tab */}
            <TabsContent value="testing" className="space-y-6 mt-6">
              <Card>
                <CardHeader>
                  <CardTitle>PWA Testing & Diagnostics</CardTitle>
                  <CardDescription>
                    Test and evaluate PWA features and performance
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <PWAEnhancementSystem />
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>
        </div>
      </div>
    </PhysicsModuleLayout>
  );
}