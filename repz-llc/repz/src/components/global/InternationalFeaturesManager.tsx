import React, { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/ui/molecules/Card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/ui/molecules/Tabs';
import { Button } from '@/ui/atoms/Button';
import { Badge } from '@/ui/atoms/Badge';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/ui/molecules/Select';
import { Switch } from '@/ui/atoms/Switch';
import { Progress } from '@/components/ui/progress';
import { Globe, Languages, DollarSign, Clock, MapPin, Users, Shield, Zap } from 'lucide-react';

interface LocalizationStatus {
  language: string;
  code: string;
  progress: number;
  translators: number;
  status: 'complete' | 'in-progress' | 'pending';
  users: number;
}

interface RegionalFeature {
  region: string;
  feature: string;
  status: 'enabled' | 'disabled' | 'testing';
  compliance: boolean;
  usage: number;
}

export const InternationalFeaturesManager: React.FC = () => {
  const [selectedRegion, setSelectedRegion] = useState('global');
  const [autoTranslation, setAutoTranslation] = useState(true);
  const [complianceMode, setComplianceMode] = useState(true);

  const languages: LocalizationStatus[] = [
    { language: 'English (US)', code: 'en-US', progress: 100, translators: 5, status: 'complete', users: 28340 },
    { language: 'Spanish', code: 'es', progress: 98, translators: 3, status: 'complete', users: 12780 },
    { language: 'French', code: 'fr', progress: 95, translators: 2, status: 'complete', users: 8920 },
    { language: 'German', code: 'de', progress: 92, translators: 2, status: 'in-progress', users: 7540 },
    { language: 'Portuguese (BR)', code: 'pt-BR', progress: 89, translators: 2, status: 'in-progress', users: 6890 },
    { language: 'Japanese', code: 'ja', progress: 85, translators: 3, status: 'in-progress', users: 9840 },
    { language: 'Korean', code: 'ko', progress: 78, translators: 2, status: 'in-progress', users: 4560 },
    { language: 'Chinese (Simplified)', code: 'zh-CN', progress: 72, translators: 4, status: 'in-progress', users: 11230 },
    { language: 'Arabic', code: 'ar', progress: 45, translators: 2, status: 'pending', users: 2340 },
    { language: 'Hindi', code: 'hi', progress: 38, translators: 2, status: 'pending', users: 3450 }
  ];

  const regionalFeatures: RegionalFeature[] = [
    { region: 'North America', feature: 'HIPAA Compliance', status: 'enabled', compliance: true, usage: 87 },
    { region: 'Europe', feature: 'GDPR Compliance', status: 'enabled', compliance: true, usage: 95 },
    { region: 'Asia Pacific', feature: 'Local Payment Gateways', status: 'enabled', compliance: true, usage: 73 },
    { region: 'Latin America', feature: 'Currency Localization', status: 'enabled', compliance: true, usage: 82 },
    { region: 'Middle East', feature: 'Cultural Adaptations', status: 'testing', compliance: true, usage: 34 },
    { region: 'Africa', feature: 'Offline Mode', status: 'testing', compliance: true, usage: 28 }
  ];

  const currencies = [
    { code: 'USD', name: 'US Dollar', symbol: '$', users: 28340 },
    { code: 'EUR', name: 'Euro', symbol: '€', users: 16460 },
    { code: 'GBP', name: 'British Pound', symbol: '£', users: 5230 },
    { code: 'JPY', name: 'Japanese Yen', symbol: '¥', users: 9840 },
    { code: 'CAD', name: 'Canadian Dollar', symbol: 'C$', users: 3420 },
    { code: 'AUD', name: 'Australian Dollar', symbol: 'A$', users: 2890 },
    { code: 'BRL', name: 'Brazilian Real', symbol: 'R$', users: 6890 },
    { code: 'INR', name: 'Indian Rupee', symbol: '₹', users: 3450 }
  ];

  const timezones = [
    { region: 'Americas', zones: 9, active: 8 },
    { region: 'Europe/Africa', zones: 12, active: 11 },
    { region: 'Asia/Pacific', zones: 15, active: 13 },
    { region: 'Middle East', zones: 4, active: 3 }
  ];

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'complete': case 'enabled': return 'default';
      case 'in-progress': case 'testing': return 'secondary';
      case 'pending': case 'disabled': return 'outline';
      default: return 'secondary';
    }
  };

  const getProgressColor = (progress: number) => {
    if (progress >= 95) return 'bg-green-500';
    if (progress >= 80) return 'bg-yellow-500';
    return 'bg-red-500';
  };

  return (
    <div className="space-y-6">
      {/* Global Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Supported Languages</CardTitle>
            <Languages className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{languages.length}</div>
            <p className="text-xs text-muted-foreground">
              {languages.filter(l => l.status === 'complete').length} fully localized
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Global Users</CardTitle>
            <Users className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">93,380</div>
            <p className="text-xs text-muted-foreground">
              Across 47 countries
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Currencies</CardTitle>
            <DollarSign className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{currencies.length}</div>
            <p className="text-xs text-muted-foreground">
              Multi-currency support
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Compliance</CardTitle>
            <Shield className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">100%</div>
            <p className="text-xs text-muted-foreground">
              Regional compliance met
            </p>
          </CardContent>
        </Card>
      </div>

      <Tabs defaultValue="localization" className="w-full">
        <TabsList className="grid w-full grid-cols-5">
          <TabsTrigger value="localization">
            <Languages className="h-4 w-4 mr-2" />
            Localization
          </TabsTrigger>
          <TabsTrigger value="currencies">
            <DollarSign className="h-4 w-4 mr-2" />
            Currencies
          </TabsTrigger>
          <TabsTrigger value="timezones">
            <Clock className="h-4 w-4 mr-2" />
            Time Zones
          </TabsTrigger>
          <TabsTrigger value="regional">
            <MapPin className="h-4 w-4 mr-2" />
            Regional
          </TabsTrigger>
          <TabsTrigger value="compliance">
            <Shield className="h-4 w-4 mr-2" />
            Compliance
          </TabsTrigger>
        </TabsList>

        <TabsContent value="localization" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center justify-between">
                Language Localization Status
                <div className="flex items-center space-x-4">
                  <div className="flex items-center space-x-2">
                    <span className="text-sm">Auto Translation</span>
                    <Switch checked={autoTranslation} onCheckedChange={setAutoTranslation} />
                  </div>
                  <Button size="sm" onClick={() => console.log("InternationalFeaturesManager button clicked")}>Add Language</Button>
                </div>
              </CardTitle>
              <CardDescription>
                Track translation progress and manage localization across all supported languages
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {languages.map((lang) => (
                  <div key={lang.code} className="flex items-center justify-between p-4 border rounded-lg">
                    <div className="flex items-center space-x-4">
                      <div className="min-w-[120px]">
                        <div className="font-medium">{lang.language}</div>
                        <div className="text-sm text-muted-foreground">{lang.code}</div>
                      </div>
                      <div className="min-w-[100px]">
                        <Progress value={lang.progress} className="h-2" />
                        <div className="text-xs text-muted-foreground mt-1">{lang.progress}% complete</div>
                      </div>
                    </div>
                    <div className="flex items-center space-x-6">
                      <div className="text-center">
                        <div className="text-sm font-medium">{lang.users.toLocaleString()}</div>
                        <div className="text-xs text-muted-foreground">Users</div>
                      </div>
                      <div className="text-center">
                        <div className="text-sm font-medium">{lang.translators}</div>
                        <div className="text-xs text-muted-foreground">Translators</div>
                      </div>
                      <Badge variant={getStatusColor(lang.status)}>
                        {lang.status}
                      </Badge>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="currencies" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Multi-Currency Support</CardTitle>
              <CardDescription>
                Manage currency localization and payment processing
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid gap-4">
                {currencies.map((currency) => (
                  <div key={currency.code} className="flex items-center justify-between p-4 border rounded-lg">
                    <div className="flex items-center space-x-4">
                      <div className="w-12 h-12 bg-primary/10 rounded-full flex items-center justify-center">
                        <span className="text-lg font-bold">{currency.symbol}</span>
                      </div>
                      <div>
                        <div className="font-medium">{currency.name}</div>
                        <div className="text-sm text-muted-foreground">{currency.code}</div>
                      </div>
                    </div>
                    <div className="flex items-center space-x-6">
                      <div className="text-center">
                        <div className="text-sm font-medium">{currency.users.toLocaleString()}</div>
                        <div className="text-xs text-muted-foreground">Active Users</div>
                      </div>
                      <Button variant="outline" size="sm" onClick={() => console.log("InternationalFeaturesManager button clicked")}>
                        Configure
                      </Button>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="timezones" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Global Time Zone Management</CardTitle>
              <CardDescription>
                Handle scheduling and time-sensitive features across regions
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-6">
                <div className="grid grid-cols-2 gap-6">
                  {timezones.map((tz) => (
                    <div key={tz.region} className="p-4 border rounded-lg">
                      <div className="flex items-center justify-between mb-2">
                        <h4 className="font-medium">{tz.region}</h4>
                        <Badge variant="outline">{tz.active}/{tz.zones}</Badge>
                      </div>
                      <div className="space-y-2">
                        <div className="flex justify-between text-sm">
                          <span>Total Zones</span>
                          <span className="font-medium">{tz.zones}</span>
                        </div>
                        <div className="flex justify-between text-sm">
                          <span>Active Zones</span>
                          <span className="font-medium text-green-600">{tz.active}</span>
                        </div>
                        <Progress value={(tz.active / tz.zones) * 100} className="h-2" />
                      </div>
                    </div>
                  ))}
                </div>
                
                <div className="space-y-2">
                  <h4 className="font-medium">Smart Scheduling Features</h4>
                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div className="flex justify-between">
                      <span>Auto Time Detection</span>
                      <span className="font-medium text-green-600">Enabled</span>
                    </div>
                    <div className="flex justify-between">
                      <span>DST Handling</span>
                      <span className="font-medium text-green-600">Automatic</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Meeting Optimization</span>
                      <span className="font-medium text-green-600">Active</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Notification Timing</span>
                      <span className="font-medium text-green-600">Localized</span>
                    </div>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="regional" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Regional Feature Management</CardTitle>
              <CardDescription>
                Configure region-specific features and adaptations
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {regionalFeatures.map((feature) => (
                  <div key={`${feature.region}-${feature.feature}`} className="flex items-center justify-between p-4 border rounded-lg">
                    <div className="flex items-center space-x-4">
                      <div>
                        <div className="font-medium">{feature.feature}</div>
                        <div className="text-sm text-muted-foreground">{feature.region}</div>
                      </div>
                    </div>
                    <div className="flex items-center space-x-6">
                      <div className="text-center">
                        <div className="text-sm font-medium">{feature.usage}%</div>
                        <div className="text-xs text-muted-foreground">Usage</div>
                      </div>
                      <div className="flex items-center space-x-2">
                        {feature.compliance && (
                          <Badge variant="outline" className="text-green-600">
                            Compliant
                          </Badge>
                        )}
                        <Badge variant={getStatusColor(feature.status)}>
                          {feature.status}
                        </Badge>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="compliance" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center justify-between">
                Global Compliance Dashboard
                <div className="flex items-center space-x-2">
                  <span className="text-sm">Compliance Mode</span>
                  <Switch checked={complianceMode} onCheckedChange={setComplianceMode} />
                </div>
              </CardTitle>
              <CardDescription>
                Ensure regulatory compliance across all operating regions
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-6">
                <div className="grid grid-cols-2 gap-6">
                  <div className="space-y-4">
                    <h4 className="font-medium">Data Protection</h4>
                    <div className="space-y-2">
                      <div className="flex items-center justify-between p-2 border rounded">
                        <span className="text-sm">GDPR (EU)</span>
                        <Badge variant="default">Compliant</Badge>
                      </div>
                      <div className="flex items-center justify-between p-2 border rounded">
                        <span className="text-sm">CCPA (California)</span>
                        <Badge variant="default">Compliant</Badge>
                      </div>
                      <div className="flex items-center justify-between p-2 border rounded">
                        <span className="text-sm">PIPEDA (Canada)</span>
                        <Badge variant="default">Compliant</Badge>
                      </div>
                    </div>
                  </div>
                  <div className="space-y-4">
                    <h4 className="font-medium">Healthcare</h4>
                    <div className="space-y-2">
                      <div className="flex items-center justify-between p-2 border rounded">
                        <span className="text-sm">HIPAA (US)</span>
                        <Badge variant="default">Compliant</Badge>
                      </div>
                      <div className="flex items-center justify-between p-2 border rounded">
                        <span className="text-sm">ISO 27001</span>
                        <Badge variant="default">Certified</Badge>
                      </div>
                      <div className="flex items-center justify-between p-2 border rounded">
                        <span className="text-sm">SOC 2 Type II</span>
                        <Badge variant="default">Certified</Badge>
                      </div>
                    </div>
                  </div>
                </div>
                
                <div className="space-y-2">
                  <h4 className="font-medium">Audit Trail</h4>
                  <div className="grid grid-cols-3 gap-4 text-sm">
                    <div className="flex justify-between">
                      <span>Last Audit</span>
                      <span className="font-medium">15 days ago</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Findings</span>
                      <span className="font-medium text-green-600">0 Critical</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Next Review</span>
                      <span className="font-medium">45 days</span>
                    </div>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
};