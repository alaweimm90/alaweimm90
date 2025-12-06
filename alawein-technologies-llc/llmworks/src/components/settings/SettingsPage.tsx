import { useState } from "react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { 
  Settings,
  Bell,
  Shield,
  Download,
  Upload,
  Trash2,
  Save,
  RefreshCw,
  Globe,
  Moon,
  Sun,
  Monitor
} from "lucide-react";

export const SettingsPage = () => {
  const [notifications, setNotifications] = useState({
    evaluationComplete: true,
    modelErrors: true,
    weeklyReports: false,
    systemUpdates: true
  });

  const [preferences, setPreferences] = useState({
    theme: "system",
    language: "en",
    timezone: "UTC",
    defaultTimeout: 300,
    maxConcurrentEvals: 3
  });

  const [exportSettings, setExportSettings] = useState({
    format: "json",
    includeMetadata: true,
    includeRawOutputs: false,
    compression: true
  });

  const toggleNotification = (key: string) => {
    setNotifications(prev => ({
      ...prev,
      [key]: !prev[key as keyof typeof prev]
    }));
  };

  const exportData = () => {
    const data = {
      evaluations: "mock_evaluation_data",
      models: "mock_model_configs",
      settings: preferences,
      exportedAt: new Date().toISOString()
    };
    
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `llmworks-export-${Date.now()}.json`;
    a.click();
  };

  const importData = () => {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = '.json';
    input.onchange = (e) => {
      const file = (e.target as HTMLInputElement).files?.[0];
      if (file) {
        const reader = new FileReader();
        reader.onload = (e) => {
          try {
            const data = JSON.parse(e.target?.result as string);
            console.log('Imported data:', data);
            // Handle import logic here
          } catch (error) {
            console.error('Failed to parse import file:', error);
          }
        };
        reader.readAsText(file);
      }
    };
    input.click();
  };

  const resetSettings = () => {
    setPreferences({
      theme: "system",
      language: "en", 
      timezone: "UTC",
      defaultTimeout: 300,
      maxConcurrentEvals: 3
    });
    setNotifications({
      evaluationComplete: true,
      modelErrors: true,
      weeklyReports: false,
      systemUpdates: true
    });
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-foreground flex items-center gap-3">
            Settings
            <Badge className="bg-trust/10 text-trust border-trust/20">Control Center</Badge>
          </h1>
          <p className="text-muted-foreground">Configure LLM Works evaluation parameters and preferences</p>
        </div>
        <div className="flex gap-2">
          <Button onClick={() => console.log('Settings saved')} variant="trust" className="hover-scale">
            <Save className="h-4 w-4 mr-2" />
            Save Configuration
          </Button>
          <Button onClick={resetSettings} variant="outline" className="hover:border-risk hover:text-risk">
            <RefreshCw className="h-4 w-4 mr-2" />
            Reset to Defaults
          </Button>
        </div>
      </div>

      <Tabs defaultValue="general" className="w-full">
        <TabsList className="grid w-full grid-cols-5">
          <TabsTrigger value="general">General</TabsTrigger>
          <TabsTrigger value="notifications">Notifications</TabsTrigger>
          <TabsTrigger value="privacy">Privacy</TabsTrigger>
          <TabsTrigger value="data">Data</TabsTrigger>
          <TabsTrigger value="advanced">Advanced</TabsTrigger>
        </TabsList>

        <TabsContent value="general" className="space-y-6">
          <Card className="p-6 gradient-surface border-trust/20">
            <div className="flex items-center gap-2 mb-4">
              <Settings className="h-5 w-5 text-trust" />
              <h3 className="text-lg font-bold text-foreground">System Configuration</h3>
            </div>

            <div className="grid md:grid-cols-2 gap-6">
              <div className="space-y-4">
                <div>
                  <Label htmlFor="theme">Theme</Label>
                  <Select value={preferences.theme} onValueChange={(value) => setPreferences({...preferences, theme: value})}>
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="light">
                        <div className="flex items-center gap-2">
                          <Sun className="h-4 w-4" />
                          Light
                        </div>
                      </SelectItem>
                      <SelectItem value="dark">
                        <div className="flex items-center gap-2">
                          <Moon className="h-4 w-4" />
                          Dark
                        </div>
                      </SelectItem>
                      <SelectItem value="system">
                        <div className="flex items-center gap-2">
                          <Monitor className="h-4 w-4" />
                          System
                        </div>
                      </SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div>
                  <Label htmlFor="language">Language</Label>
                  <Select value={preferences.language} onValueChange={(value) => setPreferences({...preferences, language: value})}>
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="en">English</SelectItem>
                      <SelectItem value="es">Spanish</SelectItem>
                      <SelectItem value="fr">French</SelectItem>
                      <SelectItem value="de">German</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>

              <div className="space-y-4">
                <div>
                  <Label htmlFor="timezone">Timezone</Label>
                  <Select value={preferences.timezone} onValueChange={(value) => setPreferences({...preferences, timezone: value})}>
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="UTC">UTC</SelectItem>
                      <SelectItem value="PST">Pacific (PST)</SelectItem>
                      <SelectItem value="EST">Eastern (EST)</SelectItem>
                      <SelectItem value="GMT">Greenwich (GMT)</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div>
                  <Label htmlFor="timeout">Default Timeout (seconds)</Label>
                  <Input
                    id="timeout"
                    type="number"
                    value={preferences.defaultTimeout}
                    onChange={(e) => setPreferences({...preferences, defaultTimeout: parseInt(e.target.value)})}
                  />
                </div>
              </div>
            </div>
          </Card>
        </TabsContent>

        <TabsContent value="notifications" className="space-y-6">
          <Card className="p-6 gradient-surface border-neutral/20">
            <div className="flex items-center gap-2 mb-4">
              <Bell className="h-5 w-5 text-neutral" />
              <h3 className="text-lg font-bold text-foreground">Alert Configuration</h3>
            </div>

            <div className="space-y-4">
              {Object.entries(notifications).map(([key, value]) => (
                <div key={key} className="flex items-center justify-between p-4 rounded-lg border border-border">
                  <div>
                    <div className="font-medium text-foreground">
                      {key === 'evaluationComplete' && 'Evaluation Complete'}
                      {key === 'modelErrors' && 'Model Errors'}
                      {key === 'weeklyReports' && 'Weekly Reports'}
                      {key === 'systemUpdates' && 'System Updates'}
                    </div>
                    <div className="text-sm text-muted-foreground">
                      {key === 'evaluationComplete' && 'Notify when evaluations finish running'}
                      {key === 'modelErrors' && 'Alert when models encounter errors'}
                      {key === 'weeklyReports' && 'Receive weekly performance summaries'}
                      {key === 'systemUpdates' && 'Get notified about platform updates'}
                    </div>
                  </div>
                  <Switch
                    checked={value}
                    onCheckedChange={() => toggleNotification(key)}
                  />
                </div>
              ))}
            </div>
          </Card>
        </TabsContent>

        <TabsContent value="privacy" className="space-y-6">
          <Card className="p-6 gradient-surface border-trust/20">
            <div className="flex items-center gap-2 mb-4">
              <Shield className="h-5 w-5 text-trust" />
              <h3 className="text-lg font-bold text-foreground">Security Protocols</h3>
            </div>

            <div className="space-y-4">
              <div className="flex items-center justify-between p-4 rounded-lg border border-border">
                <div>
                  <div className="font-medium text-foreground">Data Analytics</div>
                  <div className="text-sm text-muted-foreground">Help improve LLM Works by sharing anonymous usage data</div>
                </div>
                <Switch defaultChecked />
              </div>

              <div className="flex items-center justify-between p-4 rounded-lg border border-border">
                <div>
                  <div className="font-medium text-foreground">Error Reporting</div>
                  <div className="text-sm text-muted-foreground">Automatically send error reports to help fix issues</div>
                </div>
                <Switch defaultChecked />
              </div>

              <div className="flex items-center justify-between p-4 rounded-lg border border-border">
                <div>
                  <div className="font-medium text-foreground">Model Data Retention</div>
                  <div className="text-sm text-muted-foreground">How long to keep evaluation data (30 days recommended)</div>
                </div>
                <Select defaultValue="30">
                  <SelectTrigger className="w-32">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="7">7 days</SelectItem>
                    <SelectItem value="30">30 days</SelectItem>
                    <SelectItem value="90">90 days</SelectItem>
                    <SelectItem value="365">1 year</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>
          </Card>
        </TabsContent>

        <TabsContent value="data" className="space-y-6">
          <Card className="p-6 gradient-surface border-neutral/20">
            <div className="flex items-center gap-2 mb-4">
              <Download className="h-5 w-5 text-neutral" />
              <h3 className="text-lg font-bold text-foreground">Data Operations</h3>
            </div>

            <div className="grid md:grid-cols-2 gap-6">
              <div className="space-y-4">
                <h4 className="font-medium text-foreground">Export Data</h4>
                <p className="text-sm text-muted-foreground">Download your evaluation data and settings</p>
                
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <Label>Include metadata</Label>
                    <Switch 
                      checked={exportSettings.includeMetadata}
                      onCheckedChange={(checked) => setExportSettings({...exportSettings, includeMetadata: checked})}
                    />
                  </div>
                  <div className="flex items-center justify-between">
                    <Label>Include raw outputs</Label>
                    <Switch 
                      checked={exportSettings.includeRawOutputs}
                      onCheckedChange={(checked) => setExportSettings({...exportSettings, includeRawOutputs: checked})}
                    />
                  </div>
                </div>

                <Button onClick={exportData} variant="outline" className="w-full">
                  <Download className="h-4 w-4 mr-2" />
                  Export Data
                </Button>
              </div>

              <div className="space-y-4">
                <h4 className="font-medium text-foreground">Import Data</h4>
                <p className="text-sm text-muted-foreground">Import previously exported data</p>

                <Button onClick={importData} variant="outline" className="w-full">
                  <Upload className="h-4 w-4 mr-2" />
                  Import Data
                </Button>

                <div className="p-3 rounded-lg bg-destructive/10 border border-destructive/20">
                  <h5 className="font-medium text-destructive mb-1">Danger Zone</h5>
                  <p className="text-sm text-muted-foreground mb-2">Permanently delete all evaluation data</p>
                  <Button variant="destructive" size="sm">
                    <Trash2 className="h-4 w-4 mr-2" />
                    Delete All Data
                  </Button>
                </div>
              </div>
            </div>
          </Card>
        </TabsContent>

        <TabsContent value="advanced" className="space-y-6">
          <Card className="p-6 gradient-surface border-risk/20">
            <div className="flex items-center gap-2 mb-4">
              <Globe className="h-5 w-5 text-risk" />
              <h3 className="text-lg font-bold text-foreground">Advanced Configuration</h3>
              <Badge className="bg-risk/10 text-risk">Expert Mode</Badge>
            </div>

            <div className="space-y-4">
              <div>
                <Label htmlFor="concurrent">Max Concurrent Evaluations</Label>
                <Input
                  id="concurrent"
                  type="number"
                  value={preferences.maxConcurrentEvals}
                  onChange={(e) => setPreferences({...preferences, maxConcurrentEvals: parseInt(e.target.value)})}
                  className="w-32"
                />
                <p className="text-sm text-muted-foreground mt-1">Higher values may impact performance</p>
              </div>

              <div>
                <Label htmlFor="api-endpoint">Custom API Endpoint</Label>
                <Input
                  id="api-endpoint"
                  placeholder="https://api.example.com"
                  className="mt-1"
                />
                <p className="text-sm text-muted-foreground mt-1">For enterprise deployments</p>
              </div>

              <div>
                <Label htmlFor="debug-mode">Debug Mode</Label>
                <div className="flex items-center justify-between mt-2">
                  <span className="text-sm text-muted-foreground">Enable detailed logging and debugging</span>
                  <Switch />
                </div>
              </div>
            </div>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
};