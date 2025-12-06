import { useState, useEffect } from 'react';
import { Settings as SettingsIcon, Key, Shield, Save, Trash2 } from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Switch } from '@/components/ui/switch';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Textarea } from '@/components/ui/textarea';
import { useAppStore } from '@/store';
import { useToast } from '@/hooks/use-toast';
import { Sliders } from 'lucide-react';
import { Slider } from '@/components/ui/slider';
import { supabase } from '@/integrations/supabase/client';
import { useAuth } from '@/hooks/useAuth';
import { useNavigate } from 'react-router-dom';
import { useSEO } from '@/hooks/useSEO';
import NeuralBackground from '@/components/dev/NeuralBackground';
import AnimatedGrid from '@/components/dev/AnimatedGrid';
import AISettingsSection from '@/components/ai/AISettingsSection';

const Settings = () => {
  const { apiKeys, analysisOptions, scoringWeights, updateApiKey, updateAnalysisOptions, updateScoringWeights } = useAppStore();
  const { toast } = useToast();
  const { user } = useAuth();
  const navigate = useNavigate();
  
  const [openaiKey, setOpenaiKey] = useState(apiKeys.openai || '');
  const [anthropicKey, setAnthropicKey] = useState(apiKeys.anthropic || '');
  const [localModelName, setLocalModelName] = useState(analysisOptions.localModelName || 'Xenova/gpt2');

  useEffect(() => {
    const loadKeys = async () => {
      if (!user) return;
      const { data, error } = await supabase
        .from('user_api_keys')
        .select('openai_key, anthropic_key')
        .eq('user_id', user.id)
        .single();
      if (!error && data) {
        setOpenaiKey(data.openai_key || '');
        setAnthropicKey(data.anthropic_key || '');
      }
    };
    loadKeys();
  }, [user]);

  const handleSaveKeys = async () => {
    if (!user) {
      toast({ title: 'Local-only mode', description: 'API keys stored locally in browser only. Full attribution analysis works without external services.' });
      return;
    }
    if (openaiKey !== apiKeys.openai) {
      updateApiKey('openai', openaiKey);
    }
    if (anthropicKey !== apiKeys.anthropic) {
      updateApiKey('anthropic', anthropicKey);
    }

    // Persist to Supabase if authenticated
    if (user) {
      const { error } = await supabase
        .from('user_api_keys')
        .upsert({ user_id: user.id, openai_key: openaiKey || null, anthropic_key: anthropicKey || null });
      if (error) {
        toast({ title: 'Failed to save to Supabase', description: error.message, variant: 'destructive' });
      } else {
        toast({ title: 'Saved to Supabase', description: 'API keys stored securely in your account.' });
      }
    }

    if (localModelName !== analysisOptions.localModelName) {
      updateAnalysisOptions({ localModelName });
      
      // Attempt to validate/reload model
      try {
        const { ensureModelWithProgress } = await import('@/lib/nlp/tokenizers');
        await ensureModelWithProgress(localModelName);
        toast({
          title: "Model updated",
          description: `Successfully loaded ${localModelName}`,
        });
      } catch (error) {
        toast({
          title: "Model unavailable",
          description: `${localModelName} failed to load, using fallback`,
          variant: "destructive"
        });
      }
    }
    
    toast({
      title: "Settings saved",
      description: "Your API keys and preferences have been updated.",
    });
  };

  const handleClearKeys = async () => {
    try {
      // Clear local state and store
      setOpenaiKey('');
      setAnthropicKey('');
      updateApiKey('openai', '');
      updateApiKey('anthropic', '');

      if (user) {
        const { error } = await supabase
          .from('user_api_keys')
          .upsert({ user_id: user.id, openai_key: null, anthropic_key: null });
        if (error) {
          toast({ title: 'Failed to clear from account', description: error.message, variant: 'destructive' });
        } else {
          toast({ title: 'API keys removed', description: 'Keys cleared from your account and this device.' });
        }
      } else {
        toast({ title: 'API keys removed', description: 'Keys cleared from this device.' });
      }
    } catch (e: unknown) {
      const message = e instanceof Error ? e.message : 'Could not remove keys';
      toast({ title: 'Error', description: message, variant: 'destructive' });
    }
  };

  const maskKey = (key: string) => {
    if (!key || key.length < 8) return key;
    return key.substring(0, 8) + '•'.repeat(Math.min(key.length - 8, 20));
  };

  const getModelName = (): string => {
    const w = window as Window & { __AttributaModelName?: string; __TRANSFORMERS_MODEL_NAME?: string };
    return w.__AttributaModelName || w.__TRANSFORMERS_MODEL_NAME || 'Local (mock)';
  };

  const clearModelCaches = async () => {
    try {
      if ('caches' in window) {
        const keys = await caches.keys();
        await Promise.all(keys.map((k) => caches.delete(k)));
      }
      interface IDBWithDatabases extends IDBFactory {
        databases?: () => Promise<Array<{ name: string; version: number }>>;
      }
      const idbExtended = indexedDB as IDBWithDatabases;
      let names: string[] = [];
      if (idbExtended.databases) {
        const dbs = await idbExtended.databases();
        names = (dbs || []).map((d) => d.name).filter(Boolean);
      }
      const fallback = ['transformers-cache', 'transformers', 'huggingface', 'whisper-cache', 'onnx-cache'];
      const toDelete = Array.from(new Set([...(names as string[]), ...fallback])).filter((n) => /transformers|huggingface|whisper|onnx/i.test(n));
      await Promise.all(
        toDelete.map(
          (name) =>
            new Promise<void>((resolve) => {
              const req = indexedDB.deleteDatabase(name);
              req.onsuccess = req.onerror = req.onblocked = () => resolve();
            })
        )
      );
      for (let i = localStorage.length - 1; i >= 0; i--) {
        const key = localStorage.key(i)!;
        if (/transformers|huggingface|whisper|onnx/i.test(key)) localStorage.removeItem(key);
      }
      for (let i = sessionStorage.length - 1; i >= 0; i--) {
        const key = sessionStorage.key(i)!;
        if (/transformers|huggingface|whisper|onnx/i.test(key)) sessionStorage.removeItem(key);
      }
      toast({ title: 'Model cache cleared', description: 'Model files removed. Reloading...' });
      setTimeout(() => window.location.reload(), 700);
    } catch (e) {
      toast({ title: 'Failed to clear cache', description: 'Please try again.', variant: 'destructive' });
    }
  };

  useSEO({
    title: 'Settings — API & Preferences | Attributa.dev',
    description: 'Configure API keys, model settings, and privacy options.',
  });

  return (
    <div className="relative min-h-screen overflow-hidden bg-gradient-to-br from-background via-background to-secondary engineering-pattern">
      <NeuralBackground />
      <AnimatedGrid />
      
      <div className="relative z-10 container mx-auto px-4 py-8 max-w-4xl">
        <div className="space-y-6 animate-fade-in">
        <div className="space-y-1">
          <h1 className="text-2xl font-bold tracking-tight display-tight font-mono flex items-center space-x-2">
            <SettingsIcon className="h-6 w-6" />
            <span>Settings</span>
          </h1>
          <p className="text-muted-foreground">
            Configure API keys and analysis preferences
          </p>
        </div>


        <Card className="glass-card">
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <Shield className="h-5 w-5" />
              <span>Cloud Models</span>
              <Badge variant="secondary">Optional</Badge>
            </CardTitle>
            <CardDescription>
              Use OpenAI or Anthropic via your own API keys. Account is optional and only needed for cloud sync/models.
            </CardDescription>
          </CardHeader>
          <CardContent className="flex items-center justify-between">
            <p className="text-sm text-muted-foreground">Local Mode stays free and private.</p>
            {user ? (
              <Button size="sm" onClick={() => {
                const el = document.getElementById('api-keys-section');
                el?.scrollIntoView({ behavior: 'smooth' });
              }}>Manage API keys</Button>
            ) : (
              <Button size="sm" onClick={() => navigate('/auth')}>Enable cloud models (optional)</Button>
            )}
          </CardContent>
        </Card>

        {/* API Keys */}
        <Card id="api-keys-section" className="glass-card">
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <Key className="h-5 w-5" />
              <span>API Keys</span>
            </CardTitle>
            <CardDescription>
              {user ? (
                <>API keys are linked to your account and used only when you enable external APIs.</>
              ) : (
                <>Optional sign-in lets you sync keys and enable cloud models. Local Mode remains private.</>
              )}
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            <div className="space-y-2">
              <Label htmlFor="openai-key">OpenAI API Key</Label>
              <Input
                id="openai-key"
                type="password"
                placeholder="sk-..."
                value={openaiKey}
                onChange={(e) => setOpenaiKey(e.target.value)}
                disabled={!user}
              />
              {apiKeys.openai && (
                <p className="text-xs text-muted-foreground">
                  Current: {maskKey(apiKeys.openai)}
                </p>
              )}
            </div>

            <div className="space-y-2">
              <Label htmlFor="anthropic-key">Anthropic API Key</Label>
              <Input
                id="anthropic-key"
                type="password"
                placeholder="sk-ant-..."
                value={anthropicKey}
                onChange={(e) => setAnthropicKey(e.target.value)}
                disabled={!user}
              />
              {apiKeys.anthropic && (
                <p className="text-xs text-muted-foreground">
                  Current: {maskKey(apiKeys.anthropic)}
                </p>
              )}
            </div>
            
            <div className="space-y-2">
              <div className="flex justify-between">
                <Label>Minimum characters per prose segment</Label>
                <span className="text-sm text-muted-foreground">{analysisOptions.minChars}</span>
              </div>
              <Slider
                value={[analysisOptions.minChars]}
                onValueChange={(values) => updateAnalysisOptions({ minChars: values[0] })}
                min={600}
                max={2000}
                step={50}
                className="w-full"
              />
              <p className="text-xs text-muted-foreground">
                Segments shorter than this will be merged with neighbors before analysis.
              </p>
            </div>

            <div className="flex gap-2">
              <Button onClick={user ? handleSaveKeys : () => navigate('/auth')}>
                <Save className="h-4 w-4 mr-2" />
                {user ? 'Save API Keys' : 'Sign in to sync API keys'}
              </Button>
              <Button variant="outline" onClick={handleClearKeys}>
                <Trash2 className="h-4 w-4 mr-2" />
                Remove API Keys
              </Button>
            </div>
          </CardContent>
        </Card>

        {/* Analysis Options */}
        <Card className="glass-card">
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <Shield className="h-5 w-5" />
              <span>Privacy & Analysis Options</span>
            </CardTitle>
            <CardDescription>
              Control how analysis is performed and what external services are used
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            <div className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="local-model-name">Local causal LM (Hugging Face id)</Label>
                <Input
                  id="local-model-name"
                  placeholder="Xenova/gpt2"
                  value={localModelName}
                  onChange={(e) => setLocalModelName(e.target.value)}
                  aria-label="Local causal language model identifier"
                />
                <p className="text-xs text-muted-foreground">
                  Must be a <strong>causal</strong> (left-to-right) model. If unavailable, falls back to the default.
                </p>
              </div>
              
              <div className="flex items-center justify-between rounded-md border p-3">
                <div className="space-y-0.5">
                  <Label>Detected Model</Label>
                  <p className="text-sm text-muted-foreground">{getModelName()}</p>
                </div>
                <Button variant="outline" size="sm" onClick={clearModelCaches}>Clear model cache</Button>
              </div>
            </div>
            <div className="flex items-center justify-between">
              <div className="space-y-1 flex-1">
                <Label htmlFor="enable-external">Enable external model calls</Label>
                <p className="text-sm text-muted-foreground">
                  Allow analysis to use external APIs when keys are provided. 
                  When disabled, only local/mock analysis is performed.
                </p>
              </div>
              <Switch
                id="enable-external"
                checked={analysisOptions.useExternalApis}
                onCheckedChange={(checked) => updateAnalysisOptions({ useExternalApis: checked })}
              />
            </div>

            <div className="flex items-center justify-between">
              <div className="space-y-1 flex-1">
                <Label htmlFor="enable-citations">Enable citation lookups</Label>
                <p className="text-sm text-muted-foreground">
                  Attempt to validate citations using external databases like Crossref. 
                  In MVP, this is mocked.
                </p>
              </div>
              <div className="flex items-center space-x-2">
                <Badge variant="outline" className="text-xs">
                  MVP: Mocked
                </Badge>
                <Switch
                  id="enable-citations"
                  checked={true}
                  disabled
                />
              </div>
            </div>

            <div className="flex items-center justify-between">
              <div className="space-y-1 flex-1">
                <Label htmlFor="watermark-detection">Watermark detection</Label>
                <p className="text-sm text-muted-foreground">
                  Attempt to detect statistical watermarks in text. 
                  Experimental feature with limited accuracy.
                </p>
              </div>
              <Switch
                id="watermark-detection"
                checked={analysisOptions.tryWatermark}
                onCheckedChange={(checked) => updateAnalysisOptions({ tryWatermark: checked })}
              />
            </div>

            <div className="flex items-center justify-between">
              <div className="space-y-1 flex-1">
                <Label htmlFor="local-only">Local-only mode</Label>
                <p className="text-sm text-muted-foreground">
                  Process all content locally without any external API calls. 
                  Recommended for sensitive content.
                </p>
              </div>
              <Switch
                id="local-only"
                checked={analysisOptions.useLocalOnly}
                onCheckedChange={(checked) => updateAnalysisOptions({ useLocalOnly: checked })}
              />
            </div>
          </CardContent>
        </Card>

        {/* Scoring Weights */}
        <Card className="glass-card">
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <Sliders className="h-5 w-5" />
              <span>Scoring Weights</span>
            </CardTitle>
            <CardDescription>
              Adjust the importance of different analysis signals in the composite score
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            <div className="space-y-4">
              <div className="space-y-2">
                <div className="flex justify-between">
                  <Label>GLTR Token Analysis</Label>
                  <span className="text-sm text-muted-foreground">{Math.round(scoringWeights.gltr * 100)}%</span>
                </div>
                <Slider
                  value={[scoringWeights.gltr * 100]}
                  onValueChange={(values) => updateScoringWeights({ gltr: values[0] / 100 })}
                  max={50}
                  step={1}
                  className="w-full"
                />
              </div>
              
              <div className="space-y-2">
                <div className="flex justify-between">
                  <Label>DetectGPT Curvature</Label>
                  <span className="text-sm text-muted-foreground">{Math.round(scoringWeights.detectgpt * 100)}%</span>
                </div>
                <Slider
                  value={[scoringWeights.detectgpt * 100]}
                  onValueChange={(values) => updateScoringWeights({ detectgpt: values[0] / 100 })}
                  max={50}
                  step={1}
                  className="w-full"
                />
              </div>
              
              <div className="space-y-2">
                <div className="flex justify-between">
                  <Label>Citation Validity</Label>
                  <span className="text-sm text-muted-foreground">{Math.round(scoringWeights.citations * 100)}%</span>
                </div>
                <Slider
                  value={[scoringWeights.citations * 100]}
                  onValueChange={(values) => updateScoringWeights({ citations: values[0] / 100 })}
                  max={50}
                  step={1}
                  className="w-full"
                />
              </div>
              
              <div className="space-y-2">
                <div className="flex justify-between">
                  <Label>Watermark Detection</Label>
                  <span className="text-sm text-muted-foreground">{Math.round(scoringWeights.watermark * 100)}%</span>
                </div>
                <Slider
                  value={[scoringWeights.watermark * 100]}
                  onValueChange={(values) => updateScoringWeights({ watermark: values[0] / 100 })}
                  max={50}
                  step={1}
                  className="w-full"
                />
              </div>
              
              <div className="space-y-2">
                <div className="flex justify-between">
                  <Label>Code Security (CWE)</Label>
                  <span className="text-sm text-muted-foreground">{Math.round(scoringWeights.cwe * 100)}%</span>
                </div>
                <Slider
                  value={[scoringWeights.cwe * 100]}
                  onValueChange={(values) => updateScoringWeights({ cwe: values[0] / 100 })}
                  max={30}
                  step={1}
                  className="w-full"
                />
              </div>
              
              <div className="space-y-2">
                <div className="flex justify-between">
                  <Label>Short Segment Penalty</Label>
                  <span className="text-sm text-muted-foreground">{Math.round(scoringWeights.shortPenalty * 100)}%</span>
                </div>
                <Slider
                  value={[scoringWeights.shortPenalty * 100]}
                  onValueChange={(values) => updateScoringWeights({ shortPenalty: values[0] / 100 })}
                  max={10}
                  step={1}
                  className="w-full"
                />
              </div>
            </div>
            
            <div className="pt-4 border-t">
              <div className="flex items-center justify-between mb-4">
                <div className="text-sm text-muted-foreground">
                  <p className="mb-2">Total weight: {Math.round((Object.values(scoringWeights).reduce((a, b) => a + b, 0)) * 100)}%</p>
                  <p>Weights are automatically normalized during scoring.</p>
                </div>
                <Button 
                  variant="outline" 
                  size="sm"
                  onClick={() => {
                    updateScoringWeights({
                      gltr: 0.22,
                      detectgpt: 0.22,
                      citations: 0.25,
                      watermark: 0.18,
                      cwe: 0.10,
                      shortPenalty: 0.03
                    });
                    toast({
                      title: "Weights reset",
                      description: "Scoring weights have been reset to defaults.",
                    });
                  }}
                >
                  Reset to defaults
                </Button>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Status */}
        <Card>
          <CardHeader>
            <CardTitle>Current Status</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid md:grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label>Analysis Mode</Label>
                <div className="flex flex-wrap gap-2">
                  {analysisOptions.useLocalOnly ? (
                    <Badge className="bg-success/10 text-success border-success/20">
                      Local-only mode
                    </Badge>
                  ) : (
                    <Badge className="bg-warning/10 text-warning border-warning/20">
                      External APIs enabled
                    </Badge>
                  )}
                  
                  {analysisOptions.tryWatermark && (
                    <Badge variant="outline">Watermark detection</Badge>
                  )}
                </div>
              </div>

              <div className="space-y-2">
                <Label>API Keys Configured</Label>
                <div className="flex flex-wrap gap-2">
                  {apiKeys.openai ? (
                    <Badge variant="secondary">OpenAI ✓</Badge>
                  ) : (
                    <Badge variant="outline">OpenAI</Badge>
                  )}
                  
                  {apiKeys.anthropic ? (
                    <Badge variant="secondary">Anthropic ✓</Badge>
                  ) : (
                    <Badge variant="outline">Anthropic</Badge>
                  )}
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* AI Assistant Settings */}
        <AISettingsSection />

        {/* Disclaimer */}
        <Card className="border-muted-foreground/20">
          <CardContent className="pt-6">
            <p className="text-sm text-muted-foreground">
              <strong>Privacy Notice:</strong> All settings and API keys are stored locally in your browser.
              No data is transmitted to external servers unless you explicitly enable external API calls
              and provide the necessary keys. Analysis results are not stored on any external servers.
            </p>
          </CardContent>
        </Card>
        </div>
      </div>
    </div>
  );
};

export default Settings;