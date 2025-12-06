/**
 * AI Assistant Settings Section for Settings Page
 */

import { useState } from 'react';
import { Bot, Check, ExternalLink, Eye, EyeOff, Sparkles } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Switch } from '@/components/ui/switch';
import { Card } from '@/components/ui/card';
import { Slider } from '@/components/ui/slider';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { useAIAssistant } from '@/hooks/useAIAssistant';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';

export default function AISettingsSection() {
  const { settings, updateSettings, validateApiKey } = useAIAssistant();
  const [apiKey, setApiKey] = useState(settings.apiKey);
  const [showApiKey, setShowApiKey] = useState(false);
  const [isValidating, setIsValidating] = useState(false);
  const [validationStatus, setValidationStatus] = useState<'idle' | 'valid' | 'invalid'>('idle');

  const handleSaveApiKey = async () => {
    updateSettings({ apiKey });
    setValidationStatus('idle');
  };

  const handleValidateApiKey = async () => {
    if (!apiKey) return;
    setIsValidating(true);
    setValidationStatus('idle');
    const isValid = await validateApiKey(apiKey);
    setValidationStatus(isValid ? 'valid' : 'invalid');
    setIsValidating(false);
  };

  const handleToggleEnabled = (enabled: boolean) => {
    if (enabled && !apiKey) {
      // Show alert that API key is needed
      return;
    }
    updateSettings({ enabled });
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center gap-3">
        <div className="h-10 w-10 rounded-full bg-primary/10 flex items-center justify-center">
          <Bot className="h-5 w-5 text-primary" />
        </div>
        <div>
          <h2 className="text-xl font-semibold">AI Assistant</h2>
          <p className="text-sm text-muted-foreground">
            Configure Claude AI to help analyze results and answer questions
          </p>
        </div>
      </div>

      {/* Enable Toggle */}
      <Card className="p-4">
        <div className="flex items-center justify-between">
          <div className="space-y-0.5">
            <Label htmlFor="ai-enabled" className="text-base">
              Enable AI Assistant
            </Label>
            <p className="text-sm text-muted-foreground">
              Show AI assistant throughout the app
            </p>
          </div>
          <Switch
            id="ai-enabled"
            checked={settings.enabled}
            onCheckedChange={handleToggleEnabled}
            disabled={!apiKey}
          />
        </div>
      </Card>

      {/* Info Card */}
      <Alert>
        <Sparkles className="h-4 w-4" />
        <AlertTitle>Claude API Required</AlertTitle>
        <AlertDescription>
          Get your API key from{' '}
          <a
            href="https://console.anthropic.com/"
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-1 underline hover:text-primary"
          >
            console.anthropic.com
            <ExternalLink className="h-3 w-3" />
          </a>
          . Your key is stored locally and never sent to Attributa servers.
        </AlertDescription>
      </Alert>

      {/* API Key */}
      <Card className="p-4 space-y-4">
        <div className="space-y-2">
          <Label htmlFor="api-key">API Key</Label>
          <div className="flex gap-2">
            <div className="relative flex-1">
              <Input
                id="api-key"
                type={showApiKey ? 'text' : 'password'}
                value={apiKey}
                onChange={(e) => setApiKey(e.target.value)}
                placeholder="sk-ant-..."
                className="pr-10"
              />
              <Button
                variant="ghost"
                size="icon"
                className="absolute right-0 top-0 h-full"
                onClick={() => setShowApiKey(!showApiKey)}
                aria-label={showApiKey ? 'Hide API key' : 'Show API key'}
              >
                {showApiKey ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
              </Button>
            </div>
            <Button
              onClick={handleSaveApiKey}
              disabled={apiKey === settings.apiKey}
              variant="outline"
            >
              Save
            </Button>
          </div>
        </div>

        {/* Validation */}
        {apiKey && (
          <div className="flex items-center gap-2">
            <Button
              variant="secondary"
              size="sm"
              onClick={handleValidateApiKey}
              disabled={isValidating || !apiKey}
            >
              {isValidating ? 'Validating...' : 'Test Connection'}
            </Button>
            {validationStatus === 'valid' && (
              <span className="text-sm text-green-600 dark:text-green-400 flex items-center gap-1">
                <Check className="h-4 w-4" /> Valid
              </span>
            )}
            {validationStatus === 'invalid' && (
              <span className="text-sm text-destructive">Invalid API key</span>
            )}
          </div>
        )}
      </Card>

      {/* Advanced Settings */}
      <Card className="p-4 space-y-4">
        <h3 className="font-semibold">Advanced Settings</h3>

        {/* Model Selection */}
        <div className="space-y-2">
          <Label htmlFor="model">Model</Label>
          <Select
            value={settings.model}
            onValueChange={(model) => updateSettings({ model })}
          >
            <SelectTrigger id="model">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="claude-3-5-sonnet-20241022">
                Claude 3.5 Sonnet (Recommended)
              </SelectItem>
              <SelectItem value="claude-3-opus-20240229">
                Claude 3 Opus (Most Capable)
              </SelectItem>
              <SelectItem value="claude-3-haiku-20240307">
                Claude 3 Haiku (Fastest)
              </SelectItem>
            </SelectContent>
          </Select>
          <p className="text-xs text-muted-foreground">
            Sonnet offers the best balance of speed and quality
          </p>
        </div>

        {/* Temperature */}
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <Label htmlFor="temperature">Temperature</Label>
            <span className="text-sm text-muted-foreground">{settings.temperature}</span>
          </div>
          <Slider
            id="temperature"
            min={0}
            max={1}
            step={0.1}
            value={[settings.temperature]}
            onValueChange={([temperature]) => updateSettings({ temperature })}
          />
          <p className="text-xs text-muted-foreground">
            Higher values make responses more creative, lower values more focused
          </p>
        </div>

        {/* Max Tokens */}
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <Label htmlFor="max-tokens">Max Tokens</Label>
            <span className="text-sm text-muted-foreground">{settings.maxTokens}</span>
          </div>
          <Slider
            id="max-tokens"
            min={1024}
            max={8192}
            step={1024}
            value={[settings.maxTokens]}
            onValueChange={([maxTokens]) => updateSettings({ maxTokens })}
          />
          <p className="text-xs text-muted-foreground">
            Maximum length of AI responses (higher = longer responses but slower)
          </p>
        </div>
      </Card>

      {/* Privacy Notice */}
      <Alert variant="default">
        <AlertDescription className="text-xs">
          <strong>Privacy:</strong> Your API key and conversations are stored locally in your
          browser. Messages are sent directly to Anthropic's API using your key. Attributa
          does not have access to your conversations or API key.
        </AlertDescription>
      </Alert>
    </div>
  );
}
