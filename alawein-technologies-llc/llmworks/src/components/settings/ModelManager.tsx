import { useState } from "react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Switch } from "@/components/ui/switch";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { 
  Plus, 
  Settings, 
  Trash2, 
  Edit,
  CheckCircle,
  AlertCircle,
  Zap,
  Brain,
  Globe,
  Key,
  Server
} from "lucide-react";

interface ModelConfig {
  id: string;
  name: string;
  provider: string;
  apiKey?: string;
  endpoint?: string;
  isActive: boolean;
  maxTokens: number;
  temperature: number;
  description: string;
}

export const ModelManager = () => {
  const [models, setModels] = useState<ModelConfig[]>([
    {
      id: "gpt-4o",
      name: "GPT-4o",
      provider: "OpenAI",
      isActive: true,
      maxTokens: 4096,
      temperature: 0.7,
      description: "Latest GPT-4 optimized model"
    },
    {
      id: "claude-3.5-sonnet",
      name: "Claude 3.5 Sonnet", 
      provider: "Anthropic",
      isActive: true,
      maxTokens: 4096,
      temperature: 0.7,
      description: "Anthropic's most capable model"
    },
    {
      id: "gemini-1.5-pro",
      name: "Gemini 1.5 Pro",
      provider: "Google",
      isActive: false,
      maxTokens: 2048,
      temperature: 0.7,
      description: "Google's latest multimodal model"
    }
  ]);

  const [newModel, setNewModel] = useState<Partial<ModelConfig>>({
    name: "",
    provider: "",
    maxTokens: 4096,
    temperature: 0.7,
    description: ""
  });

  const [editingModel, setEditingModel] = useState<string | null>(null);

  const providers = ["OpenAI", "Anthropic", "Google", "Meta", "Cohere", "Custom"];

  const addModel = () => {
    if (!newModel.name || !newModel.provider) return;

    const model: ModelConfig = {
      id: `model-${Date.now()}`,
      name: newModel.name,
      provider: newModel.provider,
      isActive: false,
      maxTokens: newModel.maxTokens || 4096,
      temperature: newModel.temperature || 0.7,
      description: newModel.description || ""
    };

    setModels([...models, model]);
    setNewModel({
      name: "",
      provider: "",
      maxTokens: 4096,
      temperature: 0.7,
      description: ""
    });
  };

  const toggleModel = (id: string) => {
    setModels(models.map(model => 
      model.id === id ? { ...model, isActive: !model.isActive } : model
    ));
  };

  const deleteModel = (id: string) => {
    setModels(models.filter(model => model.id !== id));
  };

  const updateModel = (id: string, updates: Partial<ModelConfig>) => {
    setModels(models.map(model => 
      model.id === id ? { ...model, ...updates } : model
    ));
  };

  const testConnection = async (modelId: string) => {
    // Mock API test
    console.log(`Testing connection for model: ${modelId}`);
  };

  return (
    <div className="space-y-6">
      {/* Add New Model */}
      <Card className="p-6">
        <div className="flex items-center gap-2 mb-4">
          <Plus className="h-5 w-5 text-primary" />
          <h3 className="text-lg font-bold text-foreground">Add New Model</h3>
        </div>

        <div className="grid md:grid-cols-2 gap-4 mb-4">
          <div>
            <Label htmlFor="model-name">Model Name</Label>
            <Input
              id="model-name"
              value={newModel.name || ""}
              onChange={(e) => setNewModel({ ...newModel, name: e.target.value })}
              placeholder="e.g., GPT-4 Turbo"
            />
          </div>
          <div>
            <Label htmlFor="provider">Provider</Label>
            <Select 
              value={newModel.provider || ""} 
              onValueChange={(value) => setNewModel({ ...newModel, provider: value })}
            >
              <SelectTrigger>
                <SelectValue placeholder="Select provider" />
              </SelectTrigger>
              <SelectContent>
                {providers.map(provider => (
                  <SelectItem key={provider} value={provider}>{provider}</SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        </div>

        <div className="grid md:grid-cols-2 gap-4 mb-4">
          <div>
            <Label htmlFor="max-tokens">Max Tokens</Label>
            <Input
              id="max-tokens"
              type="number"
              value={newModel.maxTokens || 4096}
              onChange={(e) => setNewModel({ ...newModel, maxTokens: parseInt(e.target.value) })}
            />
          </div>
          <div>
            <Label htmlFor="temperature">Temperature</Label>
            <Input
              id="temperature"
              type="number"
              step="0.1"
              min="0"
              max="2"
              value={newModel.temperature || 0.7}
              onChange={(e) => setNewModel({ ...newModel, temperature: parseFloat(e.target.value) })}
            />
          </div>
        </div>

        <div className="mb-4">
          <Label htmlFor="description">Description</Label>
          <Textarea
            id="description"
            value={newModel.description || ""}
            onChange={(e) => setNewModel({ ...newModel, description: e.target.value })}
            placeholder="Brief description of the model..."
          />
        </div>

        <Button onClick={addModel} variant="gradient">
          <Plus className="h-4 w-4 mr-2" />
          Add Model
        </Button>
      </Card>

      {/* Model List */}
      <div className="space-y-4">
        <h3 className="text-lg font-bold text-foreground">Configured Models</h3>
        
        {models.map((model) => (
          <Card key={model.id} className="p-6">
            {editingModel === model.id ? (
              <div className="space-y-4">
                <div className="grid md:grid-cols-2 gap-4">
                  <div>
                    <Label>Model Name</Label>
                    <Input
                      value={model.name}
                      onChange={(e) => updateModel(model.id, { name: e.target.value })}
                    />
                  </div>
                  <div>
                    <Label>Provider</Label>
                    <Select 
                      value={model.provider} 
                      onValueChange={(value) => updateModel(model.id, { provider: value })}
                    >
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        {providers.map(provider => (
                          <SelectItem key={provider} value={provider}>{provider}</SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                </div>

                <div className="grid md:grid-cols-2 gap-4">
                  <div>
                    <Label>Max Tokens</Label>
                    <Input
                      type="number"
                      value={model.maxTokens}
                      onChange={(e) => updateModel(model.id, { maxTokens: parseInt(e.target.value) })}
                    />
                  </div>
                  <div>
                    <Label>Temperature</Label>
                    <Input
                      type="number"
                      step="0.1"
                      value={model.temperature}
                      onChange={(e) => updateModel(model.id, { temperature: parseFloat(e.target.value) })}
                    />
                  </div>
                </div>

                <div>
                  <Label>Description</Label>
                  <Textarea
                    value={model.description}
                    onChange={(e) => updateModel(model.id, { description: e.target.value })}
                  />
                </div>

                <div className="flex gap-2">
                  <Button onClick={() => setEditingModel(null)} variant="default">
                    Save Changes
                  </Button>
                  <Button onClick={() => setEditingModel(null)} variant="outline">
                    Cancel
                  </Button>
                </div>
              </div>
            ) : (
              <div className="flex items-start justify-between">
                <div className="flex items-start gap-4 flex-1">
                  <div className="p-2 rounded-lg bg-primary/10">
                    <Brain className="h-5 w-5 text-primary" />
                  </div>
                  
                  <div className="flex-1">
                    <div className="flex items-center gap-3 mb-2">
                      <h4 className="font-semibold text-foreground">{model.name}</h4>
                      <Badge variant="outline">{model.provider}</Badge>
                      <Badge className={model.isActive ? "bg-accent/10 text-accent" : "bg-muted text-muted-foreground"}>
                        {model.isActive ? "Active" : "Inactive"}
                      </Badge>
                    </div>
                    
                    <p className="text-sm text-muted-foreground mb-3">{model.description}</p>
                    
                    <div className="flex items-center gap-4 text-xs text-muted-foreground">
                      <span>Max Tokens: {model.maxTokens}</span>
                      <span>Temperature: {model.temperature}</span>
                    </div>
                  </div>
                </div>

                <div className="flex items-center gap-2">
                  <Switch
                    checked={model.isActive}
                    onCheckedChange={() => toggleModel(model.id)}
                  />
                  <Button
                    onClick={() => testConnection(model.id)}
                    variant="ghost"
                    size="sm"
                  >
                    <Zap className="h-4 w-4" />
                  </Button>
                  <Button
                    onClick={() => setEditingModel(model.id)}
                    variant="ghost"
                    size="sm"
                  >
                    <Edit className="h-4 w-4" />
                  </Button>
                  <Button
                    onClick={() => deleteModel(model.id)}
                    variant="ghost"
                    size="sm"
                    className="text-destructive hover:text-destructive"
                  >
                    <Trash2 className="h-4 w-4" />
                  </Button>
                </div>
              </div>
            )}
          </Card>
        ))}
      </div>

      {/* Connection Status */}
      <Card className="p-6">
        <div className="flex items-center gap-2 mb-4">
          <Server className="h-5 w-5 text-accent" />
          <h3 className="text-lg font-bold text-foreground">Connection Status</h3>
        </div>

        <div className="space-y-3">
          {models.filter(m => m.isActive).map((model) => (
            <div key={model.id} className="flex items-center justify-between p-3 rounded-lg bg-muted/50">
              <div className="flex items-center gap-3">
                <CheckCircle className="h-4 w-4 text-accent" />
                <span className="font-medium text-foreground">{model.name}</span>
              </div>
              <Badge className="bg-accent/10 text-accent">Connected</Badge>
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
};