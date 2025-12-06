import { useState } from "react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { 
  Upload, 
  FileText, 
  Plus, 
  Trash2, 
  Play, 
  Settings,
  CheckCircle,
  AlertCircle,
  Download
} from "lucide-react";

interface CustomTest {
  id: string;
  name: string;
  description: string;
  prompts: Array<{
    prompt: string;
    expectedOutput: string;
    criteria: string;
  }>;
  createdAt: Date;
}

export const CustomTestBuilder = () => {
  const [testName, setTestName] = useState("");
  const [testDescription, setTestDescription] = useState("");
  const [prompts, setPrompts] = useState([
    { prompt: "", expectedOutput: "", criteria: "" }
  ]);
  const [savedTests, setSavedTests] = useState<CustomTest[]>([
    {
      id: "test-1",
      name: "Medical Diagnosis Accuracy",
      description: "Test model's ability to provide accurate medical information",
      prompts: [
        {
          prompt: "What are the symptoms of Type 2 diabetes?",
          expectedOutput: "Increased thirst, frequent urination, blurred vision, fatigue",
          criteria: "Accuracy of medical information, completeness of symptom list"
        }
      ],
      createdAt: new Date("2024-01-15")
    },
    {
      id: "test-2", 
      name: "Legal Reasoning Test",
      description: "Evaluate legal analysis and reasoning capabilities",
      prompts: [
        {
          prompt: "Explain the concept of negligence in tort law",
          expectedOutput: "Duty of care, breach of duty, causation, damages",
          criteria: "Legal accuracy, clarity of explanation, completeness"
        }
      ],
      createdAt: new Date("2024-01-10")
    }
  ]);
  const [isRunning, setIsRunning] = useState(false);
  const [selectedTest, setSelectedTest] = useState<string>("");
  const [selectedModels, setSelectedModels] = useState<string[]>([]);

  const models = ["GPT-4o", "Claude 3.5 Sonnet", "Gemini 1.5 Pro", "Llama 3.1 70B"];

  const addPrompt = () => {
    setPrompts([...prompts, { prompt: "", expectedOutput: "", criteria: "" }]);
  };

  const removePrompt = (index: number) => {
    if (prompts.length > 1) {
      setPrompts(prompts.filter((_, i) => i !== index));
    }
  };

  const updatePrompt = (index: number, field: string, value: string) => {
    const updated = prompts.map((prompt, i) => 
      i === index ? { ...prompt, [field]: value } : prompt
    );
    setPrompts(updated);
  };

  const saveTest = () => {
    if (!testName || prompts.some(p => !p.prompt)) return;

    const newTest: CustomTest = {
      id: `test-${Date.now()}`,
      name: testName,
      description: testDescription,
      prompts: prompts.filter(p => p.prompt.trim()),
      createdAt: new Date()
    };

    setSavedTests([newTest, ...savedTests]);
    setTestName("");
    setTestDescription("");
    setPrompts([{ prompt: "", expectedOutput: "", criteria: "" }]);
  };

  const runCustomTest = async () => {
    if (!selectedTest || selectedModels.length === 0) return;
    
    setIsRunning(true);
    
    // Mock test execution
    setTimeout(() => {
      setIsRunning(false);
      // In a real implementation, this would show results
    }, 3000);
  };

  const exportTest = (test: CustomTest) => {
    const data = {
      name: test.name,
      description: test.description,
      prompts: test.prompts,
      exportedAt: new Date().toISOString()
    };
    
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${test.name.toLowerCase().replace(/\s+/g, '-')}-test.json`;
    a.click();
  };

  return (
    <div className="space-y-6">
      {/* Test Builder */}
      <Card className="p-6 gradient-surface border-trust/20 hover:border-trust/40 transition-colors">
        <div className="flex items-center gap-2 mb-6">
          <Plus className="h-5 w-5 text-trust" />
          <h3 className="text-lg font-bold text-foreground">Forge Custom Evaluation</h3>
          <Badge className="bg-trust/10 text-trust">Builder Mode</Badge>
        </div>

        <div className="grid md:grid-cols-2 gap-4 mb-6">
          <div>
            <Label htmlFor="test-name" className="text-sm font-medium text-foreground">Test Name</Label>
            <Input
              id="test-name"
              value={testName}
              onChange={(e) => setTestName(e.target.value)}
              placeholder="Enter test name..."
              className="mt-1"
            />
          </div>
          <div>
            <Label htmlFor="test-description" className="text-sm font-medium text-foreground">Description</Label>
            <Input
              id="test-description"
              value={testDescription}
              onChange={(e) => setTestDescription(e.target.value)}
              placeholder="Brief description of the test..."
              className="mt-1"
            />
          </div>
        </div>

        {/* Prompts Builder */}
        <div className="space-y-4 mb-6">
          <div className="flex items-center justify-between">
            <h4 className="font-medium text-foreground">Test Prompts</h4>
            <Button onClick={addPrompt} variant="outline" size="sm">
              <Plus className="h-4 w-4 mr-1" />
              Add Prompt
            </Button>
          </div>

          {prompts.map((prompt, index) => (
            <Card key={index} className="p-4 border border-border">
              <div className="flex items-center justify-between mb-3">
                <Label className="text-sm font-medium text-foreground">Prompt {index + 1}</Label>
                {prompts.length > 1 && (
                  <Button
                    onClick={() => removePrompt(index)}
                    variant="ghost"
                    size="sm"
                    className="text-destructive hover:text-destructive"
                  >
                    <Trash2 className="h-4 w-4" />
                  </Button>
                )}
              </div>

              <div className="space-y-3">
                <div>
                  <Label className="text-xs text-muted-foreground">Prompt</Label>
                  <Textarea
                    value={prompt.prompt}
                    onChange={(e) => updatePrompt(index, "prompt", e.target.value)}
                    placeholder="Enter the prompt to test..."
                    className="mt-1"
                  />
                </div>

                <div className="grid md:grid-cols-2 gap-3">
                  <div>
                    <Label className="text-xs text-muted-foreground">Expected Output (Optional)</Label>
                    <Textarea
                      value={prompt.expectedOutput}
                      onChange={(e) => updatePrompt(index, "expectedOutput", e.target.value)}
                      placeholder="Expected response or key points..."
                      className="mt-1"
                      rows={3}
                    />
                  </div>
                  <div>
                    <Label className="text-xs text-muted-foreground">Evaluation Criteria</Label>
                    <Textarea
                      value={prompt.criteria}
                      onChange={(e) => updatePrompt(index, "criteria", e.target.value)}
                      placeholder="How should this be evaluated..."
                      className="mt-1"
                      rows={3}
                    />
                  </div>
                </div>
              </div>
            </Card>
          ))}
        </div>

        <Button 
          onClick={saveTest} 
          disabled={!testName || prompts.some(p => !p.prompt)}
          variant="trust"
          className="hover-scale"
        >
          <CheckCircle className="h-4 w-4 mr-2" />
          Forge Evaluation
        </Button>
      </Card>

      {/* Saved Tests & Runner */}
      <div className="grid lg:grid-cols-2 gap-6">
        {/* Saved Tests */}
        <Card className="p-6">
          <div className="flex items-center gap-2 mb-4">
            <FileText className="h-5 w-5 text-accent" />
            <h3 className="text-lg font-bold text-foreground">Saved Tests</h3>
          </div>

          <div className="space-y-3">
            {savedTests.map((test) => (
              <Card key={test.id} className="p-4 border border-border hover:shadow-medium transition-shadow">
                <div className="flex items-start justify-between mb-2">
                  <div className="flex-1">
                    <h4 className="font-medium text-foreground">{test.name}</h4>
                    <p className="text-sm text-muted-foreground mb-2">{test.description}</p>
                    <div className="flex items-center gap-4 text-xs text-muted-foreground">
                      <span>{test.prompts.length} prompts</span>
                      <span>Created {test.createdAt.toLocaleDateString()}</span>
                    </div>
                  </div>
                  <Button
                    onClick={() => exportTest(test)}
                    variant="ghost"
                    size="sm"
                  >
                    <Download className="h-4 w-4" />
                  </Button>
                </div>
                
                <Button
                  onClick={() => setSelectedTest(test.id)}
                  variant={selectedTest === test.id ? "default" : "outline"}
                  size="sm"
                  className="w-full mt-2"
                >
                  {selectedTest === test.id ? "Selected" : "Select for Testing"}
                </Button>
              </Card>
            ))}
          </div>
        </Card>

        {/* Test Runner */}
        <Card className="p-6">
          <div className="flex items-center gap-2 mb-4">
            <Play className="h-5 w-5 text-primary" />
            <h3 className="text-lg font-bold text-foreground">Run Custom Test</h3>
          </div>

          <div className="space-y-4">
            <div>
              <Label className="text-sm font-medium text-foreground mb-2 block">Selected Test</Label>
              {selectedTest ? (
                <div className="p-3 rounded-lg bg-accent/10 border border-accent/20">
                  <div className="font-medium text-accent">
                    {savedTests.find(t => t.id === selectedTest)?.name}
                  </div>
                  <div className="text-sm text-muted-foreground">
                    {savedTests.find(t => t.id === selectedTest)?.prompts.length} prompts
                  </div>
                </div>
              ) : (
                <div className="p-3 rounded-lg bg-muted/50 border border-border">
                  <div className="text-sm text-muted-foreground">No test selected</div>
                </div>
              )}
            </div>

            <div>
              <Label className="text-sm font-medium text-foreground mb-2 block">Select Models</Label>
              <div className="space-y-2">
                {models.map((model) => (
                  <label key={model} className="flex items-center space-x-2 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={selectedModels.includes(model)}
                      onChange={(e) => {
                        if (e.target.checked) {
                          setSelectedModels([...selectedModels, model]);
                        } else {
                          setSelectedModels(selectedModels.filter(m => m !== model));
                        }
                      }}
                      className="rounded"
                    />
                    <span className="text-sm text-foreground">{model}</span>
                  </label>
                ))}
              </div>
            </div>

            {isRunning && (
              <div className="space-y-2">
                <div className="flex items-center justify-between text-sm">
                  <span className="text-foreground">Running custom test...</span>
                  <span className="text-muted-foreground">Progress</span>
                </div>
                <Progress value={65} className="h-2" />
              </div>
            )}

            <Button 
              onClick={runCustomTest}
              disabled={!selectedTest || selectedModels.length === 0 || isRunning}
              variant="trust"
              className="w-full hover-scale"
            >
              {isRunning ? (
                <>
                  <Settings className="h-4 w-4 mr-2 animate-spin" />
                  Evaluating Performance...
                </>
              ) : (
                <>
                  <Play className="h-4 w-4 mr-2" />
                  Execute Evaluation
                </>
              )}
            </Button>
          </div>
        </Card>
      </div>
    </div>
  );
};