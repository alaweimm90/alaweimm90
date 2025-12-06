import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Checkbox } from '@/components/ui/checkbox';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog';
import { Label } from '@/components/ui/label';
import { Separator } from '@/components/ui/separator';
import { Download, FileSpreadsheet, FileText, FileJson, FileImage, Settings } from 'lucide-react';
import { useToast } from '@/hooks/use-toast';

interface Attribution {
  id: string;
  attribution_type: string;
  confidence_score: number;
  confidence_level: string;
  similarity_score?: number;
  sources?: {
    title?: string;
    source_type?: string;
  };
  rationale?: string[];
  signals?: {
    gltr?: unknown;
    detectgpt?: unknown;
    semantic?: unknown;
  };
}

interface ExportOptionsProps {
  attributions: Attribution[];
  projectName?: string;
}

interface ExportConfig {
  format: 'json' | 'csv' | 'pdf' | 'png';
  includeRationale: boolean;
  includeSignals: boolean;
  includeMetrics: boolean;
  confidenceThreshold: number;
  maxResults: number;
}

const defaultConfig: ExportConfig = {
  format: 'json',
  includeRationale: true,
  includeSignals: true,
  includeMetrics: true,
  confidenceThreshold: 0,
  maxResults: 1000
};

export default function ExportOptions({ attributions, projectName = 'project' }: ExportOptionsProps) {
  const [config, setConfig] = useState<ExportConfig>(defaultConfig);
  const [isExporting, setIsExporting] = useState(false);
  const { toast } = useToast();

  const filteredAttributions = attributions.filter(
    attr => (attr.confidence_score * 100) >= config.confidenceThreshold
  ).slice(0, config.maxResults);

  const exportData = async () => {
    setIsExporting(true);
    try {
      const timestamp = new Date().toISOString().split('T')[0];
      const filename = `${projectName}-attributions-${timestamp}`;

      switch (config.format) {
        case 'json':
          await exportAsJSON(filteredAttributions, config, filename);
          break;
        case 'csv':
          await exportAsCSV(filteredAttributions, config, filename);
          break;
        case 'pdf':
          await exportAsPDF(filteredAttributions, config, filename);
          break;
        case 'png':
          await exportAsPNG(filteredAttributions, config, filename);
          break;
      }
      
      toast({
        title: "Export completed",
        description: `Downloaded ${config.format.toUpperCase()} file with ${filteredAttributions.length} attributions`
      });
    } catch (error) {
      toast({
        title: "Export failed",
        description: "There was an error generating the export file",
        variant: "destructive"
      });
    } finally {
      setIsExporting(false);
    }
  };

  return (
    <Dialog>
      <DialogTrigger asChild>
        <Button variant="outline" className="flex items-center gap-2">
          <Download className="h-4 w-4" />
          Export Data
        </Button>
      </DialogTrigger>
      
      <DialogContent className="max-w-2xl">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Settings className="h-5 w-5" />
            Export Attribution Data
          </DialogTitle>
          <DialogDescription>
            Configure export format and data to include in your attribution report
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-6">
          {/* Export Format */}
          <div className="space-y-3">
            <Label className="text-base font-medium">Export Format</Label>
            <div className="grid grid-cols-2 gap-3">
              {[
                { value: 'json', icon: FileJson, label: 'JSON', desc: 'Structured data' },
                { value: 'csv', icon: FileSpreadsheet, label: 'CSV', desc: 'Spreadsheet data' },
                { value: 'pdf', icon: FileText, label: 'PDF', desc: 'Report document' },
                { value: 'png', icon: FileImage, label: 'PNG', desc: 'Visual chart' }
              ].map(({ value, icon: Icon, label, desc }) => (
                <Card
                  key={value}
                  className={`cursor-pointer transition-colors ${
                    config.format === value ? 'ring-2 ring-primary bg-primary/5' : 'hover:bg-muted/50'
                  }`}
                  onClick={() => setConfig({ ...config, format: value as ExportConfig['format'] })}
                >
                  <CardContent className="p-4 text-center">
                    <Icon className="h-8 w-8 mx-auto mb-2 text-muted-foreground" />
                    <div className="font-medium">{label}</div>
                    <div className="text-xs text-muted-foreground">{desc}</div>
                  </CardContent>
                </Card>
              ))}
            </div>
          </div>

          <Separator />

          {/* Data Options */}
          <div className="space-y-4">
            <Label className="text-base font-medium">Include Data</Label>
            <div className="grid grid-cols-2 gap-4">
              <div className="flex items-center space-x-2">
                <Checkbox
                  id="rationale"
                  checked={config.includeRationale}
                  onCheckedChange={(checked) => 
                    setConfig({ ...config, includeRationale: !!checked })
                  }
                />
                <Label htmlFor="rationale" className="text-sm">Analysis rationale</Label>
              </div>
              
              <div className="flex items-center space-x-2">
                <Checkbox
                  id="signals"
                  checked={config.includeSignals}
                  onCheckedChange={(checked) => 
                    setConfig({ ...config, includeSignals: !!checked })
                  }
                />
                <Label htmlFor="signals" className="text-sm">Signal details</Label>
              </div>
              
              <div className="flex items-center space-x-2">
                <Checkbox
                  id="metrics"
                  checked={config.includeMetrics}
                  onCheckedChange={(checked) => 
                    setConfig({ ...config, includeMetrics: !!checked })
                  }
                />
                <Label htmlFor="metrics" className="text-sm">Performance metrics</Label>
              </div>
            </div>
          </div>

          <Separator />

          {/* Filter Options */}
          <div className="space-y-4">
            <Label className="text-base font-medium">Filters</Label>
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label htmlFor="threshold" className="text-sm">Min Confidence (%)</Label>
                <Select
                  value={config.confidenceThreshold.toString()}
                  onValueChange={(value) => 
                    setConfig({ ...config, confidenceThreshold: parseInt(value) })
                  }
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="0">All results (0%)</SelectItem>
                    <SelectItem value="25">Low confidence (25%)</SelectItem>
                    <SelectItem value="50">Medium confidence (50%)</SelectItem>
                    <SelectItem value="75">High confidence (75%)</SelectItem>
                    <SelectItem value="90">Very high (90%)</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              
              <div className="space-y-2">
                <Label htmlFor="maxResults" className="text-sm">Max Results</Label>
                <Select
                  value={config.maxResults.toString()}
                  onValueChange={(value) => 
                    setConfig({ ...config, maxResults: parseInt(value) })
                  }
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="100">100 results</SelectItem>
                    <SelectItem value="500">500 results</SelectItem>
                    <SelectItem value="1000">1,000 results</SelectItem>
                    <SelectItem value="5000">5,000 results</SelectItem>
                    <SelectItem value="-1">All results</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>
          </div>

          {/* Export Summary */}
          <Card className="bg-muted/30">
            <CardContent className="p-4">
              <div className="flex items-center justify-between">
                <div>
                  <div className="font-medium">Export Summary</div>
                  <div className="text-sm text-muted-foreground">
                    {filteredAttributions.length} of {attributions.length} attributions will be exported
                  </div>
                </div>
                <Badge variant="secondary">
                  {config.format.toUpperCase()}
                </Badge>
              </div>
            </CardContent>
          </Card>

          {/* Export Button */}
          <div className="flex justify-end gap-3">
            <Button variant="outline" onClick={() => setConfig(defaultConfig)}>
              Reset
            </Button>
            <Button 
              onClick={exportData} 
              disabled={isExporting || filteredAttributions.length === 0}
              className="min-w-24"
            >
              {isExporting ? "Exporting..." : "Export"}
            </Button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}

// Export helper functions
async function exportAsJSON(data: Attribution[], config: ExportConfig, filename: string) {
  const processedData = data.map(item => ({
    id: item.id,
    attribution_type: item.attribution_type,
    confidence_score: item.confidence_score,
    confidence_level: item.confidence_level,
    similarity_score: item.similarity_score,
    source: item.sources,
    ...(config.includeRationale && { rationale: item.rationale }),
    ...(config.includeSignals && { signals: item.signals }),
    ...(config.includeMetrics && { 
      metrics: {
        gltr: item.signals?.gltr,
        detectgpt: item.signals?.detectgpt,
        semantic: item.signals?.semantic
      }
    })
  }));

  const blob = new Blob([JSON.stringify(processedData, null, 2)], { 
    type: 'application/json' 
  });
  downloadBlob(blob, `${filename}.json`);
}

async function exportAsCSV(data: Attribution[], config: ExportConfig, filename: string) {
  const headers = [
    'ID', 'Type', 'Confidence Score', 'Confidence Level', 'Similarity Score', 'Source Title', 'Source Type'
  ];
  
  if (config.includeRationale) headers.push('Rationale');
  if (config.includeSignals) headers.push('GLTR Score', 'DetectGPT Score', 'Semantic Score');

  const rows = data.map(item => {
    const row = [
      item.id,
      item.attribution_type,
      item.confidence_score,
      item.confidence_level,
      item.similarity_score || '',
      item.sources?.title || '',
      item.sources?.source_type || ''
    ];
    
    if (config.includeRationale) {
      row.push(item.rationale?.join('; ') || '');
    }
    
    if (config.includeSignals) {
      row.push(
        item.signals?.gltr?.tailTokenShare || '',
        item.signals?.detectgpt?.curvature || '',
        item.signals?.semantic?.similarity || ''
      );
    }
    
    return row;
  });

  const csvContent = [headers, ...rows]
    .map(row => row.map(cell => `"${cell}"`).join(','))
    .join('\n');

  const blob = new Blob([csvContent], { type: 'text/csv' });
  downloadBlob(blob, `${filename}.csv`);
}

async function exportAsPDF(data: Attribution[], config: ExportConfig, filename: string) {
  // For now, export as text-based PDF content
  const content = generateReportContent(data, config);
  const blob = new Blob([content], { type: 'text/plain' });
  downloadBlob(blob, `${filename}.txt`);
}

async function exportAsPNG(data: Attribution[], config: ExportConfig, filename: string) {
  // Create a simple visualization canvas
  const canvas = document.createElement('canvas');
  canvas.width = 800;
  canvas.height = 600;
  const ctx = canvas.getContext('2d')!;
  
  // Helpers to use design tokens (HSL vars) in canvas
  const css = getComputedStyle(document.documentElement);
  const hslVar = (name: string, fallback: string) => {
    const val = css.getPropertyValue(name).trim();
    return `hsl(${val || fallback})`;
  };

  // Background
  ctx.fillStyle = hslVar('--card', '0 0% 100%');
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  
  // Title
  ctx.fillStyle = hslVar('--card-foreground', '0 0% 0%');
  ctx.font = '24px Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial';
  ctx.fillText('Attribution Analysis Report', 50, 50);
  
  // Simple bar chart of confidence levels
  const confidenceBuckets = { high: 0, medium: 0, low: 0 };
  data.forEach(item => {
    confidenceBuckets[item.confidence_level.toLowerCase() as keyof typeof confidenceBuckets]++;
  });
  
  const colors = {
    high: hslVar('--destructive', '0 84% 60%'),
    medium: hslVar('--warning', '35 91% 65%'),
    low: hslVar('--success', '142 71% 45%')
  } as const;
  let x = 100;
  
  Object.entries(confidenceBuckets).forEach(([level, count]) => {
    const height = data.length ? (count / data.length) * 300 : 0;
    ctx.fillStyle = colors[level as keyof typeof colors];
    ctx.fillRect(x, 400 - height, 80, height);

    ctx.fillStyle = hslVar('--muted-foreground', '0 0% 0%');
    ctx.font = '14px Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial';
    ctx.fillText(level, x + 10, 420);
    ctx.fillText(count.toString(), x + 10, 440);
    x += 120;
  });
  
  canvas.toBlob((blob) => {
    if (blob) downloadBlob(blob, `${filename}.png`);
  });
}

function generateReportContent(data: Attribution[], config: ExportConfig): string {
  let content = `Attribution Analysis Report\n`;
  content += `Generated: ${new Date().toISOString()}\n`;
  content += `Total Attributions: ${data.length}\n\n`;
  
  data.forEach((item, index) => {
    content += `${index + 1}. ${item.attribution_type.toUpperCase()} Attribution\n`;
    content += `   Confidence: ${(item.confidence_score * 100).toFixed(1)}% (${item.confidence_level})\n`;
    content += `   Source: ${item.sources?.title || 'Unknown'}\n`;
    
    if (config.includeRationale && item.rationale) {
      content += `   Rationale: ${item.rationale.join('; ')}\n`;
    }
    
    content += '\n';
  });
  
  return content;
}

function downloadBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}