import React, { useState, useRef } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Textarea } from '@/components/ui/textarea';
import { Separator } from '@/components/ui/separator';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { 
  Download, 
  FileText, 
  Image, 
  Camera, 
  Database,
  Settings,
  Share2,
  Copy,
  CheckCircle
} from 'lucide-react';
import { useToast } from '@/hooks/use-toast';
import html2canvas from 'html2canvas';

interface ExportData {
  type: 'csv' | 'json' | 'pdf' | 'png' | 'svg';
  filename: string;
  data: any;
  metadata?: {
    timestamp: string;
    source: string;
    parameters: Record<string, any>;
    description?: string;
  };
}

interface ReportTemplate {
  id: string;
  name: string;
  description: string;
  sections: string[];
}

const reportTemplates: ReportTemplate[] = [
  {
    id: 'basic',
    name: 'Basic Report',
    description: 'Simple report with data and basic analysis',
    sections: ['title', 'parameters', 'data', 'conclusion']
  },
  {
    id: 'academic',
    name: 'Academic Paper',
    description: 'Formal academic structure with methodology and references',
    sections: ['abstract', 'introduction', 'methodology', 'results', 'discussion', 'references']
  },
  {
    id: 'presentation',
    name: 'Presentation Summary',
    description: 'Concise format suitable for presentations',
    sections: ['executive_summary', 'key_findings', 'visualizations', 'recommendations']
  }
];

export function ScientificDataExport() {
  const [exportFormat, setExportFormat] = useState<'csv' | 'json' | 'pdf'>('csv');
  const [filename, setFilename] = useState('simulation_data');
  const [includeMetadata, setIncludeMetadata] = useState(true);
  const [selectedTemplate, setSelectedTemplate] = useState<ReportTemplate>(reportTemplates[0]);
  const [reportTitle, setReportTitle] = useState('Physics Simulation Analysis');
  const [reportDescription, setReportDescription] = useState('');
  const [isExporting, setIsExporting] = useState(false);
  const captureRef = useRef<HTMLDivElement>(null);
  const { toast } = useToast();

  // Sample simulation data
  const simulationData = [
    { time: 0, position: 0, velocity: 0, energy: 100 },
    { time: 1, position: 2.5, velocity: 5, energy: 98.5 },
    { time: 2, position: 7.5, velocity: 7.5, energy: 95.2 },
    { time: 3, position: 15, velocity: 10, energy: 90.1 },
    { time: 4, position: 25, velocity: 12.5, energy: 83.2 }
  ];

  const exportToCSV = (data: any[], filename: string) => {
    if (!data.length) return;
    
    const headers = Object.keys(data[0]);
    const csvContent = [
      headers.join(','),
      ...data.map(row => headers.map(header => row[header]).join(','))
    ].join('\n');

    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${filename}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const exportToJSON = (data: any, filename: string) => {
    const exportData: ExportData = {
      type: 'json',
      filename,
      data,
      metadata: includeMetadata ? {
        timestamp: new Date().toISOString(),
        source: 'SimCore Scientific Computing Platform',
        parameters: {
          simulationType: 'quantum_tunneling',
          timeStep: 0.1,
          iterations: 1000
        },
        description: reportDescription || 'Simulation data export'
      } : undefined
    };

    const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${filename}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const captureScreenshot = async () => {
    if (!captureRef.current) return;

    try {
      setIsExporting(true);
      const canvas = await html2canvas(captureRef.current, {
        backgroundColor: '#ffffff',
        scale: 2,
        useCORS: true
      });
      
      canvas.toBlob((blob) => {
        if (blob) {
          const url = URL.createObjectURL(blob);
          const a = document.createElement('a');
          a.href = url;
          a.download = `${filename}_screenshot.png`;
          a.click();
          URL.revokeObjectURL(url);
          
          toast({
            title: "Screenshot Captured!",
            description: "High-resolution screenshot saved successfully.",
          });
        }
      });
    } catch (error) {
      toast({
        title: "Export Failed",
        description: "Could not capture screenshot.",
        variant: "destructive",
      });
    } finally {
      setIsExporting(false);
    }
  };

  const generateReport = () => {
    const reportContent = `
# ${reportTitle}

**Generated:** ${new Date().toLocaleString()}
**Template:** ${selectedTemplate.name}

## Description
${reportDescription || 'No description provided'}

## Simulation Parameters
- Simulation Type: Quantum Tunneling
- Time Step: 0.1s
- Total Time: 4s
- Data Points: ${simulationData.length}

## Data Summary
- Initial Energy: ${simulationData[0]?.energy}J
- Final Energy: ${simulationData[simulationData.length - 1]?.energy}J
- Max Velocity: ${Math.max(...simulationData.map(d => d.velocity))}m/s

## Raw Data
${simulationData.map(d => `Time: ${d.time}s, Position: ${d.position}m, Velocity: ${d.velocity}m/s, Energy: ${d.energy}J`).join('\n')}

## Conclusions
Energy is conserved within expected tolerances. The simulation demonstrates classical behavior in the macroscopic limit.
    `.trim();

    const blob = new Blob([reportContent], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${filename}_report.md`;
    a.click();
    URL.revokeObjectURL(url);

    toast({
      title: "Report Generated!",
      description: "Scientific report saved in Markdown format.",
    });
  };

  const handleExport = () => {
    setIsExporting(true);
    
    setTimeout(() => {
      try {
        switch (exportFormat) {
          case 'csv':
            exportToCSV(simulationData, filename);
            break;
          case 'json':
            exportToJSON(simulationData, filename);
            break;
          case 'pdf':
            generateReport();
            break;
        }
        
        toast({
          title: "Export Complete!",
          description: `Data exported as ${exportFormat.toUpperCase()} successfully.`,
        });
      } catch (error) {
        toast({
          title: "Export Failed",
          description: "Could not export data. Please try again.",
          variant: "destructive",
        });
      } finally {
        setIsExporting(false);
      }
    }, 1000);
  };

  const copyDataToClipboard = async () => {
    try {
      const dataString = JSON.stringify(simulationData, null, 2);
      await navigator.clipboard.writeText(dataString);
      toast({
        title: "Copied!",
        description: "Simulation data copied to clipboard.",
      });
    } catch (error) {
      toast({
        title: "Copy Failed",
        description: "Could not copy data to clipboard.",
        variant: "destructive",
      });
    }
  };

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Download className="h-5 w-5" />
            Scientific Data Export
          </CardTitle>
        </CardHeader>
        <CardContent>
          <Tabs defaultValue="data" className="w-full">
            <TabsList className="grid w-full grid-cols-3">
              <TabsTrigger value="data">Data Export</TabsTrigger>
              <TabsTrigger value="visualization">Visualization</TabsTrigger>
              <TabsTrigger value="report">Report Generation</TabsTrigger>
            </TabsList>

            <TabsContent value="data" className="space-y-6">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="space-y-4">
                  <div>
                    <Label htmlFor="filename">Filename</Label>
                    <Input
                      id="filename"
                      value={filename}
                      onChange={(e) => setFilename(e.target.value)}
                      placeholder="Enter filename"
                    />
                  </div>

                  <div>
                    <Label htmlFor="format">Export Format</Label>
                    <Select value={exportFormat} onValueChange={(value: any) => setExportFormat(value)}>
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="csv">CSV (Comma Separated)</SelectItem>
                        <SelectItem value="json">JSON (JavaScript Object)</SelectItem>
                        <SelectItem value="pdf">PDF Report</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>

                  <div className="flex items-center space-x-2">
                    <input
                      type="checkbox"
                      id="metadata"
                      checked={includeMetadata}
                      onChange={(e) => setIncludeMetadata(e.target.checked)}
                      className="rounded border-gray-300"
                    />
                    <Label htmlFor="metadata">Include metadata and parameters</Label>
                  </div>
                </div>

                <div className="space-y-4">
                  <div>
                    <Label>Data Preview</Label>
                    <div className="bg-muted/50 rounded-lg p-3 text-xs font-mono">
                      <div className="space-y-1">
                        {simulationData.slice(0, 3).map((row, i) => (
                          <div key={i} className="text-muted-foreground">
                            {JSON.stringify(row)}
                          </div>
                        ))}
                        <div className="text-muted-foreground">... and {simulationData.length - 3} more rows</div>
                      </div>
                    </div>
                  </div>

                  <div className="flex flex-col gap-2">
                    <Button onClick={handleExport} disabled={isExporting} className="w-full">
                      {isExporting ? "Exporting..." : "Export Data"}
                      <Download className="h-4 w-4 ml-2" />
                    </Button>
                    
                    <Button variant="outline" onClick={copyDataToClipboard} className="w-full">
                      <Copy className="h-4 w-4 mr-2" />
                      Copy to Clipboard
                    </Button>
                  </div>
                </div>
              </div>
            </TabsContent>

            <TabsContent value="visualization" className="space-y-6">
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <div className="space-y-4">
                  <h3 className="font-semibold">Screenshot Capture</h3>
                  <p className="text-sm text-muted-foreground">
                    Capture high-resolution screenshots of your simulations and visualizations.
                  </p>
                  
                  <div className="space-y-3">
                    <Button onClick={captureScreenshot} disabled={isExporting} className="w-full">
                      <Camera className="h-4 w-4 mr-2" />
                      {isExporting ? "Capturing..." : "Capture Screenshot"}
                    </Button>
                    
                    <div className="text-xs text-muted-foreground space-y-1">
                      <div>• High resolution (2x scale)</div>
                      <div>• PNG format with transparency</div>
                      <div>• Includes all visible elements</div>
                    </div>
                  </div>
                </div>

                <div ref={captureRef} className="border rounded-lg p-4 bg-gradient-to-br from-blue-50 to-purple-50 dark:from-blue-950/20 dark:to-purple-950/20">
                  <h4 className="font-semibold mb-3">Sample Visualization</h4>
                  <div className="space-y-2">
                    <div className="h-24 bg-gradient-to-r from-blue-200 to-purple-200 dark:from-blue-800 dark:to-purple-800 rounded opacity-75"></div>
                    <div className="flex justify-between text-xs text-muted-foreground">
                      <span>Energy: 83.2J</span>
                      <span>Time: 4.0s</span>
                      <span>Position: 25m</span>
                    </div>
                  </div>
                </div>
              </div>
            </TabsContent>

            <TabsContent value="report" className="space-y-6">
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <div className="space-y-4">
                  <div>
                    <Label htmlFor="title">Report Title</Label>
                    <Input
                      id="title"
                      value={reportTitle}
                      onChange={(e) => setReportTitle(e.target.value)}
                    />
                  </div>

                  <div>
                    <Label htmlFor="description">Description</Label>
                    <Textarea
                      id="description"
                      value={reportDescription}
                      onChange={(e) => setReportDescription(e.target.value)}
                      placeholder="Describe your simulation and findings..."
                      rows={3}
                    />
                  </div>

                  <div>
                    <Label>Report Template</Label>
                    <div className="space-y-2 mt-2">
                      {reportTemplates.map((template) => (
                        <Card 
                          key={template.id}
                          className={`cursor-pointer transition-all ${
                            selectedTemplate.id === template.id ? 'ring-2 ring-primary' : ''
                          }`}
                          onClick={() => setSelectedTemplate(template)}
                        >
                          <CardContent className="p-3">
                            <div className="flex items-center justify-between">
                              <div>
                                <h4 className="font-medium">{template.name}</h4>
                                <p className="text-xs text-muted-foreground">{template.description}</p>
                              </div>
                              <Badge variant="outline">{template.sections.length} sections</Badge>
                            </div>
                          </CardContent>
                        </Card>
                      ))}
                    </div>
                  </div>
                </div>

                <div className="space-y-4">
                  <div>
                    <Label>Report Sections</Label>
                    <div className="space-y-2 mt-2">
                      {selectedTemplate.sections.map((section, index) => (
                        <div key={section} className="flex items-center gap-2 text-sm">
                          <CheckCircle className="h-4 w-4 text-green-500" />
                          <span className="capitalize">{section.replace('_', ' ')}</span>
                        </div>
                      ))}
                    </div>
                  </div>

                  <Button onClick={generateReport} className="w-full">
                    <FileText className="h-4 w-4 mr-2" />
                    Generate Report
                  </Button>
                </div>
              </div>
            </TabsContent>
          </Tabs>
        </CardContent>
      </Card>
    </div>
  );
}