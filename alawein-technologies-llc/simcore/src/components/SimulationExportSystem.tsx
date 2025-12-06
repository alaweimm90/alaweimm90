/**
 * Simulation Export System
 * Handles exporting simulation data, results, and configurations
 */

import React, { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Checkbox } from '@/components/ui/checkbox';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Progress } from '@/components/ui/progress';
import { 
  Download, 
  FileText, 
  Image, 
  Database, 
  Share2, 
  Copy,
  CheckCircle,
  ExternalLink
} from 'lucide-react';
import { useToast } from '@/hooks/use-toast';

interface ExportOptions {
  format: 'json' | 'csv' | 'pdf' | 'png' | 'svg';
  includeParameters: boolean;
  includeResults: boolean;
  includeVisualization: boolean;
  includeMetadata: boolean;
  includeAnalysis: boolean;
}

interface SimulationData {
  parameters: Record<string, any>;
  results: any[];
  metadata: Record<string, any>;
  analysis?: any[];
}

interface SimulationExportSystemProps {
  simulationData: SimulationData;
  simulationType: string;
  onExport?: (format: string, data: any) => void;
}

export function SimulationExportSystem({ 
  simulationData, 
  simulationType,
  onExport 
}: SimulationExportSystemProps) {
  const [exportOptions, setExportOptions] = useState<ExportOptions>({
    format: 'json',
    includeParameters: true,
    includeResults: true,
    includeVisualization: false,
    includeMetadata: true,
    includeAnalysis: false
  });
  
  const [isExporting, setIsExporting] = useState(false);
  const [exportProgress, setExportProgress] = useState(0);
  const [shareableLink, setShareableLink] = useState('');
  const [projectName, setProjectName] = useState('');
  const [description, setDescription] = useState('');
  const { toast } = useToast();

  const formatOptions = [
    { value: 'json', label: 'JSON Data', icon: FileText },
    { value: 'csv', label: 'CSV Spreadsheet', icon: Database },
    { value: 'pdf', label: 'PDF Report', icon: FileText },
    { value: 'png', label: 'PNG Image', icon: Image },
    { value: 'svg', label: 'SVG Vector', icon: Image }
  ];

  const updateOption = (key: keyof ExportOptions, value: any) => {
    setExportOptions(prev => ({ ...prev, [key]: value }));
  };

  const generateExportData = () => {
    const exportData: any = {
      timestamp: new Date().toISOString(),
      simulationType,
      projectName: projectName || `${simulationType}-simulation`,
      description
    };

    if (exportOptions.includeParameters) {
      exportData.parameters = simulationData.parameters;
    }

    if (exportOptions.includeResults) {
      exportData.results = simulationData.results;
    }

    if (exportOptions.includeMetadata) {
      exportData.metadata = {
        ...simulationData.metadata,
        exportedBy: 'Physics Simulation Platform',
        exportVersion: '1.0.0'
      };
    }

    if (exportOptions.includeAnalysis && simulationData.analysis) {
      exportData.analysis = simulationData.analysis;
    }

    return exportData;
  };

  const exportAsJSON = (data: any) => {
    const blob = new Blob([JSON.stringify(data, null, 2)], { 
      type: 'application/json' 
    });
    downloadFile(blob, `${data.projectName}.json`);
  };

  const exportAsCSV = (data: any) => {
    // Convert results to CSV format
    if (!data.results || !Array.isArray(data.results)) {
      toast({
        title: "Export Error",
        description: "No tabular data available for CSV export",
        variant: "destructive"
      });
      return;
    }

    const headers = Object.keys(data.results[0] || {});
    const csvContent = [
      headers.join(','),
      ...data.results.map(row => 
        headers.map(header => {
          const value = row[header];
          return typeof value === 'string' ? `"${value}"` : value;
        }).join(',')
      )
    ].join('\n');

    const blob = new Blob([csvContent], { type: 'text/csv' });
    downloadFile(blob, `${data.projectName}.csv`);
  };

  const exportAsPDF = async (data: any) => {
    // Simulate PDF generation
    const pdfContent = `
Physics Simulation Report
========================

Project: ${data.projectName}
Type: ${data.simulationType}
Generated: ${new Date().toLocaleString()}

${data.description ? `Description:\n${data.description}\n\n` : ''}

Parameters:
${Object.entries(data.parameters || {}).map(([key, value]) => 
  `${key}: ${value}`
).join('\n')}

Results Summary:
${data.results ? `${data.results.length} data points collected` : 'No results available'}

${data.analysis ? `Analysis performed with ${data.analysis.length} insights` : ''}
    `;

    const blob = new Blob([pdfContent], { type: 'text/plain' });
    downloadFile(blob, `${data.projectName}.txt`); // Simulated as text file
  };

  const downloadFile = (blob: Blob, filename: string) => {
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const handleExport = async () => {
    setIsExporting(true);
    setExportProgress(0);

    // Simulate export process
    const steps = [
      'Preparing data...',
      'Processing parameters...',
      'Formatting results...',
      'Generating output...',
      'Finalizing export...'
    ];

    for (let i = 0; i < steps.length; i++) {
      await new Promise(resolve => setTimeout(resolve, 300));
      setExportProgress(((i + 1) / steps.length) * 100);
    }

    const exportData = generateExportData();

    try {
      switch (exportOptions.format) {
        case 'json':
          exportAsJSON(exportData);
          break;
        case 'csv':
          exportAsCSV(exportData);
          break;
        case 'pdf':
          await exportAsPDF(exportData);
          break;
        default:
          toast({
            title: "Export Format",
            description: `${exportOptions.format.toUpperCase()} export simulated`,
          });
      }

      onExport?.(exportOptions.format, exportData);
      
      toast({
        title: "Export Complete",
        description: `Simulation data exported as ${exportOptions.format.toUpperCase()}`,
      });
    } catch (error) {
      toast({
        title: "Export Failed",
        description: "There was an error exporting the simulation data",
        variant: "destructive"
      });
    }

    setIsExporting(false);
  };

  const generateShareableLink = () => {
    // Simulate link generation
    const linkId = Math.random().toString(36).substring(2, 15);
    const link = `https://physics-sim.com/shared/${linkId}`;
    setShareableLink(link);
    
    toast({
      title: "Shareable Link Generated",
      description: "Link copied to clipboard",
    });
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
    toast({
      title: "Copied",
      description: "Link copied to clipboard",
    });
  };

  const getEstimatedSize = () => {
    const dataSize = JSON.stringify(generateExportData()).length;
    if (dataSize < 1024) return `${dataSize} B`;
    if (dataSize < 1024 * 1024) return `${(dataSize / 1024).toFixed(1)} KB`;
    return `${(dataSize / (1024 * 1024)).toFixed(1)} MB`;
  };

  return (
    <div className="space-y-6">
      {/* Export Configuration */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Download className="h-5 w-5" />
            Export Configuration
          </CardTitle>
          <CardDescription>
            Configure what data to include in your export
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* Project Details */}
          <div className="space-y-4">
            <div className="grid gap-4 md:grid-cols-2">
              <div className="space-y-2">
                <Label htmlFor="projectName">Project Name</Label>
                <Input
                  id="projectName"
                  value={projectName}
                  onChange={(e) => setProjectName(e.target.value)}
                  placeholder={`${simulationType} Simulation`}
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="format">Export Format</Label>
                <Select value={exportOptions.format} onValueChange={(value: any) => updateOption('format', value)}>
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {formatOptions.map(option => (
                      <SelectItem key={option.value} value={option.value}>
                        <div className="flex items-center gap-2">
                          <option.icon className="h-4 w-4" />
                          {option.label}
                        </div>
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            </div>
            
            <div className="space-y-2">
              <Label htmlFor="description">Description (Optional)</Label>
              <Textarea
                id="description"
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                placeholder="Add a description for this simulation export..."
                rows={3}
              />
            </div>
          </div>

          {/* Data Selection */}
          <div className="space-y-4">
            <h4 className="font-medium">Include in Export</h4>
            <div className="grid gap-4 md:grid-cols-2">
              <div className="flex items-center space-x-2">
                <Checkbox
                  id="parameters"
                  checked={exportOptions.includeParameters}
                  onCheckedChange={(checked) => updateOption('includeParameters', checked)}
                />
                <Label htmlFor="parameters">Simulation Parameters</Label>
              </div>
              
              <div className="flex items-center space-x-2">
                <Checkbox
                  id="results"
                  checked={exportOptions.includeResults}
                  onCheckedChange={(checked) => updateOption('includeResults', checked)}
                />
                <Label htmlFor="results">Simulation Results</Label>
              </div>
              
              <div className="flex items-center space-x-2">
                <Checkbox
                  id="visualization"
                  checked={exportOptions.includeVisualization}
                  onCheckedChange={(checked) => updateOption('includeVisualization', checked)}
                />
                <Label htmlFor="visualization">Visualization Data</Label>
              </div>
              
              <div className="flex items-center space-x-2">
                <Checkbox
                  id="metadata"
                  checked={exportOptions.includeMetadata}
                  onCheckedChange={(checked) => updateOption('includeMetadata', checked)}
                />
                <Label htmlFor="metadata">Metadata & Settings</Label>
              </div>
              
              <div className="flex items-center space-x-2">
                <Checkbox
                  id="analysis"
                  checked={exportOptions.includeAnalysis}
                  onCheckedChange={(checked) => updateOption('includeAnalysis', checked)}
                />
                <Label htmlFor="analysis">ML Analysis Results</Label>
              </div>
            </div>
          </div>

          {/* Export Info */}
          <div className="flex items-center justify-between p-4 bg-muted rounded-lg">
            <div>
              <p className="font-medium">Estimated Export Size</p>
              <p className="text-sm text-muted-foreground">{getEstimatedSize()}</p>
            </div>
            <Badge variant="outline">
              {Object.values(exportOptions).filter(Boolean).length - 1} items selected
            </Badge>
          </div>
        </CardContent>
      </Card>

      {/* Export Actions */}
      <Card>
        <CardContent className="pt-6 space-y-4">
          {isExporting && (
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span>Exporting...</span>
                <span>{Math.round(exportProgress)}%</span>
              </div>
              <Progress value={exportProgress} className="w-full" />
            </div>
          )}
          
          <div className="flex gap-4">
            <Button 
              onClick={handleExport}
              disabled={isExporting}
              className="flex items-center gap-2"
            >
              <Download className="h-4 w-4" />
              {isExporting ? 'Exporting...' : 'Export Data'}
            </Button>
            
            <Button
              onClick={generateShareableLink}
              variant="outline"
              className="flex items-center gap-2"
            >
              <Share2 className="h-4 w-4" />
              Generate Share Link
            </Button>
          </div>

          {shareableLink && (
            <div className="p-4 border rounded-lg space-y-2">
              <div className="flex items-center gap-2">
                <CheckCircle className="h-4 w-4 text-green-600" />
                <span className="font-medium">Shareable Link Generated</span>
              </div>
              <div className="flex items-center gap-2">
                <Input value={shareableLink} readOnly className="flex-1" />
                <Button
                  onClick={() => copyToClipboard(shareableLink)}
                  variant="outline"
                  size="sm"
                >
                  <Copy className="h-4 w-4" />
                </Button>
                <Button
                  onClick={() => window.open(shareableLink, '_blank')}
                  variant="outline"
                  size="sm"
                >
                  <ExternalLink className="h-4 w-4" />
                </Button>
              </div>
              <p className="text-sm text-muted-foreground">
                This link allows others to view and download your simulation data
              </p>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}

export default SimulationExportSystem;