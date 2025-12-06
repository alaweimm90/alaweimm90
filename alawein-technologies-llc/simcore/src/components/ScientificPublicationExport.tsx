import React, { useState, useMemo } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Checkbox } from '@/components/ui/checkbox';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';
import { Progress } from '@/components/ui/progress';
import { 
  Download, 
  FileText, 
  Image, 
  Video,
  BookOpen,
  Zap,
  CheckCircle,
  Clock,
  Share2,
  Settings,
  Info
} from 'lucide-react';
import { useToast } from '@/hooks/use-toast';
import { semanticColors } from '@/theme/tokens';
import { cn } from '@/lib/utils';

interface PublicationFormat {
  id: string;
  name: string;
  description: string;
  icon: React.ElementType;
  fileExtension: string;
  features: string[];
  requirements: string[];
  qualityLevels: string[];
}

interface ExportJob {
  id: string;
  title: string;
  format: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  progress: number;
  createdAt: Date;
  downloadUrl?: string;
  fileSize?: number;
}

interface PublicationMetadata {
  title: string;
  authors: string[];
  abstract: string;
  keywords: string[];
  doi?: string;
  journal?: string;
  institution: string;
  correspondingAuthor: string;
  license: string;
  acknowledgments: string;
}

const PUBLICATION_FORMATS: PublicationFormat[] = [
  {
    id: 'latex',
    name: 'LaTeX Document',
    description: 'Complete LaTeX manuscript with figures',
    icon: FileText,
    fileExtension: 'tex',
    features: ['High-quality equations', 'Vector graphics', 'Bibliography support', 'Journal templates'],
    requirements: ['LaTeX installation required'],
    qualityLevels: ['draft', 'preprint', 'journal-ready']
  },
  {
    id: 'pdf',
    name: 'PDF Report',
    description: 'Professional PDF with embedded plots',
    icon: FileText,
    fileExtension: 'pdf',
    features: ['Self-contained', 'High-resolution figures', 'Searchable text', 'Print-ready'],
    requirements: ['None'],
    qualityLevels: ['low', 'medium', 'high', 'print']
  },
  {
    id: 'jupyter',
    name: 'Jupyter Notebook',
    description: 'Interactive notebook with code and results',
    icon: BookOpen,
    fileExtension: 'ipynb',
    features: ['Executable code', 'Interactive plots', 'Markdown cells', 'Version control friendly'],
    requirements: ['Jupyter installation'],
    qualityLevels: ['basic', 'enhanced', 'presentation']
  },
  {
    id: 'supplementary',
    name: 'Supplementary Package',
    description: 'Complete data package with analysis',
    icon: Download,
    fileExtension: 'zip',
    features: ['Raw data', 'Analysis scripts', 'Figures', 'Documentation'],
    requirements: ['None'],
    qualityLevels: ['minimal', 'standard', 'comprehensive']
  }
];

const JOURNAL_TEMPLATES = [
  'Nature Physics',
  'Physical Review Letters',
  'Science',
  'Nature Communications',
  'Physical Review B',
  'New Journal of Physics',
  'Scientific Reports',
  'ACS Nano',
  'Advanced Materials',
  'Custom Template'
];

interface ScientificPublicationExportProps {
  simulationData?: any;
  plotData?: any;
  parameters?: Record<string, any>;
  className?: string;
}

export const ScientificPublicationExport: React.FC<ScientificPublicationExportProps> = ({
  simulationData,
  plotData,
  parameters,
  className
}) => {
  const { toast } = useToast();
  
  const [selectedFormat, setSelectedFormat] = useState<string>('pdf');
  const [qualityLevel, setQualityLevel] = useState<string>('high');
  const [journalTemplate, setJournalTemplate] = useState<string>('Physical Review Letters');
  const [exportJobs, setExportJobs] = useState<ExportJob[]>([]);
  
  const [metadata, setMetadata] = useState<PublicationMetadata>({
    title: 'Physics Simulation Results from SimCore',
    authors: [''],
    abstract: '',
    keywords: [],
    institution: '',
    correspondingAuthor: '',
    license: 'CC BY 4.0',
    acknowledgments: ''
  });

  const [exportOptions, setExportOptions] = useState({
    includeCode: true,
    includeRawData: false,
    includePlots: true,
    includeParameters: true,
    includeMethodology: true,
    includeReferences: true,
    figureDPI: 300,
    colorScheme: 'publication'
  });

  const selectedFormatInfo = PUBLICATION_FORMATS.find(f => f.id === selectedFormat);

  // Estimate export complexity and time
  const exportEstimate = useMemo(() => {
    let complexity = 1;
    let estimatedTime = 10; // base seconds

    if (exportOptions.includeCode) complexity += 0.5;
    if (exportOptions.includeRawData) complexity += 1;
    if (exportOptions.includePlots) complexity += 0.8;
    if (exportOptions.figureDPI > 150) complexity += 0.3;

    const qualityMultiplier = {
      draft: 0.5,
      low: 0.5,
      basic: 0.5,
      minimal: 0.5,
      preprint: 0.8,
      medium: 0.8,
      enhanced: 0.8,
      standard: 0.8,
      'journal-ready': 1.2,
      high: 1.2,
      presentation: 1.2,
      comprehensive: 1.5,
      print: 1.5
    }[qualityLevel] || 1;

    estimatedTime *= complexity * qualityMultiplier;

    return {
      complexity: Math.round(complexity * 10) / 10,
      time: Math.ceil(estimatedTime),
      fileSize: Math.round(complexity * qualityMultiplier * 2.5) // MB
    };
  }, [selectedFormat, qualityLevel, exportOptions]);

  const startExport = async () => {
    if (!metadata.title.trim()) {
      toast({
        title: "Title Required",
        description: "Please provide a title for your publication",
        variant: "destructive"
      });
      return;
    }

    if (metadata.authors.filter(a => a.trim()).length === 0) {
      toast({
        title: "Author Required",
        description: "Please add at least one author",
        variant: "destructive"
      });
      return;
    }

    const jobId = `pub_${Date.now()}`;
    const newJob: ExportJob = {
      id: jobId,
      title: metadata.title,
      format: selectedFormatInfo?.name || selectedFormat,
      status: 'pending',
      progress: 0,
      createdAt: new Date()
    };

    setExportJobs(prev => [newJob, ...prev]);

    toast({
      title: "Publication Export Started",
      description: `Generating ${selectedFormatInfo?.name} publication...`
    });

    // Simulate publication generation process
    const updateProgress = (progress: number, status: ExportJob['status'] = 'processing') => {
      setExportJobs(prev => prev.map(job => 
        job.id === jobId 
          ? { 
              ...job, 
              progress, 
              status,
              ...(status === 'completed' ? { 
                downloadUrl: `#download-${jobId}`,
                fileSize: exportEstimate.fileSize * 1024 * 1024
              } : {})
            }
          : job
      ));
    };

    // Simulate processing stages
    setTimeout(() => updateProgress(15), 1000);
    setTimeout(() => updateProgress(35), 2000);
    setTimeout(() => updateProgress(55), 3500);
    setTimeout(() => updateProgress(75), 5000);
    setTimeout(() => updateProgress(90), 6500);
    setTimeout(() => updateProgress(100, 'completed'), exportEstimate.time * 1000);

    setTimeout(() => {
      toast({
        title: "Publication Ready",
        description: `"${metadata.title}" is ready for download`,
        action: (
          <Button 
            variant="outline" 
            size="sm" 
            onClick={() => downloadPublication(newJob)}
          >
            Download
          </Button>
        )
      });
    }, exportEstimate.time * 1000 + 500);
  };

  const downloadPublication = (job: ExportJob) => {
    // Generate publication content
    const publicationData = {
      metadata,
      simulationData: exportOptions.includeRawData ? simulationData : null,
      plotData: exportOptions.includePlots ? plotData : null,
      parameters: exportOptions.includeParameters ? parameters : null,
      exportOptions,
      format: selectedFormat,
      qualityLevel,
      journalTemplate,
      generatedAt: new Date().toISOString()
    };

    const content = generatePublicationContent(publicationData);
    const blob = new Blob([content], { 
      type: selectedFormat === 'pdf' ? 'application/pdf' : 'text/plain' 
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${metadata.title.replace(/[^a-zA-Z0-9]/g, '_')}.${selectedFormatInfo?.fileExtension}`;
    a.click();
    URL.revokeObjectURL(url);

    toast({
      title: "Download Started",
      description: `Downloading ${selectedFormatInfo?.name}`
    });
  };

  const generatePublicationContent = (data: any) => {
    // Generate different formats based on selection
    switch (selectedFormat) {
      case 'latex':
        return generateLaTeXContent(data);
      case 'jupyter':
        return generateJupyterContent(data);
      case 'supplementary':
        return JSON.stringify(data, null, 2);
      default:
        return generateMarkdownContent(data);
    }
  };

  const generateLaTeXContent = (data: any) => {
    return `\\documentclass[aps,prl,reprint]{revtex4-2}
\\usepackage{graphicx}
\\usepackage{amsmath}
\\usepackage{hyperref}

\\begin{document}

\\title{${data.metadata.title}}
\\author{${data.metadata.authors.filter((a: string) => a.trim()).join(', ')}}
\\affiliation{${data.metadata.institution}}

\\begin{abstract}
${data.metadata.abstract}
\\end{abstract}

\\keywords{${data.metadata.keywords.join(', ')}}

\\maketitle

\\section{Introduction}
This document presents simulation results generated using SimCore, a comprehensive physics simulation platform.

\\section{Methodology}
${data.exportOptions.includeMethodology ? 'The simulation was performed using state-of-the-art numerical methods implemented in SimCore.' : ''}

\\section{Results}
${data.exportOptions.includePlots ? '% Figures would be inserted here' : ''}

\\section{Conclusion}
The simulation results demonstrate the effectiveness of the computational approach.

${data.metadata.acknowledgments ? `\\section{Acknowledgments}
${data.metadata.acknowledgments}` : ''}

\\end{document}`;
  };

  const generateJupyterContent = (data: any) => {
    return JSON.stringify({
      cells: [
        {
          cell_type: "markdown",
          metadata: {},
          source: [`# ${data.metadata.title}\n\n**Authors:** ${data.metadata.authors.join(', ')}\n\n**Abstract:** ${data.metadata.abstract}`]
        },
        {
          cell_type: "code",
          execution_count: 1,
          metadata: {},
          outputs: [],
          source: ["# SimCore Physics Simulation Results\nimport numpy as np\nimport matplotlib.pyplot as plt\n\n# Simulation parameters\nparameters = ", JSON.stringify(data.parameters, null, 2)]
        }
      ],
      metadata: {
        kernelspec: {
          display_name: "Python 3",
          language: "python",
          name: "python3"
        }
      },
      nbformat: 4,
      nbformat_minor: 4
    }, null, 2);
  };

  const generateMarkdownContent = (data: any) => {
    return `# ${data.metadata.title}

**Authors:** ${data.metadata.authors.filter((a: string) => a.trim()).join(', ')}
**Institution:** ${data.metadata.institution}
**Generated:** ${new Date().toLocaleDateString()}

## Abstract
${data.metadata.abstract}

## Keywords
${data.metadata.keywords.join(', ')}

## Simulation Parameters
${data.exportOptions.includeParameters ? JSON.stringify(data.parameters, null, 2) : 'Parameters excluded from this export.'}

## Results
${data.exportOptions.includePlots ? 'Plots and visualizations would be embedded here.' : ''}

${data.exportOptions.includeCode ? '## Code\nSimulation code and analysis scripts would be included here.' : ''}

${data.metadata.acknowledgments ? `## Acknowledgments\n${data.metadata.acknowledgments}` : ''}

---
*Generated using SimCore Physics Simulation Platform*`;
  };

  const addAuthor = () => {
    setMetadata(prev => ({
      ...prev,
      authors: [...prev.authors, '']
    }));
  };

  const updateAuthor = (index: number, value: string) => {
    setMetadata(prev => ({
      ...prev,
      authors: prev.authors.map((author, i) => i === index ? value : author)
    }));
  };

  const removeAuthor = (index: number) => {
    setMetadata(prev => ({
      ...prev,
      authors: prev.authors.filter((_, i) => i !== index)
    }));
  };

  const addKeyword = (keyword: string) => {
    if (keyword.trim() && !metadata.keywords.includes(keyword.trim())) {
      setMetadata(prev => ({
        ...prev,
        keywords: [...prev.keywords, keyword.trim()]
      }));
    }
  };

  const removeKeyword = (keyword: string) => {
    setMetadata(prev => ({
      ...prev,
      keywords: prev.keywords.filter(k => k !== keyword)
    }));
  };

  return (
    <div className={cn('w-full space-y-6', className)}>
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <BookOpen className="w-5 h-5" />
            Scientific Publication Export
          </CardTitle>
          <CardDescription>
            Generate publication-ready documents from your physics simulations
          </CardDescription>
        </CardHeader>
        <CardContent>
          <Tabs defaultValue="format" className="w-full">
            <TabsList className="grid w-full grid-cols-4">
              <TabsTrigger value="format">Format</TabsTrigger>
              <TabsTrigger value="metadata">Metadata</TabsTrigger>
              <TabsTrigger value="options">Options</TabsTrigger>
              <TabsTrigger value="queue">Export Queue</TabsTrigger>
            </TabsList>

            <TabsContent value="format" className="space-y-6">
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <div className="space-y-4">
                  <div>
                    <Label>Publication Format</Label>
                    <Select value={selectedFormat} onValueChange={setSelectedFormat}>
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        {PUBLICATION_FORMATS.map(format => {
                          const Icon = format.icon;
                          return (
                            <SelectItem key={format.id} value={format.id}>
                              <div className="flex items-center gap-2">
                                <Icon className="w-4 h-4" />
                                <span>{format.name}</span>
                              </div>
                            </SelectItem>
                          );
                        })}
                      </SelectContent>
                    </Select>
                  </div>

                  <div>
                    <Label>Quality Level</Label>
                    <Select value={qualityLevel} onValueChange={setQualityLevel}>
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        {selectedFormatInfo?.qualityLevels.map(level => (
                          <SelectItem key={level} value={level}>
                            {level.charAt(0).toUpperCase() + level.slice(1).replace('-', ' ')}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>

                  {selectedFormat === 'latex' && (
                    <div>
                      <Label>Journal Template</Label>
                      <Select value={journalTemplate} onValueChange={setJournalTemplate}>
                        <SelectTrigger>
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          {JOURNAL_TEMPLATES.map(journal => (
                            <SelectItem key={journal} value={journal}>
                              {journal}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>
                  )}
                </div>

                {selectedFormatInfo && (
                  <Card>
                    <CardHeader>
                      <CardTitle className="text-lg flex items-center gap-2">
                        <selectedFormatInfo.icon className="w-5 h-5" />
                        {selectedFormatInfo.name}
                      </CardTitle>
                      <CardDescription>
                        {selectedFormatInfo.description}
                      </CardDescription>
                    </CardHeader>
                    <CardContent className="space-y-4">
                      <div>
                        <h4 className="font-medium mb-2">Features</h4>
                        <div className="flex flex-wrap gap-2">
                          {selectedFormatInfo.features.map(feature => (
                            <Badge key={feature} variant="outline" className="text-xs">
                              {feature}
                            </Badge>
                          ))}
                        </div>
                      </div>
                      
                      <div>
                        <h4 className="font-medium mb-2">Requirements</h4>
                        <ul className="text-sm text-textSecondary">
                          {selectedFormatInfo.requirements.map(req => (
                            <li key={req}>• {req}</li>
                          ))}
                        </ul>
                      </div>

                      <div>
                        <h4 className="font-medium mb-2">Export Estimate</h4>
                        <div className="space-y-2 text-sm">
                          <div className="flex justify-between">
                            <span>Processing time:</span>
                            <Badge variant="outline">{exportEstimate.time}s</Badge>
                          </div>
                          <div className="flex justify-between">
                            <span>Estimated size:</span>
                            <Badge variant="outline">{exportEstimate.fileSize}MB</Badge>
                          </div>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                )}
              </div>
            </TabsContent>

            <TabsContent value="metadata" className="space-y-6">
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <div className="space-y-4">
                  <div>
                    <Label htmlFor="title">Publication Title *</Label>
                    <Input
                      id="title"
                      value={metadata.title}
                      onChange={(e) => setMetadata(prev => ({ ...prev, title: e.target.value }))}
                      placeholder="Enter publication title"
                    />
                  </div>

                  <div>
                    <Label>Authors *</Label>
                    <div className="space-y-2">
                      {metadata.authors.map((author, index) => (
                        <div key={index} className="flex gap-2">
                          <Input
                            value={author}
                            onChange={(e) => updateAuthor(index, e.target.value)}
                            placeholder={`Author ${index + 1}`}
                          />
                          {metadata.authors.length > 1 && (
                            <Button
                              variant="outline"
                              size="sm"
                              onClick={() => removeAuthor(index)}
                            >
                              Remove
                            </Button>
                          )}
                        </div>
                      ))}
                      <Button variant="outline" onClick={addAuthor}>
                        Add Author
                      </Button>
                    </div>
                  </div>

                  <div>
                    <Label htmlFor="institution">Institution</Label>
                    <Input
                      id="institution"
                      value={metadata.institution}
                      onChange={(e) => setMetadata(prev => ({ ...prev, institution: e.target.value }))}
                      placeholder="Research institution or university"
                    />
                  </div>

                  <div>
                    <Label htmlFor="corresponding">Corresponding Author</Label>
                    <Input
                      id="corresponding"
                      value={metadata.correspondingAuthor}
                      onChange={(e) => setMetadata(prev => ({ ...prev, correspondingAuthor: e.target.value }))}
                      placeholder="Email of corresponding author"
                    />
                  </div>
                </div>

                <div className="space-y-4">
                  <div>
                    <Label htmlFor="abstract">Abstract</Label>
                    <Textarea
                      id="abstract"
                      value={metadata.abstract}
                      onChange={(e) => setMetadata(prev => ({ ...prev, abstract: e.target.value }))}
                      placeholder="Brief summary of the research and findings..."
                      rows={6}
                    />
                  </div>

                  <div>
                    <Label>Keywords</Label>
                    <div className="flex flex-wrap gap-2 mb-2">
                      {metadata.keywords.map(keyword => (
                        <Badge key={keyword} variant="secondary" className="gap-1">
                          {keyword}
                          <button
                            onClick={() => removeKeyword(keyword)}
                            className="ml-1 hover:bg-destructive hover:text-destructive-foreground rounded-full"
                          >
                            ×
                          </button>
                        </Badge>
                      ))}
                    </div>
                    <Input
                      placeholder="Add keyword and press Enter"
                      onKeyDown={(e) => {
                        if (e.key === 'Enter') {
                          addKeyword(e.currentTarget.value);
                          e.currentTarget.value = '';
                        }
                      }}
                    />
                  </div>

                  <div>
                    <Label htmlFor="acknowledgments">Acknowledgments</Label>
                    <Textarea
                      id="acknowledgments"
                      value={metadata.acknowledgments}
                      onChange={(e) => setMetadata(prev => ({ ...prev, acknowledgments: e.target.value }))}
                      placeholder="Funding sources, collaborators, etc."
                      rows={3}
                    />
                  </div>
                </div>
              </div>
            </TabsContent>

            <TabsContent value="options" className="space-y-6">
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <Card>
                  <CardHeader>
                    <CardTitle>Content Options</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="flex items-center space-x-2">
                      <Checkbox
                        id="include-plots"
                        checked={exportOptions.includePlots}
                        onCheckedChange={(checked) => 
                          setExportOptions(prev => ({ ...prev, includePlots: !!checked }))
                        }
                      />
                      <Label htmlFor="include-plots">Include plots and visualizations</Label>
                    </div>

                    <div className="flex items-center space-x-2">
                      <Checkbox
                        id="include-parameters"
                        checked={exportOptions.includeParameters}
                        onCheckedChange={(checked) => 
                          setExportOptions(prev => ({ ...prev, includeParameters: !!checked }))
                        }
                      />
                      <Label htmlFor="include-parameters">Include simulation parameters</Label>
                    </div>

                    <div className="flex items-center space-x-2">
                      <Checkbox
                        id="include-methodology"
                        checked={exportOptions.includeMethodology}
                        onCheckedChange={(checked) => 
                          setExportOptions(prev => ({ ...prev, includeMethodology: !!checked }))
                        }
                      />
                      <Label htmlFor="include-methodology">Include methodology section</Label>
                    </div>

                    <div className="flex items-center space-x-2">
                      <Checkbox
                        id="include-code"
                        checked={exportOptions.includeCode}
                        onCheckedChange={(checked) => 
                          setExportOptions(prev => ({ ...prev, includeCode: !!checked }))
                        }
                      />
                      <Label htmlFor="include-code">Include analysis code</Label>
                    </div>

                    <div className="flex items-center space-x-2">
                      <Checkbox
                        id="include-raw-data"
                        checked={exportOptions.includeRawData}
                        onCheckedChange={(checked) => 
                          setExportOptions(prev => ({ ...prev, includeRawData: !!checked }))
                        }
                      />
                      <Label htmlFor="include-raw-data">Include raw data</Label>
                    </div>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader>
                    <CardTitle>Quality Settings</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div>
                      <Label htmlFor="figure-dpi">Figure DPI: {exportOptions.figureDPI}</Label>
                      <input
                        type="range"
                        id="figure-dpi"
                        min="150"
                        max="600"
                        step="50"
                        value={exportOptions.figureDPI}
                        onChange={(e) => setExportOptions(prev => ({ 
                          ...prev, 
                          figureDPI: parseInt(e.target.value) 
                        }))}
                        className="w-full"
                      />
                      <div className="flex justify-between text-xs text-textSecondary">
                        <span>Web (150)</span>
                        <span>Print (300)</span>
                        <span>High (600)</span>
                      </div>
                    </div>

                    <div>
                      <Label>Color Scheme</Label>
                      <Select 
                        value={exportOptions.colorScheme} 
                        onValueChange={(value) => setExportOptions(prev => ({ 
                          ...prev, 
                          colorScheme: value 
                        }))}
                      >
                        <SelectTrigger>
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="publication">Publication (B&W friendly)</SelectItem>
                          <SelectItem value="presentation">Presentation (High contrast)</SelectItem>
                          <SelectItem value="web">Web (Colorful)</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                  </CardContent>
                </Card>
              </div>

              <div className="flex justify-center">
                <Button 
                  onClick={startExport} 
                  size="lg" 
                  className="gap-2"
                  disabled={!metadata.title.trim() || metadata.authors.filter(a => a.trim()).length === 0}
                >
                  <Zap className="w-4 h-4" />
                  Generate Publication
                </Button>
              </div>
            </TabsContent>

            <TabsContent value="queue" className="space-y-4">
              {exportJobs.length === 0 ? (
                <Card>
                  <CardContent className="flex items-center justify-center h-32">
                    <div className="text-center text-textSecondary">
                      <Clock className="w-8 h-8 mx-auto mb-2 opacity-50" />
                      <p>No publications in queue</p>
                    </div>
                  </CardContent>
                </Card>
              ) : (
                exportJobs.map(job => (
                  <Card key={job.id}>
                    <CardContent className="p-4">
                      <div className="flex items-center justify-between mb-3">
                        <div>
                          <h4 className="font-medium">{job.title}</h4>
                          <p className="text-sm text-textSecondary">
                            {job.format} • {job.createdAt.toLocaleString()}
                          </p>
                        </div>
                        <Badge variant={
                          job.status === 'completed' ? 'default' :
                          job.status === 'failed' ? 'destructive' :
                          'secondary'
                        }>
                          {job.status === 'completed' && <CheckCircle className="w-3 h-3 mr-1" />}
                          {job.status}
                        </Badge>
                      </div>
                      
                      {job.status === 'processing' && (
                        <div className="mb-3">
                          <Progress value={job.progress} className="h-2" />
                          <p className="text-xs text-textSecondary mt-1">
                            {job.progress}% complete
                          </p>
                        </div>
                      )}

                      {job.status === 'completed' && (
                        <div className="flex items-center justify-between">
                          <span className="text-sm text-textSecondary">
                            {job.fileSize && `${(job.fileSize / 1024 / 1024).toFixed(1)} MB`} • Ready for download
                          </span>
                          <div className="flex gap-2">
                            <Button size="sm" variant="outline" onClick={() => {}}>
                              <Share2 className="w-3 h-3 mr-1" />
                              Share
                            </Button>
                            <Button size="sm" onClick={() => downloadPublication(job)}>
                              <Download className="w-3 h-3 mr-1" />
                              Download
                            </Button>
                          </div>
                        </div>
                      )}
                    </CardContent>
                  </Card>
                ))
              )}
            </TabsContent>
          </Tabs>
        </CardContent>
      </Card>
    </div>
  );
};

export default ScientificPublicationExport;