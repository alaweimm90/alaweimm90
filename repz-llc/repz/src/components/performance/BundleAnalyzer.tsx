import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from "@/ui/molecules/Card";
import { Button } from "@/ui/atoms/Button";
import { Badge } from "@/ui/atoms/Badge";
import { Progress } from "@/components/ui/progress";
import { 
  Package, 
  FileText, 
  Zap, 
  Download,
  AlertTriangle,
  CheckCircle,
  TrendingDown,
  Layers
} from 'lucide-react';

interface BundleChunk {
  name: string;
  size: number;
  compressed: number;
  type: 'vendor' | 'app' | 'css' | 'assets';
  modules: string[];
  loadTime: number;
}

interface BundleAnalysis {
  totalSize: number;
  compressedSize: number;
  chunks: BundleChunk[];
  recommendations: string[];
  performanceScore: number;
}

export function BundleAnalyzer() {
  const [analysis, setAnalysis] = useState<BundleAnalysis | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  // Mock bundle analysis data
  const mockAnalysis: BundleAnalysis = {
    totalSize: 845000,
    compressedSize: 312000,
    performanceScore: 78,
    chunks: [
      {
        name: 'vendor.js',
        size: 456000,
        compressed: 168000,
        type: 'vendor',
        modules: ['react', 'react-dom', 'lucide-react', '@radix-ui/*'],
        loadTime: 2.3
      },
      {
        name: 'app.js',
        size: 234000,
        compressed: 89000,
        type: 'app',
        modules: ['src/components/*', 'src/pages/*', 'src/hooks/*'],
        loadTime: 1.2
      },
      {
        name: 'styles.css',
        size: 89000,
        compressed: 32000,
        type: 'css',
        modules: ['tailwindcss', 'custom styles'],
        loadTime: 0.8
      },
      {
        name: 'assets',
        size: 66000,
        compressed: 23000,
        type: 'assets',
        modules: ['images', 'fonts', 'icons'],
        loadTime: 0.5
      }
    ],
    recommendations: [
      'Consider lazy loading non-critical components',
      'Optimize vendor chunk by splitting large libraries',
      'Implement tree shaking for unused code',
      'Compress images with WebP format',
      'Enable Brotli compression on server'
    ]
  };

  const analyzeBundles = async () => {
    setIsAnalyzing(true);
    
    // Simulate analysis time
    await new Promise(resolve => setTimeout(resolve, 2000));
    
    setAnalysis(mockAnalysis);
    setIsAnalyzing(false);
  };

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return `${parseFloat((bytes / Math.pow(k, i)).toFixed(1))} ${sizes[i]}`;
  };

  const getChunkIcon = (type: string) => {
    switch (type) {
      case 'vendor': return <Package className="h-4 w-4" />;
      case 'app': return <FileText className="h-4 w-4" />;
      case 'css': return <Layers className="h-4 w-4" />;
      case 'assets': return <Download className="h-4 w-4" />;
      default: return <FileText className="h-4 w-4" />;
    }
  };

  const getChunkColor = (type: string) => {
    switch (type) {
      case 'vendor': return 'bg-blue-500';
      case 'app': return 'bg-green-500';
      case 'css': return 'bg-purple-500';
      case 'assets': return 'bg-orange-500';
      default: return 'bg-gray-500';
    }
  };

  const compressionRatio = analysis ? ((analysis.totalSize - analysis.compressedSize) / analysis.totalSize * 100) : 0;

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold">Bundle Analyzer</h2>
          <p className="text-muted-foreground">JavaScript Bundle Size Analysis & Optimization</p>
        </div>
        <Button 
          onClick={analyzeBundles}
          disabled={isAnalyzing}
          className="flex items-center gap-2"
        >
          <Package className={`h-4 w-4 ${isAnalyzing ? 'animate-pulse' : ''}`} />
          {isAnalyzing ? 'Analyzing...' : 'Analyze Bundles'}
        </Button>
      </div>

      {analysis && (
        <>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <Card>
              <CardContent className="p-6">
                <div className="flex items-center gap-3">
                  <FileText className="h-8 w-8 text-primary" />
                  <div>
                    <h3 className="font-semibold">Total Bundle Size</h3>
                    <p className="text-2xl font-bold">{formatFileSize(analysis.totalSize)}</p>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardContent className="p-6">
                <div className="flex items-center gap-3">
                  <TrendingDown className="h-8 w-8 text-success" />
                  <div>
                    <h3 className="font-semibold">Compressed Size</h3>
                    <p className="text-2xl font-bold">{formatFileSize(analysis.compressedSize)}</p>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardContent className="p-6">
                <div className="flex items-center gap-3">
                  <Zap className="h-8 w-8 text-warning" />
                  <div>
                    <h3 className="font-semibold">Performance Score</h3>
                    <p className="text-2xl font-bold">{analysis.performanceScore}%</p>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>

          <Card>
            <CardHeader>
              <CardTitle>Compression Analysis</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <span>Compression Ratio</span>
                  <span className="font-bold text-success">{compressionRatio.toFixed(1)}%</span>
                </div>
                <Progress value={compressionRatio} className="h-3" />
                <p className="text-sm text-muted-foreground">
                  Bundle reduced from {formatFileSize(analysis.totalSize)} to {formatFileSize(analysis.compressedSize)}
                </p>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Bundle Chunks</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {analysis.chunks.map((chunk, index) => (
                  <div key={index} className="border rounded-lg p-4">
                    <div className="flex items-center justify-between mb-3">
                      <div className="flex items-center gap-3">
                        {getChunkIcon(chunk.type)}
                        <div>
                          <h4 className="font-semibold">{chunk.name}</h4>
                          <p className="text-sm text-muted-foreground">
                            Load time: {chunk.loadTime}s
                          </p>
                        </div>
                      </div>
                      <Badge variant="outline" className="capitalize">
                        {chunk.type}
                      </Badge>
                    </div>
                    
                    <div className="grid grid-cols-2 gap-4 mb-3">
                      <div>
                        <span className="text-sm text-muted-foreground">Original Size</span>
                        <p className="font-medium">{formatFileSize(chunk.size)}</p>
                      </div>
                      <div>
                        <span className="text-sm text-muted-foreground">Compressed Size</span>
                        <p className="font-medium text-success">{formatFileSize(chunk.compressed)}</p>
                      </div>
                    </div>

                    <div className="relative h-4 bg-muted rounded-full overflow-hidden">
                      <div 
                        className={`h-full ${getChunkColor(chunk.type)} transition-all duration-300`}
                        style={{ width: `${(chunk.size / analysis.totalSize) * 100}%` }}
                      />
                    </div>
                    
                    <div className="mt-3">
                      <span className="text-sm text-muted-foreground">Main modules:</span>
                      <div className="flex flex-wrap gap-1 mt-1">
                        {chunk.modules.slice(0, 3).map((module, i) => (
                          <Badge key={i} variant="secondary" className="text-xs">
                            {module}
                          </Badge>
                        ))}
                        {chunk.modules.length > 3 && (
                          <Badge variant="secondary" className="text-xs">
                            +{chunk.modules.length - 3} more
                          </Badge>
                        )}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Optimization Recommendations</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {analysis.recommendations.map((recommendation, index) => (
                  <div key={index} className="flex items-start gap-3">
                    <AlertTriangle className="h-4 w-4 text-warning mt-0.5" />
                    <div>
                      <p className="text-sm">{recommendation}</p>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Bundle Size Targets</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <span>Initial Bundle</span>
                    <span className={analysis.compressedSize <= 200000 ? 'text-success' : 'text-warning'}>
                      {formatFileSize(analysis.compressedSize)} / 200KB
                    </span>
                  </div>
                  <Progress 
                    value={Math.min((analysis.compressedSize / 200000) * 100, 100)} 
                    className="h-2"
                  />
                </div>
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <span>Total Bundle</span>
                    <span className={analysis.compressedSize <= 1000000 ? 'text-success' : 'text-warning'}>
                      {formatFileSize(analysis.compressedSize)} / 1MB
                    </span>
                  </div>
                  <Progress 
                    value={Math.min((analysis.compressedSize / 1000000) * 100, 100)} 
                    className="h-2"
                  />
                </div>
              </div>
            </CardContent>
          </Card>
        </>
      )}
    </div>
  );
}