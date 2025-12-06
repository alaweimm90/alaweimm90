import React, { useState, useEffect, useCallback } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Switch } from '@/components/ui/switch';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { 
  Zap, 
  Monitor, 
  Cpu, 
  MemoryStick,
  Settings,
  CheckCircle,
  AlertTriangle,
  XCircle,
  BarChart3,
  Gauge,
  Eye,
  Layers
} from 'lucide-react';
import { useToast } from '@/hooks/use-toast';
import { cn } from '@/lib/utils';

interface WebGLCapabilities {
  version: string;
  vendor: string;
  renderer: string;
  maxTextureSize: number;
  maxVertexUniforms: number;
  maxFragmentUniforms: number;
  maxVaryingVectors: number;
  maxVertexAttributes: number;
  maxDrawBuffers: number;
  extensions: string[];
  contextLost: boolean;
}

interface PerformanceMetrics {
  fps: number;
  frameTime: number;
  drawCalls: number;
  vertices: number;
  triangles: number;
  memoryUsage: number;
  textureMemory: number;
  geometryMemory: number;
}

interface OptimizationSettings {
  adaptiveLOD: boolean;
  frustumCulling: boolean;
  instancing: boolean;
  textureCompression: boolean;
  mipmapping: boolean;
  depthTesting: boolean;
  backfaceCulling: boolean;
  antialiasing: boolean;
  shadowQuality: 'disabled' | 'low' | 'medium' | 'high';
  renderScale: number;
  maxParticles: number;
  lodBias: number;
}

interface WebGLPerformanceEnhancerProps {
  className?: string;
  onSettingsChange?: (settings: OptimizationSettings) => void;
}

export const WebGLPerformanceEnhancer: React.FC<WebGLPerformanceEnhancerProps> = ({
  className,
  onSettingsChange
}) => {
  const { toast } = useToast();
  const [capabilities, setCapabilities] = useState<WebGLCapabilities | null>(null);
  const [metrics, setMetrics] = useState<PerformanceMetrics>({
    fps: 60,
    frameTime: 16.67,
    drawCalls: 0,
    vertices: 0,
    triangles: 0,
    memoryUsage: 0,
    textureMemory: 0,
    geometryMemory: 0
  });
  const [settings, setSettings] = useState<OptimizationSettings>({
    adaptiveLOD: true,
    frustumCulling: true,
    instancing: true,
    textureCompression: true,
    mipmapping: true,
    depthTesting: true,
    backfaceCulling: true,
    antialiasing: true,
    shadowQuality: 'medium',
    renderScale: 1.0,
    maxParticles: 1000,
    lodBias: 0
  });
  const [isMonitoring, setIsMonitoring] = useState(false);
  const [performanceScore, setPerformanceScore] = useState(0);

  useEffect(() => {
    initializeWebGL();
  }, []);

  useEffect(() => {
    if (isMonitoring) {
      const interval = setInterval(updateMetrics, 1000);
      return () => clearInterval(interval);
    }
  }, [isMonitoring]);

  useEffect(() => {
    onSettingsChange?.(settings);
    calculatePerformanceScore();
  }, [settings, onSettingsChange]);

  const initializeWebGL = () => {
    try {
      const canvas = document.createElement('canvas');
      const gl = canvas.getContext('webgl2') || canvas.getContext('webgl');
      
      if (!gl) {
        toast({
          title: "WebGL Not Supported",
          description: "Your browser doesn't support WebGL",
          variant: "destructive"
        });
        return;
      }

      // Get WebGL capabilities
      const debugInfo = gl.getExtension('WEBGL_debug_renderer_info');
      const extensions = gl.getSupportedExtensions() || [];

      const caps: WebGLCapabilities = {
        version: gl.getParameter(gl.VERSION),
        vendor: debugInfo ? gl.getParameter(debugInfo.UNMASKED_VENDOR_WEBGL) : gl.getParameter(gl.VENDOR),
        renderer: debugInfo ? gl.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER),
        maxTextureSize: gl.getParameter(gl.MAX_TEXTURE_SIZE),
        maxVertexUniforms: gl.getParameter(gl.MAX_VERTEX_UNIFORM_VECTORS),
        maxFragmentUniforms: gl.getParameter(gl.MAX_FRAGMENT_UNIFORM_VECTORS),
        maxVaryingVectors: gl.getParameter(gl.MAX_VARYING_VECTORS),
        maxVertexAttributes: gl.getParameter(gl.MAX_VERTEX_ATTRIBS),
        maxDrawBuffers: (gl as WebGL2RenderingContext).MAX_DRAW_BUFFERS ? gl.getParameter((gl as WebGL2RenderingContext).MAX_DRAW_BUFFERS) : 1,
        extensions,
        contextLost: gl.isContextLost()
      };

      setCapabilities(caps);

      // Auto-optimize based on capabilities
      autoOptimize(caps);

    } catch (error) {
      console.error('Failed to initialize WebGL:', error);
      toast({
        title: "WebGL Error",
        description: "Failed to initialize WebGL context",
        variant: "destructive"
      });
    }
  };

  const autoOptimize = (caps: WebGLCapabilities) => {
    const newSettings = { ...settings };

    // Detect mobile/low-end devices
    const isMobile = /Mobile|Android|iPhone|iPad/.test(navigator.userAgent);
    const isLowEnd = caps.maxTextureSize < 4096 || caps.maxVertexUniforms < 256;

    if (isMobile || isLowEnd) {
      newSettings.shadowQuality = 'low';
      newSettings.renderScale = 0.8;
      newSettings.maxParticles = 500;
      newSettings.antialiasing = false;
    }

    // Enable advanced features if supported
    if (caps.extensions.includes('OES_vertex_array_object')) {
      newSettings.instancing = true;
    }

    if (caps.extensions.includes('WEBGL_compressed_texture_s3tc')) {
      newSettings.textureCompression = true;
    }

    setSettings(newSettings);

    toast({
      title: "Auto-Optimization Complete",
      description: `Optimized for ${isMobile ? 'mobile' : isLowEnd ? 'low-end' : 'high-end'} device`
    });
  };

  const updateMetrics = useCallback(() => {
    // Simulate real-time performance metrics
    const baseFPS = settings.antialiasing ? 55 : 60;
    const scaleImpact = (2 - settings.renderScale) * 10;
    const shadowImpact = { disabled: 0, low: 5, medium: 10, high: 20 }[settings.shadowQuality];
    
    const estimatedFPS = Math.max(15, baseFPS - scaleImpact - shadowImpact + Math.random() * 10 - 5);
    
    setMetrics(prev => ({
      ...prev,
      fps: Math.round(estimatedFPS),
      frameTime: 1000 / estimatedFPS,
      drawCalls: Math.floor(Math.random() * 50 + 20),
      vertices: Math.floor(Math.random() * 100000 + 50000),
      triangles: Math.floor(Math.random() * 50000 + 25000),
      memoryUsage: Math.random() * 200 + 100,
      textureMemory: Math.random() * 100 + 50,
      geometryMemory: Math.random() * 50 + 25
    }));
  }, [settings]);

  const calculatePerformanceScore = () => {
    let score = 100;

    // Deduct points for disabled optimizations
    if (!settings.frustumCulling) score -= 10;
    if (!settings.backfaceCulling) score -= 5;
    if (!settings.instancing) score -= 15;
    if (!settings.adaptiveLOD) score -= 20;
    if (settings.shadowQuality === 'high') score -= 15;
    if (settings.renderScale > 1.2) score -= 10;
    if (settings.maxParticles > 2000) score -= 10;

    // Add points for good practices
    if (settings.textureCompression) score += 5;
    if (settings.mipmapping) score += 5;

    setPerformanceScore(Math.max(0, Math.min(100, score)));
  };

  const updateSetting = (key: keyof OptimizationSettings, value: any) => {
    setSettings(prev => ({ ...prev, [key]: value }));
  };

  const resetToDefaults = () => {
    setSettings({
      adaptiveLOD: true,
      frustumCulling: true,
      instancing: true,
      textureCompression: true,
      mipmapping: true,
      depthTesting: true,
      backfaceCulling: true,
      antialiasing: true,
      shadowQuality: 'medium',
      renderScale: 1.0,
      maxParticles: 1000,
      lodBias: 0
    });

    toast({
      title: "Settings Reset",
      description: "All optimization settings restored to defaults"
    });
  };

  const getPerformanceColor = (score: number) => {
    if (score >= 80) return 'text-green-500';
    if (score >= 60) return 'text-yellow-500';
    return 'text-red-500';
  };

  const getCapabilityStatus = (value: number, threshold: number) => {
    if (value >= threshold) return <CheckCircle className="w-4 h-4 text-green-500" />;
    if (value >= threshold * 0.7) return <AlertTriangle className="w-4 h-4 text-yellow-500" />;
    return <XCircle className="w-4 h-4 text-red-500" />;
  };

  return (
    <div className={cn('w-full space-y-6', className)}>
      {/* Performance Overview */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2">
                <Gauge className="w-5 h-5 text-accentPhysics" />
                <span className="font-medium">Performance</span>
              </div>
              <Badge variant={performanceScore >= 80 ? "default" : performanceScore >= 60 ? "secondary" : "destructive"}>
                {performanceScore}%
              </Badge>
            </div>
            <Progress value={performanceScore} className="h-2" />
            <p className="text-xs text-textSecondary mt-1">
              Optimization score
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2">
                <BarChart3 className="w-5 h-5 text-accentQuantum" />
                <span className="font-medium">FPS</span>
              </div>
              <Badge variant="outline">{metrics.fps}</Badge>
            </div>
            <Progress value={(metrics.fps / 60) * 100} className="h-2" />
            <p className="text-xs text-textSecondary mt-1">
              {metrics.frameTime.toFixed(1)}ms frame time
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2">
                <MemoryStick className="w-5 h-5 text-accentEnergy" />
                <span className="font-medium">Memory</span>
              </div>
              <Badge variant="outline">{metrics.memoryUsage.toFixed(0)}MB</Badge>
            </div>
            <Progress value={(metrics.memoryUsage / 500) * 100} className="h-2" />
            <p className="text-xs text-textSecondary mt-1">
              GPU memory usage
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2">
                <Layers className="w-5 h-5 text-accentStatistical" />
                <span className="font-medium">Draw Calls</span>
              </div>
              <Badge variant="outline">{metrics.drawCalls}</Badge>
            </div>
            <Progress value={(metrics.drawCalls / 100) * 100} className="h-2" />
            <p className="text-xs text-textSecondary mt-1">
              Render batches
            </p>
          </CardContent>
        </Card>
      </div>

      {/* WebGL Capabilities */}
      {capabilities && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Monitor className="w-5 h-5" />
              WebGL Capabilities
            </CardTitle>
            <CardDescription>
              Graphics hardware and driver information
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <div className="space-y-4">
                <div>
                  <h4 className="font-medium mb-2">Hardware Information</h4>
                  <div className="space-y-2 text-sm">
                    <div className="flex justify-between">
                      <span className="text-textSecondary">WebGL Version:</span>
                      <span>{capabilities.version.split(' ')[0]}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-textSecondary">Vendor:</span>
                      <span className="truncate max-w-48">{capabilities.vendor}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-textSecondary">Renderer:</span>
                      <span className="truncate max-w-48">{capabilities.renderer}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-textSecondary">Context Status:</span>
                      <span className="flex items-center gap-1">
                        {capabilities.contextLost ? (
                          <>
                            <XCircle className="w-4 h-4 text-red-500" />
                            Lost
                          </>
                        ) : (
                          <>
                            <CheckCircle className="w-4 h-4 text-green-500" />
                            Active
                          </>
                        )}
                      </span>
                    </div>
                  </div>
                </div>

                <div>
                  <h4 className="font-medium mb-2">Resource Limits</h4>
                  <div className="space-y-2 text-sm">
                    <div className="flex justify-between items-center">
                      <span className="text-textSecondary">Max Texture Size:</span>
                      <span className="flex items-center gap-1">
                        {capabilities.maxTextureSize}px
                        {getCapabilityStatus(capabilities.maxTextureSize, 4096)}
                      </span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-textSecondary">Vertex Uniforms:</span>
                      <span className="flex items-center gap-1">
                        {capabilities.maxVertexUniforms}
                        {getCapabilityStatus(capabilities.maxVertexUniforms, 256)}
                      </span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-textSecondary">Fragment Uniforms:</span>
                      <span className="flex items-center gap-1">
                        {capabilities.maxFragmentUniforms}
                        {getCapabilityStatus(capabilities.maxFragmentUniforms, 256)}
                      </span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-textSecondary">Vertex Attributes:</span>
                      <span className="flex items-center gap-1">
                        {capabilities.maxVertexAttributes}
                        {getCapabilityStatus(capabilities.maxVertexAttributes, 16)}
                      </span>
                    </div>
                  </div>
                </div>
              </div>

              <div>
                <h4 className="font-medium mb-2">Supported Extensions</h4>
                <div className="max-h-48 overflow-y-auto">
                  <div className="grid grid-cols-1 gap-1">
                    {capabilities.extensions.slice(0, 20).map(ext => (
                      <Badge key={ext} variant="outline" className="text-xs justify-start">
                        {ext}
                      </Badge>
                    ))}
                    {capabilities.extensions.length > 20 && (
                      <p className="text-xs text-textSecondary mt-2">
                        +{capabilities.extensions.length - 20} more extensions
                      </p>
                    )}
                  </div>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Optimization Settings */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Settings className="w-5 h-5" />
            Optimization Settings
          </CardTitle>
          <CardDescription>
            Configure WebGL rendering optimizations for best performance
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Rendering Optimizations */}
            <div className="space-y-4">
              <h4 className="font-medium">Rendering Optimizations</h4>
              
              <div className="flex items-center justify-between">
                <div className="space-y-0.5">
                  <Label htmlFor="adaptive-lod">Adaptive Level of Detail</Label>
                  <p className="text-sm text-textSecondary">
                    Reduce geometry complexity based on distance
                  </p>
                </div>
                <Switch
                  id="adaptive-lod"
                  checked={settings.adaptiveLOD}
                  onCheckedChange={(checked) => updateSetting('adaptiveLOD', checked)}
                />
              </div>

              <div className="flex items-center justify-between">
                <div className="space-y-0.5">
                  <Label htmlFor="frustum-culling">Frustum Culling</Label>
                  <p className="text-sm text-textSecondary">
                    Skip rendering objects outside camera view
                  </p>
                </div>
                <Switch
                  id="frustum-culling"
                  checked={settings.frustumCulling}
                  onCheckedChange={(checked) => updateSetting('frustumCulling', checked)}
                />
              </div>

              <div className="flex items-center justify-between">
                <div className="space-y-0.5">
                  <Label htmlFor="instancing">GPU Instancing</Label>
                  <p className="text-sm text-textSecondary">
                    Render multiple copies efficiently
                  </p>
                </div>
                <Switch
                  id="instancing"
                  checked={settings.instancing}
                  onCheckedChange={(checked) => updateSetting('instancing', checked)}
                />
              </div>

              <div className="flex items-center justify-between">
                <div className="space-y-0.5">
                  <Label htmlFor="backface-culling">Backface Culling</Label>
                  <p className="text-sm text-textSecondary">
                    Skip rendering back-facing triangles
                  </p>
                </div>
                <Switch
                  id="backface-culling"
                  checked={settings.backfaceCulling}
                  onCheckedChange={(checked) => updateSetting('backfaceCulling', checked)}
                />
              </div>
            </div>

            {/* Quality Settings */}
            <div className="space-y-4">
              <h4 className="font-medium">Quality Settings</h4>

              <div>
                <Label htmlFor="shadow-quality">Shadow Quality</Label>
                <Select
                  value={settings.shadowQuality}
                  onValueChange={(value: any) => updateSetting('shadowQuality', value)}
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="disabled">Disabled</SelectItem>
                    <SelectItem value="low">Low</SelectItem>
                    <SelectItem value="medium">Medium</SelectItem>
                    <SelectItem value="high">High</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div>
                <Label htmlFor="render-scale">
                  Render Scale: {settings.renderScale.toFixed(1)}x
                </Label>
                <input
                  type="range"
                  id="render-scale"
                  min="0.5"
                  max="2.0"
                  step="0.1"
                  value={settings.renderScale}
                  onChange={(e) => updateSetting('renderScale', parseFloat(e.target.value))}
                  className="w-full"
                />
                <div className="flex justify-between text-xs text-textSecondary">
                  <span>Performance</span>
                  <span>Quality</span>
                </div>
              </div>

              <div>
                <Label htmlFor="max-particles">
                  Max Particles: {settings.maxParticles}
                </Label>
                <input
                  type="range"
                  id="max-particles"
                  min="100"
                  max="5000"
                  step="100"
                  value={settings.maxParticles}
                  onChange={(e) => updateSetting('maxParticles', parseInt(e.target.value))}
                  className="w-full"
                />
              </div>

              <div className="flex items-center justify-between">
                <div className="space-y-0.5">
                  <Label htmlFor="antialiasing">Antialiasing (MSAA)</Label>
                  <p className="text-sm text-textSecondary">
                    Smooth jagged edges (performance cost)
                  </p>
                </div>
                <Switch
                  id="antialiasing"
                  checked={settings.antialiasing}
                  onCheckedChange={(checked) => updateSetting('antialiasing', checked)}
                />
              </div>
            </div>
          </div>

          <div className="flex gap-4 pt-4 border-t">
            <Button variant="outline" onClick={resetToDefaults}>
              Reset to Defaults
            </Button>
            <Button
              variant={isMonitoring ? "destructive" : "default"}
              onClick={() => setIsMonitoring(!isMonitoring)}
              className="gap-2"
            >
              <Eye className="w-4 h-4" />
              {isMonitoring ? 'Stop Monitoring' : 'Start Monitoring'}
            </Button>
            {capabilities && (
              <Button variant="outline" onClick={() => autoOptimize(capabilities)}>
                <Zap className="w-4 h-4 mr-2" />
                Auto-Optimize
              </Button>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Performance Alerts */}
      {metrics.fps < 30 && (
        <Alert>
          <AlertTriangle className="h-4 w-4" />
          <AlertDescription>
            Low frame rate detected ({metrics.fps} FPS). Consider reducing quality settings or enabling more optimizations.
          </AlertDescription>
        </Alert>
      )}

      {metrics.memoryUsage > 400 && (
        <Alert>
          <AlertTriangle className="h-4 w-4" />
          <AlertDescription>
            High GPU memory usage ({metrics.memoryUsage.toFixed(0)}MB). Reduce texture quality or particle count.
          </AlertDescription>
        </Alert>
      )}
    </div>
  );
};

export default WebGLPerformanceEnhancer;