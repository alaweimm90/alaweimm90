import React, { useState, useRef, useCallback, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/ui/molecules/Card';
import { Button } from '@/ui/atoms/Button';
import { Badge } from '@/ui/atoms/Badge';
import { Progress } from '@/components/ui/progress';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/ui/molecules/Select';
import { TierGate } from '@/components/auth/TierGate';
import { supabase } from '@/integrations/supabase/client';
import { useToast } from '@/ui/molecules/useToast';
import { 
  Camera, 
  Video, 
  Play, 
  Pause, 
  Square, 
  RotateCcw,
  CheckCircle2,
  AlertTriangle,
  Target,
  TrendingUp,
  Eye
} from 'lucide-react';

interface FormAnalysis {
  overallScore: number;
  phase: string;
  keyPointsAnalysis: Array<{
    point: string;
    score: number;
    feedback: string;
    status: 'good' | 'needs_work' | 'poor';
  }>;
  criticalErrors: Array<{
    error: string;
    severity: 'minor' | 'moderate' | 'severe';
    correction: string;
  }>;
  recommendations: string[];
  confidenceLevel: number;
  exerciseType: string;
  timestamp: string;
}

const EXERCISE_TYPES = {
  squat: {
    name: 'Squat',
    description: 'Bodyweight or weighted squat analysis',
    icon: Target
  },
  deadlift: {
    name: 'Deadlift',
    description: 'Deadlift form and technique analysis',
    icon: TrendingUp
  },
  bench_press: {
    name: 'Bench Press',
    description: 'Bench press form analysis',
    icon: Target
  },
  push_up: {
    name: 'Push-up',
    description: 'Push-up form and alignment check',
    icon: Target
  }
};

export default function VideoAnalysisSystem() {
  const { toast } = useToast();
  const [isRecording, setIsRecording] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [selectedExercise, setSelectedExercise] = useState<string>('squat');
  const [analysis, setAnalysis] = useState<FormAnalysis | null>(null);
  const [recordedFrames, setRecordedFrames] = useState<string[]>([]);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const frameIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const streamRef = useRef<MediaStream | null>(null);

  // Initialize camera
  const initializeCamera = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 1280 },
          height: { ideal: 720 },
          facingMode: 'user'
        }
      });
      
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
      
      toast({
        title: "Camera Ready",
        description: "Video analysis system initialized successfully"
      });
      
    } catch (error) {
      console.error('Error accessing camera:', error);
      toast({
        title: "Camera Error",
        description: "Could not access camera. Please check permissions.",
        variant: "destructive"
      });
    }
  }, [toast]);

  // Capture frame from video
  const captureFrame = useCallback(() => {
    if (!videoRef.current || !canvasRef.current) return null;
    
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return null;
    
    canvas.width = videoRef.current.videoWidth;
    canvas.height = videoRef.current.videoHeight;
    ctx.drawImage(videoRef.current, 0, 0);
    
    return canvas.toDataURL('image/jpeg', 0.8);
  }, []);

  // Start recording
  const startRecording = useCallback(async () => {
    if (!streamRef.current) {
      await initializeCamera();
      return;
    }
    
    setIsRecording(true);
    setRecordedFrames([]);
    setAnalysis(null);
    
    // Capture frames every 200ms during recording
    frameIntervalRef.current = setInterval(() => {
      const frame = captureFrame();
      if (frame) {
        setRecordedFrames(prev => [...prev, frame]);
      }
    }, 200);
    
    toast({
      title: "Recording Started",
      description: `Recording ${EXERCISE_TYPES[selectedExercise as keyof typeof EXERCISE_TYPES].name} for analysis`
    });
  }, [selectedExercise, captureFrame, initializeCamera, toast]);

  // Stop recording
  const stopRecording = useCallback(() => {
    setIsRecording(false);
    
    if (frameIntervalRef.current) {
      clearInterval(frameIntervalRef.current);
      frameIntervalRef.current = null;
    }
    
    toast({
      title: "Recording Stopped",
      description: `Captured ${recordedFrames.length} frames for analysis`
    });
  }, [recordedFrames.length, toast]);

  // Analyze recorded video
  const analyzeVideo = useCallback(async () => {
    if (recordedFrames.length === 0) {
      toast({
        title: "No Video Data",
        description: "Please record a video first",
        variant: "destructive"
      });
      return;
    }
    
    setIsAnalyzing(true);
    
    try {
      // Take every 3rd frame to reduce processing load
      const sampleFrames = recordedFrames.filter((_, index) => index % 3 === 0);
      
      const { data, error } = await supabase.functions.invoke('video-analysis', {
        body: {
          videoFrames: sampleFrames.slice(0, 10), // Max 10 frames
          exerciseType: selectedExercise,
          analysisType: 'form_check'
        }
      });

      if (error) throw error;

      setAnalysis(data);
      
      toast({
        title: "Analysis Complete",
        description: `Form score: ${data.overallScore}/100`
      });
      
    } catch (error) {
      console.error('Error analyzing video:', error);
      toast({
        title: "Analysis Error",
        description: "Could not analyze video. Please try again.",
        variant: "destructive"
      });
    } finally {
      setIsAnalyzing(false);
    }
  }, [recordedFrames, selectedExercise, toast]);

  // Get score color
  const getScoreColor = (score: number) => {
    if (score >= 80) return 'text-green-500';
    if (score >= 60) return 'text-yellow-500';
    return 'text-red-500';
  };

  // Get status badge variant
  const getStatusVariant = (status: string) => {
    switch (status) {
      case 'good': return 'default';
      case 'needs_work': return 'secondary';
      case 'poor': return 'destructive';
      default: return 'outline';
    }
  };

  // Initialize camera on mount
  useEffect(() => {
    initializeCamera();
    
    return () => {
      // Cleanup
      if (frameIntervalRef.current) {
        clearInterval(frameIntervalRef.current);
      }
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
      }
    };
  }, [initializeCamera]);

  return (
    <TierGate requiredTier="adaptive" feature="video_analysis">
      <div className="space-y-6">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-3">
              <Video className="h-6 w-6" />
              AI Video Form Analysis
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-6">
            {/* Exercise Selection */}
            <div className="space-y-3">
              <label className="text-sm font-medium">Exercise Type</label>
              <Select value={selectedExercise} onValueChange={setSelectedExercise}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {Object.entries(EXERCISE_TYPES).map(([key, exercise]) => (
                    <SelectItem key={key} value={key}>
                      <div className="flex items-center gap-2">
                        <exercise.icon className="h-4 w-4" />
                        <div>
                          <p className="font-medium">{exercise.name}</p>
                          <p className="text-xs text-muted-foreground">{exercise.description}</p>
                        </div>
                      </div>
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            {/* Video Feed */}
            <div className="relative bg-gray-900 rounded-lg overflow-hidden aspect-video">
              <video
                ref={videoRef}
                autoPlay
                muted
                playsInline
                className="w-full h-full object-cover"
              />
              <canvas ref={canvasRef} className="hidden" />
              
              {/* Recording Indicator */}
              {isRecording && (
                <div className="absolute top-4 left-4 flex items-center gap-2 bg-red-500 text-white px-3 py-1 rounded-full">
                  <div className="w-2 h-2 bg-white rounded-full animate-pulse" />
                  <span className="text-sm font-medium">Recording</span>
                </div>
              )}
              
              {/* Frame Counter */}
              {recordedFrames.length > 0 && (
                <div className="absolute top-4 right-4 bg-black/50 text-white px-3 py-1 rounded-full text-sm">
                  {recordedFrames.length} frames
                </div>
              )}
            </div>

            {/* Controls */}
            <div className="flex flex-wrap gap-3 justify-center">
              {!isRecording ? (
                <Button onClick={startRecording} className="flex items-center gap-2">
                  <Camera className="h-4 w-4" />
                  Start Recording
                </Button>
              ) : (
                <Button onClick={stopRecording} variant="destructive" className="flex items-center gap-2">
                  <Square className="h-4 w-4" />
                  Stop Recording
                </Button>
              )}
              
              <Button 
                onClick={analyzeVideo} 
                disabled={recordedFrames.length === 0 || isAnalyzing}
                variant="outline"
                className="flex items-center gap-2"
              >
                {isAnalyzing ? (
                  <RotateCcw className="h-4 w-4 animate-spin" />
                ) : (
                  <Eye className="h-4 w-4" />
                )}
                {isAnalyzing ? 'Analyzing...' : 'Analyze Form'}
              </Button>
            </div>
          </CardContent>
        </Card>

        {/* Analysis Results */}
        {analysis && (
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-3">
                <Target className="h-6 w-6" />
                Form Analysis Results
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              {/* Overall Score */}
              <div className="text-center space-y-2">
                <div className={`text-4xl font-bold ${getScoreColor(analysis.overallScore)}`}>
                  {analysis.overallScore}/100
                </div>
                <p className="text-muted-foreground">Overall Form Score</p>
                <Progress value={analysis.overallScore} className="w-full max-w-md mx-auto" />
              </div>

              {/* Key Points Analysis */}
              <div className="space-y-4">
                <h3 className="text-lg font-semibold">Form Breakdown</h3>
                <div className="grid gap-3">
                  {analysis.keyPointsAnalysis.map((point, index) => (
                    <div key={index} className="flex items-center justify-between p-3 bg-muted rounded-lg">
                      <div className="flex-1">
                        <div className="flex items-center gap-3">
                          <h4 className="font-medium">{point.point}</h4>
                          <Badge variant={getStatusVariant(point.status)}>
                            {point.status.replace('_', ' ')}
                          </Badge>
                        </div>
                        <p className="text-sm text-muted-foreground mt-1">{point.feedback}</p>
                      </div>
                      <div className={`text-lg font-bold ${getScoreColor(point.score)}`}>
                        {point.score}
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Critical Errors */}
              {analysis.criticalErrors && analysis.criticalErrors.length > 0 && (
                <div className="space-y-3">
                  <h3 className="text-lg font-semibold flex items-center gap-2">
                    <AlertTriangle className="h-5 w-5 text-yellow-500" />
                    Areas for Improvement
                  </h3>
                  <div className="space-y-2">
                    {analysis.criticalErrors.map((error, index) => (
                      <div key={index} className="p-3 border-l-4 border-yellow-500 bg-yellow-50 rounded">
                        <div className="flex items-center gap-2 mb-1">
                          <h4 className="font-medium">{error.error}</h4>
                          <Badge variant={error.severity === 'severe' ? 'destructive' : 'secondary'}>
                            {error.severity}
                          </Badge>
                        </div>
                        <p className="text-sm text-muted-foreground">{error.correction}</p>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Recommendations */}
              {analysis.recommendations && analysis.recommendations.length > 0 && (
                <div className="space-y-3">
                  <h3 className="text-lg font-semibold flex items-center gap-2">
                    <CheckCircle2 className="h-5 w-5 text-green-500" />
                    Recommendations
                  </h3>
                  <ul className="space-y-2">
                    {analysis.recommendations.map((rec, index) => (
                      <li key={index} className="flex items-start gap-2">
                        <div className="w-2 h-2 bg-primary rounded-full mt-2 flex-shrink-0" />
                        <span className="text-sm">{rec}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {/* Analysis Metadata */}
              <div className="flex justify-between text-xs text-muted-foreground pt-4 border-t">
                <span>Confidence: {analysis.confidenceLevel}%</span>
                <span>Analyzed: {new Date(analysis.timestamp).toLocaleString()}</span>
              </div>
            </CardContent>
          </Card>
        )}
      </div>
    </TierGate>
  );
}