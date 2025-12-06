import React, { useState, useRef } from 'react';
import { Button } from "@/ui/atoms/Button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/ui/molecules/Card";
import { Badge } from "@/ui/atoms/Badge";
import { Input } from "@/ui/atoms/Input";
import { Label } from "@/ui/atoms/Label";
import { Textarea } from "@/components/ui/textarea";
import { 
  Video, 
  Upload, 
  Play, 
  Pause, 
  RotateCcw, 
  Download,
  Eye,
  BarChart3,
  Target,
  AlertCircle,
  CheckCircle
} from 'lucide-react';
import { toast } from 'sonner';
import { useTierAccess } from '@/hooks/useTierAccess';

interface VideoAnalysisProps {
  className?: string;
}

interface VideoAnalysisResult {
  id: string;
  exerciseName: string;
  timestamp: Date;
  overallScore: number;
  feedback: {
    technique: string;
    improvements: string[];
    strengths: string[];
  };
  metrics: {
    depthConsistency: number;
    tempoControl: number;
    formStability: number;
    rangeOfMotion: number;
  };
}

export const VideoAnalysis: React.FC<VideoAnalysisProps> = ({ className }) => {
  const [selectedVideo, setSelectedVideo] = useState<File | null>(null);
  const [exerciseType, setExerciseType] = useState('');
  const [analysisNotes, setAnalysisNotes] = useState('');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisResults, setAnalysisResults] = useState<VideoAnalysisResult[]>([]);
  const [playingVideo, setPlayingVideo] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  
  const tierAccess = useTierAccess();

  const exerciseTypes = [
    'Squat', 'Deadlift', 'Bench Press', 'Overhead Press', 'Pull-up',
    'Push-up', 'Lunge', 'Row', 'Clean', 'Snatch', 'Other'
  ];

  const handleVideoUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      if (file.size > 100 * 1024 * 1024) { // 100MB limit
        toast.error('Video file is too large. Please upload a file smaller than 100MB.');
        return;
      }
      
      if (!file.type.startsWith('video/')) {
        toast.error('Please upload a valid video file.');
        return;
      }
      
      setSelectedVideo(file);
      toast.success('Video uploaded successfully!');
    }
  };

  const simulateAIAnalysis = async (): Promise<VideoAnalysisResult> => {
    // Simulate AI processing time
    await new Promise(resolve => setTimeout(resolve, 3000));
    
    return {
      id: crypto.randomUUID(),
      exerciseName: exerciseType,
      timestamp: new Date(),
      overallScore: Math.floor(Math.random() * 30) + 70, // 70-100
      feedback: {
        technique: "Good overall form with minor adjustments needed in the bottom position.",
        improvements: [
          "Maintain chest up throughout the movement",
          "Control descent speed for better muscle activation",
          "Ensure knees track over toes"
        ],
        strengths: [
          "Consistent depth across all reps",
          "Good core engagement",
          "Proper breathing pattern"
        ]
      },
      metrics: {
        depthConsistency: Math.floor(Math.random() * 20) + 80,
        tempoControl: Math.floor(Math.random() * 25) + 75,
        formStability: Math.floor(Math.random() * 15) + 85,
        rangeOfMotion: Math.floor(Math.random() * 20) + 80
      }
    };
  };

  const handleAnalyze = async () => {
    if (!selectedVideo || !exerciseType) {
      toast.error('Please upload a video and select an exercise type.');
      return;
    }

    if (!tierAccess.hasMinimumTier('adaptive')) {
      toast.error('Video analysis requires Adaptive tier or higher.');
      return;
    }

    setIsAnalyzing(true);
    try {
      const result = await simulateAIAnalysis();
      setAnalysisResults(prev => [result, ...prev]);
      toast.success('Video analysis completed!');
      
      // Reset form
      setSelectedVideo(null);
      setExerciseType('');
      setAnalysisNotes('');
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    } catch (error) {
      toast.error('Analysis failed. Please try again.');
    } finally {
      setIsAnalyzing(false);
    }
  };

  const getScoreColor = (score: number) => {
    if (score >= 90) return 'text-green-600';
    if (score >= 80) return 'text-yellow-600';
    return 'text-red-600';
  };

  const getScoreBadge = (score: number) => {
    if (score >= 90) return 'bg-green-100 text-green-800';
    if (score >= 80) return 'bg-yellow-100 text-yellow-800';
    return 'bg-red-100 text-red-800';
  };

  if (!tierAccess.hasMinimumTier('adaptive')) {
    return (
      <Card className={className}>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Video className="h-5 w-5" />
            Video Form Analysis
          </CardTitle>
          <CardDescription>
            AI-powered exercise form analysis and feedback
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="text-center py-8">
            <Video className="h-12 w-12 text-gray-400 mx-auto mb-4" />
            <h3 className="font-semibold text-gray-900 mb-2">Premium Feature</h3>
            <p className="text-gray-600 mb-4">
              Video analysis is available with Adaptive tier or higher.
            </p>
            <Button variant="outline" onClick={() => console.log("VideoAnalysis button clicked")}>Upgrade Plan</Button>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className={`space-y-6 ${className}`}>
      {/* Upload and Analysis Section */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Video className="h-5 w-5" />
            Video Form Analysis
          </CardTitle>
          <CardDescription>
            Upload exercise videos for AI-powered form analysis and personalized feedback
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* Video Upload */}
          <div className="space-y-4">
            <Label htmlFor="video-upload">Upload Exercise Video</Label>
            <div className="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center">
              <input
                ref={fileInputRef}
                id="video-upload"
                type="file"
                accept="video/*"
                onChange={handleVideoUpload}
                className="hidden"
              />
              {selectedVideo ? (
                <div className="space-y-4">
                  <Video className="h-12 w-12 text-green-600 mx-auto" />
                  <div>
                    <p className="font-medium text-gray-900">{selectedVideo.name}</p>
                    <p className="text-sm text-gray-600">
                      {(selectedVideo.size / (1024 * 1024)).toFixed(2)} MB
                    </p>
                  </div>
                  <Button
                    variant="outline"
                    onClick={() => fileInputRef.current?.click()}
                  >
                    Change Video
                  </Button>
                </div>
              ) : (
                <div className="space-y-4">
                  <Upload className="h-12 w-12 text-gray-400 mx-auto" />
                  <div>
                    <p className="text-lg font-medium text-gray-900">Upload a video</p>
                    <p className="text-gray-600">MP4, MOV, or AVI up to 100MB</p>
                  </div>
                  <Button onClick={() => fileInputRef.current?.click()}>
                    Select Video File
                  </Button>
                </div>
              )}
            </div>
          </div>

          {/* Exercise Type Selection */}
          <div className="space-y-2">
            <Label htmlFor="exercise-type">Exercise Type</Label>
            <select
              id="exercise-type"
              value={exerciseType}
              onChange={(e) => setExerciseType(e.target.value)}
              className="w-full p-3 border border-gray-300 rounded-md focus:ring-2 focus:ring-repz-orange focus:border-repz-orange"
            >
              <option value="">Select an exercise</option>
              {exerciseTypes.map((exercise) => (
                <option key={exercise} value={exercise}>
                  {exercise}
                </option>
              ))}
            </select>
          </div>

          {/* Analysis Notes */}
          <div className="space-y-2">
            <Label htmlFor="analysis-notes">Additional Notes (Optional)</Label>
            <Textarea
              id="analysis-notes"
              placeholder="Any specific concerns or areas you'd like analyzed..."
              value={analysisNotes}
              onChange={(e) => setAnalysisNotes(e.target.value)}
              rows={3}
            />
          </div>

          {/* Analyze Button */}
          <Button
            onClick={handleAnalyze}
            disabled={!selectedVideo || !exerciseType || isAnalyzing}
            className="w-full"
          >
            {isAnalyzing ? (
              <>
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                Analyzing Video...
              </>
            ) : (
              <>
                <BarChart3 className="h-4 w-4 mr-2" />
                Analyze Form
              </>
            )}
          </Button>
        </CardContent>
      </Card>

      {/* Analysis Results */}
      {analysisResults.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Analysis History</CardTitle>
            <CardDescription>
              Your recent video analysis results and feedback
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-6">
              {analysisResults.map((result) => (
                <div key={result.id} className="border rounded-lg p-6 space-y-4">
                  {/* Header */}
                  <div className="flex items-center justify-between">
                    <div>
                      <h3 className="font-semibold text-lg">{result.exerciseName}</h3>
                      <p className="text-sm text-gray-600">
                        {result.timestamp.toLocaleDateString()} at{' '}
                        {result.timestamp.toLocaleTimeString()}
                      </p>
                    </div>
                    <Badge className={getScoreBadge(result.overallScore)}>
                      {result.overallScore}/100
                    </Badge>
                  </div>

                  {/* Metrics */}
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    {Object.entries(result.metrics).map(([key, value]) => (
                      <div key={key} className="text-center">
                        <div className={`text-2xl font-bold ${getScoreColor(value)}`}>
                          {value}%
                        </div>
                        <div className="text-sm text-gray-600 capitalize">
                          {key.replace(/([A-Z])/g, ' $1').trim()}
                        </div>
                      </div>
                    ))}
                  </div>

                  {/* Feedback */}
                  <div className="space-y-4">
                    <div>
                      <h4 className="font-medium text-gray-900 mb-2">Overall Feedback</h4>
                      <p className="text-gray-700">{result.feedback.technique}</p>
                    </div>

                    <div className="grid md:grid-cols-2 gap-4">
                      <div>
                        <h4 className="font-medium text-gray-900 mb-2 flex items-center gap-2">
                          <AlertCircle className="h-4 w-4 text-yellow-600" />
                          Areas for Improvement
                        </h4>
                        <ul className="space-y-1">
                          {result.feedback.improvements.map((improvement, index) => (
                            <li key={index} className="text-sm text-gray-700 flex items-start gap-2">
                              <span className="text-yellow-600 mt-1">•</span>
                              {improvement}
                            </li>
                          ))}
                        </ul>
                      </div>

                      <div>
                        <h4 className="font-medium text-gray-900 mb-2 flex items-center gap-2">
                          <CheckCircle className="h-4 w-4 text-green-600" />
                          Strengths
                        </h4>
                        <ul className="space-y-1">
                          {result.feedback.strengths.map((strength, index) => (
                            <li key={index} className="text-sm text-gray-700 flex items-start gap-2">
                              <span className="text-green-600 mt-1">•</span>
                              {strength}
                            </li>
                          ))}
                        </ul>
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
};