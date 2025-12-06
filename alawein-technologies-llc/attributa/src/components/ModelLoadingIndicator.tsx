import { useEffect } from 'react';
import { Progress } from '@/components/ui/progress';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Download, AlertTriangle, CheckCircle } from 'lucide-react';
import { useModelStore } from '@/store/modelStore';

export default function ModelLoadingIndicator() {
  const { isLoading, progress, modelName, error } = useModelStore();

  // Only render when actively loading, errored, or we have progress
  if (!isLoading && !error && progress <= 0) {
    return null;
  }

  return (
    <Card className="mb-6">
      <CardHeader className="pb-3">
        <CardTitle className="flex items-center space-x-2">
          {error ? (
            <AlertTriangle className="h-5 w-5 text-destructive" />
          ) : progress === 100 ? (
            <CheckCircle className="h-5 w-5 text-success" />
          ) : (
            <Download className="h-5 w-5 text-info" />
          )}
          <span>
            {error ? 'Model Loading Failed' : 
             progress === 100 ? 'Model Ready' : 
             'Loading Local AI Model'}
          </span>
        </CardTitle>
        {modelName && (
          <CardDescription>
            {modelName} {!error && '(~124MB, cached after first download)'}
          </CardDescription>
        )}
      </CardHeader>
      <CardContent>
        {error ? (
          <Alert variant="destructive">
            <AlertTriangle className="h-4 w-4" />
            <AlertDescription>
              {error}. Using fallback analysis mode.
            </AlertDescription>
          </Alert>
        ) : (
          <div className="space-y-2">
            <Progress value={progress} className="h-2" />
            <div className="flex justify-between text-sm text-muted-foreground">
              <span>
                {progress < 100 ? 'Downloading and caching model...' : 'Model loaded successfully!'}
              </span>
              <span>{Math.round(progress)}%</span>
            </div>
            {progress < 100 && (
              <p className="text-xs text-muted-foreground">
                This only happens once. Subsequent loads use the cached model.
              </p>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
}