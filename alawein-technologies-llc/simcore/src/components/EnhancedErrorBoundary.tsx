/**
 * Enhanced Error Boundary with Recovery and Reporting
 * Provides comprehensive error handling for scientific modules
 */

import React, { Component, ErrorInfo, ReactNode } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible';
import { AlertTriangle, RefreshCw, Bug, ChevronDown, Copy } from 'lucide-react';
import { toast } from '@/hooks/use-toast';

interface Props {
  children: ReactNode;
  fallback?: ReactNode;
  onError?: (error: Error, errorInfo: ErrorInfo) => void;
  moduleName?: string;
  enableRecovery?: boolean;
  showErrorDetails?: boolean;
}

interface State {
  hasError: boolean;
  error: Error | null;
  errorInfo: ErrorInfo | null;
  errorId: string;
  retryCount: number;
  isRecovering: boolean;
}

export class EnhancedErrorBoundary extends Component<Props, State> {
  private maxRetries = 3;

  constructor(props: Props) {
    super(props);
    this.state = {
      hasError: false,
      error: null,
      errorInfo: null,
      errorId: '',
      retryCount: 0,
      isRecovering: false
    };
  }

  static getDerivedStateFromError(error: Error): Partial<State> {
    const errorId = `ERR_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    return {
      hasError: true,
      error,
      errorId
    };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    this.setState({
      error,
      errorInfo
    });

    // Report error to analytics/monitoring service
    this.reportError(error, errorInfo);

    // Call optional error callback
    this.props.onError?.(error, errorInfo);

    console.error('Enhanced Error Boundary caught an error:', error, errorInfo);
  }

  private reportError = (error: Error, errorInfo: ErrorInfo) => {
    const errorReport = {
      message: error.message,
      stack: error.stack,
      componentStack: errorInfo.componentStack,
      moduleName: this.props.moduleName,
      timestamp: new Date().toISOString(),
      userAgent: navigator.userAgent,
      url: window.location.href,
      errorId: this.state.errorId
    };

    // In a real app, send to error reporting service
    console.warn('Error Report:', errorReport);
  };

  private handleRetry = async () => {
    if (this.state.retryCount >= this.maxRetries) {
      toast({
        title: "Maximum retries reached",
        description: "Please refresh the page or contact support",
        variant: "destructive"
      });
      return;
    }

    this.setState({ isRecovering: true });

    // Add delay for recovery
    await new Promise(resolve => setTimeout(resolve, 1000));

    this.setState({
      hasError: false,
      error: null,
      errorInfo: null,
      retryCount: this.state.retryCount + 1,
      isRecovering: false
    });

    toast({
      title: "Recovery attempted",
      description: `Retry ${this.state.retryCount + 1}/${this.maxRetries}`,
    });
  };

  private handleRefresh = () => {
    window.location.reload();
  };

  private copyErrorDetails = () => {
    const errorDetails = {
      errorId: this.state.errorId,
      message: this.state.error?.message,
      stack: this.state.error?.stack,
      componentStack: this.state.errorInfo?.componentStack,
      moduleName: this.props.moduleName,
      timestamp: new Date().toISOString()
    };

    navigator.clipboard.writeText(JSON.stringify(errorDetails, null, 2));
    toast({
      title: "Error details copied",
      description: "Copied to clipboard for support"
    });
  };

  render() {
    if (this.state.hasError) {
      // Use custom fallback if provided
      if (this.props.fallback) {
        return this.props.fallback;
      }

      return (
        <Card className="border-destructive/50 bg-destructive/5">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-destructive">
              <AlertTriangle className="h-5 w-5" />
              {this.props.moduleName ? `${this.props.moduleName} Error` : 'Application Error'}
            </CardTitle>
            <CardDescription>
              Something went wrong while rendering this component
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex items-center gap-2">
              <Badge variant="destructive" className="font-mono text-xs">
                {this.state.errorId}
              </Badge>
              <span className="text-sm text-muted-foreground">
                Retry {this.state.retryCount}/{this.maxRetries}
              </span>
            </div>

            <div className="flex flex-wrap gap-2">
              {this.props.enableRecovery !== false && this.state.retryCount < this.maxRetries && (
                <Button
                  onClick={this.handleRetry}
                  disabled={this.state.isRecovering}
                  size="sm"
                  className="flex items-center gap-2"
                >
                  <RefreshCw className={`h-4 w-4 ${this.state.isRecovering ? 'animate-spin' : ''}`} />
                  {this.state.isRecovering ? 'Recovering...' : 'Try Again'}
                </Button>
              )}

              <Button
                onClick={this.handleRefresh}
                variant="outline"
                size="sm"
                className="flex items-center gap-2"
              >
                <RefreshCw className="h-4 w-4" />
                Refresh Page
              </Button>

              <Button
                onClick={this.copyErrorDetails}
                variant="outline"
                size="sm"
                className="flex items-center gap-2"
              >
                <Copy className="h-4 w-4" />
                Copy Details
              </Button>
            </div>

            {this.props.showErrorDetails !== false && this.state.error && (
              <Collapsible>
                <CollapsibleTrigger asChild>
                  <Button variant="ghost" size="sm" className="flex items-center gap-2">
                    <Bug className="h-4 w-4" />
                    Error Details
                    <ChevronDown className="h-4 w-4" />
                  </Button>
                </CollapsibleTrigger>
                <CollapsibleContent className="space-y-2">
                  <div className="rounded-md bg-muted p-3 text-sm font-mono">
                    <div className="font-semibold text-destructive mb-2">
                      {this.state.error.message}
                    </div>
                    <div className="text-muted-foreground text-xs whitespace-pre-wrap max-h-32 overflow-y-auto">
                      {this.state.error.stack}
                    </div>
                  </div>
                  
                  {this.state.errorInfo?.componentStack && (
                    <div className="rounded-md bg-muted p-3 text-sm font-mono">
                      <div className="font-semibold mb-2">Component Stack:</div>
                      <div className="text-muted-foreground text-xs whitespace-pre-wrap max-h-32 overflow-y-auto">
                        {this.state.errorInfo.componentStack}
                      </div>
                    </div>
                  )}
                </CollapsibleContent>
              </Collapsible>
            )}

            <div className="text-xs text-muted-foreground">
              If this error persists, please report it with the error ID above.
            </div>
          </CardContent>
        </Card>
      );
    }

    return this.props.children;
  }
}

export default EnhancedErrorBoundary;