import React, { Component, ErrorInfo, ReactNode } from 'react';
import { Button } from '@/components/ui/button';
import { AlertTriangle, RefreshCw, Home } from 'lucide-react';
import { logger } from '@/lib/logger';

interface Props {
  children: ReactNode;
  fallback?: ReactNode;
}

interface State {
  hasError: boolean;
  error?: Error;
  errorInfo?: ErrorInfo;
}

export class QuantumErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    this.setState({
      error,
      errorInfo
    });

    // Log to analytics in production
    if (process.env.NODE_ENV === 'production') {
      logger.error('Quantum Error Boundary caught an error', { error, errorInfo });
    }
  }

  handleReset = () => {
    this.setState({ hasError: false, error: undefined, errorInfo: undefined });
  };

  handleGoHome = () => {
    window.location.href = '/';
  };

  render() {
    if (this.state.hasError) {
      if (this.props.fallback) {
        return this.props.fallback;
      }

      return (
        <div className="min-h-screen bg-slate-950 flex items-center justify-center p-6">
          <div className="max-w-2xl w-full">
            {/* Quantum Error Visual */}
            <div className="text-center mb-8">
              <div className="relative inline-block">
                <div className="w-24 h-24 mx-auto mb-6 rounded-full border-2 border-red-400/50 bg-gradient-to-br from-red-500/20 via-slate-900/50 to-slate-900/30 flex items-center justify-center animate-quantum-pulse-glow">
                  <AlertTriangle className="w-12 h-12 text-red-400" />
                  
                  {/* Quantum orbit rings */}
                  <div className="absolute inset-0 border border-red-400/30 rounded-full animate-quantum-orbit"></div>
                  <div className="absolute inset-2 border border-red-400/20 rounded-full animate-quantum-orbit" style={{ animationDuration: '6s', animationDirection: 'reverse' }}></div>
                </div>
                
                <h1 className="text-4xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-red-400 via-red-300 to-red-400 mb-4">
                  Quantum State Collapsed
                </h1>
                
                <p className="text-xl text-slate-400 mb-2">
                  A quantum error occurred in the system
                </p>
                <p className="text-sm text-slate-500">
                  The quantum superposition has been disrupted. Let's restore the system to a stable state.
                </p>
              </div>
            </div>

            {/* Error Details Card */}
            <div className="rounded-3xl border border-red-400/30 bg-gradient-to-br from-red-500/10 via-slate-900/50 to-slate-900/30 backdrop-blur-sm shadow-2xl p-8 mb-8">
              <h2 className="text-xl font-semibold text-red-300 mb-4">Error Details</h2>
              
              <div className="space-y-3">
                <div className="p-4 rounded-xl bg-slate-800/50 border border-slate-700/50">
                  <div className="text-sm font-mono text-red-300">
                    {this.state.error?.name || 'Unknown Error'}
                  </div>
                  <div className="text-sm text-slate-400 mt-1">
                    {this.state.error?.message || 'An unexpected error occurred in the quantum system'}
                  </div>
                </div>

                {process.env.NODE_ENV === 'development' && this.state.errorInfo && (
                  <details className="p-4 rounded-xl bg-slate-800/50 border border-slate-700/50">
                    <summary className="text-sm font-semibold text-slate-300 cursor-pointer hover:text-slate-200">
                      Stack Trace (Development)
                    </summary>
                    <pre className="text-xs text-slate-500 mt-3 overflow-auto max-h-40 whitespace-pre-wrap">
                      {this.state.errorInfo.componentStack}
                    </pre>
                  </details>
                )}
              </div>
            </div>

            {/* Recovery Actions */}
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <Button
                onClick={this.handleReset}
                className="flex items-center gap-3 h-12 px-8 bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-500 hover:to-purple-500 text-white rounded-xl font-medium transition-all duration-300 shadow-lg hover:shadow-xl"
              >
                <RefreshCw className="w-5 h-5" />
                Restore Quantum State
              </Button>
              
              <Button
                onClick={this.handleGoHome}
                variant="outline"
                className="flex items-center gap-3 h-12 px-8 border-slate-600 text-slate-300 hover:bg-slate-800 hover:text-white hover:border-slate-500 rounded-xl font-medium transition-all duration-300"
              >
                <Home className="w-5 h-5" />
                Return to Laboratory
              </Button>
            </div>

            {/* Quantum Particles for Visual Effect */}
            <div className="absolute inset-0 pointer-events-none overflow-hidden">
              <div className="absolute top-1/4 left-1/4 w-1 h-1 bg-red-400 rounded-full animate-quantum-particle-drift opacity-60"></div>
              <div className="absolute top-1/2 right-1/3 w-1 h-1 bg-red-300 rounded-full animate-quantum-particle-drift opacity-40" style={{ animationDelay: '1s' }}></div>
              <div className="absolute bottom-1/3 left-1/2 w-1 h-1 bg-red-500 rounded-full animate-quantum-particle-drift opacity-50" style={{ animationDelay: '2s' }}></div>
            </div>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}