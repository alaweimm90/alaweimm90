import React, { useState, useCallback, useRef, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Slider } from '@/components/ui/slider';
import { useIsMobile } from '@/hooks/use-mobile';
import { Smartphone, Zap, RotateCcw, Play, Pause } from 'lucide-react';

interface MobilePhysicsControlsProps {
  parameters: Record<string, { value: number; min: number; max: number; step: number; label: string }>;
  onParameterChange: (key: string, value: number) => void;
  onPlay?: () => void;
  onPause?: () => void;
  onReset?: () => void;
  isPlaying?: boolean;
}

interface TouchGesture {
  startX: number;
  startY: number;
  currentX: number;
  currentY: number;
  startTime: number;
  parameter?: string;
  initialValue?: number;
}

export const MobilePhysicsControls: React.FC<MobilePhysicsControlsProps> = ({
  parameters,
  onParameterChange,
  onPlay,
  onPause,
  onReset,
  isPlaying = false
}) => {
  const isMobile = useIsMobile();
  const [activeGesture, setActiveGesture] = useState<TouchGesture | null>(null);
  const [gestureParameter, setGestureParameter] = useState<string | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  // Touch gesture handling
  const handleTouchStart = useCallback((e: React.TouchEvent, paramKey: string) => {
    if (!isMobile) return;
    
    const touch = e.touches[0];
    const gesture: TouchGesture = {
      startX: touch.clientX,
      startY: touch.clientY,
      currentX: touch.clientX,
      currentY: touch.clientY,
      startTime: performance.now(),
      parameter: paramKey,
      initialValue: parameters[paramKey]?.value
    };
    
    setActiveGesture(gesture);
    setGestureParameter(paramKey);
    e.preventDefault();
  }, [isMobile, parameters]);

  const handleTouchMove = useCallback((e: React.TouchEvent) => {
    if (!activeGesture || !activeGesture.parameter) return;
    
    const touch = e.touches[0];
    const deltaX = touch.clientX - activeGesture.startX;
    const deltaY = touch.clientY - activeGesture.startY;
    
    // Use horizontal movement for value adjustment
    const param = parameters[activeGesture.parameter];
    if (!param) return;
    
    const range = param.max - param.min;
    const sensitivity = range / 200; // 200px for full range
    const delta = deltaX * sensitivity;
    const newValue = Math.max(param.min, Math.min(param.max, 
      (activeGesture.initialValue || param.value) + delta));
    
    onParameterChange(activeGesture.parameter, newValue);
    
    setActiveGesture({
      ...activeGesture,
      currentX: touch.clientX,
      currentY: touch.clientY
    });
    
    e.preventDefault();
  }, [activeGesture, parameters, onParameterChange]);

  const handleTouchEnd = useCallback((e: React.TouchEvent) => {
    setActiveGesture(null);
    setGestureParameter(null);
    e.preventDefault();
  }, []);

  // Format parameter value for display
  const formatValue = (value: number, step: number): string => {
    const decimals = step < 1 ? 2 : step < 0.1 ? 3 : 1;
    return value.toFixed(decimals);
  };

  // Get touch-optimized button size
  const getButtonClass = () => {
    return isMobile 
      ? "h-12 px-6 text-base touch-manipulation" 
      : "h-10 px-4 text-sm";
  };

  // Get touch-optimized slider class
  const getSliderClass = () => {
    return isMobile 
      ? "touch-manipulation" 
      : "";
  };

  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Smartphone className="w-5 h-5 text-accent" />
          Physics Controls
          {isMobile && (
            <span className="text-xs text-muted-foreground ml-auto">
              Touch & Drag
            </span>
          )}
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-6">
          {/* Playback Controls */}
          <div className="flex gap-3 justify-center">
            <Button
              variant={isPlaying ? "secondary" : "default"}
              className={getButtonClass()}
              onClick={isPlaying ? onPause : onPlay}
            >
              {isPlaying ? (
                <>
                  <Pause className="w-4 h-4 mr-2" />
                  Pause
                </>
              ) : (
                <>
                  <Play className="w-4 h-4 mr-2" />
                  Play
                </>
              )}
            </Button>
            <Button
              variant="outline"
              className={getButtonClass()}
              onClick={onReset}
            >
              <RotateCcw className="w-4 h-4 mr-2" />
              Reset
            </Button>
          </div>

          {/* Parameter Controls */}
          <div 
            ref={containerRef}
            className="space-y-4"
            onTouchMove={handleTouchMove}
            onTouchEnd={handleTouchEnd}
          >
            {Object.entries(parameters).map(([key, param]) => (
              <div key={key} className="space-y-2">
                <div className="flex justify-between items-center">
                  <label className="text-sm font-medium">
                    {param.label}
                  </label>
                  <div className="flex items-center gap-2">
                    <span className={`text-sm font-mono px-2 py-1 rounded ${
                      gestureParameter === key 
                        ? 'bg-accent/20 text-accent' 
                        : 'bg-muted text-muted-foreground'
                    }`}>
                      {formatValue(param.value, param.step)}
                    </span>
                  </div>
                </div>
                
                {isMobile ? (
                  // Touch-optimized control area
                  <div 
                    className={`relative p-4 rounded-lg border-2 transition-colors ${
                      gestureParameter === key 
                        ? 'border-accent bg-accent/10' 
                        : 'border-muted bg-muted/20'
                    }`}
                    onTouchStart={(e) => handleTouchStart(e, key)}
                  >
                    <div className="text-center">
                      <div className="text-xs text-muted-foreground mb-1">
                        Touch & drag horizontally
                      </div>
                      <div className="h-2 bg-muted rounded-full relative overflow-hidden">
                        <div 
                          className="h-full bg-gradient-quantum rounded-full transition-all"
                          style={{
                            width: `${((param.value - param.min) / (param.max - param.min)) * 100}%`
                          }}
                        />
                      </div>
                    </div>
                  </div>
                ) : (
                  // Standard slider for desktop
                  <Slider
                    value={[param.value]}
                    onValueChange={([value]) => onParameterChange(key, value)}
                    min={param.min}
                    max={param.max}
                    step={param.step}
                    className={getSliderClass()}
                  />
                )}
              </div>
            ))}
          </div>

          {/* Mobile-specific tips */}
          {isMobile && (
            <div className="mt-6 p-3 bg-muted/30 rounded-lg">
              <div className="text-xs text-muted-foreground">
                <div className="font-medium mb-1">Touch Controls:</div>
                <div>• Drag horizontally to adjust values</div>
                <div>• Use two fingers to zoom in 3D views</div>
                <div>• Tap and hold for precise control</div>
              </div>
            </div>
          )}

          {/* Performance indicator for mobile */}
          {isMobile && (
            <div className="flex items-center justify-between text-xs text-muted-foreground">
              <span>Mobile Optimized</span>
              <div className="flex items-center gap-1">
                <div className="w-2 h-2 rounded-full bg-green-500" />
                <span>Touch Ready</span>
              </div>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
};