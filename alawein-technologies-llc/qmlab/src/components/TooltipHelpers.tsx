import React from 'react';
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";

export interface GateTooltipProps {
  children: React.ReactNode;
  gateName: string;
  description: string;
  keyboardShortcut?: string;
  disabled?: boolean;
}

export const GateTooltip: React.FC<GateTooltipProps> = ({
  children,
  gateName,
  description,
  keyboardShortcut,
  disabled = false
}) => {
  if (disabled) {
    return <>{children}</>;
  }

  return (
    <Tooltip>
      <TooltipTrigger asChild>
        {children}
      </TooltipTrigger>
      <TooltipContent side="top" className="max-w-xs">
        <div className="space-y-1">
          <div className="font-semibold text-sm">{gateName} Gate</div>
          <div className="text-xs text-muted-foreground leading-relaxed">
            {description}
          </div>
          {keyboardShortcut && (
            <div className="text-xs text-primary flex items-center gap-1 mt-2">
              <kbd className="px-1.5 py-0.5 bg-muted rounded text-xs font-mono">
                {keyboardShortcut}
              </kbd>
              <span>to add quickly</span>
            </div>
          )}
        </div>
      </TooltipContent>
    </Tooltip>
  );
};

export interface BlochTooltipProps {
  children: React.ReactNode;
  feature: string;
  description: string;
  disabled?: boolean;
}

export const BlochTooltip: React.FC<BlochTooltipProps> = ({
  children,
  feature,
  description,
  disabled = false
}) => {
  if (disabled) {
    return <>{children}</>;
  }

  return (
    <Tooltip>
      <TooltipTrigger asChild>
        {children}
      </TooltipTrigger>
      <TooltipContent side="bottom" className="max-w-xs">
        <div className="space-y-1">
          <div className="font-semibold text-sm">{feature}</div>
          <div className="text-xs text-muted-foreground leading-relaxed">
            {description}
          </div>
        </div>
      </TooltipContent>
    </Tooltip>
  );
};

export interface TrainingTooltipProps {
  children: React.ReactNode;
  parameter: string;
  description: string;
  currentValue?: string | number;
  recommendedRange?: string;
  disabled?: boolean;
}

export const TrainingTooltip: React.FC<TrainingTooltipProps> = ({
  children,
  parameter,
  description,
  currentValue,
  recommendedRange,
  disabled = false
}) => {
  if (disabled) {
    return <>{children}</>;
  }

  return (
    <Tooltip>
      <TooltipTrigger asChild>
        {children}
      </TooltipTrigger>
      <TooltipContent side="top" className="max-w-sm">
        <div className="space-y-2">
          <div className="font-semibold text-sm">{parameter}</div>
          <div className="text-xs text-muted-foreground leading-relaxed">
            {description}
          </div>
          {currentValue !== undefined && (
            <div className="text-xs">
              <span className="text-muted-foreground">Current: </span>
              <span className="text-primary font-mono">{currentValue}</span>
            </div>
          )}
          {recommendedRange && (
            <div className="text-xs">
              <span className="text-muted-foreground">Recommended: </span>
              <span className="text-secondary font-mono">{recommendedRange}</span>
            </div>
          )}
        </div>
      </TooltipContent>
    </Tooltip>
  );
};

export interface EmptyStateTooltipProps {
  children: React.ReactNode;
  title: string;
  description: string;
  suggestion: string;
  disabled?: boolean;
}

export const EmptyStateTooltip: React.FC<EmptyStateTooltipProps> = ({
  children,
  title,
  description,
  suggestion,
  disabled = false
}) => {
  if (disabled) {
    return <>{children}</>;
  }

  return (
    <Tooltip>
      <TooltipTrigger asChild>
        {children}
      </TooltipTrigger>
      <TooltipContent side="right" className="max-w-sm">
        <div className="space-y-2">
          <div className="font-semibold text-sm">{title}</div>
          <div className="text-xs text-muted-foreground leading-relaxed">
            {description}
          </div>
          <div className="text-xs text-primary border-t border-primary/20 pt-2">
            💡 {suggestion}
          </div>
        </div>
      </TooltipContent>
    </Tooltip>
  );
};

// Predefined tooltips for common quantum gates
export const QuantumGateTooltips = {
  X: {
    name: "Pauli-X",
    description: "Flips the qubit state from |0⟩ to |1⟩ or vice versa. Acts like a classical NOT gate.",
    shortcut: "X"
  },
  Y: {
    name: "Pauli-Y", 
    description: "Rotates the qubit around the Y-axis of the Bloch sphere. Combines bit-flip and phase-flip.",
    shortcut: "Y"
  },
  Z: {
    name: "Pauli-Z",
    description: "Flips the phase of the |1⟩ state. Leaves |0⟩ unchanged but adds a negative sign to |1⟩.",
    shortcut: "Z"
  },
  H: {
    name: "Hadamard",
    description: "Creates superposition by putting a qubit into an equal combination of |0⟩ and |1⟩.",
    shortcut: "H"
  },
  S: {
    name: "Phase",
    description: "Applies a 90° phase shift to the |1⟩ state. Quarter turn around Z-axis on Bloch sphere.",
    shortcut: "S"
  },
  T: {
    name: "π/8",
    description: "Applies a 45° phase shift to the |1⟩ state. Eighth of a turn around Z-axis.",
    shortcut: "T"
  },
  RX: {
    name: "X-Rotation",
    description: "Rotates the qubit by a specified angle around the X-axis of the Bloch sphere.",
    shortcut: "R"
  },
  RY: {
    name: "Y-Rotation", 
    description: "Rotates the qubit by a specified angle around the Y-axis of the Bloch sphere.",
    shortcut: "Shift+R"
  },
  RZ: {
    name: "Z-Rotation",
    description: "Rotates the qubit by a specified angle around the Z-axis (phase rotation).",
    shortcut: "Ctrl+R"
  },
  CNOT: {
    name: "Controlled-NOT",
    description: "Two-qubit gate that flips the target qubit if the control qubit is |1⟩. Creates entanglement.",
    shortcut: "C"
  }
};

// Predefined tooltips for Bloch sphere features
export const BlochSphereTooltips = {
  rotation: {
    feature: "Rotation Controls",
    description: "Click and drag to rotate the view. Use mouse wheel or pinch to zoom in/out."
  },
  state: {
    feature: "Quantum State Vector",
    description: "The red arrow shows the current quantum state. Length represents probability amplitude."
  },
  axes: {
    feature: "Coordinate System", 
    description: "X, Y, Z axes represent the three dimensions of qubit state space in the Bloch sphere."
  },
  pausePlay: {
    feature: "Animation Control",
    description: "Pause or resume the real-time animation to better observe state changes."
  },
  reset: {
    feature: "Reset View",
    description: "Return to the default camera position and zoom level."
  }
};

// Predefined tooltips for training parameters
export const TrainingTooltips = {
  learningRate: {
    parameter: "Learning Rate",
    description: "Controls how much the model weights change with each training step. Higher values train faster but may overshoot the optimal solution.",
    recommendedRange: "0.001 - 0.1"
  },
  batchSize: {
    parameter: "Batch Size", 
    description: "Number of training samples processed together in each update. Larger batches provide more stable gradients.",
    recommendedRange: "16 - 128"
  },
  epochs: {
    parameter: "Training Epochs",
    description: "Number of times the algorithm sees the entire training dataset. More epochs can improve accuracy but may cause overfitting.",
    recommendedRange: "50 - 200"
  },
  optimizer: {
    parameter: "Optimizer",
    description: "Algorithm used to update model weights. Adam is generally robust, SGD is simpler but may need tuning.",
    recommendedRange: "Adam, SGD"
  }
};