import { memo, useState, useEffect } from "react";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Card } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import { Keyboard, Command, Info } from "lucide-react";

interface KeyboardShortcut {
  keys: string[];
  description: string;
  category: string;
}

const shortcuts: KeyboardShortcut[] = [
  // Navigation
  { keys: ['Alt', 'H'], description: 'Go to home page', category: 'Navigation' },
  { keys: ['Alt', 'A'], description: 'Open Arena evaluation', category: 'Navigation' },
  { keys: ['Alt', 'B'], description: 'Open Bench evaluation', category: 'Navigation' },
  { keys: ['Alt', 'D'], description: 'Open Dashboard', category: 'Navigation' },
  { keys: ['Alt', 'S'], description: 'Open Settings', category: 'Navigation' },
  
  // General
  { keys: ['Tab'], description: 'Move to next focusable element', category: 'General' },
  { keys: ['Shift', 'Tab'], description: 'Move to previous focusable element', category: 'General' },
  { keys: ['Enter'], description: 'Activate focused element', category: 'General' },
  { keys: ['Space'], description: 'Activate buttons, checkboxes', category: 'General' },
  { keys: ['Escape'], description: 'Close modal, cancel action', category: 'General' },
  { keys: ['Alt', 'K'], description: 'Open keyboard shortcuts', category: 'General' },
  { keys: ['Alt', 'T'], description: 'Toggle accessibility toolbar', category: 'General' },
  
  // Forms
  { keys: ['Ctrl', 'Enter'], description: 'Submit form', category: 'Forms' },
  { keys: ['Ctrl', 'R'], description: 'Reset form', category: 'Forms' },
  { keys: ['Arrow Up'], description: 'Previous option in select/radio', category: 'Forms' },
  { keys: ['Arrow Down'], description: 'Next option in select/radio', category: 'Forms' },
  
  // Tables and Lists
  { keys: ['Arrow Keys'], description: 'Navigate table cells', category: 'Tables' },
  { keys: ['Home'], description: 'First cell in row', category: 'Tables' },
  { keys: ['End'], description: 'Last cell in row', category: 'Tables' },
  { keys: ['Ctrl', 'Home'], description: 'First cell in table', category: 'Tables' },
  { keys: ['Ctrl', 'End'], description: 'Last cell in table', category: 'Tables' },
  { keys: ['Ctrl', 'Space'], description: 'Select/deselect row', category: 'Tables' },
  
  // Evaluation
  { keys: ['Ctrl', 'E'], description: 'Start new evaluation', category: 'Evaluation' },
  { keys: ['Ctrl', 'P'], description: 'Pause/resume evaluation', category: 'Evaluation' },
  { keys: ['Ctrl', 'X'], description: 'Stop evaluation', category: 'Evaluation' },
  { keys: ['Ctrl', 'V'], description: 'View results', category: 'Evaluation' },
  
  // Accessibility
  { keys: ['Alt', 'C'], description: 'Toggle high contrast', category: 'Accessibility' },
  { keys: ['Alt', 'L'], description: 'Toggle large text', category: 'Accessibility' },
  { keys: ['Alt', 'M'], description: 'Toggle reduced motion', category: 'Accessibility' },
  { keys: ['Alt', 'R'], description: 'Toggle screen reader mode', category: 'Accessibility' },
];

const categories = Array.from(new Set(shortcuts.map(s => s.category)));

interface KeyboardShortcutsModalProps {
  trigger?: React.ReactNode;
}

export const KeyboardShortcutsModal = memo(({ 
  trigger 
}: KeyboardShortcutsModalProps) => {
  const [open, setOpen] = useState(false);

  // Listen for Alt+K to open shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.altKey && e.key.toLowerCase() === 'k') {
        e.preventDefault();
        setOpen(true);
      }
    };

    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, []);

  const renderKey = (key: string) => (
    <Badge 
      variant="outline" 
      className="px-2 py-1 text-xs font-mono bg-muted text-muted-foreground border-border"
      key={key}
    >
      {key}
    </Badge>
  );

  const defaultTrigger = (
    <Button 
      variant="outline" 
      size="sm" 
      className="flex items-center gap-2"
      aria-label="View keyboard shortcuts"
    >
      <Keyboard className="h-4 w-4" />
      Shortcuts
    </Button>
  );

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        {trigger || defaultTrigger}
      </DialogTrigger>
      
      <DialogContent 
        className="max-w-4xl max-h-[80vh] overflow-y-auto"
        aria-describedby="keyboard-shortcuts-description"
      >
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2 text-xl">
            <Keyboard className="h-5 w-5" />
            Keyboard Shortcuts
          </DialogTitle>
          <DialogDescription id="keyboard-shortcuts-description">
            Use these keyboard shortcuts to navigate and interact with LLM Works more efficiently. 
            Press Alt+K from anywhere to open this dialog.
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-6 mt-4">
          {/* Quick Access Info */}
          <Card className="p-4 bg-primary/5 border-primary/20">
            <div className="flex items-center gap-2 mb-2">
              <Info className="h-4 w-4 text-primary" />
              <span className="font-medium text-primary">Quick Access</span>
            </div>
            <p className="text-sm text-muted-foreground">
              Most shortcuts use the Alt key to avoid conflicts with browser shortcuts. 
              Screen reader users can navigate with standard screen reader commands.
            </p>
          </Card>

          {/* Shortcuts by Category */}
          {categories.map((category) => (
            <div key={category}>
              <h3 className="text-lg font-semibold mb-4 text-foreground">
                {category}
              </h3>
              
              <div className="grid gap-3">
                {shortcuts
                  .filter(shortcut => shortcut.category === category)
                  .map((shortcut, index) => (
                    <div 
                      key={index}
                      className="flex items-center justify-between p-3 rounded-lg border border-border hover:bg-muted/50 transition-colors"
                    >
                      <span className="text-sm text-foreground flex-1">
                        {shortcut.description}
                      </span>
                      <div className="flex items-center gap-1">
                        {shortcut.keys.map((key, keyIndex) => (
                          <span key={keyIndex} className="flex items-center gap-1">
                            {keyIndex > 0 && (
                              <span className="text-xs text-muted-foreground">+</span>
                            )}
                            {renderKey(key)}
                          </span>
                        ))}
                      </div>
                    </div>
                  ))}
              </div>
              
              {category !== categories[categories.length - 1] && (
                <Separator className="mt-6" />
              )}
            </div>
          ))}

          {/* Screen Reader Instructions */}
          <Card className="p-4 bg-secondary/50">
            <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
              <Command className="h-4 w-4" />
              Screen Reader Navigation
            </h3>
            <div className="space-y-2 text-sm text-muted-foreground">
              <p>• Use standard screen reader navigation commands (H for headings, F for forms, etc.)</p>
              <p>• Landmarks and regions are properly labeled for navigation</p>
              <p>• All interactive elements have descriptive labels</p>
              <p>• Status messages are announced automatically</p>
              <p>• Use Tab/Shift+Tab to navigate through interactive elements</p>
              <p>• Enable "Screen Reader Mode" in the accessibility toolbar for enhanced support</p>
            </div>
          </Card>

          {/* Platform-specific Notes */}
          <Card className="p-4">
            <h3 className="text-lg font-semibold mb-3">Platform Notes</h3>
            <div className="grid md:grid-cols-2 gap-4 text-sm text-muted-foreground">
              <div>
                <h4 className="font-medium text-foreground mb-2">Windows/Linux</h4>
                <p>• Alt key combinations work as described</p>
                <p>• Use Ctrl for form submissions</p>
                <p>• F6 to navigate between page regions</p>
              </div>
              <div>
                <h4 className="font-medium text-foreground mb-2">macOS</h4>
                <p>• Use Option instead of Alt</p>
                <p>• Cmd for form submissions</p>
                <p>• Control+Option for VoiceOver navigation</p>
              </div>
            </div>
          </Card>
        </div>
      </DialogContent>
    </Dialog>
  );
});

KeyboardShortcutsModal.displayName = 'KeyboardShortcutsModal';