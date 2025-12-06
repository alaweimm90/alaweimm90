import React, { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Copy, Check } from 'lucide-react';
import { Badge } from '@/components/ui/badge';

interface CodeBlockProps {
  code: string;
  language: string;
  title?: string;
  showLineNumbers?: boolean;
  highlightLines?: number[];
  maxHeight?: string;
  collapsible?: boolean;
  copyable?: boolean;
}

export default function CodeBlock({
  code,
  language,
  title,
  showLineNumbers = true,
  copyable = true
}: CodeBlockProps) {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    await navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const lines = code.split('\n');

  const syntaxHighlight = (line: string) => {
    // Simple syntax highlighting for JavaScript/TypeScript
    return line
      .replace(/(import|export|const|let|var|function|class|interface|type)\b/g, 
        '<span class="text-accent">$1</span>')
      .replace(/(from|new|await|async)\b/g, 
        '<span class="text-primary">$1</span>')
      .replace(/(['"`])((?:(?!\1)[^\\]|\\.)*)(\1)/g, 
        '<span class="text-success">$1$2$3</span>')
      .replace(/(\/\/.*)/g, 
        '<span class="text-muted-foreground">$1</span>')
      .replace(/(\{|\}|\[|\]|\(|\))/g, 
        '<span class="text-warning">$1</span>');
  };

  return (
    <div className="relative group">
      <div className="bg-background/90 border border-border/30 rounded-lg overflow-hidden backdrop-blur-sm shadow-lg">
        {title && (
          <div className="flex items-center justify-between px-4 py-3 border-b border-border/30 bg-muted/10">
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-destructive" />
              <div className="w-3 h-3 rounded-full bg-warning" />
              <div className="w-3 h-3 rounded-full bg-success" />
              <span className="ml-2 text-sm font-mono text-foreground/80">
                {title}
              </span>
            </div>
            <Badge variant="outline" className="text-xs font-mono border-accent/30 text-accent">
              {language}
            </Badge>
          </div>
        )}
        
        <div className="relative">
          <div className="p-4 overflow-x-auto font-mono text-sm leading-relaxed bg-background/50">
            {lines.map((line, index) => (
              <div key={index} className="flex">
                {showLineNumbers && (
                  <span className="select-none text-muted-foreground/70 w-8 text-right mr-4 shrink-0 font-medium">
                    {index + 1}
                  </span>
                )}
                <span 
                  className="text-foreground"
                  dangerouslySetInnerHTML={{ 
                    __html: syntaxHighlight(line) || '&nbsp;' 
                  }}
                />
              </div>
            ))}
          </div>

          {copyable && (
            <Button
              size="sm"
              variant="ghost"
              onClick={handleCopy}
              className={`
                absolute top-2 right-2 opacity-0 group-hover:opacity-100 
                transition-all duration-200 bg-background/90 backdrop-blur-sm
                hover:bg-primary/10 border border-border/30 min-h-[44px] min-w-[44px]
                focus-visible:opacity-100 focus-visible:ring-2 focus-visible:ring-primary focus-visible:ring-offset-2
                ${copied ? 'text-success' : 'text-foreground'}
              `}
              aria-label={copied ? 'Code copied to clipboard' : 'Copy code to clipboard'}
            >
              {copied ? (
                <>
                  <Check className="h-4 w-4 mr-1" />
                  Copied
                </>
              ) : (
                <>
                  <Copy className="h-4 w-4 mr-1" />
                  Copy
                </>
              )}
            </Button>
          )}
        </div>
      </div>
    </div>
  );
}