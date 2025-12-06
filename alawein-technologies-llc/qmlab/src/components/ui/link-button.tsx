import React from 'react';
import { cva, type VariantProps } from 'class-variance-authority';
import { cn } from '@/lib/utils';
import { ExternalLink, Download, FileText, Image, Archive } from 'lucide-react';

const linkButtonVariants = cva([
  // Base accessible link styles
  "inline-flex items-center gap-2 rounded-md font-medium transition-all duration-200",
  "focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-accent/50",
  "hover:underline underline-offset-2",
  "min-h-[44px] min-w-[44px] px-3 py-2",
], {
  variants: {
    variant: {
      default: "text-accent hover:text-accent/80",
      secondary: "text-muted hover:text-foreground",
      subtle: "text-muted-foreground hover:text-muted",
    }
  },
  defaultVariants: {
    variant: "default",
  }
});

export interface LinkButtonProps
  extends React.AnchorHTMLAttributes<HTMLAnchorElement>,
    VariantProps<typeof linkButtonVariants> {
  /**
   * Target URL
   */
  href: string;
  /**
   * Opens in new tab/window
   */
  external?: boolean;
  /**
   * File metadata for downloads (e.g., "PDF, 2.3 MB")
   */
  fileMeta?: string;
  /**
   * File type for appropriate icon
   */
  fileType?: 'pdf' | 'image' | 'archive' | 'text' | 'code';
  /**
   * Custom icon component
   */
  icon?: React.ComponentType<{ className?: string; 'aria-hidden'?: boolean }>;
  /**
   * Download behavior
   */
  download?: boolean | string;
}

/**
 * Get appropriate icon for file type
 */
function getFileIcon(fileType?: string, download?: boolean | string): React.ComponentType<any> | null {
  if (download) return Download;
  
  switch (fileType) {
    case 'pdf':
    case 'text':
    case 'code':
      return FileText;
    case 'image':
      return Image;
    case 'archive':
      return Archive;
    default:
      return null;
  }
}

/**
 * Accessible link button for external links and downloads
 * 
 * Guidelines:
 * - Include file metadata in visible text ("Download report (PDF, 2.3 MB)")
 * - Indicate external links ("Open docs (opens in new tab)")
 * - Use descriptive labels that make sense out of context
 * - Minimum 44x44px touch targets
 */
export const LinkButton = React.forwardRef<HTMLAnchorElement, LinkButtonProps>(
  ({ 
    className, 
    variant, 
    href, 
    external, 
    fileMeta, 
    fileType, 
    icon, 
    download, 
    children, 
    ...props 
  }, ref) => {
    // Determine if this is external (new tab behavior)
    const isExternal = external || href.startsWith('http') || href.startsWith('//');
    const isDownload = download !== undefined;
    
    // Build accessible label
    const baseText = children;
    const metaText = fileMeta ? ` (${fileMeta})` : '';
    const externalHint = isExternal && !isDownload ? ' (opens in new tab)' : '';
    const downloadHint = isDownload ? ' (download)' : '';
    
    const visibleText = `${baseText}${metaText}`;
    const accessibleLabel = `${baseText}${metaText}${externalHint}${downloadHint}`;
    
    // Determine icon
    const IconComponent = icon || getFileIcon(fileType, download) || (isExternal ? ExternalLink : null);
    
    // Link attributes
    const linkProps = {
      target: isExternal ? '_blank' : undefined,
      rel: isExternal ? 'noopener noreferrer' : undefined,
      download: download === true ? true : download || undefined,
    };

    return (
      <a
        ref={ref}
        href={href}
        className={cn(linkButtonVariants({ variant }), className)}
        aria-label={accessibleLabel}
        {...linkProps}
        {...props}
      >
        <span className="flex-1">{visibleText}</span>
        
        {IconComponent && (
          <IconComponent 
            aria-hidden="true" 
            className="w-4 h-4 flex-shrink-0" 
          />
        )}
        
        {/* Screen reader hints */}
        {isExternal && !isDownload && (
          <span className="sr-only">(opens in new tab)</span>
        )}
        {isDownload && (
          <span className="sr-only">(download)</span>
        )}
      </a>
    );
  }
);

LinkButton.displayName = "LinkButton";

/**
 * Quick link variants for common use cases
 */
export const ExternalLinkButton = React.forwardRef<HTMLAnchorElement, Omit<LinkButtonProps, 'external'>>(
  (props, ref) => <LinkButton ref={ref} external {...props} />
);

export const DownloadLinkButton = React.forwardRef<HTMLAnchorElement, Omit<LinkButtonProps, 'download'>>(
  (props, ref) => <LinkButton ref={ref} download {...props} />
);

ExternalLinkButton.displayName = "ExternalLinkButton";
DownloadLinkButton.displayName = "DownloadLinkButton";