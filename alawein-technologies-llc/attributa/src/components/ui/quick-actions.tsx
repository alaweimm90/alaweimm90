import React from 'react';
import { Button } from './button';
import { Card } from './card';
import { FileText, Upload, History, Settings, Download } from 'lucide-react';
import { Link } from 'react-router-dom';
import { useAppStore } from '@/store';

export function QuickActions() {
  const { reports = [] } = useAppStore();

  const quickAnalyzeText = (text: string) => {
    // Store sample text and navigate to scan page
    sessionStorage.setItem('quick-analyze', text);
    window.location.href = '/scan';
  };

  const sampleTexts = {
    human: "The old oak tree stood majestically in the center of the village square, its gnarled branches reaching toward the sky like ancient fingers. Children often played beneath its shade during the warm summer months, their laughter echoing through the cobblestone streets.",
    ai: "Artificial intelligence has revolutionized numerous industries by leveraging advanced algorithms and machine learning techniques. These sophisticated systems can process vast amounts of data, enabling organizations to optimize their operations and enhance decision-making processes.",
    academic: "According to Smith et al. (2023), the implementation of neural network architectures in natural language processing has demonstrated significant improvements in text classification tasks (DOI: 10.1000/sample). This finding aligns with previous research conducted by Johnson and Williams (2022)."
  };

  return (
    <Card className="p-6">
      <h3 className="text-lg font-semibold mb-4">Quick Actions</h3>
      
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-6">
        <Button variant="outline" size="sm" asChild>
          <Link to="/scan" className="flex items-center gap-2">
            <FileText className="h-4 w-4" />
            <span>New Analysis</span>
          </Link>
        </Button>
        
        <Button variant="outline" size="sm" asChild>
          <Link to="/workspace" className="flex items-center gap-2">
            <History className="h-4 w-4" />
            <span>Recent</span>
          </Link>
        </Button>
        
        <Button variant="outline" size="sm" asChild>
          <Link to="/settings" className="flex items-center gap-2">
            <Settings className="h-4 w-4" />
            <span>Settings</span>
          </Link>
        </Button>
        
        <Button 
          variant="outline" 
          size="sm"
          onClick={() => {
            const link = document.createElement('a');
            link.href = '/docs/README.md';
            link.download = 'attributa-docs.md';
            link.click();
          }}
          className="flex items-center gap-2"
        >
          <Download className="h-4 w-4" />
          <span>Docs</span>
        </Button>
      </div>

      <div className="space-y-3">
        <h4 className="text-sm font-medium text-muted-foreground">Try Sample Texts</h4>
        
        <div className="space-y-2">
          <Button 
            variant="ghost" 
            size="sm" 
            className="w-full justify-start text-left h-auto p-3"
            onClick={() => quickAnalyzeText(sampleTexts.human)}
          >
            <div>
              <div className="font-medium">Human Writing Sample</div>
              <div className="text-xs text-muted-foreground truncate">
                The old oak tree stood majestically...
              </div>
            </div>
          </Button>
          
          <Button 
            variant="ghost" 
            size="sm" 
            className="w-full justify-start text-left h-auto p-3"
            onClick={() => quickAnalyzeText(sampleTexts.ai)}
          >
            <div>
              <div className="font-medium">AI-Generated Sample</div>
              <div className="text-xs text-muted-foreground truncate">
                Artificial intelligence has revolutionized...
              </div>
            </div>
          </Button>
          
          <Button 
            variant="ghost" 
            size="sm" 
            className="w-full justify-start text-left h-auto p-3"
            onClick={() => quickAnalyzeText(sampleTexts.academic)}
          >
            <div>
              <div className="font-medium">Academic Text Sample</div>
              <div className="text-xs text-muted-foreground truncate">
                According to Smith et al. (2023)...
              </div>
            </div>
          </Button>
        </div>
      </div>

      {reports && reports.length > 0 && (
        <div className="mt-6 pt-4 border-t">
          <h4 className="text-sm font-medium text-muted-foreground mb-2">Recent Analyses</h4>
          <div className="space-y-1">
            {reports.slice(0, 3).map((report, index) => (
              <Button
                key={index}
                variant="ghost"
                size="sm"
                className="w-full justify-start text-left"
                asChild
              >
                <Link to={`/results/${report.docId}`}>
                  <div className="truncate">
                    {report.sourceTitle || `Analysis ${report.docId.slice(0, 8)}`}
                  </div>
                </Link>
              </Button>
            ))}
          </div>
        </div>
      )}
    </Card>
  );
}