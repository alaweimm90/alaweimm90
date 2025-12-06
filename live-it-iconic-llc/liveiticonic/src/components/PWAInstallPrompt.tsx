import React from 'react';
import { usePWA } from '@/hooks/usePWA';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { Download, X } from 'lucide-react';
import { useState } from 'react';

export const PWAInstallPrompt: React.FC = () => {
  const { isInstallable, install } = usePWA();
  const [dismissed, setDismissed] = useState(false);

  if (!isInstallable || dismissed) {
    return null;
  }

  const handleInstall = async () => {
    const success = await install();
    if (success) {
      setDismissed(true);
    }
  };

  return (
    <Card className="fixed bottom-4 right-4 z-50 bg-lii-ink border-lii-gold/20 max-w-sm shadow-lg">
      <CardContent className="p-4">
        <div className="flex items-start gap-4">
          <div className="flex-1">
            <h3 className="text-lii-cloud font-display font-semibold mb-1">Install App</h3>
            <p className="text-lii-ash font-ui text-sm">
              Install Live It Iconic for a better shopping experience with offline access.
            </p>
          </div>
          <button
            onClick={() => setDismissed(true)}
            className="text-lii-ash hover:text-lii-cloud transition-colors"
            aria-label="Dismiss"
          >
            <X className="w-4 h-4" />
          </button>
        </div>
        <div className="flex gap-2 mt-4">
          <Button variant="outline" size="sm" onClick={() => setDismissed(true)}>
            Not Now
          </Button>
          <Button variant="primary" size="sm" onClick={handleInstall}>
            <Download className="w-4 h-4 mr-2" />
            Install
          </Button>
        </div>
      </CardContent>
    </Card>
  );
};

export default PWAInstallPrompt;
