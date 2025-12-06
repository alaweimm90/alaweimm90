import { Link, useLocation } from 'react-router-dom';
import { History, FileSearch, BarChart3 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { SettingsDrawer } from './SettingsDrawer';
import { useAppStore } from '@/store';

export function MobileNav() {
  const location = useLocation();
  const { currentReport } = useAppStore();
  
  const isActive = (path: string) => {
    if (path === '/' && location.pathname === '/') return true;
    return path !== '/' && location.pathname.startsWith(path);
  };

  return (
    <nav className="fixed bottom-0 inset-x-0 md:hidden border-t bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/80 z-50">
      <div className="grid grid-cols-5 h-14">
        <Button
          variant={isActive('/workspace') ? 'secondary' : 'ghost'}
          size="sm"
          asChild
          className="h-full rounded-none flex-col gap-1 px-2"
        >
          <Link to="/workspace">
            <History className="h-4 w-4" />
            <span className="text-xs">Workspace</span>
          </Link>
        </Button>
        
        <Button
          variant={isActive('/scan') ? 'secondary' : 'ghost'}
          size="sm"
          asChild
          className="h-full rounded-none flex-col gap-1 px-2"
        >
          <Link to="/scan">
            <FileSearch className="h-4 w-4" />
            <span className="text-xs">Scan</span>
          </Link>
        </Button>
        
        
        <Button
          variant={isActive('/results') ? 'secondary' : 'ghost'}
          size="sm"
          asChild
          className="h-full rounded-none flex-col gap-1 px-2"
          disabled={!currentReport}
        >
          <Link to={currentReport ? `/results/${currentReport.docId}` : '#'}>
            <BarChart3 className="h-4 w-4" />
            <span className="text-xs">Results</span>
          </Link>
        </Button>
        
        <SettingsDrawer />
      </div>
    </nav>
  );
}