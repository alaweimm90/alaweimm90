import { useState } from 'react';
import { Settings as SettingsIcon, Book, Search, User, Github } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Link } from 'react-router-dom';
import FocusTrap from 'focus-trap-react';
import {
  Drawer,
  DrawerContent,
  DrawerHeader,
  DrawerTitle,
  DrawerTrigger,
} from '@/components/ui/drawer';
import { useAuth } from '@/hooks/useAuth';
import { supabase } from '@/integrations/supabase/client';
import Settings from '@/pages/Settings';

export function SettingsDrawer() {
  const [open, setOpen] = useState(false);
  const [commandPaletteOpen, setCommandPaletteOpen] = useState(false);
  const { user } = useAuth();

  return (
    <Drawer open={open} onOpenChange={setOpen}>
      <DrawerTrigger asChild>
        <Button
          variant="ghost"
          size="sm"
          className="h-9 w-9 p-0"
        >
          <SettingsIcon className="h-4 w-4" />
        </Button>
      </DrawerTrigger>
      <DrawerContent className="max-h-[85vh]">
        <FocusTrap active={open}>
          <DrawerHeader>
            <DrawerTitle>Menu</DrawerTitle>
          </DrawerHeader>
          <div className="overflow-y-auto p-4 space-y-2">
          <Button variant="ghost" className="w-full justify-start" asChild>
            <a href="#methods" onClick={() => setOpen(false)}>
              <Book className="h-4 w-4 mr-2" />
              Methods
            </a>
          </Button>
          
          <Button variant="ghost" className="w-full justify-start" onClick={() => { setCommandPaletteOpen(true); setOpen(false); }}>
            <Search className="h-4 w-4 mr-2" />
            Search
          </Button>
          
          <Button variant="ghost" className="w-full justify-start" asChild>
            <a href="https://github.com/alaweimm90/Attributa" target="_blank" rel="noopener noreferrer" onClick={() => setOpen(false)}>
              <Github className="h-4 w-4 mr-2" />
              GitHub Repository
            </a>
          </Button>
          
          <Button variant="ghost" className="w-full justify-start" asChild>
            <Link to="/settings" onClick={() => setOpen(false)}>
              <SettingsIcon className="h-4 w-4 mr-2" />
              Settings
            </Link>
          </Button>
          
          <div className="border-t pt-2">
            {user ? (
              <Button variant="ghost" className="w-full justify-start" onClick={() => { supabase.auth.signOut(); setOpen(false); }}>
                <User className="h-4 w-4 mr-2" />
                Sign out
              </Button>
            ) : (
              <Button variant="ghost" className="w-full justify-start" asChild>
                <Link to="/auth" onClick={() => setOpen(false)}>
                  <User className="h-4 w-4 mr-2" />
                  Sign in (optional)
                </Link>
              </Button>
            )}
          </div>
          </div>
        </FocusTrap>
      </DrawerContent>
    </Drawer>
  );
}