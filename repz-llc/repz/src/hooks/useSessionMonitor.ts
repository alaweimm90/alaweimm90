import { useEffect } from 'react';
import { useAuth } from '@/contexts/AuthContext';
import { supabase } from '@/integrations/supabase/client';

export const useSessionMonitor = () => {
  const { session } = useAuth();
  
  useEffect(() => {
    if (!session) return;
    
    const checkSession = async () => {
      try {
        const { data: { session: currentSession } } = await supabase.auth.getSession();
        
        if (!currentSession) {
          console.log('[SESSION] Session expired, redirecting to login');
          window.location.href = '/login';
        }
      } catch (error) {
        console.error('[SESSION] Error checking session:', error);
      }
    };
    
    // Check session every 5 minutes
    const interval = setInterval(checkSession, 5 * 60 * 1000);
    
    // Check session when tab becomes visible
    const handleVisibilityChange = () => {
      if (!document.hidden) {
        console.log('[SESSION] Tab visible, checking session');
        checkSession();
      }
    };
    
    document.addEventListener('visibilitychange', handleVisibilityChange);
    
    return () => {
      clearInterval(interval);
      document.removeEventListener('visibilitychange', handleVisibilityChange);
    };
  }, [session]);
};