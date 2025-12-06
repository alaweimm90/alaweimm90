import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/ui/molecules/Card';
import { Button } from '@/ui/atoms/Button';
import { Checkbox } from '@/ui/atoms/Checkbox';
import { Badge } from '@/ui/atoms/Badge';
import { Alert, AlertDescription } from '@/ui/molecules/Alert';
import { Progress } from '@/components/ui/progress';
import { 
  CheckCircle2, 
  AlertTriangle, 
  Rocket, 
  Database, 
  Shield, 
  Globe,
  Settings,
  Monitor,
  ExternalLink,
  Copy,
  Check
} from 'lucide-react';

interface ChecklistItem {
  id: string;
  title: string;
  description: string;
  category: 'pre-deployment' | 'deployment' | 'post-deployment';
  priority: 'critical' | 'high' | 'medium';
  completed: boolean;
  command?: string;
  link?: string;
}

const DeploymentChecklist: React.FC = () => {
  const [copied, setCopied] = useState<string>('');

  const [checklistItems, setChecklistItems] = useState<ChecklistItem[]>([
    // Pre-deployment
    {
      id: 'env-prod',
      title: 'Set Production Environment Variables',
      description: 'Configure all required environment variables for production',
      category: 'pre-deployment',
      priority: 'critical',
      completed: false,
      command: 'VITE_SUPABASE_URL=your_prod_url\nVITE_SUPABASE_ANON_KEY=your_prod_key\nVITE_STRIPE_PUBLISHABLE_KEY=your_stripe_key'
    },
    {
      id: 'supabase-prod',
      title: 'Create Supabase Production Project',
      description: 'Set up production Supabase project and apply schema',
      category: 'pre-deployment',
      priority: 'critical',
      completed: false,
      link: 'https://supabase.com/dashboard'
    },
    {
      id: 'stripe-live',
      title: 'Enable Stripe Live Mode',
      description: 'Switch from test to live Stripe keys',
      category: 'pre-deployment',
      priority: 'critical',
      completed: false,
      link: 'https://dashboard.stripe.com'
    },
    {
      id: 'build-test',
      title: 'Test Production Build',
      description: 'Ensure production build works without errors',
      category: 'pre-deployment',
      priority: 'high',
      completed: false,
      command: 'npm run build'
    },
    {
      id: 'security-audit',
      title: 'Run Security Audit',
      description: 'Check for security vulnerabilities and fix critical issues',
      category: 'pre-deployment',
      priority: 'high',
      completed: false,
      command: 'npm audit'
    },
    {
      id: 'lighthouse-audit',
      title: 'Performance Audit',
      description: 'Run Lighthouse audit and optimize performance issues',
      category: 'pre-deployment',
      priority: 'medium',
      completed: false
    },

    // Deployment
    {
      id: 'deploy-app',
      title: 'Deploy Application',
      description: 'Deploy to hosting platform (Vercel, Netlify, etc.)',
      category: 'deployment',
      priority: 'critical',
      completed: false,
      command: 'npm run deploy'
    },
    {
      id: 'configure-domain',
      title: 'Configure Custom Domain',
      description: 'Set up custom domain and SSL certificates',
      category: 'deployment',
      priority: 'high',
      completed: false
    },
    {
      id: 'supabase-functions',
      title: 'Deploy Supabase Functions',
      description: 'Deploy all edge functions to production',
      category: 'deployment',
      priority: 'high',
      completed: false,
      command: 'supabase functions deploy'
    },
    {
      id: 'database-migrate',
      title: 'Run Database Migrations',
      description: 'Apply all database migrations to production',
      category: 'deployment',
      priority: 'critical',
      completed: false,
      command: 'supabase db push'
    },

    // Post-deployment
    {
      id: 'smoke-tests',
      title: 'Run Smoke Tests',
      description: 'Test critical user flows in production',
      category: 'post-deployment',
      priority: 'critical',
      completed: false
    },
    {
      id: 'monitoring-setup',
      title: 'Set Up Monitoring',
      description: 'Configure error monitoring and analytics',
      category: 'post-deployment',
      priority: 'high',
      completed: false
    },
    {
      id: 'backup-verify',
      title: 'Verify Backups',
      description: 'Ensure database backups are working',
      category: 'post-deployment',
      priority: 'high',
      completed: false
    },
    {
      id: 'performance-metrics',
      title: 'Establish Performance Metrics',
      description: 'Record initial performance metrics',
      category: 'post-deployment',
      priority: 'medium',
      completed: false
    }
  ]);

  const toggleItem = (id: string) => {
    setChecklistItems(items =>
      items.map(item =>
        item.id === id ? { ...item, completed: !item.completed } : item
      )
    );
  };

  const copyToClipboard = (text: string, id: string) => {
    navigator.clipboard.writeText(text);
    setCopied(id);
    setTimeout(() => setCopied(''), 2000);
  };

  const getCategoryIcon = (category: ChecklistItem['category']) => {
    switch (category) {
      case 'pre-deployment': return Settings;
      case 'deployment': return Rocket;
      case 'post-deployment': return Monitor;
    }
  };

  const getPriorityColor = (priority: ChecklistItem['priority']) => {
    switch (priority) {
      case 'critical': return 'destructive';
      case 'high': return 'secondary';
      case 'medium': return 'outline';
    }
  };

  const categories = [
    { key: 'pre-deployment', title: 'Pre-Deployment' },
    { key: 'deployment', title: 'Deployment' },
    { key: 'post-deployment', title: 'Post-Deployment' }
  ];

  const getProgress = () => {
    const completed = checklistItems.filter(item => item.completed).length;
    return Math.round((completed / checklistItems.length) * 100);
  };

  const getCriticalIncomplete = () => {
    return checklistItems.filter(item => !item.completed && item.priority === 'critical').length;
  };

  const progress = getProgress();
  const criticalIncomplete = getCriticalIncomplete();

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="flex items-center gap-2">
                <Rocket className="h-5 w-5" />
                Deployment Checklist
              </CardTitle>
              <p className="text-sm text-muted-foreground mt-1">
                Step-by-step guide for production deployment
              </p>
            </div>
            <div className="text-right">
              <div className="text-2xl font-bold">{progress}%</div>
              <div className="text-sm text-muted-foreground">Complete</div>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <Progress value={progress} className="h-3" />
            
            {criticalIncomplete > 0 && (
              <Alert>
                <AlertTriangle className="h-4 w-4" />
                <AlertDescription>
                  {criticalIncomplete} critical item{criticalIncomplete !== 1 ? 's' : ''} remaining. 
                  Complete these before deploying to production.
                </AlertDescription>
              </Alert>
            )}
            
            {progress === 100 && (
              <Alert className="border-green-200 bg-green-50">
                <CheckCircle2 className="h-4 w-4 text-green-600" />
                <AlertDescription className="text-green-800">
                  All deployment steps completed! Your application is ready for production.
                </AlertDescription>
              </Alert>
            )}
          </div>
        </CardContent>
      </Card>

      {categories.map(category => {
        const categoryItems = checklistItems.filter(item => item.category === category.key);
        const CategoryIcon = getCategoryIcon(category.key as ChecklistItem['category']);
        const completed = categoryItems.filter(item => item.completed).length;
        
        return (
          <Card key={category.key}>
            <CardHeader>
              <CardTitle className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <CategoryIcon className="h-5 w-5" />
                  {category.title}
                </div>
                <Badge variant="outline">
                  {completed}/{categoryItems.length} Complete
                </Badge>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {categoryItems.map(item => (
                  <div 
                    key={item.id} 
                    className={`border rounded-lg p-4 transition-colors ${
                      item.completed ? 'bg-green-50 border-green-200' : 'bg-background'
                    }`}
                  >
                    <div className="flex items-start gap-3">
                      <Checkbox
                        checked={item.completed}
                        onCheckedChange={() => toggleItem(item.id)}
                        className="mt-1"
                      />
                      
                      <div className="flex-1 space-y-2">
                        <div className="flex items-center gap-2">
                          <h4 className={`font-medium ${item.completed ? 'line-through text-muted-foreground' : ''}`}>
                            {item.title}
                          </h4>
                          <Badge variant={getPriorityColor(item.priority)}>
                            {item.priority}
                          </Badge>
                        </div>
                        
                        <p className={`text-sm ${item.completed ? 'text-muted-foreground' : 'text-foreground'}`}>
                          {item.description}
                        </p>
                        
                        {item.command && (
                          <div className="bg-muted p-3 rounded-md">
                            <div className="flex items-center justify-between">
                              <code className="text-sm font-mono">{item.command}</code>
                              <Button
                                size="sm"
                                variant="ghost"
                                onClick={() => copyToClipboard(item.command!, item.id)}
                              >
                                {copied === item.id ? (
                                  <Check className="h-4 w-4 text-green-600" />
                                ) : (
                                  <Copy className="h-4 w-4" />
                                )}
                              </Button>
                            </div>
                          </div>
                        )}
                        
                        {item.link && (
                          <Button
                            size="sm"
                            variant="outline"
                            asChild
                           onClick={() => console.log("DeploymentChecklist button clicked")}>
                            <a href={item.link} target="_blank" rel="noopener noreferrer">
                              <ExternalLink className="h-4 w-4 mr-2" />
                              Open Resource
                            </a>
                          </Button>
                        )}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        );
      })}
    </div>
  );
};

export default DeploymentChecklist;