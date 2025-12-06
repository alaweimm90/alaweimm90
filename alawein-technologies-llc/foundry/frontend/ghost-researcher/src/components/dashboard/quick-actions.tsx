import { Upload, Search, Brain, FileText, Users, BookOpen } from 'lucide-react'
import Link from 'next/link'
import { cn } from '@/lib/utils'

const actions = [
  {
    title: 'Upload Paper',
    description: 'Import PDF or text files',
    icon: Upload,
    href: '/upload',
    color: 'bg-blue-500/10 text-blue-500 hover:bg-blue-500/20',
  },
  {
    title: 'New Research',
    description: 'Start literature review',
    icon: Search,
    href: '/research/new',
    color: 'bg-purple-500/10 text-purple-500 hover:bg-purple-500/20',
  },
  {
    title: 'Generate Hypothesis',
    description: 'AI-powered insights',
    icon: Brain,
    href: '/hypotheses/generate',
    color: 'bg-green-500/10 text-green-500 hover:bg-green-500/20',
  },
  {
    title: 'Write Paper',
    description: 'Start with AI assistance',
    icon: FileText,
    href: '/papers/new',
    color: 'bg-orange-500/10 text-orange-500 hover:bg-orange-500/20',
  },
  {
    title: 'Invite Collaborator',
    description: 'Share your research',
    icon: Users,
    href: '/collaborators/invite',
    color: 'bg-indigo-500/10 text-indigo-500 hover:bg-indigo-500/20',
  },
  {
    title: 'Browse Library',
    description: 'Explore saved papers',
    icon: BookOpen,
    href: '/library',
    color: 'bg-pink-500/10 text-pink-500 hover:bg-pink-500/20',
  },
]

export function QuickActions() {
  return (
    <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
      {actions.map((action) => (
        <Link
          key={action.title}
          href={action.href}
          className="group"
        >
          <div className="flex flex-col items-center text-center p-4 rounded-lg border transition-all hover:shadow-md hover:scale-105">
            <div className={cn(
              'w-12 h-12 rounded-lg flex items-center justify-center mb-3 transition-colors',
              action.color
            )}>
              <action.icon className="w-6 h-6" />
            </div>
            <h3 className="text-sm font-medium mb-1">{action.title}</h3>
            <p className="text-xs text-muted-foreground">
              {action.description}
            </p>
          </div>
        </Link>
      ))}
    </div>
  )
}