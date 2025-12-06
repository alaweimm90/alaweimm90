import { FileText, Brain, Users, Download, Edit, Share2 } from 'lucide-react'
import { formatRelativeTime } from '@/lib/utils'

const activities = [
  {
    id: 1,
    type: 'paper',
    action: 'Analyzed',
    title: 'Quantum Computing in Drug Discovery',
    timestamp: new Date(Date.now() - 1000 * 60 * 30), // 30 minutes ago
    icon: FileText,
    color: 'text-blue-500',
  },
  {
    id: 2,
    type: 'hypothesis',
    action: 'Generated',
    title: 'Novel protein folding mechanism',
    timestamp: new Date(Date.now() - 1000 * 60 * 60 * 2), // 2 hours ago
    icon: Brain,
    color: 'text-purple-500',
  },
  {
    id: 3,
    type: 'collaboration',
    action: 'Shared with',
    title: 'Dr. Sarah Chen',
    timestamp: new Date(Date.now() - 1000 * 60 * 60 * 5), // 5 hours ago
    icon: Share2,
    color: 'text-green-500',
  },
  {
    id: 4,
    type: 'export',
    action: 'Exported',
    title: 'Literature Review - ML in Healthcare',
    timestamp: new Date(Date.now() - 1000 * 60 * 60 * 24), // 1 day ago
    icon: Download,
    color: 'text-orange-500',
  },
  {
    id: 5,
    type: 'edit',
    action: 'Edited',
    title: 'Research proposal draft',
    timestamp: new Date(Date.now() - 1000 * 60 * 60 * 48), // 2 days ago
    icon: Edit,
    color: 'text-indigo-500',
  },
]

export function RecentActivity() {
  return (
    <div className="space-y-4">
      {activities.map((activity) => (
        <div key={activity.id} className="flex items-start gap-3">
          <div className={`mt-1 ${activity.color}`}>
            <activity.icon className="w-4 h-4" />
          </div>
          <div className="flex-1 min-w-0">
            <p className="text-sm">
              <span className="font-medium">{activity.action}</span>{' '}
              <span className="text-muted-foreground truncate">
                {activity.title}
              </span>
            </p>
            <p className="text-xs text-muted-foreground mt-1">
              {formatRelativeTime(activity.timestamp)}
            </p>
          </div>
        </div>
      ))}

      <button className="w-full text-sm text-primary hover:underline mt-4">
        View all activity
      </button>
    </div>
  )
}