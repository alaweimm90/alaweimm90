import { LucideIcon, TrendingUp, TrendingDown } from 'lucide-react'
import { Card } from '@/components/ui/card'
import { cn } from '@/lib/utils'

interface StatsCardProps {
  title: string
  value: string | number
  change: string
  icon: LucideIcon
  trend: 'up' | 'down' | 'neutral'
}

export function StatsCard({ title, value, change, icon: Icon, trend }: StatsCardProps) {
  return (
    <Card className="p-6">
      <div className="flex items-center justify-between mb-4">
        <div className="w-12 h-12 bg-primary/10 rounded-lg flex items-center justify-center">
          <Icon className="w-6 h-6 text-primary" />
        </div>
        {trend !== 'neutral' && (
          <div className={cn(
            'flex items-center gap-1 text-sm',
            trend === 'up' ? 'text-green-600' : 'text-red-600'
          )}>
            {trend === 'up' ? (
              <TrendingUp className="w-4 h-4" />
            ) : (
              <TrendingDown className="w-4 h-4" />
            )}
            <span>{change}</span>
          </div>
        )}
      </div>
      <div>
        <p className="text-2xl font-bold">{value}</p>
        <p className="text-sm text-muted-foreground mt-1">{title}</p>
        {trend === 'neutral' && (
          <p className="text-xs text-muted-foreground mt-1">{change}</p>
        )}
      </div>
    </Card>
  )
}