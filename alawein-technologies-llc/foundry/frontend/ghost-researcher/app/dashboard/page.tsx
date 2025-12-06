'use client'

import { useState } from 'react'
import { motion } from 'framer-motion'
import {
  Search, FileText, Brain, Users, TrendingUp, Clock,
  ArrowRight, Plus, Filter, Grid, List, Download
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { DashboardLayout } from '@/components/layouts/dashboard-layout'
import { StatsCard } from '@/components/dashboard/stats-card'
import { RecentActivity } from '@/components/dashboard/recent-activity'
import { QuickActions } from '@/components/dashboard/quick-actions'
import { ResearchProjects } from '@/components/dashboard/research-projects'
import { useAuth } from '@/components/providers/auth-provider'
import Link from 'next/link'

const stats = [
  {
    title: 'Papers Analyzed',
    value: '1,234',
    change: '+12%',
    icon: FileText,
    trend: 'up',
  },
  {
    title: 'Hypotheses Generated',
    value: '89',
    change: '+23%',
    icon: Brain,
    trend: 'up',
  },
  {
    title: 'Collaborators',
    value: '12',
    change: '+2',
    icon: Users,
    trend: 'up',
  },
  {
    title: 'Time Saved',
    value: '156h',
    change: 'This month',
    icon: Clock,
    trend: 'neutral',
  },
]

export default function DashboardPage() {
  const { user } = useAuth()
  const [view, setView] = useState<'grid' | 'list'>('grid')

  return (
    <DashboardLayout>
      <div className="space-y-8">
        {/* Header */}
        <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
          <div>
            <h1 className="text-3xl font-bold">
              Welcome back, {user?.name || 'Researcher'}!
            </h1>
            <p className="text-muted-foreground mt-1">
              Here's what's happening with your research today.
            </p>
          </div>
          <Link href="/research/new">
            <Button className="gap-2">
              <Plus className="w-4 h-4" />
              New Research
            </Button>
          </Link>
        </div>

        {/* Stats Grid */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
          {stats.map((stat, index) => (
            <motion.div
              key={stat.title}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.1 }}
            >
              <StatsCard {...stat} />
            </motion.div>
          ))}
        </div>

        {/* Main Content Grid */}
        <div className="grid lg:grid-cols-3 gap-6">
          {/* Left Column - Projects & Quick Actions */}
          <div className="lg:col-span-2 space-y-6">
            {/* Quick Actions */}
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.2 }}
            >
              <Card>
                <CardHeader>
                  <CardTitle>Quick Actions</CardTitle>
                  <CardDescription>
                    Jump right into your research workflow
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <QuickActions />
                </CardContent>
              </Card>
            </motion.div>

            {/* Research Projects */}
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.3 }}
            >
              <Card>
                <CardHeader>
                  <div className="flex items-center justify-between">
                    <div>
                      <CardTitle>Research Projects</CardTitle>
                      <CardDescription>
                        Your active research projects
                      </CardDescription>
                    </div>
                    <div className="flex items-center gap-2">
                      <Button
                        variant="outline"
                        size="icon"
                        onClick={() => setView('grid')}
                        className={view === 'grid' ? 'bg-secondary' : ''}
                      >
                        <Grid className="w-4 h-4" />
                      </Button>
                      <Button
                        variant="outline"
                        size="icon"
                        onClick={() => setView('list')}
                        className={view === 'list' ? 'bg-secondary' : ''}
                      >
                        <List className="w-4 h-4" />
                      </Button>
                    </div>
                  </div>
                </CardHeader>
                <CardContent>
                  <ResearchProjects view={view} />
                </CardContent>
              </Card>
            </motion.div>
          </div>

          {/* Right Column - Recent Activity */}
          <div className="space-y-6">
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.4 }}
            >
              <Card className="h-full">
                <CardHeader>
                  <CardTitle>Recent Activity</CardTitle>
                  <CardDescription>
                    Your latest research activities
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <RecentActivity />
                </CardContent>
              </Card>
            </motion.div>

            {/* Research Insights */}
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.5 }}
            >
              <Card>
                <CardHeader>
                  <CardTitle>Research Insights</CardTitle>
                  <CardDescription>
                    AI-powered recommendations
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-3">
                  <div className="p-3 bg-primary/10 rounded-lg">
                    <div className="flex items-center gap-2 mb-1">
                      <TrendingUp className="w-4 h-4 text-primary" />
                      <span className="text-sm font-medium">Trending Topic</span>
                    </div>
                    <p className="text-sm text-muted-foreground">
                      "Machine Learning in Drug Discovery" is gaining traction in your field
                    </p>
                    <Button variant="link" size="sm" className="p-0 h-auto mt-2">
                      Explore <ArrowRight className="w-3 h-3 ml-1" />
                    </Button>
                  </div>

                  <div className="p-3 bg-secondary rounded-lg">
                    <div className="flex items-center gap-2 mb-1">
                      <Brain className="w-4 h-4 text-primary" />
                      <span className="text-sm font-medium">New Connection</span>
                    </div>
                    <p className="text-sm text-muted-foreground">
                      Found potential link between your projects on protein folding and quantum computing
                    </p>
                    <Button variant="link" size="sm" className="p-0 h-auto mt-2">
                      View Details <ArrowRight className="w-3 h-3 ml-1" />
                    </Button>
                  </div>
                </CardContent>
              </Card>
            </motion.div>
          </div>
        </div>

        {/* Bottom Action Bar */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.6 }}
        >
          <Card className="bg-gradient-to-r from-primary/10 to-purple-500/10 border-none">
            <CardContent className="p-6">
              <div className="flex flex-col sm:flex-row items-center justify-between gap-4">
                <div>
                  <h3 className="text-lg font-semibold mb-1">
                    Need help with your research?
                  </h3>
                  <p className="text-sm text-muted-foreground">
                    Our AI assistant is here to help you 24/7
                  </p>
                </div>
                <div className="flex gap-3">
                  <Button variant="outline">
                    Browse Templates
                  </Button>
                  <Button>
                    <Search className="w-4 h-4 mr-2" />
                    Start Research Assistant
                  </Button>
                </div>
              </div>
            </CardContent>
          </Card>
        </motion.div>
      </div>
    </DashboardLayout>
  )
}