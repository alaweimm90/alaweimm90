import { useState } from 'react'
import { MoreVertical, Users, FileText, Calendar, TrendingUp } from 'lucide-react'
import { Card } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Progress } from '@/components/ui/progress'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'
import Link from 'next/link'

const projects = [
  {
    id: 1,
    title: 'Machine Learning in Drug Discovery',
    description: 'Analyzing the application of ML algorithms in pharmaceutical research',
    progress: 75,
    papers: 234,
    collaborators: 3,
    dueDate: '2024-02-15',
    status: 'active',
    tags: ['AI', 'Healthcare', 'Pharmacology'],
  },
  {
    id: 2,
    title: 'Quantum Computing Applications',
    description: 'Exploring quantum algorithms for optimization problems',
    progress: 45,
    papers: 156,
    collaborators: 2,
    dueDate: '2024-03-01',
    status: 'active',
    tags: ['Quantum', 'Computing', 'Physics'],
  },
  {
    id: 3,
    title: 'Climate Change Mitigation Strategies',
    description: 'Comprehensive review of carbon capture technologies',
    progress: 90,
    papers: 412,
    collaborators: 5,
    dueDate: '2024-01-30',
    status: 'review',
    tags: ['Climate', 'Environment', 'Technology'],
  },
  {
    id: 4,
    title: 'Protein Folding Mechanisms',
    description: 'Understanding protein structure prediction using AI',
    progress: 30,
    papers: 89,
    collaborators: 1,
    dueDate: '2024-04-15',
    status: 'planning',
    tags: ['Biology', 'AI', 'Proteins'],
  },
]

interface ResearchProjectsProps {
  view: 'grid' | 'list'
}

export function ResearchProjects({ view }: ResearchProjectsProps) {
  const [selectedProject, setSelectedProject] = useState<number | null>(null)

  if (view === 'list') {
    return (
      <div className="space-y-4">
        {projects.map((project) => (
          <div
            key={project.id}
            className="flex items-center justify-between p-4 border rounded-lg hover:bg-secondary/50 transition-colors"
          >
            <div className="flex-1">
              <div className="flex items-center gap-3 mb-2">
                <Link
                  href={`/projects/${project.id}`}
                  className="font-medium hover:text-primary transition-colors"
                >
                  {project.title}
                </Link>
                <span className={`text-xs px-2 py-1 rounded-full ${
                  project.status === 'active'
                    ? 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400'
                    : project.status === 'review'
                    ? 'bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-400'
                    : 'bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400'
                }`}>
                  {project.status}
                </span>
              </div>
              <p className="text-sm text-muted-foreground mb-3">
                {project.description}
              </p>
              <div className="flex items-center gap-6 text-sm text-muted-foreground">
                <div className="flex items-center gap-1">
                  <FileText className="w-4 h-4" />
                  <span>{project.papers} papers</span>
                </div>
                <div className="flex items-center gap-1">
                  <Users className="w-4 h-4" />
                  <span>{project.collaborators}</span>
                </div>
                <div className="flex items-center gap-1">
                  <Calendar className="w-4 h-4" />
                  <span>{project.dueDate}</span>
                </div>
              </div>
            </div>
            <div className="flex items-center gap-4">
              <div className="w-32">
                <Progress value={project.progress} className="h-2" />
                <p className="text-xs text-muted-foreground mt-1 text-right">
                  {project.progress}%
                </p>
              </div>
              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <Button variant="ghost" size="icon">
                    <MoreVertical className="w-4 h-4" />
                  </Button>
                </DropdownMenuTrigger>
                <DropdownMenuContent align="end">
                  <DropdownMenuItem>View Details</DropdownMenuItem>
                  <DropdownMenuItem>Edit Project</DropdownMenuItem>
                  <DropdownMenuItem>Share</DropdownMenuItem>
                  <DropdownMenuItem className="text-destructive">
                    Delete
                  </DropdownMenuItem>
                </DropdownMenuContent>
              </DropdownMenu>
            </div>
          </div>
        ))}
      </div>
    )
  }

  return (
    <div className="grid md:grid-cols-2 gap-4">
      {projects.map((project) => (
        <Card key={project.id} className="p-5 hover:shadow-lg transition-shadow">
          <div className="flex justify-between items-start mb-3">
            <div>
              <Link
                href={`/projects/${project.id}`}
                className="font-semibold text-lg hover:text-primary transition-colors"
              >
                {project.title}
              </Link>
              <span className={`ml-2 text-xs px-2 py-1 rounded-full ${
                project.status === 'active'
                  ? 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400'
                  : project.status === 'review'
                  ? 'bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-400'
                  : 'bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400'
              }`}>
                {project.status}
              </span>
            </div>
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button variant="ghost" size="icon" className="h-8 w-8">
                  <MoreVertical className="w-4 h-4" />
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end">
                <DropdownMenuItem>View Details</DropdownMenuItem>
                <DropdownMenuItem>Edit Project</DropdownMenuItem>
                <DropdownMenuItem>Share</DropdownMenuItem>
                <DropdownMenuItem className="text-destructive">
                  Delete
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          </div>

          <p className="text-sm text-muted-foreground mb-4">
            {project.description}
          </p>

          <div className="flex flex-wrap gap-2 mb-4">
            {project.tags.map((tag) => (
              <span
                key={tag}
                className="text-xs px-2 py-1 bg-secondary rounded-md"
              >
                {tag}
              </span>
            ))}
          </div>

          <Progress value={project.progress} className="h-2 mb-2" />
          <div className="flex justify-between text-xs text-muted-foreground mb-4">
            <span>Progress</span>
            <span>{project.progress}%</span>
          </div>

          <div className="flex items-center justify-between text-sm">
            <div className="flex items-center gap-4">
              <div className="flex items-center gap-1">
                <FileText className="w-4 h-4 text-muted-foreground" />
                <span>{project.papers}</span>
              </div>
              <div className="flex items-center gap-1">
                <Users className="w-4 h-4 text-muted-foreground" />
                <span>{project.collaborators}</span>
              </div>
            </div>
            <div className="flex items-center gap-1 text-muted-foreground">
              <Calendar className="w-4 h-4" />
              <span>{project.dueDate}</span>
            </div>
          </div>
        </Card>
      ))}
    </div>
  )
}