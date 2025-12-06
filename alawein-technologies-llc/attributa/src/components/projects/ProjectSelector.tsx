import { useState } from 'react';
import { Plus, FolderOpen, Settings } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';
import { Badge } from '@/components/ui/badge';
import { useProjects, useCurrentProject } from '@/hooks/useProjects';

const ProjectSelector = () => {
  const { projects, createProject, isCreating } = useProjects();
  const { currentProject, setCurrentProjectId } = useCurrentProject();
  const [isDialogOpen, setIsDialogOpen] = useState(false);
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');

  const handleCreateProject = () => {
    if (!name.trim()) return;
    
    createProject({ name: name.trim(), description: description.trim() || undefined });
    setName('');
    setDescription('');
    setIsDialogOpen(false);
  };

  if (!currentProject) {
    return (
      <Card className="max-w-2xl mx-auto">
        <CardHeader className="text-center">
          <FolderOpen className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
          <CardTitle>Select or Create Project</CardTitle>
          <CardDescription>
            Choose an existing project or create a new one to get started with attribution analysis
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {projects.length > 0 && (
            <div className="space-y-2">
              <h3 className="font-semibold">Recent Projects</h3>
              <div className="grid gap-2">
                {projects.slice(0, 3).map((project) => (
                  <Card 
                    key={project.id} 
                    className="cursor-pointer hover:shadow-md transition-shadow"
                    onClick={() => setCurrentProjectId(project.id)}
                  >
                    <CardContent className="p-4">
                      <div className="flex items-center justify-between">
                        <div>
                          <h4 className="font-medium">{project.name}</h4>
                          {project.description && (
                            <p className="text-sm text-muted-foreground">{project.description}</p>
                          )}
                        </div>
                        <Badge variant="outline">
                          {new Date(project.created_at).toLocaleDateString()}
                        </Badge>
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            </div>
          )}

          <Dialog open={isDialogOpen} onOpenChange={setIsDialogOpen}>
            <DialogTrigger asChild>
              <Button className="w-full">
                <Plus className="h-4 w-4 mr-2" />
                Create New Project
              </Button>
            </DialogTrigger>
            <DialogContent>
              <DialogHeader>
                <DialogTitle>Create New Project</DialogTitle>
                <DialogDescription>
                  Create a new attribution project to organize your content analysis
                </DialogDescription>
              </DialogHeader>
              <div className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="project-name">Project Name</Label>
                  <Input
                    id="project-name"
                    placeholder="My Attribution Project"
                    value={name}
                    onChange={(e) => setName(e.target.value)}
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="project-description">Description (optional)</Label>
                  <Textarea
                    id="project-description"
                    placeholder="Describe what this project is for..."
                    value={description}
                    onChange={(e) => setDescription(e.target.value)}
                    rows={3}
                  />
                </div>
                <div className="flex gap-2">
                  <Button 
                    onClick={handleCreateProject} 
                    disabled={!name.trim() || isCreating}
                    className="flex-1"
                  >
                    {isCreating ? 'Creating...' : 'Create Project'}
                  </Button>
                  <Button variant="outline" onClick={() => setIsDialogOpen(false)}>
                    Cancel
                  </Button>
                </div>
              </div>
            </DialogContent>
          </Dialog>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardContent className="p-4">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="font-semibold">{currentProject.name}</h3>
            {currentProject.description && (
              <p className="text-sm text-muted-foreground">{currentProject.description}</p>
            )}
          </div>
          <div className="flex items-center gap-2">
            <Button 
              variant="outline" 
              size="sm"
              onClick={() => setCurrentProjectId(null)}
            >
              <Settings className="h-4 w-4 mr-2" />
              Switch Project
            </Button>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};

export default ProjectSelector;