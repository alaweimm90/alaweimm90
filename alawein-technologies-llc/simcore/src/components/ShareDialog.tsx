import React, { useState } from 'react';
import { Share2, Download, Upload, Copy, Link2, Save, Trash2 } from 'lucide-react';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from '@/components/ui/dialog';
import {
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
} from '@/components/ui/tabs';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Switch } from '@/components/ui/switch';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import { collaborationManager, SimulationState } from '@/lib/collaboration-manager';
import { toast } from '@/hooks/use-toast';

interface ShareDialogProps {
  simulationState: SimulationState;
  onLoadSimulation: (state: SimulationState) => void;
}

// TEMPORARILY HIDDEN - Collaboration features paused
// TODO: Resume collaboration work here - ShareDialog component with:
// - Shareable URLs, export/import, local save/load, session management
// - Integration with collaboration-manager.ts
// - Full dialog with tabs for different sharing methods
export function ShareDialog({ simulationState, onLoadSimulation }: ShareDialogProps) {
  // Collaboration features enabled
  const [sessionName, setSessionName] = useState('');
  const [isPublic, setIsPublic] = useState(false);
  const [importData, setImportData] = useState('');
  const [isOpen, setIsOpen] = useState(false);
  
  const savedSimulations = collaborationManager.getSavedSimulations();
  const activeSession = collaborationManager.getActiveSession();

  const handleShare = async () => {
    try {
      const shareURL = collaborationManager.generateShareableURL(simulationState);
      await collaborationManager.copyToClipboard(shareURL);
    } catch (error) {
      console.error('Share failed:', error);
    }
  };

  const handleExport = () => {
    try {
      const exportData = collaborationManager.exportSimulation(simulationState);
      const filename = `${simulationState.name.replace(/\s+/g, '-').toLowerCase()}-${Date.now()}.json`;
      collaborationManager.downloadAsFile(exportData, filename);
    } catch (error) {
      console.error('Export failed:', error);
    }
  };

  const handleImport = () => {
    try {
      if (!importData.trim()) {
        toast({
          title: "Import Error",
          description: "Please paste simulation data",
          variant: "destructive",
        });
        return;
      }
      
      const imported = collaborationManager.importSimulation(importData);
      onLoadSimulation(imported);
      setImportData('');
      setIsOpen(false);
    } catch (error) {
      console.error('Import failed:', error);
    }
  };

  const handleSave = () => {
    try {
      collaborationManager.saveSimulation(simulationState);
    } catch (error) {
      console.error('Save failed:', error);
    }
  };

  const handleLoadSaved = (state: SimulationState) => {
    onLoadSimulation(state);
    setIsOpen(false);
  };

  const handleCreateSession = () => {
    if (!sessionName.trim()) {
      toast({
        title: "Session Error",
        description: "Please enter a session name",
        variant: "destructive",
      });
      return;
    }
    
    collaborationManager.createSession(sessionName, isPublic);
    setSessionName('');
  };

  const formatDate = (timestamp: number) => {
    return new Date(timestamp).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  return (
    <Dialog open={isOpen} onOpenChange={setIsOpen}>
      <DialogTrigger asChild>
        <Button variant="outline" size="sm" className="gap-2">
          <Share2 className="h-4 w-4" />
          Share
        </Button>
      </DialogTrigger>
      <DialogContent className="max-w-2xl max-h-[80vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Share2 className="h-5 w-5" />
            Research Collaboration Hub
          </DialogTitle>
          <DialogDescription>
            Share computational states, archive research sessions, and enable real-time collaboration
          </DialogDescription>
        </DialogHeader>

        <Tabs defaultValue="share" className="w-full">
          <TabsList className="grid w-full grid-cols-4">
            <TabsTrigger value="share">Share</TabsTrigger>
            <TabsTrigger value="save">Save/Load</TabsTrigger>
            <TabsTrigger value="import">Import</TabsTrigger>
            <TabsTrigger value="collaborate">Collaborate</TabsTrigger>
          </TabsList>

          <TabsContent value="share" className="space-y-4 mt-6">
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="text-lg font-medium">Share Simulation</h3>
                  <p className="text-sm text-muted-foreground">
                    Generate a shareable link with current parameters
                  </p>
                </div>
                <Button onClick={handleShare} className="gap-2">
                  <Link2 className="h-4 w-4" />
                  Copy Link
                </Button>
              </div>
              
              <Separator />
              
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="text-lg font-medium">Export Research Data</h3>
                  <p className="text-sm text-muted-foreground">
                    Archive complete simulation state with metadata
                  </p>
                </div>
                <Button onClick={handleExport} variant="outline" className="gap-2">
                  <Download className="h-4 w-4" />
                  Archive Data
                </Button>
              </div>

              <Card>
                <CardHeader>
                  <CardTitle className="text-sm">Current Simulation</CardTitle>
                </CardHeader>
                <CardContent className="space-y-2">
                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div>
                      <Label className="text-muted-foreground">Name</Label>
                      <p className="font-medium">{simulationState.name}</p>
                    </div>
                    <div>
                      <Label className="text-muted-foreground">Module</Label>
                      <p className="font-medium">{simulationState.module}</p>
                    </div>
                    <div>
                      <Label className="text-muted-foreground">Parameters</Label>
                      <p className="font-medium">{Object.keys(simulationState.parameters).length} settings</p>
                    </div>
                    <div>
                      <Label className="text-muted-foreground">Modified</Label>
                      <p className="font-medium">{formatDate(simulationState.timestamp)}</p>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          <TabsContent value="save" className="space-y-4 mt-6">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-lg font-medium">Save Current Simulation</h3>
                <p className="text-sm text-muted-foreground">
                  Save to local storage for quick access
                </p>
              </div>
              <Button onClick={handleSave} className="gap-2">
                <Save className="h-4 w-4" />
                Save Session
              </Button>
            </div>

            <Separator />

            <div>
              <h3 className="text-lg font-medium mb-4">Saved Simulations</h3>
              {Object.keys(savedSimulations).length === 0 ? (
                <Card>
                  <CardContent className="pt-6">
                    <p className="text-center text-muted-foreground">
                      No saved simulations yet
                    </p>
                  </CardContent>
                </Card>
              ) : (
                <div className="grid gap-3">
                  {Object.values(savedSimulations).map((saved) => (
                    <Card key={saved.id} className="hover:bg-muted/50 transition-colors">
                      <CardContent className="pt-4">
                        <div className="flex items-center justify-between">
                          <div className="space-y-1">
                            <div className="flex items-center gap-2">
                              <h4 className="font-medium">{saved.name}</h4>
                              <Badge variant="secondary">{saved.module}</Badge>
                            </div>
                            <p className="text-sm text-muted-foreground">
                              Saved {formatDate(saved.timestamp)}
                            </p>
                          </div>
                          <div className="flex gap-2">
                            <Button
                              size="sm"
                              onClick={() => handleLoadSaved(saved)}
                              className="gap-2"
                            >
                              <Upload className="h-3 w-3" />
                              Load
                            </Button>
                            <Button
                              size="sm"
                              variant="outline"
                              onClick={() => collaborationManager.deleteSimulation(saved.id)}
                            >
                              <Trash2 className="h-3 w-3" />
                            </Button>
                          </div>
                        </div>
                      </CardContent>
                    </Card>
                  ))}
                </div>
              )}
            </div>
          </TabsContent>

          <TabsContent value="import" className="space-y-4 mt-6">
            <div>
              <h3 className="text-lg font-medium">Import Simulation</h3>
              <p className="text-sm text-muted-foreground mb-4">
                Paste JSON data from exported simulation file
              </p>
              
              <div className="space-y-4">
                <div>
                  <Label htmlFor="import-data">Simulation Data (JSON)</Label>
                  <textarea
                    id="import-data"
                    className="w-full min-h-[200px] p-3 mt-2 border rounded-md font-mono text-sm"
                    placeholder="Paste simulation JSON data here..."
                    value={importData}
                    onChange={(e) => setImportData(e.target.value)}
                  />
                </div>
                
                <div className="flex gap-2">
                  <Button onClick={handleImport} className="gap-2">
                    <Upload className="h-4 w-4" />
                    Import Session
                  </Button>
                  <Button
                    variant="outline"
                    onClick={() => setImportData('')}
                  >
                    Clear
                  </Button>
                </div>
              </div>
            </div>
          </TabsContent>

          <TabsContent value="collaborate" className="space-y-4 mt-6">
            <div>
              <h3 className="text-lg font-medium">Collaboration Sessions</h3>
              <p className="text-sm text-muted-foreground mb-4">
                Create or join real-time collaboration sessions
              </p>

              {activeSession && (
                <Card className="mb-4">
                  <CardHeader>
                    <CardTitle className="text-sm flex items-center gap-2">
                      <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
                      Active Session
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-2">
                      <div className="flex items-center justify-between">
                        <span className="font-medium">{activeSession.name}</span>
                        <Badge variant={activeSession.isPublic ? "default" : "secondary"}>
                          {activeSession.isPublic ? "Public" : "Private"}
                        </Badge>
                      </div>
                      <p className="text-sm text-muted-foreground">
                        Created {formatDate(activeSession.createdAt)}
                      </p>
                      <p className="text-sm text-muted-foreground">
                        {activeSession.participants.length} participant(s)
                      </p>
                    </div>
                  </CardContent>
                </Card>
              )}

              <div className="space-y-4">
                <div>
                  <Label htmlFor="session-name">New Session Name</Label>
                  <Input
                    id="session-name"
                    placeholder="e.g., Quantum Tunneling Research"
                    value={sessionName}
                    onChange={(e) => setSessionName(e.target.value)}
                    className="mt-2"
                  />
                </div>

                <div className="flex items-center space-x-2">
                  <Switch
                    id="public-session"
                    checked={isPublic}
                    onCheckedChange={setIsPublic}
                  />
                  <Label htmlFor="public-session">Make session public</Label>
                </div>

                <Button onClick={handleCreateSession} className="w-full gap-2">
                  <Share2 className="h-4 w-4" />
                  Create Session
                </Button>
              </div>

              <Card>
                <CardContent className="pt-6">
                  <div className="text-center text-muted-foreground space-y-2">
                    <p className="text-sm">
                      <strong>Coming Soon:</strong> Real-time parameter sync
                    </p>
                    <p className="text-sm">
                      👥 Multi-user collaborative editing
                    </p>
                    <p className="text-sm">
                      💬 In-simulation chat and annotations
                    </p>
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>
        </Tabs>
      </DialogContent>
    </Dialog>
  );
}