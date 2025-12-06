import { useState, useEffect } from 'react';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Badge } from '@/components/ui/badge';
import { useToast } from '@/hooks/use-toast';
import { supabase } from '@/integrations/supabase/client';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { Search, Eye, CheckCircle, Mail, Download, Filter, Send, FileText, Loader2 } from 'lucide-react';
import { Textarea } from '@/components/ui/textarea';
import { downloadTrainingPlanPDF } from '@/utils/pdfGenerator';

interface NonPortalClient {
  id: string;
  email: string;
  full_name: string;
  phone: string;
  intake_data: Record<string, any>;
  payment_type: string;
  payment_status: string;
  plan_delivered_at: string | null;
  notes: string | null;
  created_at: string;
  updated_at: string;
}

export default function NonPortalClients() {
  const { toast } = useToast();
  const [clients, setClients] = useState<NonPortalClient[]>([]);
  const [filteredClients, setFilteredClients] = useState<NonPortalClient[]>([]);
  const [loading, setLoading] = useState(true);
  const [searchTerm, setSearchTerm] = useState('');
  const [statusFilter, setStatusFilter] = useState<string>('all');
  const [selectedClient, setSelectedClient] = useState<NonPortalClient | null>(null);
  const [showDetailModal, setShowDetailModal] = useState(false);
  const [adminNotes, setAdminNotes] = useState('');
  const [planPdfUrl, setPlanPdfUrl] = useState('');
  const [showDeliverModal, setShowDeliverModal] = useState(false);
  const [deliveryNotes, setDeliveryNotes] = useState('');

  useEffect(() => {
    loadClients();
  }, []);

  useEffect(() => {
    filterClients();
  }, [clients, searchTerm, statusFilter]);

  const loadClients = async () => {
    try {
      const { data, error } = await supabase
        .from('non_portal_clients')
        .select('*')
        .order('created_at', { ascending: false });

      if (error) throw error;

      setClients(data || []);
    } catch (error) {
      console.error('Error loading clients:', error);
      toast({
        title: 'Error',
        description: 'Failed to load non-portal clients',
        variant: 'destructive',
      });
    } finally {
      setLoading(false);
    }
  };

  const filterClients = () => {
    let filtered = clients;

    // Apply search filter
    if (searchTerm) {
      filtered = filtered.filter(
        (client) =>
          client.full_name.toLowerCase().includes(searchTerm.toLowerCase()) ||
          client.email.toLowerCase().includes(searchTerm.toLowerCase())
      );
    }

    // Apply status filter
    if (statusFilter !== 'all') {
      filtered = filtered.filter((client) => client.payment_status === statusFilter);
    }

    setFilteredClients(filtered);
  };

  const viewClientDetails = (client: NonPortalClient) => {
    setSelectedClient(client);
    setAdminNotes(client.notes || '');
    setShowDetailModal(true);
  };

  const markPlanDelivered = async (clientId: string) => {
    try {
      const { error } = await supabase
        .from('non_portal_clients')
        .update({ plan_delivered_at: new Date().toISOString() })
        .eq('id', clientId);

      if (error) throw error;

      toast({
        title: 'Success',
        description: 'Plan marked as delivered',
      });

      loadClients();
    } catch (error) {
      console.error('Error updating client:', error);
      toast({
        title: 'Error',
        description: 'Failed to update client status',
        variant: 'destructive',
      });
    }
  };

  const updatePaymentStatus = async (clientId: string, status: string) => {
    try {
      const { error } = await supabase
        .from('non_portal_clients')
        .update({ payment_status: status })
        .eq('id', clientId);

      if (error) throw error;

      toast({
        title: 'Success',
        description: 'Payment status updated',
      });

      loadClients();
    } catch (error) {
      console.error('Error updating payment status:', error);
      toast({
        title: 'Error',
        description: 'Failed to update payment status',
        variant: 'destructive',
      });
    }
  };

  const saveAdminNotes = async () => {
    if (!selectedClient) return;

    try {
      const { error } = await supabase
        .from('non_portal_clients')
        .update({ notes: adminNotes })
        .eq('id', selectedClient.id);

      if (error) throw error;

      toast({
        title: 'Success',
        description: 'Notes saved successfully',
      });

      loadClients();
    } catch (error) {
      console.error('Error saving notes:', error);
      toast({
        title: 'Error',
        description: 'Failed to save notes',
        variant: 'destructive',
      });
    }
  };

  const deliverPlan = async () => {
    if (!selectedClient || !planPdfUrl) {
      toast({
        title: 'Error',
        description: 'Please provide a plan PDF URL',
        variant: 'destructive',
      });
      return;
    }

    try {
      const { error } = await supabase.functions.invoke('deliver-plan', {
        body: {
          clientId: selectedClient.id,
          planPdfUrl,
          notes: deliveryNotes,
        },
      });

      if (error) throw error;

      toast({
        title: 'Success',
        description: 'Plan delivered successfully via email',
      });

      setShowDeliverModal(false);
      setPlanPdfUrl('');
      setDeliveryNotes('');
      loadClients();
    } catch (error) {
      console.error('Error delivering plan:', error);
      toast({
        title: 'Error',
        description: 'Failed to deliver plan',
        variant: 'destructive',
      });
    }
  };

  const downloadIntakeData = (client: NonPortalClient) => {
    const dataStr = JSON.stringify(client.intake_data, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `intake-${client.full_name.replace(/\s+/g, '-')}-${client.id}.json`;
    link.click();
    URL.revokeObjectURL(url);
  };

  const getPaymentStatusColor = (status: string) => {
    switch (status) {
      case 'paid':
        return 'bg-success/20 text-success';
      case 'pending':
        return 'bg-warning/20 text-warning';
      case 'failed':
        return 'bg-error/20 text-error';
      default:
        return 'bg-neutral/20 text-neutral';
    }
  };

  const getPaymentTypeName = (type: string) => {
    const names = {
      'one-time-basic': 'Basic ($299)',
      'one-time-premium': 'Premium ($599)',
      'one-time-concierge': 'Concierge ($1,499)',
    };
    return names[type] || type;
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="text-white/60">Loading...</div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold text-white">Non-Portal Clients</h1>
          <p className="text-white/60 mt-1">Manage email-based intake submissions</p>
        </div>
      </div>

      {/* Filters */}
      <Card className="p-4">
        <div className="flex gap-4 items-center">
          <div className="flex-1 relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-white/40" />
            <Input
              placeholder="Search by name or email..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="pl-10"
            />
          </div>

          <div className="flex items-center gap-2">
            <Filter className="h-4 w-4 text-white/60" />
            <select
              value={statusFilter}
              onChange={(e) => setStatusFilter(e.target.value)}
              className="flex h-10 rounded-md border border-input bg-background px-3 py-2 text-sm"
            >
              <option value="all">All Status</option>
              <option value="pending">Pending</option>
              <option value="paid">Paid</option>
              <option value="failed">Failed</option>
            </select>
          </div>
        </div>
      </Card>

      {/* Statistics */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card className="p-4">
          <div className="text-white/60 text-sm">Total Submissions</div>
          <div className="text-2xl font-bold text-white mt-1">{clients.length}</div>
        </Card>
        <Card className="p-4">
          <div className="text-white/60 text-sm">Pending Payment</div>
          <div className="text-2xl font-bold text-warning mt-1">
            {clients.filter((c) => c.payment_status === 'pending').length}
          </div>
        </Card>
        <Card className="p-4">
          <div className="text-white/60 text-sm">Paid</div>
          <div className="text-2xl font-bold text-success mt-1">
            {clients.filter((c) => c.payment_status === 'paid').length}
          </div>
        </Card>
        <Card className="p-4">
          <div className="text-white/60 text-sm">Plans Delivered</div>
          <div className="text-2xl font-bold text-white mt-1">
            {clients.filter((c) => c.plan_delivered_at).length}
          </div>
        </Card>
      </div>

      {/* Clients Table */}
      <Card>
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Name</TableHead>
              <TableHead>Email</TableHead>
              <TableHead>Plan Type</TableHead>
              <TableHead>Payment Status</TableHead>
              <TableHead>Plan Delivered</TableHead>
              <TableHead>Submitted</TableHead>
              <TableHead>Actions</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {filteredClients.length === 0 ? (
              <TableRow>
                <TableCell colSpan={7} className="text-center text-white/60">
                  No clients found
                </TableCell>
              </TableRow>
            ) : (
              filteredClients.map((client) => (
                <TableRow key={client.id}>
                  <TableCell className="font-medium text-white">{client.full_name}</TableCell>
                  <TableCell className="text-white/70">{client.email}</TableCell>
                  <TableCell>
                    <Badge variant="outline">{getPaymentTypeName(client.payment_type)}</Badge>
                  </TableCell>
                  <TableCell>
                    <Badge className={getPaymentStatusColor(client.payment_status)}>
                      {client.payment_status}
                    </Badge>
                  </TableCell>
                  <TableCell>
                    {client.plan_delivered_at ? (
                      <Badge className="bg-success/20 text-success">
                        {new Date(client.plan_delivered_at).toLocaleDateString()}
                      </Badge>
                    ) : (
                      <span className="text-white/40">Not delivered</span>
                    )}
                  </TableCell>
                  <TableCell className="text-white/60">
                    {new Date(client.created_at).toLocaleDateString()}
                  </TableCell>
                  <TableCell>
                    <div className="flex gap-2">
                      <Button
                        size="sm"
                        variant="ghost"
                        onClick={() => viewClientDetails(client)}
                      >
                        <Eye className="h-4 w-4" />
                      </Button>
                      {!client.plan_delivered_at && client.payment_status === 'paid' && (
                        <Button
                          size="sm"
                          variant="ghost"
                          onClick={() => {
                            setSelectedClient(client);
                            setShowDeliverModal(true);
                          }}
                          title="Deliver plan"
                        >
                          <Send className="h-4 w-4" />
                        </Button>
                      )}
                      <Button
                        size="sm"
                        variant="ghost"
                        onClick={() => downloadIntakeData(client)}
                        title="Download intake data"
                      >
                        <Download className="h-4 w-4" />
                      </Button>
                    </div>
                  </TableCell>
                </TableRow>
              ))
            )}
          </TableBody>
        </Table>
      </Card>

      {/* Detail Modal */}
      <Dialog open={showDetailModal} onOpenChange={setShowDetailModal}>
        <DialogContent className="max-w-4xl max-h-[90vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle>Client Details: {selectedClient?.full_name}</DialogTitle>
            <DialogDescription>Complete intake information and admin tools</DialogDescription>
          </DialogHeader>

          {selectedClient && (
            <div className="space-y-6">
              {/* Contact Information */}
              <div>
                <h3 className="text-lg font-semibold text-white mb-3">Contact Information</h3>
                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <span className="text-white/60">Email:</span>
                    <div className="text-white">{selectedClient.email}</div>
                  </div>
                  <div>
                    <span className="text-white/60">Phone:</span>
                    <div className="text-white">{selectedClient.phone || 'Not provided'}</div>
                  </div>
                </div>
              </div>

              {/* Payment Information */}
              <div>
                <h3 className="text-lg font-semibold text-white mb-3">Payment Information</h3>
                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <span className="text-white/60">Plan Type:</span>
                    <div className="text-white">{getPaymentTypeName(selectedClient.payment_type)}</div>
                  </div>
                  <div>
                    <span className="text-white/60">Payment Status:</span>
                    <div className="mt-1">
                      <select
                        value={selectedClient.payment_status}
                        onChange={(e) => updatePaymentStatus(selectedClient.id, e.target.value)}
                        className="flex h-9 rounded-md border border-input bg-background px-3 py-1 text-sm"
                      >
                        <option value="pending">Pending</option>
                        <option value="paid">Paid</option>
                        <option value="failed">Failed</option>
                        <option value="refunded">Refunded</option>
                      </select>
                    </div>
                  </div>
                </div>
              </div>

              {/* Intake Data Summary */}
              <div>
                <h3 className="text-lg font-semibold text-white mb-3">Intake Summary</h3>
                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <span className="text-white/60">Age / Sex:</span>
                    <div className="text-white">
                      {selectedClient.intake_data.age} / {selectedClient.intake_data.sex}
                    </div>
                  </div>
                  <div>
                    <span className="text-white/60">Height / Weight:</span>
                    <div className="text-white">
                      {selectedClient.intake_data.height}" / {selectedClient.intake_data.weight} lbs
                    </div>
                  </div>
                  <div>
                    <span className="text-white/60">Activity Level:</span>
                    <div className="text-white">{selectedClient.intake_data.activityLevel}</div>
                  </div>
                  <div>
                    <span className="text-white/60">Training Years:</span>
                    <div className="text-white">{selectedClient.intake_data.trainingYears}</div>
                  </div>
                  <div className="col-span-2">
                    <span className="text-white/60">Primary Goal:</span>
                    <div className="text-white">{selectedClient.intake_data.primaryGoal}</div>
                  </div>
                  <div className="col-span-2">
                    <span className="text-white/60">Health Conditions:</span>
                    <div className="text-white">
                      {selectedClient.intake_data.healthConditions || 'None reported'}
                    </div>
                  </div>
                </div>
              </div>

              {/* Admin Notes */}
              <div>
                <h3 className="text-lg font-semibold text-white mb-3">Admin Notes</h3>
                <Textarea
                  value={adminNotes}
                  onChange={(e) => setAdminNotes(e.target.value)}
                  placeholder="Add notes about this client..."
                  rows={4}
                />
                <Button
                  onClick={saveAdminNotes}
                  className="mt-2 bg-repz-orange hover:bg-repz-orange-dark"
                >
                  Save Notes
                </Button>
              </div>

              {/* Action Buttons */}
              <div className="flex flex-wrap gap-3 pt-4 border-t border-surface-elevated">
                <Button
                  onClick={() => {
                    downloadTrainingPlanPDF(
                      selectedClient.id,
                      selectedClient.full_name,
                      selectedClient.intake_data,
                      selectedClient.payment_type,
                      adminNotes || undefined
                    );
                    toast({
                      title: 'PDF Generated',
                      description: 'Training plan PDF has been downloaded',
                    });
                  }}
                  className="flex items-center gap-2 bg-green-600 hover:bg-green-700"
                >
                  <FileText className="h-4 w-4" />
                  Generate PDF Plan
                </Button>
                <Button
                  onClick={() => downloadIntakeData(selectedClient)}
                  variant="outline"
                  className="flex items-center gap-2"
                >
                  <Download className="h-4 w-4" />
                  Download Raw Data
                </Button>
                {!selectedClient.plan_delivered_at && selectedClient.payment_status === 'paid' && (
                  <Button
                    onClick={() => {
                      setShowDetailModal(false);
                      setShowDeliverModal(true);
                    }}
                    className="flex items-center gap-2 bg-repz-orange hover:bg-repz-orange-dark"
                  >
                    <Send className="h-4 w-4" />
                    Deliver Plan
                  </Button>
                )}
              </div>
            </div>
          )}
        </DialogContent>
      </Dialog>

      {/* Deliver Plan Modal */}
      <Dialog open={showDeliverModal} onOpenChange={setShowDeliverModal}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Deliver Training Plan</DialogTitle>
            <DialogDescription>
              Send the training plan to {selectedClient?.full_name} via email
            </DialogDescription>
          </DialogHeader>

          <div className="space-y-4">
            <div>
              <label className="text-sm font-medium text-white">Plan PDF URL</label>
              <Input
                placeholder="https://your-storage.com/plan.pdf"
                value={planPdfUrl}
                onChange={(e) => setPlanPdfUrl(e.target.value)}
                className="mt-1"
              />
              <p className="text-xs text-white/60 mt-1">
                Upload the PDF to your storage and paste the public URL here
              </p>
            </div>

            <div>
              <label className="text-sm font-medium text-white">Coach's Notes (Optional)</label>
              <Textarea
                placeholder="Add personal notes for the client..."
                value={deliveryNotes}
                onChange={(e) => setDeliveryNotes(e.target.value)}
                rows={4}
                className="mt-1"
              />
            </div>

            <div className="flex gap-3 pt-4">
              <Button
                variant="outline"
                onClick={() => {
                  setShowDeliverModal(false);
                  setPlanPdfUrl('');
                  setDeliveryNotes('');
                }}
              >
                Cancel
              </Button>
              <Button
                onClick={deliverPlan}
                disabled={!planPdfUrl}
                className="bg-repz-orange hover:bg-repz-orange-dark"
              >
                <Send className="h-4 w-4 mr-2" />
                Send Plan
              </Button>
            </div>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
}
