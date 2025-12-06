import React, { useState, useMemo } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/ui/molecules/Card';
import { Badge } from '@/ui/atoms/Badge';
import { Button } from '@/ui/atoms/Button';
import { Input } from '@/ui/atoms/Input';
import {
  Search,
  Filter,
  Mail,
  MessageSquare,
  Clock,
  CheckCircle,
  AlertCircle,
  User,
  Send,
  Paperclip,
  MoreVertical,
  Eye,
  Trash2,
  RefreshCw,
  FileText
} from 'lucide-react';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger
} from '@/ui/molecules/DropdownMenu';
import { cn } from '@/lib/utils';

interface Ticket {
  id: string;
  subject: string;
  customer: {
    name: string;
    email: string;
    tier: string;
  };
  status: 'open' | 'in-progress' | 'resolved' | 'closed';
  priority: 'low' | 'medium' | 'high' | 'urgent';
  category: 'technical' | 'billing' | 'account' | 'feature-request' | 'general';
  createdAt: string;
  lastUpdate: string;
  assignedTo?: string;
  messageCount: number;
}

interface Message {
  id: string;
  author: string;
  role: 'customer' | 'support';
  message: string;
  timestamp: string;
  attachments?: string[];
}

interface ResponseTemplate {
  id: string;
  name: string;
  subject: string;
  body: string;
  category: string;
}

const SupportTicketHandler: React.FC = () => {
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedStatus, setSelectedStatus] = useState<string>('all');
  const [selectedPriority, setSelectedPriority] = useState<string>('all');
  const [selectedTicket, setSelectedTicket] = useState<Ticket | null>(null);
  const [replyMessage, setReplyMessage] = useState('');
  const [showTemplates, setShowTemplates] = useState(false);

  // Mock tickets data
  const [tickets] = useState<Ticket[]>([
    {
      id: 'TKT-001',
      subject: 'Unable to sync wearable device',
      customer: {
        name: 'John Smith',
        email: 'john.smith@example.com',
        tier: 'performance'
      },
      status: 'open',
      priority: 'high',
      category: 'technical',
      createdAt: '2025-11-12 09:30:00',
      lastUpdate: '2025-11-12 09:30:00',
      messageCount: 1
    },
    {
      id: 'TKT-002',
      subject: 'Question about subscription billing',
      customer: {
        name: 'Sarah Johnson',
        email: 'sarah.j@example.com',
        tier: 'adaptive'
      },
      status: 'in-progress',
      priority: 'medium',
      category: 'billing',
      createdAt: '2025-11-12 08:15:00',
      lastUpdate: '2025-11-12 10:45:00',
      assignedTo: 'Support Team',
      messageCount: 3
    },
    {
      id: 'TKT-003',
      subject: 'Request for workout plan modification',
      customer: {
        name: 'Mike Davis',
        email: 'mike.davis@example.com',
        tier: 'core'
      },
      status: 'open',
      priority: 'low',
      category: 'general',
      createdAt: '2025-11-11 16:20:00',
      lastUpdate: '2025-11-11 16:20:00',
      messageCount: 1
    },
    {
      id: 'TKT-004',
      subject: 'AI assistant not responding',
      customer: {
        name: 'Emily Brown',
        email: 'emily.brown@example.com',
        tier: 'longevity'
      },
      status: 'open',
      priority: 'urgent',
      category: 'technical',
      createdAt: '2025-11-12 11:00:00',
      lastUpdate: '2025-11-12 11:00:00',
      messageCount: 2
    },
    {
      id: 'TKT-005',
      subject: 'Feature request: Dark mode',
      customer: {
        name: 'Chris Wilson',
        email: 'chris.w@example.com',
        tier: 'adaptive'
      },
      status: 'in-progress',
      priority: 'low',
      category: 'feature-request',
      createdAt: '2025-11-10 14:30:00',
      lastUpdate: '2025-11-12 09:15:00',
      assignedTo: 'Product Team',
      messageCount: 5
    },
    {
      id: 'TKT-006',
      subject: 'Password reset not working',
      customer: {
        name: 'Jessica Lee',
        email: 'jessica.lee@example.com',
        tier: 'performance'
      },
      status: 'resolved',
      priority: 'high',
      category: 'account',
      createdAt: '2025-11-11 10:00:00',
      lastUpdate: '2025-11-11 14:30:00',
      assignedTo: 'Support Team',
      messageCount: 4
    }
  ]);

  // Mock conversation history
  const mockConversation: Message[] = [
    {
      id: 'MSG-001',
      author: 'John Smith',
      role: 'customer',
      message: 'Hi, I\'ve been trying to sync my Apple Watch with the REPZ app for the past hour, but it keeps failing. I\'ve tried restarting both devices but no luck. Can you help?',
      timestamp: '2025-11-12 09:30:00'
    },
    {
      id: 'MSG-002',
      author: 'Support Team',
      role: 'support',
      message: 'Hello John, thank you for reaching out. I\'m sorry to hear you\'re having trouble syncing your Apple Watch. Let me help you troubleshoot this issue. Could you please confirm which version of iOS and watchOS you\'re running?',
      timestamp: '2025-11-12 09:35:00'
    },
    {
      id: 'MSG-003',
      author: 'John Smith',
      role: 'customer',
      message: 'I\'m running iOS 17.1 and watchOS 10.1. Both are the latest versions.',
      timestamp: '2025-11-12 09:40:00'
    }
  ];

  // Response templates
  const responseTemplates: ResponseTemplate[] = [
    {
      id: 'TPL-001',
      name: 'Technical Issue - Initial Response',
      subject: 'Re: {subject}',
      body: 'Hello {customer_name},\n\nThank you for reaching out to REPZ support. I\'m sorry to hear you\'re experiencing technical difficulties. We\'re here to help!\n\nTo better assist you, could you please provide the following information:\n- Device type and OS version\n- App version\n- Steps to reproduce the issue\n- Any error messages you\'ve encountered\n\nWe\'ll work quickly to resolve this for you.\n\nBest regards,\nREPZ Support Team',
      category: 'technical'
    },
    {
      id: 'TPL-002',
      name: 'Billing Question',
      subject: 'Re: {subject}',
      body: 'Hello {customer_name},\n\nThank you for contacting REPZ regarding your billing question. I\'m happy to help clarify this for you.\n\nYour current subscription:\n- Tier: {tier}\n- Billing Cycle: Monthly\n- Next Payment: {next_payment_date}\n\nIf you have any specific questions about your billing or would like to make changes to your subscription, please let me know.\n\nBest regards,\nREPZ Support Team',
      category: 'billing'
    },
    {
      id: 'TPL-003',
      name: 'Issue Resolved',
      subject: 'Re: {subject}',
      body: 'Hello {customer_name},\n\nGreat news! We\'ve resolved the issue you reported. Please try again and confirm that everything is working as expected.\n\nIf you continue to experience any problems or have additional questions, please don\'t hesitate to reach out.\n\nThank you for your patience!\n\nBest regards,\nREPZ Support Team',
      category: 'general'
    },
    {
      id: 'TPL-004',
      name: 'Feature Request Acknowledgment',
      subject: 'Re: {subject}',
      body: 'Hello {customer_name},\n\nThank you for your feature request! We truly appreciate you taking the time to share your feedback with us.\n\nI\'ve forwarded your suggestion to our product team for consideration. While we can\'t guarantee when or if this feature will be implemented, we do carefully review all feedback from our community.\n\nYou\'ll receive updates if this feature is added to our roadmap.\n\nBest regards,\nREPZ Support Team',
      category: 'feature-request'
    }
  ];

  // Filter tickets
  const filteredTickets = useMemo(() => {
    return tickets.filter((ticket) => {
      const matchesSearch =
        ticket.subject.toLowerCase().includes(searchQuery.toLowerCase()) ||
        ticket.customer.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
        ticket.customer.email.toLowerCase().includes(searchQuery.toLowerCase()) ||
        ticket.id.toLowerCase().includes(searchQuery.toLowerCase());

      const matchesStatus = selectedStatus === 'all' || ticket.status === selectedStatus;
      const matchesPriority = selectedPriority === 'all' || ticket.priority === selectedPriority;

      return matchesSearch && matchesStatus && matchesPriority;
    });
  }, [tickets, searchQuery, selectedStatus, selectedPriority]);

  const getStatusBadgeClass = (status: string) => {
    switch (status) {
      case 'open':
        return 'bg-blue-100 text-blue-800';
      case 'in-progress':
        return 'bg-yellow-100 text-yellow-800';
      case 'resolved':
        return 'bg-green-100 text-green-800';
      case 'closed':
        return 'bg-gray-100 text-gray-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  };

  const getPriorityBadgeClass = (priority: string) => {
    switch (priority) {
      case 'urgent':
        return 'bg-red-100 text-red-800';
      case 'high':
        return 'bg-orange-100 text-orange-800';
      case 'medium':
        return 'bg-yellow-100 text-yellow-800';
      case 'low':
        return 'bg-gray-100 text-gray-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  };

  const handleSendReply = () => {
    if (!replyMessage.trim()) return;
    console.log('Sending reply:', replyMessage);
    setReplyMessage('');
  };

  const handleStatusChange = (ticketId: string, newStatus: string) => {
    console.log(`Changing ticket ${ticketId} status to ${newStatus}`);
  };

  const handleUseTemplate = (template: ResponseTemplate) => {
    setReplyMessage(template.body);
    setShowTemplates(false);
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleString('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  const getRelativeTime = (dateString: string) => {
    const date = new Date(dateString);
    const now = new Date();
    const diffInMs = now.getTime() - date.getTime();
    const diffInMinutes = Math.floor(diffInMs / (1000 * 60));

    if (diffInMinutes < 1) return 'Just now';
    if (diffInMinutes < 60) return `${diffInMinutes}m ago`;
    const diffInHours = Math.floor(diffInMinutes / 60);
    if (diffInHours < 24) return `${diffInHours}h ago`;
    const diffInDays = Math.floor(diffInHours / 24);
    return `${diffInDays}d ago`;
  };

  return (
    <div className="flex h-screen bg-gray-50">
      {/* Tickets List */}
      <div className="w-96 bg-white border-r border-gray-200 flex flex-col">
        <div className="p-4 border-b border-gray-200">
          <h1 className="text-2xl font-bold text-gray-900 mb-4">Support Tickets</h1>

          {/* Search */}
          <div className="relative mb-3">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
            <Input
              placeholder="Search tickets..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="pl-10"
            />
          </div>

          {/* Filters */}
          <div className="grid grid-cols-2 gap-2">
            <select
              value={selectedStatus}
              onChange={(e) => setSelectedStatus(e.target.value)}
              className="h-9 px-3 rounded-md border border-gray-300 bg-white text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value="all">All Status</option>
              <option value="open">Open</option>
              <option value="in-progress">In Progress</option>
              <option value="resolved">Resolved</option>
              <option value="closed">Closed</option>
            </select>

            <select
              value={selectedPriority}
              onChange={(e) => setSelectedPriority(e.target.value)}
              className="h-9 px-3 rounded-md border border-gray-300 bg-white text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value="all">All Priority</option>
              <option value="urgent">Urgent</option>
              <option value="high">High</option>
              <option value="medium">Medium</option>
              <option value="low">Low</option>
            </select>
          </div>
        </div>

        {/* Tickets List */}
        <div className="flex-1 overflow-y-auto">
          {filteredTickets.map((ticket) => (
            <div
              key={ticket.id}
              onClick={() => setSelectedTicket(ticket)}
              className={cn(
                'p-4 border-b border-gray-200 cursor-pointer hover:bg-gray-50 transition-colors',
                selectedTicket?.id === ticket.id && 'bg-blue-50 border-l-4 border-l-blue-600'
              )}
            >
              <div className="flex items-start justify-between mb-2">
                <div className="flex-1">
                  <div className="flex items-center gap-2 mb-1">
                    <Badge className={cn('text-xs', getPriorityBadgeClass(ticket.priority))}>
                      {ticket.priority.toUpperCase()}
                    </Badge>
                    <Badge className={cn('text-xs', getStatusBadgeClass(ticket.status))}>
                      {ticket.status.replace('-', ' ').toUpperCase()}
                    </Badge>
                  </div>
                  <p className="font-medium text-gray-900 text-sm mb-1 line-clamp-2">{ticket.subject}</p>
                  <p className="text-xs text-gray-600">{ticket.customer.name}</p>
                </div>
              </div>
              <div className="flex items-center justify-between text-xs text-gray-500">
                <div className="flex items-center gap-1">
                  <MessageSquare className="h-3 w-3" />
                  <span>{ticket.messageCount}</span>
                </div>
                <span>{getRelativeTime(ticket.lastUpdate)}</span>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Ticket Detail */}
      <div className="flex-1 flex flex-col">
        {selectedTicket ? (
          <>
            {/* Ticket Header */}
            <div className="p-6 bg-white border-b border-gray-200">
              <div className="flex items-start justify-between mb-4">
                <div className="flex-1">
                  <div className="flex items-center gap-2 mb-2">
                    <Badge className={cn('text-xs', getPriorityBadgeClass(selectedTicket.priority))}>
                      {selectedTicket.priority.toUpperCase()}
                    </Badge>
                    <Badge className={cn('text-xs', getStatusBadgeClass(selectedTicket.status))}>
                      {selectedTicket.status.replace('-', ' ').toUpperCase()}
                    </Badge>
                    <span className="text-sm text-gray-600">{selectedTicket.category}</span>
                  </div>
                  <h2 className="text-2xl font-bold text-gray-900 mb-2">{selectedTicket.subject}</h2>
                  <div className="flex items-center gap-4 text-sm text-gray-600">
                    <div className="flex items-center gap-1">
                      <User className="h-4 w-4" />
                      <span>{selectedTicket.customer.name}</span>
                    </div>
                    <span>{selectedTicket.customer.email}</span>
                    <span className="text-xs bg-gray-100 px-2 py-1 rounded">
                      {selectedTicket.customer.tier.toUpperCase()} Tier
                    </span>
                  </div>
                </div>
                <DropdownMenu>
                  <DropdownMenuTrigger asChild>
                    <Button variant="outline" size="sm">
                      <MoreVertical className="h-4 w-4" />
                    </Button>
                  </DropdownMenuTrigger>
                  <DropdownMenuContent align="end">
                    <DropdownMenuItem onClick={() => handleStatusChange(selectedTicket.id, 'in-progress')}>
                      Mark as In Progress
                    </DropdownMenuItem>
                    <DropdownMenuItem onClick={() => handleStatusChange(selectedTicket.id, 'resolved')}>
                      Mark as Resolved
                    </DropdownMenuItem>
                    <DropdownMenuItem onClick={() => handleStatusChange(selectedTicket.id, 'closed')}>
                      Close Ticket
                    </DropdownMenuItem>
                    <DropdownMenuItem className="text-red-600">
                      <Trash2 className="h-4 w-4 mr-2" />
                      Delete Ticket
                    </DropdownMenuItem>
                  </DropdownMenuContent>
                </DropdownMenu>
              </div>
            </div>

            {/* Conversation History */}
            <div className="flex-1 overflow-y-auto p-6 space-y-4">
              {mockConversation.map((message) => (
                <div
                  key={message.id}
                  className={cn(
                    'flex gap-3',
                    message.role === 'support' && 'flex-row-reverse'
                  )}
                >
                  <div
                    className={cn(
                      'h-10 w-10 rounded-full flex items-center justify-center flex-shrink-0',
                      message.role === 'customer' ? 'bg-gray-200' : 'bg-blue-600'
                    )}
                  >
                    {message.role === 'customer' ? (
                      <User className="h-5 w-5 text-gray-600" />
                    ) : (
                      <Mail className="h-5 w-5 text-white" />
                    )}
                  </div>
                  <div className={cn('flex-1 max-w-2xl', message.role === 'support' && 'text-right')}>
                    <div className="flex items-center gap-2 mb-1">
                      {message.role === 'support' && <span className="text-sm text-gray-600">{formatDate(message.timestamp)}</span>}
                      <span className="text-sm font-medium text-gray-900">{message.author}</span>
                      {message.role === 'customer' && <span className="text-sm text-gray-600">{formatDate(message.timestamp)}</span>}
                    </div>
                    <div
                      className={cn(
                        'inline-block p-4 rounded-lg',
                        message.role === 'customer'
                          ? 'bg-gray-100 text-gray-900'
                          : 'bg-blue-600 text-white'
                      )}
                    >
                      <p className="text-sm whitespace-pre-wrap">{message.message}</p>
                    </div>
                  </div>
                </div>
              ))}
            </div>

            {/* Response Templates Modal */}
            {showTemplates && (
              <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
                <Card className="w-full max-w-2xl m-4">
                  <CardHeader>
                    <div className="flex items-center justify-between">
                      <CardTitle>Response Templates</CardTitle>
                      <Button variant="ghost" size="sm" onClick={() => setShowTemplates(false)}>
                        ✕
                      </Button>
                    </div>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-2">
                      {responseTemplates.map((template) => (
                        <div
                          key={template.id}
                          onClick={() => handleUseTemplate(template)}
                          className="p-3 border border-gray-200 rounded-lg cursor-pointer hover:bg-gray-50"
                        >
                          <p className="font-medium text-gray-900 mb-1">{template.name}</p>
                          <p className="text-sm text-gray-600 line-clamp-2">{template.body}</p>
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </Card>
              </div>
            )}

            {/* Reply Box */}
            <div className="p-6 bg-white border-t border-gray-200">
              <div className="flex items-center gap-2 mb-3">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setShowTemplates(true)}
                >
                  <FileText className="h-4 w-4 mr-2" />
                  Use Template
                </Button>
                <Button variant="outline" size="sm">
                  <Paperclip className="h-4 w-4 mr-2" />
                  Attach File
                </Button>
              </div>
              <div className="flex gap-3">
                <textarea
                  value={replyMessage}
                  onChange={(e) => setReplyMessage(e.target.value)}
                  placeholder="Type your response..."
                  className="flex-1 min-h-32 p-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 resize-y"
                />
              </div>
              <div className="flex justify-end gap-2 mt-3">
                <Button variant="outline" onClick={() => setReplyMessage('')}>
                  Clear
                </Button>
                <Button onClick={handleSendReply} disabled={!replyMessage.trim()}>
                  <Send className="h-4 w-4 mr-2" />
                  Send Reply
                </Button>
              </div>
            </div>
          </>
        ) : (
          <div className="flex-1 flex items-center justify-center text-gray-500">
            <div className="text-center">
              <MessageSquare className="h-16 w-16 mx-auto mb-4 text-gray-400" />
              <p className="text-lg font-medium">Select a ticket to view details</p>
              <p className="text-sm">Choose a ticket from the list to view conversation history and respond</p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default SupportTicketHandler;
