import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/ui/molecules/Card';
import { Button } from '@/ui/atoms/Button';
import { Badge } from '@/ui/atoms/Badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/ui/molecules/Tabs';
import { Input } from '@/ui/atoms/Input';
import { Textarea } from '@/components/ui/textarea';
import { LoadingButton } from '@/components/ui/loading-button';
import { Avatar, AvatarFallback, AvatarImage } from '@/ui/atoms/Avatar';
import { useToast } from '@/hooks/use-toast';
import { useTierAccess } from '@/hooks/useTierAccess';
import { supabase } from '@/integrations/supabase/client';
import { 
  Crown, 
  MessageSquare, 
  Calendar, 
  MapPin, 
  Clock, 
  Star,
  Send,
  CheckCircle,
  AlertCircle,
  User,
  Heart,
  Brain,
  Trophy,
  Zap
} from 'lucide-react';

interface ConciergeRequest {
  id: string;
  type: 'booking' | 'nutrition' | 'lifestyle' | 'emergency' | 'general';
  title: string;
  description: string;
  priority: 'low' | 'medium' | 'high' | 'urgent';
  status: 'pending' | 'processing' | 'completed' | 'cancelled';
  created_at: string;
  updated_at: string;
  estimated_completion?: string;
  concierge_notes?: string;
}

interface ConciergeProfile {
  name: string;
  role: string;
  avatar: string;
  specialties: string[];
  response_time: string;
  rating: number;
}

export const AIConciergeService: React.FC = () => {
  const [requests, setRequests] = useState<ConciergeRequest[]>([]);
  const [newRequest, setNewRequest] = useState({
    type: 'general' as const,
    title: '',
    description: '',
    priority: 'medium' as const
  });
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [loading, setLoading] = useState(true);
  const { toast } = useToast();
  const { hasFeature } = useTierAccess();

  const conciergeTeam: ConciergeProfile[] = [
    {
      name: "Dr. Sarah Chen",
      role: "Chief Wellness Concierge",
      avatar: "/lovable-uploads/user-profile.png",
      specialties: ["Longevity", "Biomarkers", "Executive Health"],
      response_time: "< 30 minutes",
      rating: 4.9
    },
    {
      name: "Marcus Rodriguez",
      role: "Lifestyle Coordinator",
      avatar: "/lovable-uploads/user-profile.png",
      specialties: ["Travel Planning", "Meal Prep", "Scheduling"],
      response_time: "< 1 hour",
      rating: 4.8
    },
    {
      name: "Dr. Alex Park",
      role: "Performance Specialist",
      avatar: "/lovable-uploads/user-profile.png",
      specialties: ["Recovery", "Sleep", "Supplementation"],
      response_time: "< 2 hours",
      rating: 5.0
    }
  ];

  useEffect(() => {
    fetchRequests();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const fetchRequests = async () => {
    try {
      // Mock data for demonstration
      const mockRequests: ConciergeRequest[] = [
        {
          id: '1',
          type: 'booking',
          title: 'Schedule In-Person Training Session',
          description: 'Need to book a 90-minute session for next week, preferably Tuesday afternoon',
          priority: 'medium',
          status: 'processing',
          created_at: new Date(Date.now() - 86400000).toISOString(),
          updated_at: new Date(Date.now() - 43200000).toISOString(),
          estimated_completion: new Date(Date.now() + 86400000).toISOString(),
          concierge_notes: 'Coordinating with coach availability. Will confirm by tomorrow.'
        },
        {
          id: '2',
          type: 'nutrition',
          title: 'Custom Meal Plan for Business Travel',
          description: 'Traveling to Tokyo for 2 weeks, need meal recommendations for healthy eating while maintaining my cutting protocol',
          priority: 'high',
          status: 'completed',
          created_at: new Date(Date.now() - 172800000).toISOString(),
          updated_at: new Date(Date.now() - 86400000).toISOString(),
          concierge_notes: 'Curated list of Tokyo restaurants and meal delivery services sent to your email.'
        }
      ];
      
      setRequests(mockRequests);
    } catch (error) {
      console.error('Error fetching requests:', error);
      toast({
        title: "Error",
        description: "Failed to load concierge requests",
        variant: "destructive"
      });
    } finally {
      setLoading(false);
    }
  };

  const submitRequest = async () => {
    if (!newRequest.title.trim() || !newRequest.description.trim()) {
      toast({
        title: "Missing Information",
        description: "Please provide both a title and description for your request",
        variant: "destructive"
      });
      return;
    }

    setIsSubmitting(true);
    try {
      // Create new request
      const request: ConciergeRequest = {
        id: Date.now().toString(),
        ...newRequest,
        status: 'pending',
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString()
      };

      setRequests(prev => [request, ...prev]);
      setNewRequest({
        type: 'general',
        title: '',
        description: '',
        priority: 'medium'
      });

      toast({
        title: "Request Submitted",
        description: "Your concierge request has been submitted. Our team will respond within 30 minutes.",
      });
    } catch (error) {
      console.error('Error submitting request:', error);
      toast({
        title: "Error",
        description: "Failed to submit request. Please try again.",
        variant: "destructive"
      });
    } finally {
      setIsSubmitting(false);
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed': return <CheckCircle className="h-4 w-4 text-green-500" />;
      case 'processing': return <Clock className="h-4 w-4 text-yellow-500" />;
      case 'pending': return <AlertCircle className="h-4 w-4 text-blue-500" />;
      default: return <AlertCircle className="h-4 w-4 text-gray-500" />;
    }
  };

  const getTypeIcon = (type: string) => {
    switch (type) {
      case 'booking': return <Calendar className="h-4 w-4" />;
      case 'nutrition': return <Heart className="h-4 w-4" />;
      case 'lifestyle': return <Star className="h-4 w-4" />;
      case 'emergency': return <Zap className="h-4 w-4" />;
      default: return <MessageSquare className="h-4 w-4" />;
    }
  };

  const getPriorityColor = (priority: string) => {
    switch (priority) {
      case 'urgent': return 'bg-red-500';
      case 'high': return 'bg-orange-500';
      case 'medium': return 'bg-yellow-500';
      case 'low': return 'bg-green-500';
      default: return 'bg-gray-500';
    }
  };

  if (!hasFeature('voice_coaching')) {
    return (
      <Card className="border-gold">
        <CardHeader>
          <div className="flex items-center gap-2">
            <Crown className="h-5 w-5 text-gold" />
            <CardTitle>AI Concierge Service</CardTitle>
          </div>
          <CardDescription>
            Premium concierge services available with Longevity tier
          </CardDescription>
        </CardHeader>
        <CardContent className="text-center py-8">
          <p className="text-muted-foreground mb-4">
            Upgrade to access our 24/7 AI-powered concierge service for personalized assistance
          </p>
          <Button variant="outline" onClick={() => window.location.href = '/pricing'}>
            Upgrade to Longevity
          </Button>
        </CardContent>
      </Card>
    );
  }

  if (loading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Loading Concierge Service...</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="animate-pulse space-y-4">
            <div className="h-4 bg-gray-200 rounded w-3/4"></div>
            <div className="h-4 bg-gray-200 rounded w-1/2"></div>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="space-y-6">
      <Card className="border-gold">
        <CardHeader>
          <div className="flex items-center gap-2">
            <Crown className="h-5 w-5 text-gold" />
            <CardTitle>AI Concierge Service</CardTitle>
            <Badge variant="secondary" className="ml-auto">
              Longevity Tier
            </Badge>
          </div>
          <CardDescription>
            24/7 personalized assistance for all your health and wellness needs
          </CardDescription>
        </CardHeader>
      </Card>

      <Tabs defaultValue="requests" className="w-full">
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="requests">My Requests</TabsTrigger>
          <TabsTrigger value="new">New Request</TabsTrigger>
          <TabsTrigger value="team">Concierge Team</TabsTrigger>
        </TabsList>

        <TabsContent value="requests" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Recent Requests</CardTitle>
              <CardDescription>
                Track your concierge service requests and responses
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {requests.length === 0 ? (
                <div className="text-center py-8 text-muted-foreground">
                  <MessageSquare className="h-12 w-12 mx-auto mb-4 opacity-50" />
                  <p>No requests yet. Submit your first concierge request!</p>
                </div>
              ) : (
                requests.map((request) => (
                  <Card key={request.id} className="border-l-4 border-l-primary">
                    <CardContent className="pt-4">
                      <div className="flex items-start justify-between mb-2">
                        <div className="flex items-center gap-2">
                          {getTypeIcon(request.type)}
                          <h4 className="font-semibold">{request.title}</h4>
                          <div className={`w-2 h-2 rounded-full ${getPriorityColor(request.priority)}`} />
                        </div>
                        <div className="flex items-center gap-2">
                          {getStatusIcon(request.status)}
                          <Badge variant="outline" className="capitalize">
                            {request.status}
                          </Badge>
                        </div>
                      </div>
                      <p className="text-muted-foreground mb-3">{request.description}</p>
                      {request.concierge_notes && (
                        <div className="bg-muted p-3 rounded-md mb-3">
                          <p className="text-sm"><strong>Concierge Notes:</strong> {request.concierge_notes}</p>
                        </div>
                      )}
                      <div className="flex items-center justify-between text-sm text-muted-foreground">
                        <span>Created: {new Date(request.created_at).toLocaleDateString()}</span>
                        {request.estimated_completion && (
                          <span>Est. Completion: {new Date(request.estimated_completion).toLocaleDateString()}</span>
                        )}
                      </div>
                    </CardContent>
                  </Card>
                ))
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="new" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Submit New Request</CardTitle>
              <CardDescription>
                Our concierge team is here to assist with any health, wellness, or lifestyle needs
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="text-sm font-medium">Request Type</label>
                  <select
                    value={newRequest.type}
                    onChange={(e) => setNewRequest(prev => ({ ...prev, type: e.target.value as ConciergeRequest['type'] }))}
                    className="w-full mt-1 p-2 border rounded-md"
                  >
                    <option value="general">General Assistance</option>
                    <option value="booking">Appointment Booking</option>
                    <option value="nutrition">Nutrition Planning</option>
                    <option value="lifestyle">Lifestyle Coordination</option>
                    <option value="emergency">Urgent Support</option>
                  </select>
                </div>
                <div>
                  <label className="text-sm font-medium">Priority Level</label>
                  <select
                    value={newRequest.priority}
                    onChange={(e) => setNewRequest(prev => ({ ...prev, priority: e.target.value as ConciergeRequest['priority'] }))}
                    className="w-full mt-1 p-2 border rounded-md"
                  >
                    <option value="low">Low - Within 24 hours</option>
                    <option value="medium">Medium - Within 4 hours</option>
                    <option value="high">High - Within 1 hour</option>
                    <option value="urgent">Urgent - Within 30 minutes</option>
                  </select>
                </div>
              </div>

              <div>
                <label className="text-sm font-medium">Request Title</label>
                <Input
                  value={newRequest.title}
                  onChange={(e) => setNewRequest(prev => ({ ...prev, title: e.target.value }))}
                  placeholder="Brief description of what you need"
                  className="mt-1"
                />
              </div>

              <div>
                <label className="text-sm font-medium">Detailed Description</label>
                <Textarea
                  value={newRequest.description}
                  onChange={(e) => setNewRequest(prev => ({ ...prev, description: e.target.value }))}
                  placeholder="Provide details about your request, including any specific requirements, preferences, or deadlines..."
                  className="mt-1 min-h-[120px]"
                />
              </div>

              <LoadingButton
                onClick={submitRequest}
                loading={isSubmitting}
                className="w-full"
                disabled={!newRequest.title.trim() || !newRequest.description.trim()}
              >
                <Send className="h-4 w-4 mr-2" />
                Submit Request
              </LoadingButton>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="team" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Your Concierge Team</CardTitle>
              <CardDescription>
                Meet the professionals dedicated to your success
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {conciergeTeam.map((member) => (
                <Card key={member.name} className="border-l-4 border-l-gold">
                  <CardContent className="pt-4">
                    <div className="flex items-start gap-4">
                      <Avatar className="h-16 w-16">
                        <AvatarImage src={member.avatar} alt={member.name} />
                        <AvatarFallback>
                          {member.name.split(' ').map(n => n[0]).join('')}
                        </AvatarFallback>
                      </Avatar>
                      <div className="flex-1">
                        <div className="flex items-center gap-2 mb-1">
                          <h4 className="font-semibold">{member.name}</h4>
                          <div className="flex items-center gap-1">
                            <Star className="h-4 w-4 text-yellow-500 fill-current" />
                            <span className="text-sm">{member.rating}</span>
                          </div>
                        </div>
                        <p className="text-sm text-muted-foreground mb-2">{member.role}</p>
                        <div className="flex flex-wrap gap-1 mb-2">
                          {member.specialties.map((specialty) => (
                            <Badge key={specialty} variant="secondary" className="text-xs">
                              {specialty}
                            </Badge>
                          ))}
                        </div>
                        <div className="flex items-center gap-2 text-sm text-muted-foreground">
                          <Clock className="h-3 w-3" />
                          <span>Response time: {member.response_time}</span>
                        </div>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
};