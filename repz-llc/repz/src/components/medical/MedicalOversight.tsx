import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/ui/molecules/Card';
import { Button } from '@/ui/atoms/Button';
import { Badge } from '@/ui/atoms/Badge';
import { Alert, AlertDescription } from '@/ui/molecules/Alert';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/ui/molecules/Tabs';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from '@/ui/molecules/Dialog';
import { supabase } from '@/integrations/supabase/client';
import { useAuth } from '@/contexts/AuthContext';
import { useToast } from '@/hooks/use-toast';
import { TierGate } from '@/components/auth/TierGate';
import { 
  UserCheck, 
  Stethoscope,
  FileText,
  Calendar,
  Shield,
  AlertTriangle,
  CheckCircle,
  Clock,
  Heart,
  Brain,
  Activity,
  Phone,
  Mail,
  MapPin,
  Star,
  Award,
  Video
} from 'lucide-react';

interface MedicalProfessional {
  id: string;
  full_name: string;
  license_number: string;
  medical_specialty: string;
  credentials: string[];
  email: string;
  phone: string;
  clinic_name: string;
  areas_of_expertise: string[];
  accepts_consultations: boolean;
  consultation_fee: number;
  license_verified: boolean;
  platform_approved: boolean;
}

interface MedicalConsultation {
  id: string;
  consultation_type: string;
  consultation_date: string;
  duration_minutes: number;
  chief_complaint: string;
  medical_recommendations: string;
  protocol_approvals: Array<{ protocol_id: string; status: string; approved_at?: string }>;
  follow_up_required: boolean;
  follow_up_weeks: number;
  status: string;
  medical_professional: {
    full_name: string;
    medical_specialty: string;
  };
}

interface MedicalClearance {
  id: string;
  clearance_type: string;
  clearance_date: string;
  valid_until: string;
  approved: boolean;
  approval_notes: string;
  restrictions: string[];
  monitoring_requirements: string[];
  medical_forms_completed: boolean;
  liability_waivers_signed: boolean;
  medical_professional: {
    full_name: string;
    medical_specialty: string;
  };
}

export const MedicalOversight: React.FC = () => {
  const { user } = useAuth();
  const { toast } = useToast();
  const [medicalProfessionals, setMedicalProfessionals] = useState<MedicalProfessional[]>([]);
  const [consultations, setConsultations] = useState<MedicalConsultation[]>([]);
  const [clearances, setClearances] = useState<MedicalClearance[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (user) {
      fetchMedicalData();
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [user]);

  const fetchMedicalData = async () => {
    try {
      setLoading(true);
      
      // Fetch available medical professionals
      const { data: professionalsData, error: profError } = await supabase
        .from('medical_professionals')
        .select('*')
        .eq('platform_approved', true)
        .eq('accepts_consultations', true);

      if (profError) throw profError;

      // Fetch user's consultations
      const { data: consultationsData, error: consultError } = await supabase
        .from('medical_consultations')
        .select(`
          *,
          medical_professional:medical_professionals(full_name, medical_specialty)
        `)
        .eq('client_id', user?.id)
        .order('consultation_date', { ascending: false });

      if (consultError) throw consultError;

      // Fetch user's clearances
      const { data: clearancesData, error: clearError } = await supabase
        .from('medical_clearances')
        .select(`
          *,
          medical_professional:medical_professionals(full_name, medical_specialty)
        `)
        .eq('client_id', user?.id)
        .order('clearance_date', { ascending: false });

      if (clearError) throw clearError;

      setMedicalProfessionals(professionalsData || []);
      setConsultations(consultationsData || []);
      setClearances(clearancesData || []);

    } catch (error) {
      console.error('Error fetching medical data:', error);
      toast({
        title: "Error",
        description: "Failed to load medical oversight data",
        variant: "destructive"
      });
    } finally {
      setLoading(false);
    }
  };

  const getConsultationStatusColor = (status: string) => {
    switch (status) {
      case 'completed': return 'bg-green-100 text-green-800';
      case 'scheduled': return 'bg-blue-100 text-blue-800';
      case 'cancelled': return 'bg-red-100 text-red-800';
      case 'no_show': return 'bg-gray-100 text-gray-800';
      default: return 'bg-yellow-100 text-yellow-800';
    }
  };

  const getClearanceStatusColor = (approved: boolean, validUntil: string) => {
    const isExpired = new Date(validUntil) < new Date();
    if (approved && !isExpired) return 'bg-green-100 text-green-800';
    if (approved && isExpired) return 'bg-yellow-100 text-yellow-800';
    return 'bg-red-100 text-red-800';
  };

  const getSpecialtyIcon = (specialty: string) => {
    if (specialty.includes('hormone') || specialty.includes('endocrin')) return <Heart className="h-4 w-4" />;
    if (specialty.includes('sports') || specialty.includes('exercise')) return <Activity className="h-4 w-4" />;
    if (specialty.includes('anti-aging') || specialty.includes('longevity')) return <Shield className="h-4 w-4" />;
    return <Stethoscope className="h-4 w-4" />;
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center p-8">
        <UserCheck className="h-8 w-8 animate-pulse text-primary" />
      </div>
    );
  }

  return (
    <TierGate requiredTier="performance" feature="peds_protocols">
      <div className="space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-2xl font-bold text-gray-900 flex items-center gap-3">
              <UserCheck className="h-6 w-6 text-blue-600" />
              Medical Oversight & Safety
            </h2>
            <p className="text-gray-600 mt-1">Professional medical supervision for advanced protocols</p>
          </div>
          
          <Button onClick={() => console.log("Schedule consultation")}>
            <Calendar className="h-4 w-4 mr-2" />
            Schedule Consultation
          </Button>
        </div>

        {/* Medical Supervision Alert */}
        <Alert className="border-blue-500/30 bg-blue-500/10">
          <Stethoscope className="h-4 w-4 text-blue-500" />
          <AlertDescription>
            <strong className="text-blue-600">Medical Supervision Required:</strong> All advanced protocols (PEDs, peptides, bioregulators) 
            require medical oversight by qualified healthcare professionals. This ensures your safety and protocol optimization.
          </AlertDescription>
        </Alert>

        <Tabs defaultValue="overview" className="space-y-4">
          <TabsList>
            <TabsTrigger value="overview">Overview</TabsTrigger>
            <TabsTrigger value="consultations">Consultations</TabsTrigger>
            <TabsTrigger value="clearances">Medical Clearances</TabsTrigger>
            <TabsTrigger value="professionals">Find Doctors</TabsTrigger>
            <TabsTrigger value="monitoring">Health Monitoring</TabsTrigger>
          </TabsList>

          <TabsContent value="overview" className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {/* Active Clearances */}
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm font-medium flex items-center gap-2">
                    <Shield className="h-4 w-4 text-green-600" />
                    Active Clearances
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold text-green-600">
                    {clearances.filter(c => c.approved && new Date(c.valid_until) > new Date()).length}
                  </div>
                  <p className="text-xs text-muted-foreground">Currently approved protocols</p>
                </CardContent>
              </Card>

              {/* Upcoming Consultations */}
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm font-medium flex items-center gap-2">
                    <Calendar className="h-4 w-4 text-blue-600" />
                    Upcoming
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold text-blue-600">
                    {consultations.filter(c => c.status === 'scheduled' && new Date(c.consultation_date) > new Date()).length}
                  </div>
                  <p className="text-xs text-muted-foreground">Scheduled consultations</p>
                </CardContent>
              </Card>

              {/* Monitoring Status */}
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm font-medium flex items-center gap-2">
                    <Heart className="h-4 w-4 text-red-600" />
                    Monitoring
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold text-red-600">2</div>
                  <p className="text-xs text-muted-foreground">Active monitoring protocols</p>
                </CardContent>
              </Card>
            </div>

            {/* Recent Activity */}
            <Card>
              <CardHeader>
                <CardTitle>Recent Medical Activity</CardTitle>
                <CardDescription>Latest consultations and clearances</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {/* Recent Consultations */}
                  {consultations.slice(0, 3).map((consultation) => (
                    <div key={consultation.id} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                      <div className="flex items-center gap-3">
                        <div className="p-2 bg-blue-100 rounded-full">
                          <Stethoscope className="h-4 w-4 text-blue-600" />
                        </div>
                        <div>
                          <p className="font-medium text-sm">{consultation.consultation_type.replace('_', ' ')}</p>
                          <p className="text-xs text-muted-foreground">
                            {consultation.medical_professional?.full_name} • {new Date(consultation.consultation_date).toLocaleDateString()}
                          </p>
                        </div>
                      </div>
                      <Badge className={getConsultationStatusColor(consultation.status)}>
                        {consultation.status.toUpperCase()}
                      </Badge>
                    </div>
                  ))}

                  {/* Recent Clearances */}
                  {clearances.slice(0, 2).map((clearance) => (
                    <div key={clearance.id} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                      <div className="flex items-center gap-3">
                        <div className="p-2 bg-green-100 rounded-full">
                          <Shield className="h-4 w-4 text-green-600" />
                        </div>
                        <div>
                          <p className="font-medium text-sm">{clearance.clearance_type.replace('_', ' ')} Clearance</p>
                          <p className="text-xs text-muted-foreground">
                            Valid until {new Date(clearance.valid_until).toLocaleDateString()}
                          </p>
                        </div>
                      </div>
                      <Badge className={getClearanceStatusColor(clearance.approved, clearance.valid_until)}>
                        {clearance.approved ? 'APPROVED' : 'PENDING'}
                      </Badge>
                    </div>
                  ))}

                  {consultations.length === 0 && clearances.length === 0 && (
                    <div className="text-center py-8">
                      <UserCheck className="h-12 w-12 mx-auto mb-4 text-muted-foreground" />
                      <h3 className="text-lg font-semibold mb-2">No Medical Activity Yet</h3>
                      <p className="text-muted-foreground mb-4">Schedule your first consultation to get started</p>
                      <Button>
                        <Calendar className="h-4 w-4 mr-2" />
                        Schedule Consultation
                      </Button>
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="consultations" className="space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="text-lg font-semibold">Medical Consultations ({consultations.length})</h3>
              <Button>
                <Calendar className="h-4 w-4 mr-2" />
                Schedule New Consultation
              </Button>
            </div>

            <div className="grid gap-4">
              {consultations.map((consultation) => (
                <Card key={consultation.id}>
                  <CardHeader>
                    <div className="flex items-center justify-between">
                      <div>
                        <CardTitle className="text-lg">{consultation.consultation_type.replace('_', ' ')}</CardTitle>
                        <CardDescription>
                          Dr. {consultation.medical_professional?.full_name} • {consultation.medical_professional?.medical_specialty}
                        </CardDescription>
                      </div>
                      <Badge className={getConsultationStatusColor(consultation.status)}>
                        {consultation.status.toUpperCase()}
                      </Badge>
                    </div>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-4">
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                        <div>
                          <p className="text-muted-foreground">Date</p>
                          <p className="font-medium">{new Date(consultation.consultation_date).toLocaleDateString()}</p>
                        </div>
                        <div>
                          <p className="text-muted-foreground">Duration</p>
                          <p className="font-medium">{consultation.duration_minutes} minutes</p>
                        </div>
                        <div>
                          <p className="text-muted-foreground">Follow-up</p>
                          <p className="font-medium">
                            {consultation.follow_up_required ? `${consultation.follow_up_weeks} weeks` : 'Not required'}
                          </p>
                        </div>
                        <div>
                          <p className="text-muted-foreground">Type</p>
                          <p className="font-medium capitalize">{consultation.consultation_type.replace('_', ' ')}</p>
                        </div>
                      </div>

                      {consultation.chief_complaint && (
                        <div>
                          <h4 className="font-medium mb-1">Chief Complaint:</h4>
                          <p className="text-sm text-muted-foreground">{consultation.chief_complaint}</p>
                        </div>
                      )}

                      {consultation.medical_recommendations && (
                        <div>
                          <h4 className="font-medium mb-1">Medical Recommendations:</h4>
                          <p className="text-sm text-muted-foreground">{consultation.medical_recommendations}</p>
                        </div>
                      )}

                      <div className="flex gap-2">
                        <Button size="sm" variant="outline">
                          <FileText className="h-3 w-3 mr-1" />
                          View Notes
                        </Button>
                        {consultation.status === 'scheduled' && (
                          <Button size="sm" variant="outline">
                            <Video className="h-3 w-3 mr-1" />
                            Join Session
                          </Button>
                        )}
                      </div>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          </TabsContent>

          <TabsContent value="clearances" className="space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="text-lg font-semibold">Medical Clearances ({clearances.length})</h3>
              <Button>
                <Shield className="h-4 w-4 mr-2" />
                Request Clearance
              </Button>
            </div>

            <div className="grid gap-4">
              {clearances.map((clearance) => (
                <Card key={clearance.id}>
                  <CardHeader>
                    <div className="flex items-center justify-between">
                      <div>
                        <CardTitle className="text-lg">{clearance.clearance_type.replace('_', ' ')} Clearance</CardTitle>
                        <CardDescription>
                          Dr. {clearance.medical_professional?.full_name} • {clearance.medical_professional?.medical_specialty}
                        </CardDescription>
                      </div>
                      <Badge className={getClearanceStatusColor(clearance.approved, clearance.valid_until)}>
                        {clearance.approved ? 'APPROVED' : 'PENDING'}
                      </Badge>
                    </div>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-4">
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                        <div>
                          <p className="text-muted-foreground">Clearance Date</p>
                          <p className="font-medium">{new Date(clearance.clearance_date).toLocaleDateString()}</p>
                        </div>
                        <div>
                          <p className="text-muted-foreground">Valid Until</p>
                          <p className="font-medium">{new Date(clearance.valid_until).toLocaleDateString()}</p>
                        </div>
                        <div>
                          <p className="text-muted-foreground">Forms</p>
                          <p className="font-medium">
                            {clearance.medical_forms_completed ? 'Complete' : 'Pending'}
                          </p>
                        </div>
                        <div>
                          <p className="text-muted-foreground">Waivers</p>
                          <p className="font-medium">
                            {clearance.liability_waivers_signed ? 'Signed' : 'Pending'}
                          </p>
                        </div>
                      </div>

                      {clearance.approval_notes && (
                        <div>
                          <h4 className="font-medium mb-1">Medical Notes:</h4>
                          <p className="text-sm text-muted-foreground">{clearance.approval_notes}</p>
                        </div>
                      )}

                      {clearance.restrictions.length > 0 && (
                        <div>
                          <h4 className="font-medium mb-2">Restrictions:</h4>
                          <div className="space-y-1">
                            {clearance.restrictions.map((restriction, index) => (
                              <div key={index} className="flex items-center gap-2 text-sm">
                                <AlertTriangle className="h-3 w-3 text-yellow-500" />
                                {restriction}
                              </div>
                            ))}
                          </div>
                        </div>
                      )}

                      {clearance.monitoring_requirements.length > 0 && (
                        <div>
                          <h4 className="font-medium mb-2">Monitoring Requirements:</h4>
                          <div className="space-y-1">
                            {clearance.monitoring_requirements.map((requirement, index) => (
                              <div key={index} className="flex items-center gap-2 text-sm">
                                <Heart className="h-3 w-3 text-red-500" />
                                {requirement}
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          </TabsContent>

          <TabsContent value="professionals" className="space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="text-lg font-semibold">Available Medical Professionals ({medicalProfessionals.length})</h3>
              <Button variant="outline">
                <Shield className="h-4 w-4 mr-2" />
                Filter by Specialty
              </Button>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {medicalProfessionals.map((doctor) => (
                <Card key={doctor.id} className="hover:shadow-md transition-shadow">
                  <CardHeader>
                    <div className="flex items-start justify-between">
                      <div>
                        <CardTitle className="text-lg flex items-center gap-2">
                          {getSpecialtyIcon(doctor.medical_specialty)}
                          Dr. {doctor.full_name}
                        </CardTitle>
                        <CardDescription>{doctor.medical_specialty}</CardDescription>
                      </div>
                      <div className="flex items-center gap-1">
                        {doctor.license_verified && (
                          <Badge variant="outline" className="text-green-600">
                            <CheckCircle className="h-3 w-3 mr-1" />
                            Verified
                          </Badge>
                        )}
                      </div>
                    </div>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-4">
                      {/* Credentials */}
                      <div>
                        <div className="flex flex-wrap gap-1">
                          {doctor.credentials.slice(0, 3).map((credential) => (
                            <Badge key={credential} variant="secondary" className="text-xs">
                              {credential}
                            </Badge>
                          ))}
                        </div>
                      </div>

                      {/* Expertise */}
                      <div>
                        <h4 className="font-medium mb-2 text-sm">Areas of Expertise:</h4>
                        <div className="space-y-1">
                          {doctor.areas_of_expertise.slice(0, 3).map((area, index) => (
                            <div key={index} className="flex items-center gap-2 text-sm">
                              <Star className="h-3 w-3 text-yellow-500" />
                              {area}
                            </div>
                          ))}
                        </div>
                      </div>

                      {/* Contact & Pricing */}
                      <div className="grid grid-cols-2 gap-4 text-sm">
                        <div>
                          <p className="text-muted-foreground">Consultation Fee</p>
                          <p className="font-medium">${doctor.consultation_fee}</p>
                        </div>
                        <div>
                          <p className="text-muted-foreground">Clinic</p>
                          <p className="font-medium">{doctor.clinic_name}</p>
                        </div>
                      </div>

                      {/* Actions */}
                      <div className="flex gap-2">
                        <Button size="sm" className="flex-1">
                          <Calendar className="h-3 w-3 mr-1" />
                          Book Consultation
                        </Button>
                        <Button size="sm" variant="outline">
                          <FileText className="h-3 w-3 mr-1" />
                          View Profile
                        </Button>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>

            {medicalProfessionals.length === 0 && (
              <Card>
                <CardContent className="p-8 text-center">
                  <Stethoscope className="h-12 w-12 mx-auto mb-4 text-muted-foreground" />
                  <h3 className="text-lg font-semibold mb-2">No Doctors Available</h3>
                  <p className="text-muted-foreground">Medical professionals will be listed here when available</p>
                </CardContent>
              </Card>
            )}
          </TabsContent>

          <TabsContent value="monitoring" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Heart className="h-5 w-5 text-red-600" />
                  Health Monitoring Dashboard
                </CardTitle>
                <CardDescription>
                  Track required health metrics and monitoring protocols
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                  <div className="p-4 border rounded-lg">
                    <h4 className="font-medium mb-2">Blood Work Monitoring</h4>
                    <p className="text-sm text-muted-foreground">Regular lab work for protocol safety</p>
                    <Button size="sm" className="mt-2 w-full" variant="outline">
                      <FileText className="h-3 w-3 mr-1" />
                      Upload Results
                    </Button>
                  </div>
                  
                  <div className="p-4 border rounded-lg">
                    <h4 className="font-medium mb-2">Vital Signs Tracking</h4>
                    <p className="text-sm text-muted-foreground">Blood pressure, heart rate monitoring</p>
                    <Button size="sm" className="mt-2 w-full" variant="outline">
                      <Heart className="h-3 w-3 mr-1" />
                      Log Vitals
                    </Button>
                  </div>
                  
                  <div className="p-4 border rounded-lg">
                    <h4 className="font-medium mb-2">Side Effect Reporting</h4>
                    <p className="text-sm text-muted-foreground">Report any adverse reactions</p>
                    <Button size="sm" className="mt-2 w-full" variant="outline">
                      <AlertTriangle className="h-3 w-3 mr-1" />
                      Report Issue
                    </Button>
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    </TierGate>
  );
};