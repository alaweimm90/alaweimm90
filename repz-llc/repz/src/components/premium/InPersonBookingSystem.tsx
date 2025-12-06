import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/ui/molecules/Card';
import { Button } from '@/ui/atoms/Button';
import { Badge } from '@/ui/atoms/Badge';
import { Calendar } from '@/components/ui/calendar';
import { Input } from '@/ui/atoms/Input';
import { Textarea } from '@/components/ui/textarea';
import { LoadingButton } from '@/components/ui/loading-button';
import { useToast } from '@/hooks/use-toast';
import { useTierAccess } from '@/hooks/useTierAccess';
import { supabase } from '@/integrations/supabase/client';
import { 
  MapPin, 
  Clock, 
  Calendar as CalendarIcon, 
  User, 
  DollarSign,
  CheckCircle,
  AlertCircle,
  Car,
  Home,
  Building,
  Star,
  Crown,
  Dumbbell
} from 'lucide-react';

interface BookingSlot {
  id: string;
  date: string;
  time: string;
  duration: number;
  location: 'home' | 'gym' | 'client_location';
  available: boolean;
  price: number;
  trainer: string;
}

interface Booking {
  id: string;
  date: string;
  time: string;
  duration: number;
  location: string;
  location_address?: string;
  session_type: string;
  status: 'pending' | 'confirmed' | 'completed' | 'cancelled';
  price: number;
  notes?: string;
  trainer_notes?: string;
  created_at: string;
}

export const InPersonBookingSystem: React.FC = () => {
  const [selectedDate, setSelectedDate] = useState<Date | undefined>(new Date());
  const [availableSlots, setAvailableSlots] = useState<BookingSlot[]>([]);
  const [bookings, setBookings] = useState<Booking[]>([]);
  const [selectedSlot, setSelectedSlot] = useState<BookingSlot | null>(null);
  const [bookingDetails, setBookingDetails] = useState({
    session_type: 'personal_training',
    location_preference: 'gym',
    custom_address: '',
    notes: ''
  });
  const [isBooking, setIsBooking] = useState(false);
  const [loading, setLoading] = useState(true);
  const { toast } = useToast();
  const { hasFeature } = useTierAccess();

  const sessionTypes = [
    { id: 'personal_training', name: 'Personal Training', duration: 60, price: 150 },
    { id: 'strength_assessment', name: 'Strength Assessment', duration: 90, price: 200 },
    { id: 'form_analysis', name: 'Form Analysis', duration: 45, price: 125 },
    { id: 'mobility_session', name: 'Mobility & Recovery', duration: 60, price: 130 },
    { id: 'nutrition_consultation', name: 'Nutrition Consultation', duration: 60, price: 140 }
  ];

  const locations = [
    { id: 'gym', name: 'REPZ Gym (Palo Alto)', icon: Dumbbell, address: '123 University Ave, Palo Alto, CA' },
    { id: 'home', name: 'Trainer Comes to You', icon: Home, address: 'Your location' },
    { id: 'client_location', name: 'Your Gym/Facility', icon: Building, address: 'Custom location' }
  ];

  useEffect(() => {
    if (selectedDate) {
      fetchAvailableSlots(selectedDate);
    }
    fetchBookings();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedDate]);

  const fetchAvailableSlots = async (date: Date) => {
    try {
      // Mock available slots for demonstration
      const mockSlots: BookingSlot[] = [
        {
          id: '1',
          date: date.toISOString().split('T')[0],
          time: '09:00',
          duration: 60,
          location: 'gym',
          available: true,
          price: 150,
          trainer: 'Dr. Mike Chen'
        },
        {
          id: '2',
          date: date.toISOString().split('T')[0],
          time: '11:00',
          duration: 90,
          location: 'home',
          available: true,
          price: 200,
          trainer: 'Dr. Mike Chen'
        },
        {
          id: '3',
          date: date.toISOString().split('T')[0],
          time: '14:00',
          duration: 60,
          location: 'gym',
          available: false,
          price: 150,
          trainer: 'Dr. Mike Chen'
        },
        {
          id: '4',
          date: date.toISOString().split('T')[0],
          time: '16:00',
          duration: 60,
          location: 'client_location',
          available: true,
          price: 175,
          trainer: 'Dr. Mike Chen'
        }
      ];
      setAvailableSlots(mockSlots);
    } catch (error) {
      console.error('Error fetching slots:', error);
      toast({
        title: "Error",
        description: "Failed to load available time slots",
        variant: "destructive"
      });
    }
  };

  const fetchBookings = async () => {
    try {
      // Mock booking data
      const mockBookings: Booking[] = [
        {
          id: '1',
          date: '2024-01-15',
          time: '10:00',
          duration: 60,
          location: 'REPZ Gym',
          location_address: '123 University Ave, Palo Alto, CA',
          session_type: 'Personal Training',
          status: 'confirmed',
          price: 150,
          notes: 'Focus on deadlift form and lower body strength',
          trainer_notes: 'Client progressing well, increase weight next session',
          created_at: new Date().toISOString()
        },
        {
          id: '2',
          date: '2024-01-22',
          time: '14:00',
          duration: 90,
          location: 'Client Home',
          session_type: 'Strength Assessment',
          status: 'pending',
          price: 200,
          created_at: new Date().toISOString()
        }
      ];
      setBookings(mockBookings);
    } catch (error) {
      console.error('Error fetching bookings:', error);
    } finally {
      setLoading(false);
    }
  };

  const bookSession = async () => {
    if (!selectedSlot) return;

    setIsBooking(true);
    try {
      const newBooking: Booking = {
        id: Date.now().toString(),
        date: selectedSlot.date,
        time: selectedSlot.time,
        duration: selectedSlot.duration,
        location: locations.find(l => l.id === bookingDetails.location_preference)?.name || 'TBD',
        location_address: bookingDetails.location_preference === 'client_location' ? bookingDetails.custom_address : undefined,
        session_type: sessionTypes.find(s => s.id === bookingDetails.session_type)?.name || 'Personal Training',
        status: 'pending',
        price: selectedSlot.price,
        notes: bookingDetails.notes,
        created_at: new Date().toISOString()
      };

      setBookings(prev => [newBooking, ...prev]);
      setSelectedSlot(null);
      setBookingDetails({
        session_type: 'personal_training',
        location_preference: 'gym',
        custom_address: '',
        notes: ''
      });

      toast({
        title: "Booking Submitted",
        description: "Your session has been requested. You'll receive confirmation within 2 hours.",
      });

      // Update available slots
      fetchAvailableSlots(selectedDate!);
    } catch (error) {
      console.error('Error booking session:', error);
      toast({
        title: "Error",
        description: "Failed to book session. Please try again.",
        variant: "destructive"
      });
    } finally {
      setIsBooking(false);
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'confirmed': return <CheckCircle className="h-4 w-4 text-green-500" />;
      case 'pending': return <Clock className="h-4 w-4 text-yellow-500" />;
      case 'completed': return <CheckCircle className="h-4 w-4 text-blue-500" />;
      case 'cancelled': return <AlertCircle className="h-4 w-4 text-red-500" />;
      default: return <AlertCircle className="h-4 w-4 text-gray-500" />;
    }
  };

  const getLocationIcon = (location: string) => {
    if (location === 'gym') return <Dumbbell className="h-4 w-4" />;
    if (location === 'home') return <Home className="h-4 w-4" />;
    return <Building className="h-4 w-4" />;
  };

  if (!hasFeature('in_person_training')) {
    return (
      <Card className="border-gold">
        <CardHeader>
          <div className="flex items-center gap-2">
            <Crown className="h-5 w-5 text-gold" />
            <CardTitle>In-Person Training</CardTitle>
          </div>
          <CardDescription>
            Bay Area in-person training available with Longevity tier
          </CardDescription>
        </CardHeader>
        <CardContent className="text-center py-8">
          <p className="text-muted-foreground mb-4">
            Upgrade to access personalized in-person training sessions in the Bay Area
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
          <CardTitle>Loading Booking System...</CardTitle>
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
            <CardTitle>In-Person Training Sessions</CardTitle>
            <Badge variant="secondary" className="ml-auto">
              Bay Area Only
            </Badge>
          </div>
          <CardDescription>
            Book personalized training sessions with Dr. Mike Chen in the Bay Area
          </CardDescription>
        </CardHeader>
      </Card>

      <div className="grid lg:grid-cols-2 gap-6">
        {/* Booking Calendar */}
        <Card>
          <CardHeader>
            <CardTitle>Select Date & Time</CardTitle>
            <CardDescription>
              Choose your preferred session date and available time slot
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <Calendar
              mode="single"
              selected={selectedDate}
              onSelect={setSelectedDate}
              disabled={(date) => date < new Date() || date.getDay() === 0} // Disable past dates and Sundays
              className="rounded-md border"
            />

            {selectedDate && (
              <div className="space-y-2">
                <h4 className="font-medium">Available Times - {selectedDate.toLocaleDateString()}</h4>
                {availableSlots.filter(slot => slot.available).length === 0 ? (
                  <p className="text-muted-foreground text-sm">No available slots for this date</p>
                ) : (
                  availableSlots
                    .filter(slot => slot.available)
                    .map((slot) => (
                      <Button
                        key={slot.id}
                        variant={selectedSlot?.id === slot.id ? "default" : "outline"}
                        className="w-full justify-between"
                        onClick={() => setSelectedSlot(slot)}
                      >
                        <div className="flex items-center gap-2">
                          <Clock className="h-4 w-4" />
                          <span>{slot.time} ({slot.duration}min)</span>
                          {getLocationIcon(slot.location)}
                        </div>
                        <span>${slot.price}</span>
                      </Button>
                    ))
                )}
              </div>
            )}
          </CardContent>
        </Card>

        {/* Booking Details */}
        <Card>
          <CardHeader>
            <CardTitle>Session Details</CardTitle>
            <CardDescription>
              Configure your training session preferences
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div>
              <label className="text-sm font-medium">Session Type</label>
              <select
                value={bookingDetails.session_type}
                onChange={(e) => setBookingDetails(prev => ({ ...prev, session_type: e.target.value }))}
                className="w-full mt-1 p-2 border rounded-md"
              >
                {sessionTypes.map(type => (
                  <option key={type.id} value={type.id}>
                    {type.name} - {type.duration}min (${type.price})
                  </option>
                ))}
              </select>
            </div>

            <div>
              <label className="text-sm font-medium">Location Preference</label>
              <div className="mt-2 space-y-2">
                {locations.map(location => {
                  const IconComponent = location.icon;
                  return (
                    <label key={location.id} className="flex items-center space-x-2 cursor-pointer">
                      <input
                        type="radio"
                        name="location"
                        value={location.id}
                        checked={bookingDetails.location_preference === location.id}
                        onChange={(e) => setBookingDetails(prev => ({ ...prev, location_preference: e.target.value }))}
                      />
                      <IconComponent className="h-4 w-4" />
                      <div className="flex-1">
                        <div className="font-medium">{location.name}</div>
                        <div className="text-sm text-muted-foreground">{location.address}</div>
                      </div>
                    </label>
                  );
                })}
              </div>
            </div>

            {bookingDetails.location_preference === 'client_location' && (
              <div>
                <label className="text-sm font-medium">Custom Address</label>
                <Input
                  value={bookingDetails.custom_address}
                  onChange={(e) => setBookingDetails(prev => ({ ...prev, custom_address: e.target.value }))}
                  placeholder="Enter your gym or preferred location address"
                  className="mt-1"
                />
              </div>
            )}

            <div>
              <label className="text-sm font-medium">Session Notes</label>
              <Textarea
                value={bookingDetails.notes}
                onChange={(e) => setBookingDetails(prev => ({ ...prev, notes: e.target.value }))}
                placeholder="Any specific goals, concerns, or preferences for this session..."
                className="mt-1"
              />
            </div>

            <LoadingButton
              onClick={bookSession}
              loading={isBooking}
              disabled={!selectedSlot}
              className="w-full"
            >
              <CalendarIcon className="h-4 w-4 mr-2" />
              Book Session {selectedSlot && `($${selectedSlot.price})`}
            </LoadingButton>
          </CardContent>
        </Card>
      </div>

      {/* Booking History */}
      <Card>
        <CardHeader>
          <CardTitle>Upcoming & Past Sessions</CardTitle>
          <CardDescription>
            View your scheduled and completed training sessions
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {bookings.length === 0 ? (
            <div className="text-center py-8 text-muted-foreground">
              <CalendarIcon className="h-12 w-12 mx-auto mb-4 opacity-50" />
              <p>No sessions booked yet. Schedule your first session above!</p>
            </div>
          ) : (
            bookings.map((booking) => (
              <Card key={booking.id} className="border-l-4 border-l-primary">
                <CardContent className="pt-4">
                  <div className="flex items-start justify-between mb-2">
                    <div className="flex items-center gap-2">
                      <CalendarIcon className="h-4 w-4" />
                      <h4 className="font-semibold">{booking.session_type}</h4>
                    </div>
                    <div className="flex items-center gap-2">
                      {getStatusIcon(booking.status)}
                      <Badge variant="outline" className="capitalize">
                        {booking.status}
                      </Badge>
                    </div>
                  </div>
                  
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm text-muted-foreground mb-3">
                    <div className="flex items-center gap-1">
                      <CalendarIcon className="h-3 w-3" />
                      <span>{new Date(booking.date).toLocaleDateString()}</span>
                    </div>
                    <div className="flex items-center gap-1">
                      <Clock className="h-3 w-3" />
                      <span>{booking.time} ({booking.duration}min)</span>
                    </div>
                    <div className="flex items-center gap-1">
                      <MapPin className="h-3 w-3" />
                      <span>{booking.location}</span>
                    </div>
                    <div className="flex items-center gap-1">
                      <DollarSign className="h-3 w-3" />
                      <span>${booking.price}</span>
                    </div>
                  </div>

                  {booking.location_address && (
                    <p className="text-sm text-muted-foreground mb-2">
                      <MapPin className="h-3 w-3 inline mr-1" />
                      {booking.location_address}
                    </p>
                  )}

                  {booking.notes && (
                    <div className="bg-muted p-3 rounded-md mb-2">
                      <p className="text-sm"><strong>Your Notes:</strong> {booking.notes}</p>
                    </div>
                  )}

                  {booking.trainer_notes && (
                    <div className="bg-blue-50 p-3 rounded-md">
                      <p className="text-sm"><strong>Trainer Notes:</strong> {booking.trainer_notes}</p>
                    </div>
                  )}
                </CardContent>
              </Card>
            ))
          )}
        </CardContent>
      </Card>
    </div>
  );
};