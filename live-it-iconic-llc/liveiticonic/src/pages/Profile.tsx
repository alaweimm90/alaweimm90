import React from 'react';
import { useQuery } from '@tanstack/react-query';
import { useAuth } from '@/contexts/AuthContext';
import { orderService } from '@/services/orderService';
import SEO from '@/components/SEO';
import Navigation from '@/components/Navigation';
import Footer from '@/components/Footer';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Separator } from '@/components/ui/separator';
import OrderHistory from '@/components/OrderHistory';
import AddressBook from '@/components/AddressBook';
import { User, LogOut, Package, MapPin } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import { Skeleton } from '@/components/ui/skeleton';

const Profile: React.FC = () => {
  const { user, signOut, isLoading: authLoading, isAuthenticated } = useAuth();
  const navigate = useNavigate();

  const { data: orders, isLoading: ordersLoading } = useQuery({
    queryKey: ['customerOrders', user?.id],
    queryFn: () => orderService.getCustomerOrders(user!.id),
    enabled: !!user?.id,
  });

  const handleSignOut = async () => {
    await signOut();
    navigate('/');
  };

  if (authLoading) {
    return (
      <>
        <SEO title="My Account - Live It Iconic" description="Your account dashboard" />
        <div className="min-h-screen bg-lii-bg">
          <Navigation />
          <div className="container mx-auto px-6 py-12 pt-32">
            <Skeleton className="h-8 w-64 mb-8" />
            <Skeleton className="h-96 w-full" />
          </div>
          <Footer />
        </div>
      </>
    );
  }

  if (!isAuthenticated || !user) {
    return (
      <>
        <SEO title="Sign In - Live It Iconic" description="Sign in to your account" />
        <div className="min-h-screen bg-lii-bg">
          <Navigation />
          <div className="container mx-auto px-6 py-12 pt-32 text-center">
            <h1 className="text-2xl font-display text-lii-cloud mb-4">Please Sign In</h1>
            <p className="text-lii-ash font-ui mb-6">
              You need to be signed in to view your profile.
            </p>
            <Button onClick={() => navigate('/')} variant="primary">
              Go to Home
            </Button>
          </div>
          <Footer />
        </div>
      </>
    );
  }

  return (
    <>
      <SEO
        title="My Account - Live It Iconic"
        description="Your account dashboard and order history"
      />
      <div className="min-h-screen bg-lii-bg">
        <Navigation />
        <main className="pt-32 pb-16">
          <div className="container mx-auto px-6">
            {/* Header */}
            <div className="flex items-center justify-between mb-8">
              <div>
                <h1 className="text-3xl font-display font-semibold text-lii-cloud">My Account</h1>
                <p className="text-lii-ash font-ui text-sm mt-1">{user.email}</p>
              </div>
              <Button variant="outline" onClick={handleSignOut}>
                <LogOut className="w-4 h-4 mr-2" />
                Sign Out
              </Button>
            </div>

            <div className="grid lg:grid-cols-3 gap-8">
              {/* Main Content */}
              <div className="lg:col-span-2 space-y-8">
                {/* Profile Information */}
                <Card className="bg-lii-ink border-lii-gold/10">
                  <CardHeader>
                    <CardTitle className="flex items-center gap-3 text-lii-cloud">
                      <User className="w-5 h-5 text-lii-gold" />
                      Profile Information
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-4">
                      <div>
                        <p className="text-lii-ash font-ui text-sm mb-1">Email</p>
                        <p className="text-lii-cloud font-ui">{user.email}</p>
                      </div>
                      {user.name && (
                        <div>
                          <p className="text-lii-ash font-ui text-sm mb-1">Name</p>
                          <p className="text-lii-cloud font-ui">{user.name}</p>
                        </div>
                      )}
                      {user.createdAt && (
                        <div>
                          <p className="text-lii-ash font-ui text-sm mb-1">Member Since</p>
                          <p className="text-lii-cloud font-ui">
                            {new Date(user.createdAt).toLocaleDateString()}
                          </p>
                        </div>
                      )}
                    </div>
                  </CardContent>
                </Card>

                {/* Order History */}
                <Card className="bg-lii-ink border-lii-gold/10">
                  <CardHeader>
                    <CardTitle className="flex items-center gap-3 text-lii-cloud">
                      <Package className="w-5 h-5 text-lii-gold" />
                      Order History
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    {ordersLoading ? (
                      <div className="space-y-4">
                        <Skeleton className="h-24 w-full" />
                        <Skeleton className="h-24 w-full" />
                      </div>
                    ) : (
                      <OrderHistory orders={orders || []} />
                    )}
                  </CardContent>
                </Card>
              </div>

              {/* Sidebar */}
              <div className="space-y-6">
                {/* Saved Addresses */}
                <Card className="bg-lii-ink border-lii-gold/10">
                  <CardHeader>
                    <CardTitle className="flex items-center gap-3 text-lii-cloud">
                      <MapPin className="w-5 h-5 text-lii-gold" />
                      Saved Addresses
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <AddressBook userId={user.id} />
                  </CardContent>
                </Card>
              </div>
            </div>
          </div>
        </main>
        <Footer />
      </div>
    </>
  );
};

export default Profile;
