/**
 * Live It Iconic - Premium Lifestyle Merchandise
 * Main Application Component
 */
import { Toaster as Sonner } from '@/components/ui/sonner';
import { Toaster } from '@/components/ui/toaster';
import { TooltipProvider } from '@/components/ui/tooltip';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { HelmetProvider } from 'react-helmet-async';
import { Suspense, lazy } from 'react';
import { BrowserRouter, Route, Routes } from 'react-router-dom';
import { CartProvider } from '@/contexts/CartContext';
import { AuthProvider } from '@/contexts/AuthContext';
import { CurrencyProvider } from '@/contexts/CurrencyContext';
import PWAInstallPrompt from '@/components/PWAInstallPrompt';
import { useAnalytics } from '@/hooks/useAnalytics';
import '@/i18n/config';

// Lazy load pages for code splitting
const Index = lazy(() => import('./pages/Index.tsx'));
const About = lazy(() => import('./pages/About.tsx'));
const Lifestyle = lazy(() => import('./pages/Lifestyle.tsx'));
const Collection = lazy(() => import('./pages/Collection.tsx'));
const Shop = lazy(() => import('./pages/Shop.tsx'));
const ProductDetail = lazy(() => import('./pages/ProductDetail.tsx'));
const BrandAssets = lazy(() => import('./pages/BrandAssets.tsx'));
const BrandShowcase = lazy(() => import('./pages/BrandShowcase.tsx'));
const Launch = lazy(() => import('./pages/Launch.tsx'));
const Contact = lazy(() => import('./pages/Contact.tsx'));
const Policies = lazy(() => import('./pages/Policies.tsx'));
const Templates = lazy(() => import('./pages/Templates.tsx'));
const Checkout = lazy(() => import('./pages/Checkout.tsx'));
const CheckoutSuccess = lazy(() => import('./pages/CheckoutSuccess.tsx'));
const OrderDetails = lazy(() => import('./pages/OrderDetails.tsx'));
const Profile = lazy(() => import('./pages/Profile.tsx'));
const AdminDashboard = lazy(() => import('./pages/admin/Dashboard.tsx'));
const AdminOrders = lazy(() => import('./pages/admin/Orders.tsx'));
const AdminProducts = lazy(() => import('./pages/admin/Products.tsx'));
const AdminCustomers = lazy(() => import('./pages/admin/Customers.tsx'));
const AdminAnalytics = lazy(() => import('./pages/admin/Analytics.tsx'));
const AdminSettings = lazy(() => import('./pages/admin/Settings.tsx'));
const AdminMonitoring = lazy(() => import('./pages/admin/Monitoring.tsx'));
const NotFound = lazy(() => import('./pages/NotFound.tsx'));

// Admin layout
import { AdminLayout } from '@/components/admin/Layout/AdminLayout';

// Import modal component (not lazy loaded for better UX)
import EmailCaptureModal from '@/components/EmailCaptureModal';

// Loading fallback component
const PageLoader = () => (
  <div className="min-h-screen bg-lii-bg">
    <div className="container mx-auto px-6 pt-32">
      {/* Header skeleton */}
      <div className="max-w-4xl mx-auto mb-16">
        <div className="h-12 bg-lii-charcoal/20 rounded-lg mb-4 animate-pulse"></div>
        <div className="h-6 bg-lii-charcoal/20 rounded mb-2 animate-pulse"></div>
        <div className="h-6 bg-lii-charcoal/20 rounded w-3/4 animate-pulse"></div>
      </div>

      {/* Content skeleton */}
      <div className="grid md:grid-cols-2 gap-12 max-w-6xl mx-auto">
        <div className="space-y-6">
          <div className="h-8 bg-lii-charcoal/20 rounded animate-pulse"></div>
          <div className="h-4 bg-lii-charcoal/20 rounded animate-pulse"></div>
          <div className="h-4 bg-lii-charcoal/20 rounded animate-pulse"></div>
          <div className="h-4 bg-lii-charcoal/20 rounded w-5/6 animate-pulse"></div>
        </div>
        <div className="aspect-[4/5] bg-lii-charcoal/20 rounded-2xl animate-pulse"></div>
      </div>
    </div>
  </div>
);

const queryClient = new QueryClient();

// Analytics wrapper component that must be inside Router context
const AnalyticsWrapper = ({ children }: { children: React.ReactNode }) => {
  useAnalytics(); // Track page views globally
  return <>{children}</>;
};

const App = () => {
  return (
    <HelmetProvider>
      <QueryClientProvider client={queryClient}>
        <TooltipProvider>
          <AuthProvider>
            <CurrencyProvider>
              <CartProvider>
                <Toaster />
                <Sonner />
                <PWAInstallPrompt />
                <EmailCaptureModal />
                <BrowserRouter>
                  <AnalyticsWrapper>
                    <Suspense fallback={<PageLoader />}>
                    <Routes>
                      <Route path="/" element={<Index />} />
                      <Route path="/about" element={<About />} />
                      <Route path="/lifestyle" element={<Lifestyle />} />
                      <Route path="/collection" element={<Collection />} />
                      <Route path="/shop" element={<Shop />} />
                      <Route path="/product/:productId" element={<ProductDetail />} />
                      <Route path="/brand" element={<BrandAssets />} />
                      <Route path="/brand-showcase" element={<BrandShowcase />} />
                      <Route path="/launch" element={<Launch />} />
                      <Route path="/contact" element={<Contact />} />
                      <Route path="/policies" element={<Policies />} />
                      <Route path="/templates" element={<Templates />} />
                      <Route path="/checkout" element={<Checkout />} />
                      <Route path="/checkout/success" element={<CheckoutSuccess />} />
                      <Route path="/orders/:orderId" element={<OrderDetails />} />
                      <Route path="/profile" element={<Profile />} />

                      {/* Admin Routes with Layout */}
                      <Route path="/admin" element={<AdminLayout />}>
                        <Route index element={<AdminDashboard />} />
                        <Route path="products" element={<AdminProducts />} />
                        <Route path="orders" element={<AdminOrders />} />
                        <Route path="customers" element={<AdminCustomers />} />
                        <Route path="analytics" element={<AdminAnalytics />} />
                        <Route path="monitoring" element={<AdminMonitoring />} />
                        <Route path="settings" element={<AdminSettings />} />
                      </Route>

                      <Route path="*" element={<NotFound />} />
                    </Routes>
                    </Suspense>
                  </AnalyticsWrapper>
                </BrowserRouter>
              </CartProvider>
            </CurrencyProvider>
          </AuthProvider>
        </TooltipProvider>
      </QueryClientProvider>
    </HelmetProvider>
  );
};

export default App;
