import React, { useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { productService } from '@/services/productService';
import { useCart } from '@/contexts/CartContext';
import { Product } from '@/types/product';
import SEO from '@/components/SEO';
import Navigation from '@/components/Navigation';
import Footer from '@/components/Footer';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import { Card, CardContent } from '@/components/ui/card';
import InventoryBadge from '@/components/InventoryBadge';
import { generateProductStructuredData, generateBreadcrumbs } from '@/utils/seo';
import { ShoppingCart, Heart, ArrowLeft, Minus, Plus } from 'lucide-react';
import { Skeleton } from '@/components/ui/skeleton';
import { useToast } from '@/hooks/use-toast';
import { trackEvents } from '@/utils/analytics';
import { useAnalytics } from '@/hooks/useAnalytics';
import { useRecentlyViewed } from '@/hooks/useLocalStorage';
import ProductRecommendations from '@/components/ProductRecommendations';
import { useAuth } from '@/contexts/AuthContext';
import SizeGuide from '@/components/product/SizeGuide';
import ProductReviews from '@/components/product/ProductReviews';

const ProductDetail: React.FC = () => {
  const { productId } = useParams<{ productId: string }>();
  const navigate = useNavigate();
  const { addItem } = useCart();
  const { toast } = useToast();
  const { user } = useAuth();
  useAnalytics(); // Track page views
  const { addToRecentlyViewed } = useRecentlyViewed(user?.id);
  const [selectedImageIndex, setSelectedImageIndex] = useState(0);
  const [quantity, setQuantity] = useState(1);
  const [selectedVariant, setSelectedVariant] = useState<string | null>(null);

  const { data: product, isLoading } = useQuery<Product | null>({
    queryKey: ['product', productId],
    queryFn: () => productService.getProduct(productId!),
    enabled: !!productId,
  });

  // Track product view
  React.useEffect(() => {
    if (product) {
      trackEvents.viewProduct(product.id, product.name, product.price);
      addToRecentlyViewed(product.id);
    }
  }, [product, addToRecentlyViewed]);

  const handleAddToCart = () => {
    if (!product) return;

    if (product.inventory.trackInventory && product.inventory.quantity === 0) {
      toast({
        title: 'Out of Stock',
        description: 'This product is currently out of stock.',
      });
      return;
    }

    for (let i = 0; i < quantity; i++) {
      addItem({
        id: product.id,
        name: product.name,
        price: product.price,
        image: product.images[0]?.url || '',
        variant: selectedVariant || product.variants?.[0]?.name,
      });
    }

    toast({
      title: 'Added to Cart',
      description: `${quantity} ${product.name} added to your cart.`,
    });

    // Track analytics
    trackEvents.addToCart(product.id, quantity, product.price, product.name);
  };

  if (isLoading) {
    return (
      <>
        <SEO title="Loading Product - Live It Iconic" description="Loading product details" />
        <div className="min-h-screen bg-lii-bg">
          <Navigation />
          <div className="container mx-auto px-6 py-12 pt-32">
            <Skeleton className="h-8 w-64 mb-8" />
            <div className="grid lg:grid-cols-2 gap-8">
              <Skeleton className="aspect-square" />
              <Skeleton className="h-96" />
            </div>
          </div>
          <Footer />
        </div>
      </>
    );
  }

  if (!product) {
    return (
      <>
        <SEO title="Product Not Found - Live It Iconic" description="Product not found" />
        <div className="min-h-screen bg-lii-bg">
          <Navigation />
          <div className="container mx-auto px-6 py-12 pt-32 text-center">
            <h1 className="text-2xl font-display text-lii-cloud mb-4">Product Not Found</h1>
            <Button onClick={() => navigate('/shop')} variant="primary">
              Continue Shopping
            </Button>
          </div>
          <Footer />
        </div>
      </>
    );
  }

  const currentImage = product.images[selectedImageIndex] || product.images[0];
  const discount = product.compareAtPrice
    ? Math.round(((product.compareAtPrice - product.price) / product.compareAtPrice) * 100)
    : null;

  const breadcrumbs = generateBreadcrumbs([
    { name: 'Home', url: '/' },
    { name: 'Shop', url: '/shop' },
    { name: product.category, url: `/shop?category=${product.category}` },
    { name: product.name, url: `/product/${product.id}` },
  ]);

  const productSchema = generateProductStructuredData(product);

  return (
    <>
      <SEO
        title={`${product.name} - Premium Apparel`}
        description={product.description || `Premium ${product.category.toLowerCase()} with precision-cut design inspired by automotive discipline. Shop now at Live It Iconic.`}
        canonical={`/product/${product.id}`}
        ogType="product"
        ogImage={currentImage?.url}
        ogImageAlt={currentImage?.altText || product.name}
        keywords={`${product.name}, ${product.category}, premium apparel, luxury clothing, automotive fashion, ${product.name.toLowerCase()}`}
        structuredData={productSchema}
      />
      <script type="application/ld+json">{JSON.stringify(breadcrumbs)}</script>
      <div className="min-h-screen bg-lii-bg">
        <Navigation />
        <main className="pt-32 pb-16">
          <div className="container mx-auto px-6">
            {/* Breadcrumb */}
            <div className="flex items-center gap-2 mb-8 text-sm">
              <button
                onClick={() => navigate('/shop')}
                className="text-lii-ash hover:text-lii-gold font-ui"
              >
                Shop
              </button>
              <span className="text-lii-ash">/</span>
              <span className="text-lii-ash">{product.category}</span>
              <span className="text-lii-ash">/</span>
              <span className="text-lii-cloud">{product.name}</span>
            </div>

            {/* Back Button */}
            <Button
              variant="ghost"
              size="icon"
              onClick={() => navigate(-1)}
              className="mb-6 text-lii-ash hover:text-lii-cloud"
            >
              <ArrowLeft className="w-5 h-5" />
            </Button>

            <div className="grid lg:grid-cols-2 gap-12">
              {/* Product Images */}
              <div className="space-y-4">
                <div className="relative aspect-square overflow-hidden rounded-lg bg-gradient-to-br from-lii-charcoal/20 to-lii-ink/40">
                  {currentImage && (
                    <img
                      src={currentImage.url}
                      alt={currentImage.altText || product.name}
                      className="w-full h-full object-cover"
                    />
                  )}
                  {discount && (
                    <div className="absolute top-4 right-4 bg-lii-gold text-lii-black px-3 py-1 rounded-full text-sm font-display font-semibold">
                      {discount}% OFF
                    </div>
                  )}
                  <div className="absolute top-4 left-4">
                    <InventoryBadge
                      quantity={product.inventory.quantity}
                      trackInventory={product.inventory.trackInventory}
                    />
                  </div>
                </div>

                {/* Thumbnail Images */}
                {product.images.length > 1 && (
                  <div className="grid grid-cols-4 gap-4">
                    {product.images.map((image, index) => (
                      <button
                        key={image.id}
                        onClick={() => setSelectedImageIndex(index)}
                        className={`aspect-square overflow-hidden rounded-lg border-2 transition-all ${
                          index === selectedImageIndex
                            ? 'border-lii-gold'
                            : 'border-lii-gold/20 hover:border-lii-gold/40'
                        }`}
                      >
                        <img
                          src={image.url}
                          alt={image.altText || `${product.name} - View ${index + 1}`}
                          className="w-full h-full object-cover"
                        />
                      </button>
                    ))}
                  </div>
                )}
              </div>

              {/* Product Info */}
              <div className="space-y-6">
                <div>
                  <p className="text-lii-gold font-ui text-sm uppercase tracking-wider mb-2">
                    {product.category}
                  </p>
                  <h1 className="text-4xl font-display font-light text-lii-cloud mb-4">
                    {product.name}
                  </h1>
                  <p className="text-lii-ash font-ui leading-relaxed">{product.description}</p>
                </div>

                {/* Price */}
                <div className="flex items-center gap-4">
                  <span className="text-3xl font-display font-semibold text-lii-gold">
                    ${product.price.toFixed(2)}
                  </span>
                  {product.compareAtPrice && (
                    <span className="text-xl text-lii-ash line-through">
                      ${product.compareAtPrice.toFixed(2)}
                    </span>
                  )}
                </div>

                {/* Variants */}
                {product.variants && product.variants.length > 0 && (
                  <div className="space-y-4">
                    <p className="text-lii-cloud font-ui font-medium">Select Variant</p>
                    <div className="flex flex-wrap gap-2">
                      {product.variants.map(variant => (
                        <button
                          key={variant.id}
                          onClick={() => setSelectedVariant(variant.id)}
                          className={`px-4 py-2 rounded-lg border transition-all ${
                            selectedVariant === variant.id
                              ? 'border-lii-gold bg-lii-gold/10 text-lii-gold'
                              : 'border-lii-gold/20 text-lii-cloud hover:border-lii-gold/40'
                          }`}
                        >
                          {variant.name}
                        </button>
                      ))}
                    </div>
                  </div>
                )}

                {/* Size Guide */}
                <SizeGuide category={product.category} />

                <Separator className="bg-lii-gold/10" />

                {/* Quantity & Add to Cart */}
                <div className="space-y-4">
                  <div className="flex items-center gap-4">
                    <p className="text-lii-cloud font-ui font-medium">Quantity</p>
                    <div className="flex items-center gap-2">
                      <Button
                        variant="outline"
                        size="icon"
                        onClick={() => setQuantity(Math.max(1, quantity - 1))}
                        disabled={quantity <= 1}
                      >
                        <Minus className="w-4 h-4" />
                      </Button>
                      <span className="w-12 text-center text-lii-cloud font-ui">{quantity}</span>
                      <Button
                        variant="outline"
                        size="icon"
                        onClick={() => setQuantity(quantity + 1)}
                        disabled={
                          product.inventory.trackInventory && quantity >= product.inventory.quantity
                        }
                      >
                        <Plus className="w-4 h-4" />
                      </Button>
                    </div>
                  </div>

                  <Button
                    variant="primary"
                    size="lg"
                    onClick={handleAddToCart}
                    disabled={product.inventory.trackInventory && product.inventory.quantity === 0}
                    className="w-full font-ui font-medium"
                  >
                    <ShoppingCart className="w-5 h-5 mr-2" />
                    {product.inventory.trackInventory && product.inventory.quantity === 0
                      ? 'Out of Stock'
                      : 'Add to Cart'}
                  </Button>
                </div>

                {/* Product Details */}
                <Card className="bg-lii-ink border-lii-gold/10">
                  <CardContent className="p-6 space-y-4">
                    <div>
                      <p className="text-lii-cloud font-ui font-medium mb-2">SKU</p>
                      <p className="text-lii-ash font-ui font-mono text-sm">{product.sku}</p>
                    </div>
                    {product.metadata && Object.keys(product.metadata).length > 0 && (
                      <div>
                        <p className="text-lii-cloud font-ui font-medium mb-2">Details</p>
                        <div className="space-y-1">
                          {Object.entries(product.metadata).map(([key, value]) => (
                            <p key={key} className="text-lii-ash font-ui text-sm">
                              <span className="capitalize">{key}:</span> {String(value)}
                            </p>
                          ))}
                        </div>
                      </div>
                    )}
                  </CardContent>
                </Card>
              </div>
            </div>

            {/* Reviews */}
            <div className="mt-16">
              <ProductReviews productId={product.id} />
            </div>

            {/* Recommendations */}
            <div className="mt-16">
              <ProductRecommendations
                productId={product.id}
                userId={user?.id}
                type="frequently-bought"
              />
            </div>
          </div>
        </main>
        <Footer />
      </div>
    </>
  );
};

export default ProductDetail;
