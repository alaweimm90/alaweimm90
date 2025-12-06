import { Product } from '@/types/product';

/**
 * Comprehensive Product Catalog
 * Heritage, Performance, and Urban Collections
 * 8 Launch SKUs with complete product data
 */

export const products: Product[] = [
  // HERITAGE COLLECTION
  {
    id: 'heritage-tee-black',
    slug: 'heritage-black-tee',
    name: 'Heritage Black Tee',
    tagline: 'Cut with precision',
    description:
      'A classic crew neck tee crafted from premium Supima cotton. The Heritage Black Tee features precision-cut construction with reinforced shoulders and traditional side-seam construction for a timeless silhouette that works with any wardrobe.',
    category: 'tees',
    collection: 'heritage',
    price: 68,
    compareAtPrice: 85,
    currency: 'USD',
    colors: [
      {
        name: 'Midnight Black',
        hex: '#000000',
        images: [
          '/products/heritage-tee-black-front.jpg',
          '/products/heritage-tee-black-back.jpg',
          '/products/heritage-tee-black-detail.jpg',
        ],
      },
      {
        name: 'Cloud White',
        hex: '#FFFFFF',
        images: [
          '/products/heritage-tee-white-front.jpg',
          '/products/heritage-tee-white-back.jpg',
          '/products/heritage-tee-white-detail.jpg',
        ],
      },
    ],
    sizes: [
      { label: 'XS', inStock: true, measurements: { chest: 35, length: 26, sleeve: 32 } },
      { label: 'S', inStock: true, measurements: { chest: 37, length: 27, sleeve: 33 } },
      { label: 'M', inStock: true, measurements: { chest: 41, length: 28, sleeve: 34 } },
      { label: 'L', inStock: true, measurements: { chest: 45, length: 29, sleeve: 35 } },
      { label: 'XL', inStock: true, measurements: { chest: 49, length: 30, sleeve: 36 } },
      { label: 'XXL', inStock: false, measurements: { chest: 53, length: 31, sleeve: 37 } },
    ],
    images: [
      {
        id: 'img-1',
        url: '/products/heritage-tee-black-front.jpg',
        altText: 'Heritage Black Tee front view showcasing classic crew neck',
        position: 0,
      },
      {
        id: 'img-2',
        url: '/products/heritage-tee-black-back.jpg',
        altText: 'Heritage Black Tee back view demonstrating reinforced shoulders',
        position: 1,
      },
      {
        id: 'img-3',
        url: '/products/heritage-tee-black-detail.jpg',
        altText: 'Close-up detail of Heritage Black Tee side-seam construction',
        position: 2,
      },
      {
        id: 'img-4',
        url: '/products/heritage-tee-lifestyle.jpg',
        altText: 'Lifestyle shot of Heritage Black Tee worn by model in urban setting',
        position: 3,
      },
    ],
    materials: ['100% Supima Cotton', '220gsm weight', 'Pre-shrunk fabric'],
    features: ['Reinforced shoulders', 'Side-seam construction', 'Premium stitching', 'Crew neck collar'],
    careInstructions: [
      'Machine wash cold with like colors',
      'Turn inside out to preserve color',
      'Dry on low heat',
      'Do not bleach',
      'Avoid ironing directly on print',
    ],
    sku: 'LII-TEE-HERITAGE-001',
    inStock: true,
    isNewArrival: true,
    isFeatured: true,
    availableQuantity: 45,
    seoTitle: 'Heritage Black Tee - Premium Supima Cotton T-Shirt',
    seoDescription:
      'Classic crew neck Heritage Tee crafted from 100% Supima cotton with precision construction and reinforced shoulders for timeless style.',
    inventory: {
      quantity: 45,
      trackInventory: true,
    },
    createdAt: new Date('2024-11-01'),
    updatedAt: new Date('2024-11-01'),
  },

  {
    id: 'heritage-hoodie-charcoal',
    slug: 'heritage-crewneck-hoodie',
    name: 'Heritage Crewneck Hoodie',
    tagline: 'Elevated everyday essential',
    description:
      'An elevated everyday essential featuring a soft French Terry cotton/poly blend. The Heritage Hoodie combines comfort with refined design, featuring a hidden kangaroo pocket, reinforced hood, and beautifully ribbed cuffs for a sophisticated casual look.',
    category: 'hoodies',
    collection: 'heritage',
    price: 148,
    compareAtPrice: 185,
    currency: 'USD',
    colors: [
      {
        name: 'Charcoal',
        hex: '#36454F',
        images: [
          '/products/heritage-hoodie-charcoal-front.jpg',
          '/products/heritage-hoodie-charcoal-back.jpg',
          '/products/heritage-hoodie-charcoal-detail.jpg',
        ],
      },
      {
        name: 'Navy',
        hex: '#000080',
        images: [
          '/products/heritage-hoodie-navy-front.jpg',
          '/products/heritage-hoodie-navy-back.jpg',
          '/products/heritage-hoodie-navy-detail.jpg',
        ],
      },
      {
        name: 'Stone',
        hex: '#928E85',
        images: [
          '/products/heritage-hoodie-stone-front.jpg',
          '/products/heritage-hoodie-stone-back.jpg',
          '/products/heritage-hoodie-stone-detail.jpg',
        ],
      },
    ],
    sizes: [
      { label: 'S', inStock: true, measurements: { chest: 39, length: 28, sleeve: 33 } },
      { label: 'M', inStock: true, measurements: { chest: 43, length: 29, sleeve: 34 } },
      { label: 'L', inStock: true, measurements: { chest: 47, length: 30, sleeve: 35 } },
      { label: 'XL', inStock: true, measurements: { chest: 51, length: 31, sleeve: 36 } },
      { label: 'XXL', inStock: true, measurements: { chest: 55, length: 32, sleeve: 37 } },
    ],
    images: [
      {
        id: 'img-5',
        url: '/products/heritage-hoodie-charcoal-front.jpg',
        altText: 'Heritage Crewneck Hoodie front view in charcoal with visible kangaroo pocket',
        position: 0,
      },
      {
        id: 'img-6',
        url: '/products/heritage-hoodie-charcoal-back.jpg',
        altText: 'Heritage Crewneck Hoodie back view showcasing reinforced hood',
        position: 1,
      },
      {
        id: 'img-7',
        url: '/products/heritage-hoodie-charcoal-detail.jpg',
        altText: 'Close-up detail of Heritage Hoodie ribbed cuffs and hood construction',
        position: 2,
      },
      {
        id: 'img-8',
        url: '/products/heritage-hoodie-lifestyle.jpg',
        altText: 'Lifestyle shot of Heritage Crewneck Hoodie worn with complementary apparel',
        position: 3,
      },
    ],
    materials: ['80% Cotton / 20% Polyester', 'French Terry fabric', 'Pre-shrunk construction'],
    features: [
      'Hidden kangaroo pocket',
      'Reinforced hood with drawstrings',
      'Ribbed cuffs',
      'Ribbed hem',
      'Tonal embroidered branding',
    ],
    careInstructions: [
      'Machine wash warm with like colors',
      'Turn inside out before washing',
      'Tumble dry medium heat',
      'Avoid fabric softener',
      'Do not iron directly on design',
    ],
    sku: 'LII-HOOD-HERITAGE-001',
    inStock: true,
    isNewArrival: true,
    isFeatured: true,
    availableQuantity: 32,
    seoTitle: 'Heritage Crewneck Hoodie - Premium French Terry',
    seoDescription:
      'Premium crewneck hoodie crafted from soft French Terry blend with hidden kangaroo pocket and reinforced hood for everyday comfort.',
    inventory: {
      quantity: 32,
      trackInventory: true,
    },
    createdAt: new Date('2024-11-01'),
    updatedAt: new Date('2024-11-01'),
  },

  {
    id: 'heritage-cap-black',
    slug: 'heritage-6-panel-cap',
    name: 'Heritage 6-Panel Cap',
    tagline: 'Unstructured, low-profile style',
    description:
      'A versatile unstructured, low-profile 6-panel cap that works with any outfit. Crafted from premium garment-washed cotton twill with adjustable sizing and subtle branding, this essential accessory delivers timeless style with contemporary comfort.',
    category: 'caps',
    collection: 'heritage',
    price: 48,
    compareAtPrice: 60,
    currency: 'USD',
    colors: [
      {
        name: 'Black',
        hex: '#000000',
        images: [
          '/products/heritage-cap-black-front.jpg',
          '/products/heritage-cap-black-side.jpg',
          '/products/heritage-cap-black-back.jpg',
        ],
      },
      {
        name: 'Navy',
        hex: '#000080',
        images: [
          '/products/heritage-cap-navy-front.jpg',
          '/products/heritage-cap-navy-side.jpg',
          '/products/heritage-cap-navy-back.jpg',
        ],
      },
      {
        name: 'Olive',
        hex: '#808000',
        images: [
          '/products/heritage-cap-olive-front.jpg',
          '/products/heritage-cap-olive-side.jpg',
          '/products/heritage-cap-olive-back.jpg',
        ],
      },
    ],
    sizes: [{ label: 'ONE', inStock: true, measurements: { circumference: 56 } }],
    images: [
      {
        id: 'img-9',
        url: '/products/heritage-cap-black-front.jpg',
        altText: 'Heritage 6-Panel Cap front view in black with subtle branding',
        position: 0,
      },
      {
        id: 'img-10',
        url: '/products/heritage-cap-black-side.jpg',
        altText: 'Heritage 6-Panel Cap side profile showing unstructured design',
        position: 1,
      },
      {
        id: 'img-11',
        url: '/products/heritage-cap-black-back.jpg',
        altText: 'Heritage 6-Panel Cap back view with adjustable closure',
        position: 2,
      },
      {
        id: 'img-12',
        url: '/products/heritage-cap-lifestyle.jpg',
        altText: 'Lifestyle shot of Heritage Cap worn in casual setting',
        position: 3,
      },
    ],
    materials: ['Garment-washed cotton twill', '100% cotton construction', 'Natural fiber'],
    features: [
      '6-panel construction',
      'Unstructured crown',
      'Low-profile design',
      'Adjustable back closure',
      'Curved bill',
      'Embroidered branding',
    ],
    careInstructions: [
      'Hand wash cold water',
      'Gentle soap only',
      'Rinse thoroughly',
      'Air dry flat',
      'Do not bleach or use hot water',
    ],
    sku: 'LII-CAP-HERITAGE-001',
    inStock: true,
    isNewArrival: false,
    isFeatured: true,
    availableQuantity: 78,
    seoTitle: 'Heritage 6-Panel Cap - Premium Cotton Twill',
    seoDescription:
      'Unstructured, low-profile 6-panel cap crafted from garment-washed cotton twill with adjustable sizing for versatile everyday wear.',
    inventory: {
      quantity: 78,
      trackInventory: true,
    },
    createdAt: new Date('2024-11-01'),
    updatedAt: new Date('2024-11-01'),
  },

  // PERFORMANCE COLLECTION
  {
    id: 'performance-zip-hoodie',
    slug: 'performance-zip-hoodie',
    name: 'Performance Zip Hoodie',
    tagline: 'Technical fabric, full zip',
    description:
      'Engineered for active performance, this full-zip hoodie features moisture-wicking technical fabric that keeps you dry and comfortable. The Performance Zip Hoodie includes practical thumb loops, a media pocket for on-the-go convenience, and flatlock seams for chafe-free comfort during high-intensity activities.',
    category: 'hoodies',
    collection: 'performance',
    price: 168,
    compareAtPrice: 210,
    currency: 'USD',
    colors: [
      {
        name: 'Black',
        hex: '#000000',
        images: [
          '/products/performance-zip-black-front.jpg',
          '/products/performance-zip-black-back.jpg',
          '/products/performance-zip-black-detail.jpg',
        ],
      },
      {
        name: 'Slate Gray',
        hex: '#708090',
        images: [
          '/products/performance-zip-gray-front.jpg',
          '/products/performance-zip-gray-back.jpg',
          '/products/performance-zip-gray-detail.jpg',
        ],
      },
    ],
    sizes: [
      { label: 'XS', inStock: true, measurements: { chest: 38, length: 28, sleeve: 32 } },
      { label: 'S', inStock: true, measurements: { chest: 42, length: 29, sleeve: 33 } },
      { label: 'M', inStock: true, measurements: { chest: 46, length: 30, sleeve: 34 } },
      { label: 'L', inStock: true, measurements: { chest: 50, length: 31, sleeve: 35 } },
      { label: 'XL', inStock: false, measurements: { chest: 54, length: 32, sleeve: 36 } },
      { label: 'XXL', inStock: false, measurements: { chest: 58, length: 33, sleeve: 37 } },
    ],
    images: [
      {
        id: 'img-13',
        url: '/products/performance-zip-black-front.jpg',
        altText: 'Performance Zip Hoodie front view in black showcasing full zipper',
        position: 0,
      },
      {
        id: 'img-14',
        url: '/products/performance-zip-black-back.jpg',
        altText: 'Performance Zip Hoodie back view highlighting tech fabric',
        position: 1,
      },
      {
        id: 'img-15',
        url: '/products/performance-zip-black-detail.jpg',
        altText: 'Close-up detail of Performance Zip Hoodie thumb loops and media pocket',
        position: 2,
      },
      {
        id: 'img-16',
        url: '/products/performance-zip-lifestyle.jpg',
        altText: 'Lifestyle shot of Performance Zip Hoodie worn during active training',
        position: 3,
      },
    ],
    materials: ['88% Polyester / 12% Spandex', 'Moisture-wicking technical blend', 'Four-way stretch'],
    features: [
      'Full front zipper',
      'Thumb loops',
      'Hidden media pocket',
      'Flatlock seams',
      'Reflective accents',
      'Moisture-wicking lining',
    ],
    careInstructions: [
      'Machine wash cold water only',
      'Use mesh washing bag',
      'Air dry flat',
      'Do not use fabric softener',
      'Do not iron synthetic materials',
    ],
    sku: 'LII-HOOD-PERF-001',
    inStock: true,
    isNewArrival: true,
    isFeatured: true,
    availableQuantity: 28,
    seoTitle: 'Performance Zip Hoodie - Moisture-Wicking Technical Fabric',
    seoDescription:
      'Full-zip performance hoodie with moisture-wicking technical fabric, thumb loops, and media pocket designed for active athletes.',
    inventory: {
      quantity: 28,
      trackInventory: true,
    },
    createdAt: new Date('2024-11-01'),
    updatedAt: new Date('2024-11-01'),
  },

  {
    id: 'performance-tee',
    slug: 'performance-tee',
    name: 'Performance Tee',
    tagline: 'Athletic fit, seamless construction',
    description:
      'Engineered for peak performance, this tee features seamless construction and athletic fit tailoring. The Performance Tee incorporates laser-cut ventilation channels and anti-odor treatment, making it perfect for intense workouts and active pursuits while maintaining style.',
    category: 'tees',
    collection: 'performance',
    price: 78,
    compareAtPrice: 98,
    currency: 'USD',
    colors: [
      {
        name: 'Black',
        hex: '#000000',
        images: [
          '/products/performance-tee-black-front.jpg',
          '/products/performance-tee-black-back.jpg',
          '/products/performance-tee-black-detail.jpg',
        ],
      },
      {
        name: 'White',
        hex: '#FFFFFF',
        images: [
          '/products/performance-tee-white-front.jpg',
          '/products/performance-tee-white-back.jpg',
          '/products/performance-tee-white-detail.jpg',
        ],
      },
      {
        name: 'Ash Gray',
        hex: '#B2BEB5',
        images: [
          '/products/performance-tee-ash-front.jpg',
          '/products/performance-tee-ash-back.jpg',
          '/products/performance-tee-ash-detail.jpg',
        ],
      },
    ],
    sizes: [
      { label: 'XS', inStock: true, measurements: { chest: 36, length: 26, sleeve: 31 } },
      { label: 'S', inStock: true, measurements: { chest: 40, length: 27, sleeve: 32 } },
      { label: 'M', inStock: true, measurements: { chest: 44, length: 28, sleeve: 33 } },
      { label: 'L', inStock: true, measurements: { chest: 48, length: 29, sleeve: 34 } },
      { label: 'XL', inStock: true, measurements: { chest: 52, length: 30, sleeve: 35 } },
      { label: 'XXL', inStock: false, measurements: { chest: 56, length: 31, sleeve: 36 } },
    ],
    images: [
      {
        id: 'img-17',
        url: '/products/performance-tee-black-front.jpg',
        altText: 'Performance Tee front view in black with athletic cut',
        position: 0,
      },
      {
        id: 'img-18',
        url: '/products/performance-tee-black-back.jpg',
        altText: 'Performance Tee back view showing seamless construction',
        position: 1,
      },
      {
        id: 'img-19',
        url: '/products/performance-tee-black-detail.jpg',
        altText: 'Close-up detail of Performance Tee laser-cut ventilation',
        position: 2,
      },
      {
        id: 'img-20',
        url: '/products/performance-tee-lifestyle.jpg',
        altText: 'Lifestyle shot of Performance Tee worn during high-intensity workout',
        position: 3,
      },
    ],
    materials: ['90% Polyester / 10% Spandex', 'Technical mesh blend', 'Four-way stretch'],
    features: [
      'Seamless construction',
      'Laser-cut ventilation',
      'Anti-odor treatment',
      'Athletic fit',
      'Moisture-wicking',
      'Tagless neck',
    ],
    careInstructions: [
      'Machine wash cold with like colors',
      'Use mesh bag for delicate cycle',
      'Tumble dry low or air dry',
      'Do not bleach',
      'Do not use fabric softener',
    ],
    sku: 'LII-TEE-PERF-001',
    inStock: true,
    isNewArrival: true,
    isFeatured: false,
    availableQuantity: 52,
    seoTitle: 'Performance Tee - Seamless Athletic T-Shirt',
    seoDescription:
      'Seamless performance tee with laser-cut ventilation and anti-odor treatment for intense workouts and active training.',
    inventory: {
      quantity: 52,
      trackInventory: true,
    },
    createdAt: new Date('2024-11-01'),
    updatedAt: new Date('2024-11-01'),
  },

  {
    id: 'performance-track-pants',
    slug: 'performance-track-pants',
    name: 'Performance Track Pants',
    tagline: 'Tapered fit, zippered hems',
    description:
      'Technical track pants designed for performance and style. Features tapered fit construction with zippered hems for versatile styling, 4-way stretch fabric for full range of motion, and practical design elements including a secure zip pocket and adjustable waistband for all-day comfort.',
    category: 'accessories',
    collection: 'performance',
    price: 128,
    compareAtPrice: 160,
    currency: 'USD',
    colors: [
      {
        name: 'Black',
        hex: '#000000',
        images: [
          '/products/performance-pants-black-front.jpg',
          '/products/performance-pants-black-back.jpg',
          '/products/performance-pants-black-detail.jpg',
        ],
      },
      {
        name: 'Navy',
        hex: '#000080',
        images: [
          '/products/performance-pants-navy-front.jpg',
          '/products/performance-pants-navy-back.jpg',
          '/products/performance-pants-navy-detail.jpg',
        ],
      },
    ],
    sizes: [
      { label: 'XS', inStock: true, measurements: { chest: 30, length: 40, sleeve: 0 } },
      { label: 'S', inStock: true, measurements: { chest: 32, length: 41, sleeve: 0 } },
      { label: 'M', inStock: true, measurements: { chest: 34, length: 42, sleeve: 0 } },
      { label: 'L', inStock: true, measurements: { chest: 36, length: 43, sleeve: 0 } },
      { label: 'XL', inStock: true, measurements: { chest: 38, length: 44, sleeve: 0 } },
      { label: 'XXL', inStock: false, measurements: { chest: 40, length: 45, sleeve: 0 } },
    ],
    images: [
      {
        id: 'img-21',
        url: '/products/performance-pants-black-front.jpg',
        altText: 'Performance Track Pants front view in black with tapered cut',
        position: 0,
      },
      {
        id: 'img-22',
        url: '/products/performance-pants-black-back.jpg',
        altText: 'Performance Track Pants back view showing tapered design',
        position: 1,
      },
      {
        id: 'img-23',
        url: '/products/performance-pants-black-detail.jpg',
        altText: 'Close-up detail of Performance Pants zippered hem and pocket',
        position: 2,
      },
      {
        id: 'img-24',
        url: '/products/performance-pants-lifestyle.jpg',
        altText: 'Lifestyle shot of Performance Track Pants worn casually',
        position: 3,
      },
    ],
    materials: ['88% Polyester / 12% Spandex', '4-way stretch fabric', 'Technical blend'],
    features: [
      'Tapered fit',
      'Zippered hems',
      'Secure zip pocket',
      'Adjustable waistband',
      '4-way stretch',
      'Flat front design',
    ],
    careInstructions: [
      'Machine wash cold with like colors',
      'Turn inside out before washing',
      'Tumble dry low heat',
      'Do not bleach',
      'Avoid fabric softener',
    ],
    sku: 'LII-PANTS-PERF-001',
    inStock: true,
    isNewArrival: true,
    isFeatured: false,
    availableQuantity: 35,
    seoTitle: 'Performance Track Pants - 4-Way Stretch',
    seoDescription:
      'Technical track pants with tapered fit, zippered hems, and 4-way stretch fabric for movement and contemporary style.',
    inventory: {
      quantity: 35,
      trackInventory: true,
    },
    createdAt: new Date('2024-11-01'),
    updatedAt: new Date('2024-11-01'),
  },

  // URBAN COLLECTION
  {
    id: 'urban-bomber-jacket',
    slug: 'urban-bomber-jacket',
    name: 'Urban Bomber Jacket',
    tagline: 'Modern silhouette, quilted lining',
    description:
      'An essential modern outerwear piece featuring a contemporary bomber silhouette. The Urban Bomber Jacket combines water-resistant nylon with a plush quilted lining for warmth without bulk. The refined design includes a distinctive ribbed collar, cuffed sleeves, and interior pockets for functionality.',
    category: 'outerwear',
    collection: 'urban',
    price: 298,
    compareAtPrice: 375,
    currency: 'USD',
    colors: [
      {
        name: 'Black',
        hex: '#000000',
        images: [
          '/products/urban-bomber-black-front.jpg',
          '/products/urban-bomber-black-back.jpg',
          '/products/urban-bomber-black-detail.jpg',
        ],
      },
      {
        name: 'Olive',
        hex: '#808000',
        images: [
          '/products/urban-bomber-olive-front.jpg',
          '/products/urban-bomber-olive-back.jpg',
          '/products/urban-bomber-olive-detail.jpg',
        ],
      },
    ],
    sizes: [
      { label: 'XS', inStock: true, measurements: { chest: 38, length: 26, sleeve: 32 } },
      { label: 'S', inStock: true, measurements: { chest: 42, length: 27, sleeve: 33 } },
      { label: 'M', inStock: true, measurements: { chest: 46, length: 28, sleeve: 34 } },
      { label: 'L', inStock: true, measurements: { chest: 50, length: 29, sleeve: 35 } },
      { label: 'XL', inStock: true, measurements: { chest: 54, length: 30, sleeve: 36 } },
      { label: 'XXL', inStock: false, measurements: { chest: 58, length: 31, sleeve: 37 } },
    ],
    images: [
      {
        id: 'img-25',
        url: '/products/urban-bomber-black-front.jpg',
        altText: 'Urban Bomber Jacket front view in black showcasing modern silhouette',
        position: 0,
      },
      {
        id: 'img-26',
        url: '/products/urban-bomber-black-back.jpg',
        altText: 'Urban Bomber Jacket back view highlighting quilted construction',
        position: 1,
      },
      {
        id: 'img-27',
        url: '/products/urban-bomber-black-detail.jpg',
        altText: 'Close-up detail of Urban Bomber ribbed collar and cuffs',
        position: 2,
      },
      {
        id: 'img-28',
        url: '/products/urban-bomber-lifestyle.jpg',
        altText: 'Lifestyle shot of Urban Bomber Jacket worn in contemporary style',
        position: 3,
      },
    ],
    materials: ['100% water-resistant nylon shell', 'Quilted nylon lining', 'Polyester blend'],
    features: [
      'Water-resistant shell',
      'Quilted lining',
      'Ribbed collar',
      'Ribbed cuffs',
      'Interior pockets',
      'Full-length zipper',
      'Modern bomber cut',
    ],
    careInstructions: [
      'Spot clean exterior only',
      'Use damp cloth for spot cleaning',
      'Do not machine wash',
      'Air dry thoroughly',
      'Professional dry cleaning recommended',
    ],
    sku: 'LII-JACKET-URBAN-001',
    inStock: true,
    isNewArrival: true,
    isFeatured: true,
    availableQuantity: 18,
    seoTitle: 'Urban Bomber Jacket - Water-Resistant Quilted',
    seoDescription:
      'Contemporary bomber jacket with water-resistant nylon and quilted lining, featuring modern design and functional style.',
    inventory: {
      quantity: 18,
      trackInventory: true,
    },
    createdAt: new Date('2024-11-01'),
    updatedAt: new Date('2024-11-01'),
  },

  {
    id: 'urban-tote-bag',
    slug: 'urban-tote-bag',
    name: 'Urban Tote Bag',
    tagline: 'Structured carry-all in premium materials',
    description:
      'A versatile structured tote bag designed for everyday essentials and professional use. Crafted from durable ballistic nylon with refined leather accents, the Urban Tote features secure compartments, reinforced handles, and a timeless aesthetic that complements both casual and formal styles.',
    category: 'accessories',
    collection: 'urban',
    price: 88,
    compareAtPrice: 110,
    currency: 'USD',
    colors: [
      {
        name: 'Black',
        hex: '#000000',
        images: [
          '/products/urban-tote-black-front.jpg',
          '/products/urban-tote-black-side.jpg',
          '/products/urban-tote-black-detail.jpg',
        ],
      },
      {
        name: 'Sand',
        hex: '#C2B280',
        images: [
          '/products/urban-tote-sand-front.jpg',
          '/products/urban-tote-sand-side.jpg',
          '/products/urban-tote-sand-detail.jpg',
        ],
      },
    ],
    sizes: [{ label: 'ONE', inStock: true, measurements: { chest: 42, length: 32 } }],
    images: [
      {
        id: 'img-29',
        url: '/products/urban-tote-black-front.jpg',
        altText: 'Urban Tote Bag front view in black showing structured design',
        position: 0,
      },
      {
        id: 'img-30',
        url: '/products/urban-tote-black-side.jpg',
        altText: 'Urban Tote Bag side profile with leather handle accents',
        position: 1,
      },
      {
        id: 'img-31',
        url: '/products/urban-tote-black-detail.jpg',
        altText: 'Close-up detail of Urban Tote leather handle and compartments',
        position: 2,
      },
      {
        id: 'img-32',
        url: '/products/urban-tote-lifestyle.jpg',
        altText: 'Lifestyle shot of Urban Tote Bag carried in professional setting',
        position: 3,
      },
    ],
    materials: ['Ballistic nylon exterior', 'Leather handles', 'Reinforced seams', 'Nylon lining'],
    features: [
      'Structured construction',
      'Leather handles',
      'Multiple compartments',
      'Reinforced base',
      'Interior pockets',
      'Top zipper closure',
      'Metal hardware',
    ],
    careInstructions: [
      'Wipe clean with damp cloth',
      'Remove stains promptly',
      'Condition leather quarterly',
      'Air dry naturally',
      'Do not machine wash',
    ],
    sku: 'LII-TOTE-URBAN-001',
    inStock: true,
    isNewArrival: false,
    isFeatured: true,
    availableQuantity: 64,
    seoTitle: 'Urban Tote Bag - Ballistic Nylon with Leather',
    seoDescription:
      'Structured tote bag in durable ballistic nylon with leather accents, ideal for everyday essentials and professional use.',
    inventory: {
      quantity: 64,
      trackInventory: true,
    },
    createdAt: new Date('2024-11-01'),
    updatedAt: new Date('2024-11-01'),
  },
];

/**
 * Helper function to get products by collection
 */
export const getProductsByCollection = (collection: string) => {
  return products.filter(p => p.collection === collection);
};

/**
 * Helper function to get products by category
 */
export const getProductsByCategory = (category: string) => {
  return products.filter(p => p.category === category);
};

/**
 * Helper function to get featured products
 */
export const getFeaturedProducts = () => {
  return products.filter(p => p.isFeatured).slice(0, 6);
};

/**
 * Helper function to get new arrivals
 */
export const getNewArrivals = () => {
  return products.filter(p => p.isNewArrival).slice(0, 6);
};

/**
 * Helper function to get product by slug
 */
export const getProductBySlug = (slug: string) => {
  return products.find(p => p.slug === slug) || null;
};

/**
 * Helper function to get related products
 */
export const getRelatedProducts = (productId: string, limit: number = 4) => {
  const product = products.find(p => p.id === productId);
  if (!product) return [];

  return products
    .filter(
      p =>
        p.id !== productId &&
        (p.collection === product.collection || p.category === product.category)
    )
    .slice(0, limit);
};

export default products;
