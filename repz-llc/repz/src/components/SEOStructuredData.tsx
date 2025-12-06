import { useEffect } from 'react';

interface SEOStructuredDataProps {
  page?: 'home' | 'dashboard' | 'sessions' | 'progress';
}

const SEOStructuredData = ({ page = 'home' }: SEOStructuredDataProps) => {
  useEffect(() => {
    // Remove any existing structured data
    const existingScript = document.querySelector('script[type="application/ld+json"]');
    if (existingScript) {
      existingScript.remove();
    }

    let structuredData;

    if (page === 'home') {
      // Local Business + Service Schema for homepage
      structuredData = {
        "@context": "https://schema.org",
        "@graph": [
          {
            "@type": "LocalBusiness",
            "@id": "https://repzcoach.com/#business",
            "name": "REPZ Coach Pro",
            "image": "https://repzcoach.com/logo.png",
            "description": "Premium personal training and fitness coaching in San Francisco Bay Area. Achieve your fitness goals with personalized programs, nutrition guidance, and expert support.",
            "url": "https://repzcoach.com",
            "telephone": "+1-415-555-0123",
            "email": "info@repzcoach.com",
            "address": {
              "@type": "PostalAddress",
              "streetAddress": "123 Fitness Street",
              "addressLocality": "San Francisco",
              "addressRegion": "CA",
              "postalCode": "94102",
              "addressCountry": "US"
            },
            "geo": {
              "@type": "GeoCoordinates",
              "latitude": "37.7749",
              "longitude": "-122.4194"
            },
            "areaServed": [
              {
                "@type": "City",
                "name": "San Francisco"
              },
              {
                "@type": "City", 
                "name": "Oakland"
              },
              {
                "@type": "City",
                "name": "San Jose"
              },
              {
                "@type": "City",
                "name": "Palo Alto"
              }
            ],
            "openingHoursSpecification": [
              {
                "@type": "OpeningHoursSpecification",
                "dayOfWeek": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
                "opens": "06:00",
                "closes": "21:00"
              },
              {
                "@type": "OpeningHoursSpecification",
                "dayOfWeek": ["Saturday", "Sunday"],
                "opens": "08:00",
                "closes": "18:00"
              }
            ],
            "priceRange": "$$",
            "aggregateRating": {
              "@type": "AggregateRating",
              "ratingValue": "4.9",
              "reviewCount": "127",
              "bestRating": "5",
              "worstRating": "1"
            }
          },
          {
            "@type": "Service",
            "@id": "https://repzcoach.com/#service",
            "name": "Personal Training Services",
            "description": "Comprehensive personal training and fitness coaching services including strength training, weight loss, muscle building, and nutrition coaching.",
            "provider": {
              "@id": "https://repzcoach.com/#business"
            },
            "areaServed": {
              "@type": "State",
              "name": "California"
            },
            "hasOfferCatalog": {
              "@type": "OfferCatalog",
              "name": "Training Programs",
              "itemListElement": [
                {
                  "@type": "Offer",
                  "itemOffered": {
                    "@type": "Service",
                    "name": "Starter Tier - Foundation Building",
                    "description": "Perfect for fitness beginners and those returning to exercise"
                  },
                  "price": "149",
                  "priceCurrency": "USD",
                  "priceSpecification": {
                    "@type": "UnitPriceSpecification",
                    "price": "149",
                    "priceCurrency": "USD",
                    "unitText": "monthly"
                  }
                },
                {
                  "@type": "Offer",
                  "itemOffered": {
                    "@type": "Service",
                    "name": "Growth Tier - Accelerated Progress",
                    "description": "For intermediate fitness enthusiasts ready to level up"
                  },
                  "price": "299",
                  "priceCurrency": "USD",
                  "priceSpecification": {
                    "@type": "UnitPriceSpecification",
                    "price": "299",
                    "priceCurrency": "USD",
                    "unitText": "monthly"
                  }
                },
                {
                  "@type": "Offer",
                  "itemOffered": {
                    "@type": "Service",
                    "name": "Elite Tier - Peak Performance",
                    "description": "For serious athletes and advanced fitness enthusiasts"
                  },
                  "price": "499",
                  "priceCurrency": "USD",
                  "priceSpecification": {
                    "@type": "UnitPriceSpecification",
                    "price": "499",
                    "priceCurrency": "USD",
                    "unitText": "monthly"
                  }
                }
              ]
            }
          },
          {
            "@type": "Person",
            "@id": "https://repzcoach.com/#trainer",
            "name": "Coach Michael",
            "jobTitle": "Elite Personal Trainer & Nutrition Coach",
            "description": "NASM-CPT certified personal trainer with 8+ years of experience helping clients achieve their fitness goals in the San Francisco Bay Area.",
            "worksFor": {
              "@id": "https://repzcoach.com/#business"
            },
            "knowsAbout": [
              "Strength Training",
              "Weight Loss",
              "Muscle Building", 
              "Nutrition Coaching",
              "Athletic Performance"
            ],
            "hasCredential": [
              "NASM-CPT",
              "Precision Nutrition Level 1",
              "FMS Level 2"
            ]
          },
          {
            "@type": "WebSite",
            "@id": "https://repzcoach.com/#website",
            "url": "https://repzcoach.com",
            "name": "REPZ Coach Pro",
            "publisher": {
              "@id": "https://repzcoach.com/#business"
            },
            "potentialAction": {
              "@type": "SearchAction",
              "target": "https://repzcoach.com/search?q={search_term_string}",
              "query-input": "required name=search_term_string"
            }
          }
        ]
      };
    }

    if (structuredData) {
      const script = document.createElement('script');
      script.type = 'application/ld+json';
      script.textContent = JSON.stringify(structuredData);
      document.head.appendChild(script);
    }

    return () => {
      const scriptToRemove = document.querySelector('script[type="application/ld+json"]');
      if (scriptToRemove) {
        scriptToRemove.remove();
      }
    };
  }, [page]);

  return null;
};

export default SEOStructuredData;