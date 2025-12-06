import React from 'react';
import { useNavigate } from 'react-router-dom';
import { Button } from '@/ui/atoms/Button';
import { BrandLogo } from '@/components/shared/BrandLogo';
import { Phone, Mail, MapPin } from 'lucide-react';

export const Footer: React.FC = () => {
  const navigate = useNavigate();

  const socialLinks = [
    {
      name: 'Instagram',
      url: 'https://instagram.com/repzcoach',
      icon: (
        <svg className="w-8 h-8 md:w-10 md:h-10" viewBox="0 0 24 24" fill="currentColor">
          <path d="M12 2.163c3.204 0 3.584.012 4.85.07 3.252.148 4.771 1.691 4.919 4.919.058 1.265.069 1.645.069 4.849 0 3.205-.012 3.584-.069 4.849-.149 3.225-1.664 4.771-4.919 4.919-1.266.058-1.644.07-4.85.07-3.204 0-3.584-.012-4.849-.07-3.26-.149-4.771-1.699-4.919-4.92-.058-1.265-.07-1.644-.07-4.849 0-3.204.013-3.583.07-4.849.149-3.227 1.664-4.771 4.919-4.919 1.266-.057 1.645-.069 4.849-.069zm0-2.163c-3.259 0-3.667.014-4.947.072-4.358.2-6.78 2.618-6.98 6.98-.059 1.281-.073 1.689-.073 4.948 0 3.259.014 3.668.072 4.948.2 4.358 2.618 6.78 6.98 6.98 1.281.058 1.689.072 4.948.072 3.259 0 3.668-.014 4.948-.072 4.354-.2 6.782-2.618 6.979-6.98.059-1.28.073-1.689.073-4.948 0-3.259-.014-3.667-.072-4.947-.196-4.354-2.617-6.78-6.979-6.98-1.281-.059-1.69-.073-4.949-.073zm0 5.838c-3.403 0-6.162 2.759-6.162 6.162s2.759 6.163 6.162 6.163 6.162-2.759 6.162-6.163c0-3.403-2.759-6.162-6.162-6.162zm0 10.162c-2.209 0-4-1.79-4-4 0-2.209 1.791-4 4-4s4 1.791 4 4c0 2.21-1.791 4-4 4zm6.406-11.845c-.796 0-1.441.645-1.441 1.44s.645 1.44 1.441 1.44c.795 0 1.439-.645 1.439-1.44s-.644-1.44-1.439-1.40z"/>
        </svg>
      ),
      color: 'text-pink-500 hover:text-pink-400'
    },
    {
      name: 'Telegram',
      url: 'https://t.me/repzcoach',
      icon: (
        <svg className="w-8 h-8 md:w-10 md:h-10" viewBox="0 0 24 24" fill="currentColor">
          <path d="M11.944 0A12 12 0 0 0 0 12a12 12 0 0 0 12 12 12 12 0 0 0 12-12A12 12 0 0 0 12 0a12 12 0 0 0-.056 0zm4.962 7.224c.1-.002.321.023.465.14a.506.506 0 0 1 .171.325c.016.093.036.306.02.472-.18 1.898-.962 6.502-1.36 8.627-.168.9-.499 1.201-.82 1.23-.696.065-1.225-.46-1.9-.902-1.056-.693-1.653-1.124-2.678-1.8-1.185-.78-.417-1.21.258-1.91.177-.184 3.247-2.977 3.307-3.23.007-.032.014-.15-.056-.212s-.174-.041-.249-.024c-.106.024-1.793 1.14-5.061 3.345-.48.33-.913.49-1.302.48-.428-.008-1.252-.241-1.865-.44-.752-.245-1.349-.374-1.297-.789.027-.216.325-.437.893-.663 3.498-1.524 5.83-2.529 6.998-3.014 3.332-1.386 4.025-1.627 4.476-1.635z"/>
        </svg>
      ),
      color: 'text-blue-500 hover:text-blue-400'
    },
    {
      name: 'WhatsApp',
      url: 'https://wa.me/message/JKZKE5JEYB4MB1',
      icon: (
        <svg className="w-8 h-8 md:w-10 md:h-10" viewBox="0 0 24 24" fill="currentColor">
          <path d="M17.472 14.382c-.297-.149-1.758-.867-2.03-.967-.273-.099-.471-.148-.67.15-.197.297-.767.966-.94 1.164-.173.199-.347.223-.644.075-.297-.15-1.255-.463-2.39-1.475-.883-.788-1.48-1.761-1.653-2.059-.173-.297-.018-.458.13-.606.134-.133.298-.347.446-.52.149-.174.198-.298.298-.497.099-.198.05-.371-.025-.52-.075-.149-.669-1.612-.916-2.207-.242-.579-.487-.5-.669-.51-.173-.008-.371-.01-.57-.01-.198 0-.52.074-.792.372-.272.297-1.04 1.016-1.04 2.479 0 1.462 1.065 2.875 1.213 3.074.149.198 2.096 3.2 5.077 4.487.709.306 1.262.489 1.694.625.712.227 1.36.195 1.871.118.571-.085 1.758-.719 2.006-1.413.248-.694.248-1.289.173-1.413-.074-.124-.272-.198-.57-.347m-5.421 7.403h-.004a9.87 9.87 0 01-5.031-1.378l-.361-.214-3.741.982.998-3.648-.235-.374a9.86 9.86 0 01-1.51-5.26c.001-5.45 4.436-9.884 9.888-9.884 2.64 0 5.122 1.03 6.988 2.898a9.825 9.825 0 012.893 6.994c-.003 5.45-4.437 9.884-9.885 9.884m8.413-18.297A11.815 11.815 0 0012.05 0C5.495 0 .16 5.335.157 11.892c0 2.096.547 4.142 1.588 5.945L.057 24l6.305-1.654a11.882 11.882 0 005.683 1.448h.005c6.554 0 11.89-5.335 11.893-11.893A11.821 11.821 0 0020.885 3.488"/>
        </svg>
      ),
      color: 'text-green-500 hover:text-green-400'
    }
  ];

  return (
    <footer className="relative overflow-hidden bg-gradient-to-br from-black via-gray-950 to-slate-900">
      {/* Elegant animated background elements */}
      <div className="absolute inset-0 pointer-events-none">
        <div className="absolute top-16 left-24 w-80 h-80 bg-gradient-to-br from-repz-orange/15 via-amber-500/10 to-yellow-500/8 rounded-full blur-3xl animate-pulse"></div>
        <div className="absolute bottom-8 right-32 w-64 h-64 bg-gradient-to-br from-blue-500/12 via-cyan-500/8 to-teal-500/6 rounded-full blur-2xl animate-pulse" style={{animationDelay: '2s'}}></div>
        <div className="absolute top-1/2 left-1/2 w-56 h-56 bg-gradient-to-br from-purple-500/10 via-pink-500/7 to-repz-orange/5 rounded-full blur-3xl animate-pulse" style={{animationDelay: '4s'}}></div>
        
        {/* Geometric patterns */}
        <div className="absolute top-12 right-20 w-24 h-24 border border-repz-orange/15 rotate-45 animate-pulse" style={{animationDuration: '6s'}}></div>
        <div className="absolute bottom-16 left-16 w-20 h-20 border border-blue-400/10 rounded-full animate-ping" style={{animationDelay: '1.5s', animationDuration: '4s'}}></div>
        
        {/* Floating elements */}
        <div className="absolute top-8 right-12 w-32 h-32 bg-gradient-to-br from-repz-orange/6 to-amber-500/3 rounded-full animate-float backdrop-blur-sm"></div>
        <div className="absolute bottom-12 left-28 w-28 h-28 bg-gradient-to-br from-blue-500/5 to-cyan-400/2 rounded-full animate-pulse backdrop-blur-sm" style={{animationDelay: '1s'}}></div>
      </div>

      <div className="container mx-auto px-4 sm:px-6 lg:px-8 max-w-7xl py-12 relative z-10">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
          {/* Brand Section */}
          <div className="text-center md:border-r-4 md:border-repz-orange/30 md:pr-6">
            <div className="flex items-center justify-center gap-4 mb-4">
              <BrandLogo size="xl" />
            </div>
            <p className="text-white/90 mb-4 text-sm md:text-base">
              Science-based fitness coaching for optimal performance. Transform your fitness journey 
              with data-driven protocols and personalized optimization.
            </p>
          </div>
          
          {/* Social Media Section */}
          <div className="text-center md:border-r-4 md:border-repz-orange/30 md:pr-6">
            <h4 className="font-montserrat font-bold mb-6 text-repz-orange text-lg">Connect With Us</h4>
            {/* Mobile: Single line layout */}
            <div className="flex md:hidden justify-center items-center gap-4 mb-4">
              {socialLinks.map((social) => (
                <Button
                  key={social.name}
                  variant="ghost"
                  size="sm"
                  onClick={() => window.open(social.url, '_blank')}
                  aria-label={`Follow us on ${social.name}`}
                  className={`p-2 transition-colors ${social.color}`}
                >
                  {social.icon}
                </Button>
              ))}
            </div>
            {/* Desktop: Vertical layout */}
            <div className="hidden md:flex flex-col items-center gap-4">
              {socialLinks.map((social) => (
                <Button
                  key={social.name}
                  variant="ghost"
                  size="lg"
                  onClick={() => window.open(social.url, '_blank')}
                  aria-label={`Follow us on ${social.name}`}
                  className={`p-3 transition-colors ${social.color}`}
                >
                  {social.icon}
                </Button>
              ))}
            </div>
          </div>
          
          {/* Contact Section - Centered content */}
          <div className="text-center md:border-r-4 md:border-repz-orange/30 md:pr-6">
            <h4 className="font-montserrat font-bold mb-6 text-repz-orange text-lg">Contact</h4>
            <div className="space-y-4 flex flex-col items-center">
              <div className="text-white/90 text-sm">Mon-Fri 6AM-8PM PST</div>
              <div className="flex items-center justify-center text-white/90 text-sm">
                <Phone className="w-4 h-4 mr-2" />
                <a href="sms:+14159929792" className="hover:text-repz-orange transition-colors">
                  +1 (415) 992-9792
                </a>
              </div>
              <div className="flex items-center justify-center text-white/90 text-sm">
                <Mail className="w-4 h-4 mr-2" />
                <a href="mailto:contact@repz.com" className="hover:text-repz-orange transition-colors">
                  contact@repz.com
                </a>
              </div>
              <div className="flex items-center justify-center text-white/90 text-sm">
                <MapPin className="w-4 h-4 mr-2" />
                <span>Bay Area</span>
              </div>
            </div>
          </div>
          
          {/* Legal & Support Section */}
          <div className="text-center">
            <h4 className="font-montserrat font-bold mb-6 text-repz-orange text-lg">Legal & Support</h4>
            <div className="space-y-3 text-sm flex flex-col items-center">
              <button 
                onClick={() => navigate('/terms-of-service')} 
                className="block text-white/90 hover:text-repz-orange transition-colors"
              >
                Terms of Service
              </button>
              <button 
                onClick={() => navigate('/privacy-policy')} 
                className="block text-white/90 hover:text-repz-orange transition-colors"
              >
                Privacy Policy
              </button>
              <button 
                onClick={() => navigate('/liability-waiver')} 
                className="block text-white/90 hover:text-repz-orange transition-colors"
              >
                Liability Waiver
              </button>
              <button 
                onClick={() => navigate('/health-disclaimer')} 
                className="block text-white/90 hover:text-repz-orange transition-colors"
              >
                Health Disclaimer
              </button>
            </div>
          </div>
        </div>
        
        <div className="mt-12 pt-8 text-center border-t border-repz-orange/20">
          <p className="text-repz-orange font-bold text-sm md:text-base">
            &copy; 2025 REPZ Coaching Platform. All rights reserved.
          </p>
        </div>
      </div>
      
      {/* Bottom Boundary Line */}
      <div className="w-full h-1 bg-gradient-to-r from-transparent via-repz-orange/40 to-transparent"></div>
    </footer>
  );
};