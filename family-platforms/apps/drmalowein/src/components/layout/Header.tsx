import React, { useState } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { Menu, X, Download, Search } from 'lucide-react';

const Header: React.FC = () => {
  const location = useLocation();
  const [isMenuOpen, setIsMenuOpen] = useState(false);

  const navigation = [
    { name: 'Home', href: '/' },
    { name: 'Research', href: '/research' },
    { name: 'Publications', href: '/publications' },
    { name: 'Teaching', href: '/teaching' },
    { name: 'About', href: '/about' },
    { name: 'Contact', href: '/contact' }
  ];

  const handleDownloadCV = () => {
    // TODO: Implement CV download functionality
    window.open('/cv/DrMAlowein_CV.pdf', '_blank');
  };

  const handleSearch = () => {
    // TODO: Implement search functionality
    console.log('Search clicked');
  };

  return (
    <header className="bg-white border-b border-gray-200 sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          {/* Logo */}
          <div className="flex-shrink-0">
            <Link to="/" className="flex items-center">
              <h1 className="text-2xl font-heading text-academic-blue">
                DrMAlowein
              </h1>
            </Link>
          </div>

          {/* Desktop Navigation */}
          <nav className="hidden md:flex space-x-8">
            {navigation.map((item) => (
              <Link
                key={item.name}
                to={item.href}
                className={`text-sm font-medium transition-colors focus-ring ${
                  location.pathname === item.href
                    ? 'text-academic-blue'
                    : 'text-gray-600 hover:text-academic-blue'
                }`}
              >
                {item.name}
              </Link>
            ))}
          </nav>

          {/* Desktop Actions */}
          <div className="hidden md:flex items-center space-x-4">
            <button
              onClick={handleSearch}
              className="p-2 text-gray-600 hover:text-academic-blue transition-colors focus-ring"
              aria-label="Search"
            >
              <Search className="w-5 h-5" />
            </button>
            <button
              onClick={handleDownloadCV}
              className="bg-academic-blue text-white px-4 py-2 rounded-md text-sm font-medium hover:bg-opacity-90 transition-colors focus-ring flex items-center gap-2"
            >
              <Download className="w-4 h-4" />
              Download CV
            </button>
          </div>

          {/* Mobile menu button */}
          <div className="md:hidden">
            <button
              onClick={() => setIsMenuOpen(!isMenuOpen)}
              className="p-2 rounded-md text-gray-600 hover:text-academic-blue hover:bg-gray-100 focus-ring"
              aria-label="Toggle menu"
            >
              {isMenuOpen ? (
                <X className="w-6 h-6" />
              ) : (
                <Menu className="w-6 h-6" />
              )}
            </button>
          </div>
        </div>

        {/* Mobile Navigation */}
        {isMenuOpen && (
          <div className="md:hidden">
            <div className="px-2 pt-2 pb-3 space-y-1 sm:px-3">
              {navigation.map((item) => (
                <Link
                  key={item.name}
                  to={item.href}
                  className={`block px-3 py-2 rounded-md text-base font-medium transition-colors focus-ring ${
                    location.pathname === item.href
                      ? 'text-academic-blue bg-blue-50'
                      : 'text-gray-600 hover:text-academic-blue hover:bg-gray-50'
                  }`}
                  onClick={() => setIsMenuOpen(false)}
                >
                  {item.name}
                </Link>
              ))}
              <div className="pt-4 pb-3 border-t border-gray-200">
                <div className="flex items-center px-3 space-x-3">
                  <button
                    onClick={handleSearch}
                    className="flex-1 bg-gray-100 text-gray-700 px-4 py-2 rounded-md text-sm font-medium hover:bg-gray-200 transition-colors focus-ring flex items-center justify-center gap-2"
                  >
                    <Search className="w-4 h-4" />
                    Search
                  </button>
                  <button
                    onClick={handleDownloadCV}
                    className="flex-1 bg-academic-blue text-white px-4 py-2 rounded-md text-sm font-medium hover:bg-opacity-90 transition-colors focus-ring flex items-center justify-center gap-2"
                  >
                    <Download className="w-4 h-4" />
                    CV
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </header>
  );
};

export default Header;
