import React, { useState } from 'react';
import { Link, useLocation } from 'react-router-dom';
import {
  ShoppingCartIcon,
  UserIcon,
  HeartIcon,
  SearchIcon,
  MenuIcon,
  XMarkIcon
} from '@heroicons/react/24/outline';
import { HeartIcon as HeartSolidIcon } from '@heroicons/react/24/solid';

const FashionHeader: React.FC = () => {
  const location = useLocation();
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const [isSearchOpen, setIsSearchOpen] = useState(false);

  // Mock cart count - would come from context/store
  const cartItemCount = 0;
  const wishlistItemCount = 0;

  const navigation = [
    { name: 'Home', href: '/' },
    { name: 'Collections', href: '/collections' },
    { name: 'New Arrivals', href: '/new-arrivals' },
    { name: 'Lookbook', href: '/lookbook' },
    { name: 'About', href: '/about' },
    { name: 'Contact', href: '/contact' }
  ];

  const handleSearch = (query: string) => {
    // TODO: Implement search functionality
    console.log('Searching for:', query);
  };

  return (
    <>
      <header className="bg-white border-b border-gray-100 sticky top-0 z-40">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            {/* Logo */}
            <div className="flex-shrink-0">
              <Link to="/" className="flex items-center">
                <h1 className="text-3xl font-heading text-fashion-pink">
                  Rounaq
                </h1>
              </Link>
            </div>

            {/* Desktop Navigation */}
            <nav className="hidden lg:flex space-x-8">
              {navigation.map((item) => (
                <Link
                  key={item.name}
                  to={item.href}
                  className={`text-sm font-medium transition-colors duration-200 focus-ring ${
                    location.pathname === item.href
                      ? 'text-fashion-purple'
                      : 'text-gray-600 hover:text-fashion-purple'
                  }`}
                >
                  {item.name}
                </Link>
              ))}
            </nav>

            {/* Desktop Actions */}
            <div className="hidden lg:flex items-center space-x-4">
              <button
                onClick={() => setIsSearchOpen(true)}
                className="p-2 text-gray-600 hover:text-fashion-purple transition-colors duration-200 focus-ring"
                aria-label="Search"
              >
                <SearchIcon className="w-5 h-5" />
              </button>

              <Link
                to="/wishlist"
                className="p-2 text-gray-600 hover:text-fashion-purple transition-colors duration-200 focus-ring relative"
                aria-label="Wishlist"
              >
                {wishlistItemCount > 0 ? (
                  <HeartSolidIcon className="w-5 h-5 text-fashion-pink" />
                ) : (
                  <HeartIcon className="w-5 h-5" />
                )}
                {wishlistItemCount > 0 && (
                  <span className="absolute -top-1 -right-1 bg-fashion-pink text-white text-xs rounded-full w-5 h-5 flex items-center justify-center">
                    {wishlistItemCount}
                  </span>
                )}
              </Link>

              <Link
                to="/account"
                className="p-2 text-gray-600 hover:text-fashion-purple transition-colors duration-200 focus-ring"
                aria-label="Account"
              >
                <UserIcon className="w-5 h-5" />
              </Link>

              <Link
                to="/cart"
                className="p-2 text-gray-600 hover:text-fashion-purple transition-colors duration-200 focus-ring relative"
                aria-label="Shopping Cart"
              >
                <ShoppingCartIcon className="w-5 h-5" />
                {cartItemCount > 0 && (
                  <span className="absolute -top-1 -right-1 bg-fashion-pink text-white text-xs rounded-full w-5 h-5 flex items-center justify-center">
                    {cartItemCount}
                  </span>
                )}
              </Link>
            </div>

            {/* Mobile menu button */}
            <div className="lg:hidden">
              <button
                onClick={() => setIsMenuOpen(!isMenuOpen)}
                className="p-2 rounded-md text-gray-600 hover:text-fashion-purple hover:bg-gray-50 focus-ring"
                aria-label="Toggle menu"
              >
                {isMenuOpen ? (
                  <XMarkIcon className="w-6 h-6" />
                ) : (
                  <MenuIcon className="w-6 h-6" />
                )}
              </button>
            </div>
          </div>
        </div>

        {/* Mobile Navigation */}
        {isMenuOpen && (
          <div className="lg:hidden border-t border-gray-100">
            <div className="px-2 pt-2 pb-3 space-y-1">
              {navigation.map((item) => (
                <Link
                  key={item.name}
                  to={item.href}
                  className={`block px-3 py-2 rounded-md text-base font-medium transition-colors duration-200 focus-ring ${
                    location.pathname === item.href
                      ? 'text-fashion-purple bg-purple-50'
                      : 'text-gray-600 hover:text-fashion-purple hover:bg-gray-50'
                  }`}
                  onClick={() => setIsMenuOpen(false)}
                >
                  {item.name}
                </Link>
              ))}

              <div className="pt-4 pb-3 border-t border-gray-100">
                <div className="px-3 space-y-2">
                  <button
                    onClick={() => {
                      setIsSearchOpen(true);
                      setIsMenuOpen(false);
                    }}
                    className="w-full flex items-center justify-center gap-2 bg-gray-100 text-gray-700 px-4 py-2 rounded-md text-sm font-medium hover:bg-gray-200 transition-colors duration-200 focus-ring"
                  >
                    <SearchIcon className="w-4 h-4" />
                    Search
                  </button>

                  <Link
                    to="/wishlist"
                    className="w-full flex items-center justify-center gap-2 bg-gray-100 text-gray-700 px-4 py-2 rounded-md text-sm font-medium hover:bg-gray-200 transition-colors duration-200 focus-ring"
                    onClick={() => setIsMenuOpen(false)}
                  >
                    <HeartIcon className="w-4 h-4" />
                    Wishlist {wishlistItemCount > 0 && `(${wishlistItemCount})`}
                  </Link>

                  <Link
                    to="/account"
                    className="w-full flex items-center justify-center gap-2 bg-gray-100 text-gray-700 px-4 py-2 rounded-md text-sm font-medium hover:bg-gray-200 transition-colors duration-200 focus-ring"
                    onClick={() => setIsMenuOpen(false)}
                  >
                    <UserIcon className="w-4 h-4" />
                    Account
                  </Link>

                  <Link
                    to="/cart"
                    className="w-full flex items-center justify-center gap-2 bg-fashion-pink text-white px-4 py-2 rounded-md text-sm font-medium hover:bg-opacity-90 transition-colors duration-200 focus-ring"
                    onClick={() => setIsMenuOpen(false)}
                  >
                    <ShoppingCartIcon className="w-4 h-4" />
                    Cart {cartItemCount > 0 && `(${cartItemCount})`}
                  </Link>
                </div>
              </div>
            </div>
          </div>
        )}
      </header>

      {/* Search Overlay */}
      {isSearchOpen && (
        <div className="fixed inset-0 z-50 bg-black bg-opacity-50">
          <div className="bg-white">
            <div className="max-w-3xl mx-auto p-4">
              <div className="flex items-center gap-4">
                <SearchIcon className="w-5 h-5 text-gray-400" />
                <input
                  type="text"
                  placeholder="Search for products, collections, or styles..."
                  className="flex-1 px-4 py-3 border-b border-gray-200 focus:outline-none focus:border-fashion-purple text-lg"
                  autoFocus
                  onKeyDown={(e) => {
                    if (e.key === 'Enter') {
                      handleSearch(e.currentTarget.value);
                      setIsSearchOpen(false);
                    }
                  }}
                />
                <button
                  onClick={() => setIsSearchOpen(false)}
                  className="p-2 text-gray-600 hover:text-fashion-purple focus-ring"
                >
                  <XMarkIcon className="w-6 h-6" />
                </button>
              </div>

              {/* Search suggestions could go here */}
              <div className="mt-4 text-sm text-gray-600">
                Popular searches: "summer dresses", "handbags", "accessories", "new arrivals"
              </div>
            </div>
          </div>
        </div>
      )}
    </>
  );
};

export default FashionHeader;
