import { useEffect } from 'react';
import { useLocation } from 'react-router-dom';

const NotFound = () => {
  const location = useLocation();

  useEffect(() => {
    // Track 404 errors for analytics
  }, [location.pathname]);

  return (
    <div className="flex min-h-screen items-center justify-center bg-gray-100">
      <main>
        <div className="text-center">
          <h1 className="mb-4 text-4xl font-bold">404</h1>
          <p className="mb-4 text-xl text-gray-600">Oops! Page not found</p>
          <a href="/" className="text-blue-500 underline hover:text-blue-700">
            Return to Home
          </a>
        </div>
      </main>
    </div>
  );
};

export default NotFound;
