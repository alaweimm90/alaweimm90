import { useEffect, useState } from 'react';
import 'swagger-ui-react/swagger-ui.css';
import { swaggerSpec } from '@/api/openapi';

/**
 * API Documentation Page
 *
 * Displays interactive Swagger/OpenAPI documentation for all API endpoints.
 * Allows users to explore and test API endpoints directly from the browser.
 *
 * @component
 * @example
 * <ApiDocs />
 *
 * @returns {JSX.Element} API documentation interface
 */
export function ApiDocs() {
  const [SwaggerUI, setSwaggerUI] = useState<typeof import('swagger-ui-react').default | null>(null);

  useEffect(() => {
    // Dynamically import SwaggerUI to avoid SSR issues
    import('swagger-ui-react').then((module) => {
      setSwaggerUI(() => module.default);
    });
  }, []);

  if (!SwaggerUI) {
    return (
      <div className="container mx-auto py-12 px-4">
        <div className="animate-pulse">
          <div className="h-12 bg-gray-200 rounded mb-4"></div>
          <div className="h-64 bg-gray-100 rounded"></div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-lii-bg">
      <div className="container mx-auto py-8 px-4">
        <div className="mb-12">
          <h1 className="text-4xl font-display text-lii-bg mb-4">API Documentation</h1>
          <p className="text-lg text-lii-ash mb-6">
            RESTful API endpoints for Live It Iconic e-commerce platform. Browse, explore, and test all available endpoints.
          </p>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
            <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-200">
              <h3 className="font-semibold text-lii-bg mb-2">Base URL</h3>
              <code className="text-sm bg-gray-100 p-2 rounded block">
                https://api.liveiconic.com
              </code>
            </div>

            <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-200">
              <h3 className="font-semibold text-lii-bg mb-2">Authentication</h3>
              <p className="text-sm text-lii-ash">
                Bearer Token (JWT) in Authorization header
              </p>
            </div>

            <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-200">
              <h3 className="font-semibold text-lii-bg mb-2">Version</h3>
              <p className="text-sm text-lii-ash">API v1.0.0</p>
            </div>
          </div>

          <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 mb-8">
            <p className="text-sm text-blue-900">
              <strong>Note:</strong> To test authenticated endpoints, first sign up or sign in, then the tokens will be automatically included in subsequent requests.
            </p>
          </div>
        </div>

        <div className="bg-white rounded-lg shadow-lg overflow-hidden">
          <SwaggerUI spec={swaggerSpec} />
        </div>
      </div>
    </div>
  );
}

export default ApiDocs;
