/**
 * Bundle Analysis Configuration
 * Analyzes and optimizes webpack bundles for all applications
 */

const { BundleAnalyzerPlugin } = require('webpack-bundle-analyzer');
const CompressionPlugin = require('compression-webpack-plugin');
const TerserPlugin = require('terser-webpack-plugin');
const { WebpackManifestPlugin } = require('webpack-manifest-plugin');

module.exports = {
  // Bundle Analysis Plugin
  bundleAnalyzer: new BundleAnalyzerPlugin({
    analyzerMode: 'static',
    reportFilename: '../../../optimization/bundle-analysis/reports/bundle-report.html',
    defaultSizes: 'gzip',
    openAnalyzer: false,
    generateStatsFile: true,
    statsFilename: '../../../optimization/bundle-analysis/reports/bundle-stats.json',
    statsOptions: {
      source: false,
      chunks: true,
      modules: true,
      assets: true,
      entrypoints: true,
      chunkModules: true,
      chunkOrigins: true
    }
  }),

  // Compression Plugin
  compression: new CompressionPlugin({
    filename: '[path][base].gz',
    algorithm: 'gzip',
    test: /\.(js|css|html|svg|json)$/,
    threshold: 8192,
    minRatio: 0.8,
  }),

  // Brotli Compression
  brotliCompression: new CompressionPlugin({
    filename: '[path][base].br',
    algorithm: 'brotliCompress',
    test: /\.(js|css|html|svg|json)$/,
    compressionOptions: {
      level: 11,
    },
    threshold: 8192,
    minRatio: 0.8,
  }),

  // Optimization Configuration
  optimization: {
    minimize: true,
    minimizer: [
      new TerserPlugin({
        terserOptions: {
          compress: {
            drop_console: process.env.NODE_ENV === 'production',
            drop_debugger: true,
            pure_funcs: ['console.log', 'console.info', 'console.debug'],
            passes: 2
          },
          mangle: {
            safari10: true
          },
          format: {
            comments: false
          }
        },
        extractComments: false,
        parallel: true
      })
    ],
    moduleIds: 'deterministic',
    runtimeChunk: 'single',
    splitChunks: {
      chunks: 'all',
      maxInitialRequests: 25,
      minSize: 20000,
      maxSize: 244000,
      cacheGroups: {
        // Vendor splitting
        vendor: {
          test: /[\\/]node_modules[\\/]/,
          name(module) {
            const packageName = module.context.match(
              /[\\/]node_modules[\\/](.*?)([\\/]|$)/
            )[1];
            return `vendor.${packageName.replace('@', '')}`;
          },
          priority: 10,
          reuseExistingChunk: true
        },
        // React ecosystem
        react: {
          test: /[\\/]node_modules[\\/](react|react-dom|react-router|react-hook-form)[\\/]/,
          name: 'react-ecosystem',
          priority: 20
        },
        // UI Components
        uiComponents: {
          test: /[\\/]node_modules[\\/](@radix-ui|lucide-react|framer-motion|@dnd-kit)[\\/]/,
          name: 'ui-components',
          priority: 15
        },
        // Charts & Visualization
        charts: {
          test: /[\\/]node_modules[\\/](recharts|chart\.js|react-chartjs-2|d3|three)[\\/]/,
          name: 'charts-visualization',
          priority: 15
        },
        // State Management
        state: {
          test: /[\\/]node_modules[\\/](zustand|@tanstack[\\/]react-query)[\\/]/,
          name: 'state-management',
          priority: 15
        },
        // Utilities
        utilities: {
          test: /[\\/]node_modules[\\/](lodash|date-fns|uuid|axios|clsx|class-variance-authority)[\\/]/,
          name: 'utilities',
          priority: 10
        },
        // Common chunks
        common: {
          minChunks: 2,
          priority: 5,
          reuseExistingChunk: true,
          name: 'common'
        }
      }
    }
  },

  // Performance Hints
  performance: {
    hints: 'warning',
    maxEntrypointSize: 512000,
    maxAssetSize: 512000,
    assetFilter: function(assetFilename) {
      return assetFilename.endsWith('.js') || assetFilename.endsWith('.css');
    }
  }
};