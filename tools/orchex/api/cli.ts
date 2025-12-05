/**
 * ORCHEX API CLI
 * Command-line interface to start the REST API server
 */

import { startServer } from './server.js';

startServer().catch((error) => {
  console.error('Failed to start server:', error);
  process.exit(1);
});
