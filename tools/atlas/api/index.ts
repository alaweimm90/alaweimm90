/**
 * ATLAS REST API
 * HTTP interface for multi-agent orchestration
 */

export { createServer, startServer } from './server.js';
export type { APIRequest, APIResponse, ServerConfig } from './server.js';
export { router } from './router.js';
