import http from 'node:http';

const port = Number(process.env.PORT ?? 8080);

interface HealthResponse {
  status: 'healthy' | 'unhealthy';
  service: string;
  uptime: number;
}

interface ServiceResponse {
  service: string;
  version: string;
  timestamp: string;
}

const startTime = Date.now();

function handleHealth(_req: http.IncomingMessage, res: http.ServerResponse): void {
  const response: HealthResponse = {
    status: 'healthy',
    service: '{{PROJECT_NAME}}',
    uptime: Math.floor((Date.now() - startTime) / 1000),
  };
  res.statusCode = 200;
  res.setHeader('Content-Type', 'application/json');
  res.end(JSON.stringify(response));
}

function handleRoot(_req: http.IncomingMessage, res: http.ServerResponse): void {
  const response: ServiceResponse = {
    service: '{{PROJECT_NAME}}',
    version: '1.0.0',
    timestamp: new Date().toISOString(),
  };
  res.statusCode = 200;
  res.setHeader('Content-Type', 'application/json');
  res.end(JSON.stringify(response));
}

function handler(req: http.IncomingMessage, res: http.ServerResponse): void {
  const method = req.method ?? 'GET';
  const url = req.url ?? '/';

  console.log(`${method} ${url}`);

  if (method === 'GET' && url === '/health') {
    handleHealth(req, res);
    return;
  }

  if (method === 'GET' && url === '/') {
    handleRoot(req, res);
    return;
  }

  res.statusCode = 404;
  res.end(JSON.stringify({ error: 'Not Found' }));
}

const server = http.createServer(handler);
server.listen(port, () => {
  console.log(`{{PROJECT_NAME}} listening on port ${port}`);
});
