import http from 'node:http'

function handler(req: http.IncomingMessage, res: http.ServerResponse): void {
  const method = req.method ?? 'GET'
  const url = req.url ?? '/'
  if (method === 'GET' && url === '/health') {
    res.statusCode = 200
    res.setHeader('Content-Type', 'text/plain')
    res.end('ok')
    return
  }
  res.statusCode = 404
  res.end()
}

export function createServer(): http.Server {
  return http.createServer(handler)
}

const port = Number(process.env.PORT ?? 8080)
if (process.env.NODE_ENV !== 'test') {
  createServer().listen(port)
}
