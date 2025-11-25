import os
import psutil
from fastapi import FastAPI, Response
from prometheus_client import CollectorRegistry, Gauge, generate_latest, CONTENT_TYPE_LATEST

app = FastAPI()
registry = CollectorRegistry()
cpu = Gauge('custom_cpu_percent', 'CPU percent', registry=registry)
mem = Gauge('custom_mem_percent', 'Memory percent', registry=registry)

@app.get('/metrics')
def metrics():
  cpu.set(psutil.cpu_percent(interval=0))
  mem.set(psutil.virtual_memory().percent)
  return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)

if __name__ == '__main__':
  import uvicorn
  uvicorn.run('app:app', host='0.0.0.0', port=int(os.getenv('PORT', 8888)))
