import os
import asyncio
import httpx
from fastapi import FastAPI

app = FastAPI()

SERVICES = [
  {"name": "simcore", "url": "http://simcore:3000"},
  {"name": "repz", "url": "http://repz:8080"},
  {"name": "benchbarrier", "url": "http://benchbarrier:8081"},
  {"name": "attributa", "url": "http://attributa:3000"},
  {"name": "mag-logic", "url": "http://mag-logic:8888"},
  {"name": "custom-exporters", "url": "http://custom-exporters:8888/metrics"},
  {"name": "infra", "url": "http://infra:8000/health"},
]

@app.get("/status")
async def status():
  async with httpx.AsyncClient(timeout=3.0) as client:
    results = []
    for s in SERVICES:
      try:
        r = await client.get(s["url"]) if not s["url"].endswith("/metrics") else await client.head(s["url"]) 
        ok = 200 <= r.status_code < 300
        results.append({"name": s["name"], "url": s["url"], "status": "healthy" if ok else "unhealthy", "code": r.status_code})
      except Exception as e:
        results.append({"name": s["name"], "url": s["url"], "status": "unhealthy", "error": str(e)})
    return {"services": results}

if __name__ == "__main__":
  import uvicorn
  uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT", 4000)))
