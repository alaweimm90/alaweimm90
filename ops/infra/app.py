import os
from fastapi import FastAPI

app = FastAPI()

@app.get('/health')
def health():
  return {"status": "healthy"}

if __name__ == '__main__':
  import uvicorn
  uvicorn.run('app:app', host='0.0.0.0', port=int(os.getenv('PORT', 8000)))
