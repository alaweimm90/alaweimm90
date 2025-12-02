"""
Example FastAPI application demonstrating governance contract compliance.

This is a minimal example showing how a consumer repository
implements the governance contract patterns.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

app = FastAPI(
    title="Example Microservice",
    description="Consumer repository using governance contract",
    version="1.0.0",
)


class HealthCheck(BaseModel):
    """Health check response."""
    status: str
    version: str
    service: str


class Message(BaseModel):
    """Simple message model."""
    id: int
    content: str
    author: Optional[str] = None


# In-memory storage for example
messages_db = {}


@app.get("/health", response_model=HealthCheck)
async def health_check():
    """Health check endpoint (required per governance contract)."""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "service": "example-microservice",
    }


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Example Microservice",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/api/messages/{message_id}", response_model=Message)
async def get_message(message_id: int):
    """Get a message by ID."""
    if message_id not in messages_db:
        raise HTTPException(status_code=404, detail="Message not found")
    return messages_db[message_id]


@app.post("/api/messages", response_model=Message)
async def create_message(message: Message):
    """Create a new message."""
    messages_db[message.id] = message
    return message


@app.delete("/api/messages/{message_id}")
async def delete_message(message_id: int):
    """Delete a message."""
    if message_id not in messages_db:
        raise HTTPException(status_code=404, detail="Message not found")
    del messages_db[message_id]
    return {"status": "deleted"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
