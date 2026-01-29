"""
FastAPI application for RAG-based question answering system.
Provides endpoints for document upload and querying.
"""

import os
import time
import shutil
from typing import Dict
from collections import defaultdict
from datetime import datetime, timedelta

from fastapi import FastAPI, File, UploadFile, HTTPException, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from rag import ingest_document, query_documents, initialize_pinecone

# create fastapi app
app = FastAPI(
    title="Explainable RAG System",
    description="Upload documents and ask questions with grounded answers",
    version="1.0.0"
)

# add cors middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # configure based on your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# simple in-memory rate limiting
# in production, use redis or similar
rate_limit_storage = defaultdict(list)
RATE_LIMIT_REQUESTS = 10  # max requests per window
RATE_LIMIT_WINDOW = 60  # window in seconds


def check_rate_limit(client_ip: str):
    """
    Simple rate limiting based on client IP.
    Allows RATE_LIMIT_REQUESTS per RATE_LIMIT_WINDOW seconds.
    """
    now = datetime.now()
    
    # clean old entries outside the window
    cutoff = now - timedelta(seconds=RATE_LIMIT_WINDOW)
    rate_limit_storage[client_ip] = [
        timestamp for timestamp in rate_limit_storage[client_ip]
        if timestamp > cutoff
    ]
    
    # check if limit exceeded
    if len(rate_limit_storage[client_ip]) >= RATE_LIMIT_REQUESTS:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Max {RATE_LIMIT_REQUESTS} requests per {RATE_LIMIT_WINDOW} seconds."
        )
    
    # add current request
    rate_limit_storage[client_ip].append(now)


# pydantic models for request/response validation
class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Question to ask about uploaded documents")


class SourceDocument(BaseModel):
    content: str
    source: str
    chunk_index: int


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceDocument]
    latency_ms: float


class UploadResponse(BaseModel):
    message: str
    filename: str


# startup event to initialize pinecone
@app.on_event("startup")
async def startup_event():
    """Initialize Pinecone connection on startup"""
    try:
        initialize_pinecone()
        print("✓ Pinecone initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize Pinecone: {e}")
        raise


@app.post("/upload", response_model=UploadResponse)
async def upload_document(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """
    Upload a PDF or TXT document for ingestion.
    
    Ingestion can be slow for large files, so this runs in background.
    The endpoint returns immediately with a confirmation.
    """
    # check rate limit
    client_ip = request.client.host
    check_rate_limit(client_ip)
    
    # validate file type
    allowed_extensions = {'.pdf', '.txt'}
    file_extension = os.path.splitext(file.filename)[1].lower()
    
    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"
        )
    
    # validate file size (max 10MB)
    # read file content
    file_content = await file.read()
    file_size_mb = len(file_content) / (1024 * 1024)
    
    if file_size_mb > 10:
        raise HTTPException(
            status_code=400,
            detail="File size exceeds 10MB limit"
        )
    
    # save file temporarily
    # create uploads directory if it doesn't exist
    os.makedirs("uploads", exist_ok=True)
    
    # use timestamp to avoid filename collisions
    timestamp = int(time.time())
    safe_filename = f"{timestamp}_{file.filename}"
    file_path = os.path.join("uploads", safe_filename)
    
    with open(file_path, "wb") as f:
        f.write(file_content)
    
    # ingestion can be slow for large files, so this runs in background
    # this allows the api to return immediately
    background_tasks.add_task(ingest_document, file_path, file.filename)
    
    return UploadResponse(
        message="Document upload successful. Ingestion in progress.",
        filename=file.filename
    )


@app.post("/query", response_model=QueryResponse)
async def query(request: Request, query_request: QueryRequest):
    """
    Ask a question about uploaded documents.
    Returns an answer grounded in the document content.
    """
    # check rate limit
    client_ip = request.client.host
    check_rate_limit(client_ip)
    
    # track latency for monitoring
    start_time = time.time()
    
    try:
        # execute rag query
        result = query_documents(query_request.question)
        
        # calculate latency
        latency_ms = (time.time() - start_time) * 1000
        
        return QueryResponse(
            answer=result["answer"],
            sources=[
                SourceDocument(**source) for source in result["sources"]
            ],
            latency_ms=round(latency_ms, 2)
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )


@app.get("/health")
async def health_check():
    """Simple health check endpoint"""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
