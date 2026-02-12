from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
import os
from pinecone import Pinecone
import google.generativeai as genai
import io
from pypdf import PdfReader
import docx
import asyncio
import logging
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ---------------------------
# Configuration
# ---------------------------

# Environment variables (Render uses env vars directly, no .env file needed)
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX", "gemini-rag")

if not PINECONE_API_KEY or not GEMINI_API_KEY:
    raise RuntimeError("Missing required environment variables: PINECONE_API_KEY and/or GEMINI_API_KEY")

# Limits optimized for Render free tier (512 MB RAM)
MAX_FILE_SIZE_MB = 2  # Reduced from 5MB
CHUNK_SIZE = 500  # Reduced from 800
CHUNK_OVERLAP = 100  # Reduced from 150
BATCH_SIZE = 25  # Reduced from 50
MAX_CHUNKS_PER_FILE = 100  # Prevent memory issues
TIMEOUT_SECONDS = 15  # Timeout for API calls

# Rate limiter
limiter = Limiter(key_func=get_remote_address)


# ---------------------------
# Lifespan Events
# ---------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup resources"""
    global pc, index, client
    
    try:
        # Initialize clients
        logger.info("Initializing Pinecone and Gemini clients...")
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index(INDEX_NAME)
        genai.configure(api_key=GEMINI_API_KEY)
        logger.info("Clients initialized successfully")
        
        yield
        
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")
        raise
    finally:
        logger.info("Shutting down...")


# ---------------------------
# FastAPI App
# ---------------------------

app = FastAPI(
    title="Production RAG API",
    version="2.0.0",
    lifespan=lifespan
)

# Add rate limiter to app state
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------
# Models
# ---------------------------

class QueryRequest(BaseModel):
    question: str
    top_k: int = 3
    
    class Config:
        json_schema_extra = {
            "example": {
                "question": "What is the main topic of the document?",
                "top_k": 3
            }
        }


class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict]


class UploadResponse(BaseModel):
    message: str
    chunks_added: int
    filename: str


class HealthResponse(BaseModel):
    status: str
    total_vectors: int = 0
    message: str = ""


# ---------------------------
# Helper Functions
# ---------------------------

def validate_file_size(file: UploadFile) -> None:
    """Validate file size without loading entire file into memory"""
    if hasattr(file, 'size') and file.size:
        if file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Maximum size is {MAX_FILE_SIZE_MB}MB"
            )


def extract_text(file: UploadFile) -> str:
    """Extract text from uploaded file with memory-efficient handling"""
    try:
        # Read file content
        content = file.file.read()
        
        # Check actual size after reading
        if len(content) > MAX_FILE_SIZE_MB * 1024 * 1024:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Maximum size is {MAX_FILE_SIZE_MB}MB"
            )
        
        if file.filename.endswith(".txt"):
            return content.decode("utf-8", errors="ignore")
        
        elif file.filename.endswith(".pdf"):
            reader = PdfReader(io.BytesIO(content))
            text_parts = []
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
            return "\n".join(text_parts)
        
        elif file.filename.endswith(".docx"):
            doc = docx.Document(io.BytesIO(content))
            return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        
        else:
            raise HTTPException(
                status_code=400,
                detail="Unsupported file type. Supported: .txt, .pdf, .docx"
            )
    
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="File encoding not supported")
    except Exception as e:
        logger.error(f"Error extracting text: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")
    finally:
        # Reset file pointer
        file.file.seek(0)


def chunk_text(text: str) -> List[str]:
    """Split text into overlapping chunks"""
    if not text.strip():
        return []
    
    chunks = []
    start = 0
    
    while start < len(text) and len(chunks) < MAX_CHUNKS_PER_FILE:
        end = start + CHUNK_SIZE
        chunk = text[start:end].strip()
        
        if chunk:
            chunks.append(chunk)
        
        start = end - CHUNK_OVERLAP
    
    if len(chunks) >= MAX_CHUNKS_PER_FILE:
        logger.warning(f"File exceeded maximum chunks ({MAX_CHUNKS_PER_FILE}). Truncating.")
    
    return chunks


async def get_embedding_async(text: str) -> List[float]:
    """Get embedding with timeout and error handling"""
    try:
        loop = asyncio.get_event_loop()
        
        def _get_embedding():
            result = genai.embed_content(
                model="models/gemini-embedding-001",
                content=text,
                task_type="retrieval_document"
            )
            return result['embedding']
        
        # Run in executor with timeout
        embedding = await asyncio.wait_for(
            loop.run_in_executor(None, _get_embedding),
            timeout=TIMEOUT_SECONDS
        )
        
        return embedding
    
    except asyncio.TimeoutError:
        logger.error("Embedding generation timed out")
        raise HTTPException(status_code=504, detail="Embedding generation timed out")
    except Exception as e:
        logger.error(f"Error generating embedding: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating embedding: {str(e)}")


async def generate_answer_async(prompt: str) -> str:
    """Generate answer with timeout and error handling"""
    try:
        loop = asyncio.get_event_loop()
        
        def _generate():
            model = genai.GenerativeModel('models/gemini-2.5-flash')
            response = model.generate_content(prompt)
            return response.text
        
        # Run in executor with timeout
        answer = await asyncio.wait_for(
            loop.run_in_executor(None, _generate),
            timeout=TIMEOUT_SECONDS
        )
        
        return answer
    
    except asyncio.TimeoutError:
        logger.error("Answer generation timed out")
        raise HTTPException(status_code=504, detail="Answer generation timed out")
    except Exception as e:
        logger.error(f"Error generating answer: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating answer: {str(e)}")


async def upsert_vectors_batch(vectors: List[Dict], batch_size: int = BATCH_SIZE) -> None:
    """Upsert vectors in batches with retry logic"""
    total_batches = (len(vectors) + batch_size - 1) // batch_size
    
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i + batch_size]
        batch_num = i // batch_size + 1
        
        retries = 3
        for attempt in range(retries):
            try:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    None,
                    lambda: index.upsert(vectors=batch)
                )
                logger.info(f"Upserted batch {batch_num}/{total_batches}")
                break
            
            except Exception as e:
                if attempt == retries - 1:
                    logger.error(f"Failed to upsert batch {batch_num} after {retries} attempts: {str(e)}")
                    raise HTTPException(
                        status_code=500,
                        detail=f"Failed to upload vectors to database: {str(e)}"
                    )
                else:
                    logger.warning(f"Retry {attempt + 1}/{retries} for batch {batch_num}")
                    await asyncio.sleep(1 * (attempt + 1))  # Exponential backoff


# ---------------------------
# Endpoints
# ---------------------------

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Production RAG API for Render Free Tier",
        "version": "2.0.0",
        "endpoints": {
            "health": "/health",
            "upload": "/upload",
            "query": "/query",
            "clear": "/clear"
        }
    }


@app.get("/health", response_model=HealthResponse)
@limiter.limit("30/minute")
async def health(request: Request):
    """Health check endpoint"""
    try:
        loop = asyncio.get_event_loop()
        stats = await loop.run_in_executor(
            None,
            lambda: index.describe_index_stats()
        )
        
        return HealthResponse(
            status="healthy",
            total_vectors=stats.total_vector_count,
            message="All systems operational"
        )
    
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return HealthResponse(
            status="unhealthy",
            total_vectors=0,
            message=str(e)
        )


@app.post("/upload", response_model=UploadResponse)
@limiter.limit("5/minute")
async def upload(request: Request, file: UploadFile = File(...)):
    """
    Upload and process a document
    
    Supported formats: .txt, .pdf, .docx
    Maximum file size: 2MB
    """
    logger.info(f"Processing upload: {file.filename}")
    
    # Validate file
    validate_file_size(file)
    
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")
    
    # Extract text
    text = extract_text(file)
    
    if not text.strip():
        raise HTTPException(status_code=400, detail="File appears to be empty or unreadable")
    
    # Chunk text
    chunks = chunk_text(text)
    
    if not chunks:
        raise HTTPException(status_code=400, detail="No text chunks generated from file")
    
    logger.info(f"Generated {len(chunks)} chunks from {file.filename}")
    
    # Generate embeddings and prepare vectors
    vectors = []
    
    for i, chunk in enumerate(chunks):
        try:
            embedding = await get_embedding_async(chunk)
            
            vectors.append({
                "id": f"{file.filename}_{i}_{hash(chunk) % 100000}",
                "values": embedding,
                "metadata": {
                    "text": chunk[:1000],  # Limit metadata size
                    "filename": file.filename,
                    "chunk_index": i
                }
            })
        
        except Exception as e:
            logger.error(f"Error processing chunk {i}: {str(e)}")
            # Continue with other chunks
            continue
    
    if not vectors:
        raise HTTPException(status_code=500, detail="Failed to generate embeddings for any chunks")
    
    # Upsert to Pinecone
    await upsert_vectors_batch(vectors)
    
    logger.info(f"Successfully uploaded {len(vectors)} vectors for {file.filename}")
    
    return UploadResponse(
        message="File processed successfully",
        chunks_added=len(vectors),
        filename=file.filename
    )


@app.post("/query", response_model=QueryResponse)
@limiter.limit("20/minute")
async def query(request: Request, query_request: QueryRequest):
    """
    Query the knowledge base
    
    Returns an answer based on the most relevant document chunks
    """
    logger.info(f"Processing query: {query_request.question[:100]}")
    
    if not query_request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    # Validate top_k
    if query_request.top_k < 1 or query_request.top_k > 10:
        raise HTTPException(status_code=400, detail="top_k must be between 1 and 10")
    
    # Generate query embedding
    query_embedding = await get_embedding_async(query_request.question)
    
    # Query Pinecone
    try:
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None,
            lambda: index.query(
                vector=query_embedding,
                top_k=query_request.top_k,
                include_metadata=True
            )
        )
    except Exception as e:
        logger.error(f"Pinecone query failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Database query failed: {str(e)}")
    
    matches = results.matches
    
    if not matches:
        logger.info("No matches found for query")
        return QueryResponse(
            answer="I couldn't find any relevant information in the knowledge base to answer your question.",
            sources=[]
        )
    
    # Build context from matches
    context_parts = []
    for match in matches:
        if "text" in match.metadata:
            context_parts.append(match.metadata["text"])
    
    context = "\n\n---\n\n".join(context_parts)
    
    # Generate answer
    prompt = f"""You are a helpful assistant. Answer the question based ONLY on the context provided below.

If the answer cannot be found in the context, clearly state that you don't have enough information to answer the question.

Context:
{context}

Question: {query_request.question}

Answer:"""
    
    answer = await generate_answer_async(prompt)
    
    # Prepare sources
    sources = [
        {
            "filename": m.metadata.get("filename", "unknown"),
            "score": round(float(m.score), 4),
            "chunk_index": m.metadata.get("chunk_index", 0)
        }
        for m in matches
    ]
    
    logger.info(f"Query answered with {len(sources)} sources")
    
    return QueryResponse(
        answer=answer,
        sources=sources
    )


@app.delete("/clear")
@limiter.limit("2/minute")
async def clear(request: Request):
    """
    Clear all vectors from the database
    
    USE WITH CAUTION: This will delete all stored documents
    """
    logger.warning("Clearing database - all vectors will be deleted")
    
    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: index.delete(delete_all=True)
        )
        
        logger.info("Database cleared successfully")
        return {"message": "Database cleared successfully"}
    
    except Exception as e:
        logger.error(f"Failed to clear database: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to clear database: {str(e)}")


# ---------------------------
# Error Handlers
# ---------------------------

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for unhandled errors"""
    logger.error(f"Unhandled error: {str(exc)}", exc_info=True)
    return {
        "detail": "An internal server error occurred",
        "status_code": 500
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)