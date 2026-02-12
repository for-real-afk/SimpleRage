from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import os
from pinecone import Pinecone
import google.genai as genai
from dotenv import load_dotenv
import io
from pypdf import PdfReader
import docx

# Load env
load_dotenv()

app = FastAPI(title="Production RAG API", version="2.0.0")

# Environment variables
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX", "gemini-rag")

if not PINECONE_API_KEY or not GEMINI_API_KEY:
    raise RuntimeError("Missing required environment variables.")

# Initialize clients
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)
client = genai.Client(api_key=GEMINI_API_KEY)

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
MAX_FILE_SIZE_MB = 5


# ---------------------------
# Models
# ---------------------------

class QueryRequest(BaseModel):
    question: str
    top_k: int = 3


class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict]


# ---------------------------
# Helpers
# ---------------------------

def extract_text(file: UploadFile) -> str:
    content = file.file.read()

    if file.filename.endswith(".txt"):
        return content.decode("utf-8")

    elif file.filename.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(content))
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    elif file.filename.endswith(".docx"):
        doc = docx.Document(io.BytesIO(content))
        return "\n".join(p.text for p in doc.paragraphs)

    else:
        raise HTTPException(status_code=400, detail="Unsupported file type.")


def chunk_text(text: str) -> List[str]:
    chunks = []
    start = 0

    while start < len(text):
        end = start + CHUNK_SIZE
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - CHUNK_OVERLAP

    return chunks


def get_embedding(text: str):
    response = client.models.embed_content(
        model="text-embedding-004",
        contents=text
    )
    return response.embeddings[0].values


# ---------------------------
# Endpoints
# ---------------------------

@app.get("/")
def root():
    return {"message": "Production-safe RAG API"}


@app.get("/health")
def health():
    try:
        stats = index.describe_index_stats()
        return {
            "status": "healthy",
            "total_vectors": stats.total_vector_count
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    if file.size and file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large (max 5MB)")

    text = extract_text(file)

    if not text.strip():
        raise HTTPException(status_code=400, detail="Empty file.")

    chunks = chunk_text(text)

    vectors = []

    for i, chunk in enumerate(chunks):
        embedding = get_embedding(chunk)

        vectors.append({
            "id": f"{file.filename}_{i}",
            "values": embedding,
            "metadata": {
                "text": chunk,
                "filename": file.filename
            }
        })

    # Batch upsert
    batch_size = 50
    for i in range(0, len(vectors), batch_size):
        index.upsert(vectors=vectors[i:i+batch_size])

    return {
        "message": "File processed successfully",
        "chunks_added": len(chunks)
    }


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):

    query_embedding = get_embedding(request.question)

    results = index.query(
        vector=query_embedding,
        top_k=request.top_k,
        include_metadata=True
    )

    matches = results.matches

    if not matches:
        return QueryResponse(
            answer="No relevant information found.",
            sources=[]
        )

    context = "\n\n".join(
        match.metadata["text"] for match in matches
    )

    prompt = f"""
Answer based ONLY on the context below.
If the answer is not found, say so clearly.

Context:
{context}

Question:
{request.question}

Answer:
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    sources = [
        {
            "filename": m.metadata["filename"],
            "score": float(m.score)
        }
        for m in matches
    ]

    return QueryResponse(
        answer=response.text,
        sources=sources
    )


@app.delete("/clear")
def clear():
    index.delete(delete_all=True)
    return {"message": "Database cleared"}
