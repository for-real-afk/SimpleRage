# Production RAG API

A production-ready Retrieval-Augmented Generation (RAG) API built with FastAPI, optimized for deployment on Render's free tier. Upload documents, ask questions, and get AI-powered answers based on your knowledge base.

![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## 🌟 Features

- **📄 Document Processing**: Upload `.txt`, `.pdf`, and `.docx` files
- **🔍 Semantic Search**: Find relevant information using vector embeddings
- **🤖 AI-Powered Answers**: Generate contextual responses using Google's Gemini
- **⚡ Production-Ready**: Rate limiting, error handling, logging, and timeouts
- **💰 Free-Tier Optimized**: Designed to run on Render's 512MB RAM free tier
- **🔒 Secure**: CORS configuration, input validation, and environment-based secrets
- **📊 Monitoring**: Health checks and comprehensive logging

## 📋 Table of Contents

- [Architecture](#-architecture)
- [How It Works](#-how-it-works)
- [Quick Start](#-quick-start)
- [API Documentation](#-api-documentation)
- [Deployment](#-deployment)
- [Configuration](#-configuration)
- [Troubleshooting](#-troubleshooting)
- [Performance](#-performance)

## 🏗️ Architecture

### System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         CLIENT / USER                            │
│                    (Web, Mobile, API Client)                     │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             │ HTTP Requests
                             │ (REST API)
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                      FASTAPI APPLICATION                         │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    Middleware Layer                       │   │
│  │  • CORS Handler                                          │   │
│  │  • Rate Limiter (SlowAPI)                                │   │
│  │  • Error Handler                                         │   │
│  │  • Request Logger                                        │   │
│  └──────────────────────────────────────────────────────────┘   │
│                             │                                    │
│  ┌──────────────────────────┼────────────────────────────────┐  │
│  │              API Endpoints (Routes)                       │  │
│  │                          │                                │  │
│  │  ┌───────────┬──────────┼───────────┬─────────────┐      │  │
│  │  │           │          │           │             │      │  │
│  │  │  POST     │  POST    │   GET     │   DELETE    │      │  │
│  │  │ /upload   │ /query   │ /health   │  /clear     │      │  │
│  │  │           │          │           │             │      │  │
│  │  └─────┬─────┴────┬─────┴─────┬─────┴──────┬──────┘      │  │
│  │        │          │           │            │             │  │
│  └────────┼──────────┼───────────┼────────────┼─────────────┘  │
│           │          │           │            │                │
│  ┌────────▼──────────▼───────────▼────────────▼─────────────┐  │
│  │              Business Logic Layer                         │  │
│  │                                                            │  │
│  │  • Document Processing (extract_text)                     │  │
│  │  • Text Chunking (chunk_text)                             │  │
│  │  • Async Embedding Generation (get_embedding_async)       │  │
│  │  • Async Answer Generation (generate_answer_async)        │  │
│  │  • Vector Batch Upload (upsert_vectors_batch)             │  │
│  │  • Timeout & Retry Logic                                  │  │
│  └──────────────┬─────────────────────────┬──────────────────┘  │
│                 │                         │                     │
└─────────────────┼─────────────────────────┼─────────────────────┘
                  │                         │
         ┌────────▼────────┐       ┌────────▼─────────┐
         │                 │       │                  │
         │  PINECONE API   │       │   GEMINI API     │
         │  (Vector DB)    │       │  (Google AI)     │
         │                 │       │                  │
         │  • Store        │       │  • Embeddings    │
         │  • Search       │       │    (3072-dim)    │
         │  • Retrieve     │       │  • Text Gen      │
         │    Vectors      │       │    (Gemini 2.5)  │
         │                 │       │                  │
         └─────────────────┘       └──────────────────┘
```

### Data Flow

#### 1. Document Upload Flow

```
User uploads document
         │
         ▼
┌────────────────────┐
│  Validate file     │
│  • Type check      │
│  • Size check      │
│  • Read content    │
└────────┬───────────┘
         │
         ▼
┌────────────────────┐
│  Extract text      │
│  • PDF → text      │
│  • DOCX → text     │
│  • TXT → text      │
└────────┬───────────┘
         │
         ▼
┌────────────────────┐
│  Chunk text        │
│  • Size: 500 char  │
│  • Overlap: 100    │
│  • Max: 100 chunks │
└────────┬───────────┘
         │
         ▼
┌────────────────────┐
│  Generate          │
│  embeddings        │
│  (Gemini API)      │
│  3072 dimensions   │
└────────┬───────────┘
         │
         ▼
┌────────────────────┐
│  Batch upsert      │
│  to Pinecone       │
│  • Retry logic     │
│  • 25 per batch    │
└────────┬───────────┘
         │
         ▼
    Success! ✅
```

#### 2. Query Flow

```
User asks question
         │
         ▼
┌────────────────────┐
│  Generate query    │
│  embedding         │
│  (Gemini API)      │
│  3072 dimensions   │
└────────┬───────────┘
         │
         ▼
┌────────────────────┐
│  Search Pinecone   │
│  • Cosine          │
│    similarity      │
│  • Top K results   │
│  • Return metadata │
└────────┬───────────┘
         │
         ▼
┌────────────────────┐
│  Build context     │
│  from top matches  │
│  • Combine chunks  │
│  • Prepare prompt  │
└────────┬───────────┘
         │
         ▼
┌────────────────────┐
│  Generate answer   │
│  (Gemini 2.5)      │
│  • Context-aware   │
│  • Grounded        │
└────────┬───────────┘
         │
         ▼
┌────────────────────┐
│  Return response   │
│  • Answer text     │
│  • Source docs     │
│  • Similarity      │
│    scores          │
└────────────────────┘
         │
         ▼
    Success! ✅
```

### Component Details

#### FastAPI Application Layer

```python
┌─────────────────────────────────────────┐
│         FastAPI Application             │
│                                         │
│  Lifespan Management:                   │
│  • Initialize Pinecone client           │
│  • Configure Gemini API                 │
│  • Setup logging                        │
│  • Graceful shutdown                    │
│                                         │
│  Middleware:                            │
│  • CORS (Cross-Origin Resource Sharing) │
│  • Rate Limiting (SlowAPI)              │
│  • Error Handling                       │
│  • Request Logging                      │
└─────────────────────────────────────────┘
```

#### Vector Storage (Pinecone)

```
┌─────────────────────────────────────┐
│        Pinecone Vector Index         │
│                                     │
│  Index: "gemini-rag"                │
│  Dimensions: 3072                   │
│  Metric: Cosine Similarity          │
│                                     │
│  Vector Structure:                  │
│  {                                  │
│    id: "filename_chunk_hash",       │
│    values: [3072 floats],           │
│    metadata: {                      │
│      text: "chunk content...",      │
│      filename: "doc.pdf",           │
│      chunk_index: 0                 │
│    }                                │
│  }                                  │
└─────────────────────────────────────┘
```

#### AI Models (Google Gemini)

```
┌──────────────────────────────────────┐
│         Gemini Embedding Model        │
│                                      │
│  Model: gemini-embedding-001         │
│  Output: 3072-dimensional vector     │
│  Task Type: retrieval_document       │
│  Use: Convert text to embeddings     │
└──────────────────────────────────────┘

┌──────────────────────────────────────┐
│      Gemini Generation Model         │
│                                      │
│  Model: gemini-2.5-flash             │
│  Use: Generate contextual answers    │
│  Features:                           │
│  • Fast inference                    │
│  • Context-aware                     │
│  • Grounded responses                │
└──────────────────────────────────────┘
```

## 🔄 How It Works

### Understanding RAG (Retrieval-Augmented Generation)

RAG combines information retrieval with AI text generation to provide accurate, grounded answers:

1. **Indexing Phase** (Upload):
   ```
   Document → Split into chunks → Generate embeddings → Store in vector DB
   ```

2. **Retrieval Phase** (Query):
   ```
   Question → Generate embedding → Find similar chunks → Retrieve context
   ```

3. **Generation Phase** (Answer):
   ```
   Context + Question → LLM → Grounded answer + sources
   ```

### Why RAG?

- ✅ **Accurate**: Answers based on your documents, not hallucinations
- ✅ **Traceable**: Every answer includes source documents
- ✅ **Up-to-date**: Update knowledge by uploading new documents
- ✅ **Private**: Your data stays in your control

### Similarity Scoring

The system uses **cosine similarity** to measure relevance:

```python
Score Range: 0.0 to 1.0

0.9 - 1.0  ████████████ Excellent match
0.8 - 0.9  ██████████   Very good match
0.7 - 0.8  ████████     Good match
0.6 - 0.7  ██████       Fair match
< 0.6      ████         Weak match
```

**Example from your test:**
```json
{
  "answer": "The system uses Pinecone for vector storage.",
  "sources": [
    {"filename": "test_document.txt", "score": 0.8104},  // 81% relevant
    {"filename": "test_document.txt", "score": 0.7914}   // 79% relevant
  ]
}
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11 or 3.12 (Python 3.13 has limited package support)
- Pinecone account ([sign up](https://app.pinecone.io/))
- Google AI Studio account ([get API key](https://aistudio.google.com/app/apikey))

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd rag-api
   ```

2. **Create virtual environment**
   ```bash
   python3.11 -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   ```

   Edit `.env` and add your API keys:
   ```env
   PINECONE_API_KEY=your_pinecone_key_here
   GEMINI_API_KEY=your_gemini_key_here
   PINECONE_INDEX=gemini-rag
   ```

5. **Create Pinecone index**

   Go to [Pinecone Console](https://app.pinecone.io/):
   - Click "Create Index"
   - Name: `gemini-rag`
   - Dimensions: `3072` ⚠️ **CRITICAL**
   - Metric: `cosine`
   - Click "Create"

6. **Run the server**
   ```bash
   uvicorn main:app --reload
   ```

7. **Test the API**
   ```bash
   python test_api.py
   ```

   You should see:
   ```
   ✅ All tests completed!
   Your API is ready for deployment to Render!
   ```

8. **Access API documentation**

   Open your browser: [http://localhost:8000/docs](http://localhost:8000/docs)

## 📡 API Documentation

### Base URL

```
Local: http://localhost:8000
Production: https://your-app.onrender.com
```

### Endpoints

#### 1. Health Check

```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "total_vectors": 150,
  "message": "All systems operational"
}
```

**Rate Limit:** 30 requests/minute

---

#### 2. Upload Document

```http
POST /upload
Content-Type: multipart/form-data
```

**Parameters:**
- `file` (required): Document file (.txt, .pdf, .docx)

**Example:**
```bash
curl -X POST "http://localhost:8000/upload" \
  -F "file=@document.pdf"
```

**Response:**
```json
{
  "message": "File processed successfully",
  "chunks_added": 15,
  "filename": "document.pdf"
}
```

**Constraints:**
- Max file size: 2MB
- Supported formats: `.txt`, `.pdf`, `.docx`
- Max chunks per file: 100
- Chunk size: 500 characters
- Chunk overlap: 100 characters

**Rate Limit:** 5 requests/minute

---

#### 3. Query Knowledge Base

```http
POST /query
Content-Type: application/json
```

**Request Body:**
```json
{
  "question": "What is the main topic?",
  "top_k": 3
}
```

**Parameters:**
- `question` (required): Your question as a string
- `top_k` (optional): Number of results to retrieve (1-10, default: 3)

**Example:**
```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the key features?",
    "top_k": 3
  }'
```

**Response:**
```json
{
  "answer": "Based on the documents, the key features are...",
  "sources": [
    {
      "filename": "document.pdf",
      "score": 0.8542,
      "chunk_index": 0
    },
    {
      "filename": "document.pdf",
      "score": 0.8123,
      "chunk_index": 1
    }
  ]
}
```

**Rate Limit:** 20 requests/minute

---

#### 4. Clear Database

```http
DELETE /clear
```

⚠️ **Warning:** This deletes ALL vectors from the database!

**Response:**
```json
{
  "message": "Database cleared successfully"
}
```

**Rate Limit:** 2 requests/minute

---

### Error Responses

All errors follow this format:

```json
{
  "detail": "Error message here"
}
```

**Common HTTP Status Codes:**
- `200` - Success
- `400` - Bad Request (invalid input)
- `404` - Not Found
- `500` - Internal Server Error
- `504` - Gateway Timeout (operation took too long)
- `429` - Too Many Requests (rate limit exceeded)

## 🌐 Deployment

### Deploy to Render (Free Tier)

#### Step 1: Prepare Your Code

1. Push code to GitHub:
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin <your-github-repo-url>
   git push -u origin main
   ```

2. Make sure `.env` is in `.gitignore` (already configured)

#### Step 2: Create Render Account

Sign up at [render.com](https://render.com)

#### Step 3: Create Web Service

1. Click "New +" → "Web Service"
2. Connect your GitHub repository
3. Configure:

   | Setting | Value |
   |---------|-------|
   | **Name** | `rag-api` |
   | **Region** | Choose closest to you |
   | **Branch** | `main` |
   | **Runtime** | `Python 3` |
   | **Build Command** | `pip install -r requirements.txt` |
   | **Start Command** | `uvicorn main:app --host 0.0.0.0 --port $PORT` |
   | **Instance Type** | `Free` |

#### Step 4: Set Environment Variables

In Render dashboard, add these environment variables:

| Variable | Value |
|----------|-------|
| `PINECONE_API_KEY` | Your Pinecone API key |
| `GEMINI_API_KEY` | Your Gemini API key |
| `PINECONE_INDEX` | `gemini-rag` |
| `PYTHON_VERSION` | `3.11.0` |

#### Step 5: Deploy

Click "Create Web Service" - Render will:
1. Clone your repository
2. Install dependencies
3. Start your application
4. Give you a URL: `https://your-app.onrender.com`

#### Step 6: Test Production

```bash
# Upload a document
curl -X POST "https://your-app.onrender.com/upload" \
  -F "file=@test.txt"

# Query
curl -X POST "https://your-app.onrender.com/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "test question"}'
```

### Render Free Tier Limits

- ✅ 512 MB RAM
- ✅ Shared CPU
- ✅ 750 hours/month
- ⚠️ App sleeps after 15 min inactivity
- ⚠️ Cold start: ~30 seconds

### Keep Your App Awake (Optional)

Use a cron job service like [cron-job.org](https://cron-job.org):
- Ping `https://your-app.onrender.com/health` every 10 minutes
- Prevents cold starts

## ⚙️ Configuration

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `PINECONE_API_KEY` | Yes | - | Your Pinecone API key |
| `GEMINI_API_KEY` | Yes | - | Your Google Gemini API key |
| `PINECONE_INDEX` | No | `gemini-rag` | Name of your Pinecone index |

### Performance Tuning

Edit these constants in `main.py`:

```python
# File processing limits
MAX_FILE_SIZE_MB = 2          # Maximum upload size
CHUNK_SIZE = 500              # Characters per chunk
CHUNK_OVERLAP = 100           # Overlap between chunks
MAX_CHUNKS_PER_FILE = 100     # Max chunks to process

# API settings
BATCH_SIZE = 25               # Vectors per batch upload
TIMEOUT_SECONDS = 15          # API call timeout

# Rate limits (requests per minute)
@limiter.limit("30/minute")   # Health endpoint
@limiter.limit("5/minute")    # Upload endpoint
@limiter.limit("20/minute")   # Query endpoint
@limiter.limit("2/minute")    # Clear endpoint
```

### For Higher Traffic

If you need to handle more requests:

1. **Upgrade Render tier** ($7/month for 512MB, $25/month for 2GB)
2. **Increase rate limits** in code
3. **Add Redis** for distributed rate limiting
4. **Use load balancer** for multiple instances

## 🔧 Troubleshooting

### Common Issues

#### 1. Dimension Mismatch Error

```
Vector dimension 3072 does not match the dimension of the index 768
```

**Solution:** Delete and recreate your Pinecone index with 3072 dimensions.

---

#### 2. Model Not Found Error

```
404 models/gemini-embedding-001 is not found
```

**Solution:** Run `python test_models.py` to see available models for your API key.

---

#### 3. Import Error

```
ModuleNotFoundError: No module named 'google.generativeai'
```

**Solution:**
```bash
pip install google-generativeai==0.8.3
```

---

#### 4. Timeout Errors

```
504: Embedding generation timed out
```

**Solution:**
- Reduce `CHUNK_SIZE` to 400
- Increase `TIMEOUT_SECONDS` to 30
- Use smaller documents

---

#### 5. Rate Limit Exceeded

```
429: Too Many Requests
```

**Solution:**
- Wait 1 minute before retrying
- Reduce request frequency
- Check Gemini API quotas

---

#### 6. Memory Issues (Render Free Tier)

**Solution:**
```python
MAX_FILE_SIZE_MB = 1     # Reduce from 2
CHUNK_SIZE = 400         # Reduce from 500
BATCH_SIZE = 10          # Reduce from 25
```

## 📊 Performance

### Benchmarks (Render Free Tier)

| Metric | Value |
|--------|-------|
| Upload (1MB PDF) | ~8-12 seconds |
| Query Response | ~2-4 seconds |
| Cold Start | ~30 seconds |
| Warm Response | <1 second |
| Max Concurrent | 2-3 requests |

### Optimization Tips

1. **Use smaller chunks** for faster processing
2. **Batch uploads** instead of individual files
3. **Cache frequently asked questions**
4. **Use webhooks** instead of polling
5. **Implement request queuing** for high load

### Scaling Strategy

```
Free Tier (512MB)
  ↓
Starter ($7/mo, 512MB + more CPU)
  ↓
Standard ($25/mo, 2GB RAM)
  ↓
Pro ($85/mo, 4GB RAM)
  ↓
Custom (Redis + Load Balancer)
```

## 🛡️ Security Best Practices

1. **Never commit `.env` file** - Already in `.gitignore`
2. **Use environment variables** for all secrets
3. **Update CORS origins** in production:
   ```python
   allow_origins=["https://yourdomain.com"]
   ```
4. **Add API key authentication** for public deployments
5. **Implement request signing** for sensitive data
6. **Enable HTTPS only** in production
7. **Monitor rate limits** and add IP blocking if needed

## 📈 Monitoring

### Logs

Check logs in Render dashboard:
1. Go to your web service
2. Click "Logs" tab
3. Monitor for errors

### Health Checks

Render automatically pings `/health` endpoint.

### Metrics to Track

- Request latency
- Error rate
- Vector count growth
- API quota usage
- Memory usage

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details

## 🆘 Support

- **Issues**: [GitHub Issues](https://github.com/for-real-afk/SimpleRage.git)
- **Discussions**: [GitHub Discussions](https://github.com/for-real-afk/SimpleRage.git)
- **Pinecone Docs**: [docs.pinecone.io](https://docs.pinecone.io)
- **Gemini Docs**: [ai.google.dev](https://ai.google.dev)
- **Render Docs**: [render.com/docs](https://render.com/docs)

## 🎯 Roadmap

- [ ] Add support for more file formats (Excel, PPT)
- [ ] Implement conversation memory
- [ ] Add user authentication
- [ ] Build web interface
- [ ] Support multiple indexes
- [ ] Add analytics dashboard
- [ ] Implement caching layer
- [ ] Support streaming responses

---

**Built with ❤️ for the AI community**

Deploy your own RAG system in minutes! 🚀
