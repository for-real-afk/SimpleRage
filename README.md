# AI PDF/DocX Analysis API

A FastAPI-powered backend for processing documents and interacting with Gemini & Pinecone.

## 🚀 Getting Started

### Prerequisites
- Python 3.12+ (Recommended) or 3.14
- A Pinecone API Key
- A Google Gemini API Key

### Installation
1. Clone the repo:
   ```bash
   git clone <your-repo-url>
   cd <project-folder>
Create and activate a virtual environment:

Bash
python -m venv env
source env/bin/activate  # On Windows: .\env\Scripts\activate
Install dependencies:

Bash
pip install -r requirements.txt
🛠 Deployment (Render)
Build Command: pip install -r requirements.txt

Start Command: uvicorn main:app --host 0.0.0.0 --port $PORT

📄 License
MIT


---

### 4. Where to setup `.env` variables

#### Local Development:
Create a file named `.env` in your **root directory** (the same folder as `main.py`).
```env
PINECONE_API_KEY=your_key_here
GEMINI_API_KEY=your_key_here
DATABASE_URL=...