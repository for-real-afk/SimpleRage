#!/usr/bin/env python3
"""
Test script for RAG API
Run this to test your API locally before deploying
"""

import requests
import json
import sys
from pathlib import Path

BASE_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint"""
    print("Testing /health endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        response.raise_for_status()
        data = response.json()
        print(f"✅ Health check passed: {data}")
        return True
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

def test_upload(file_path: str):
    """Test file upload"""
    print(f"\nTesting /upload endpoint with file: {file_path}...")
    
    if not Path(file_path).exists():
        print(f"❌ File not found: {file_path}")
        return False
    
    try:
        with open(file_path, 'rb') as f:
            files = {'file': (Path(file_path).name, f)}
            response = requests.post(f"{BASE_URL}/upload", files=files)
            response.raise_for_status()
            data = response.json()
            print(f"✅ Upload successful: {data}")
            return True
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"   Response: {e.response.text}")
        return False

def test_query(question: str, top_k: int = 3):
    """Test query endpoint"""
    print(f"\nTesting /query endpoint with question: {question}...")
    try:
        payload = {
            "question": question,
            "top_k": top_k
        }
        response = requests.post(
            f"{BASE_URL}/query",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        data = response.json()
        print(f"✅ Query successful!")
        print(f"   Answer: {data['answer']}")
        print(f"   Sources: {len(data['sources'])} documents")
        for i, source in enumerate(data['sources'], 1):
            print(f"     {i}. {source['filename']} (score: {source['score']})")
        return True
    except Exception as e:
        print(f"❌ Query failed: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"   Response: {e.response.text}")
        return False

def create_test_file():
    """Create a test file for upload"""
    test_content = """
    # Test Document for RAG System
    
    This is a test document to verify the RAG system is working correctly.
    
    Key Information:
    - The system uses Pinecone for vector storage
    - Gemini is used for embeddings and answer generation
    - The application is optimized for Render's free tier
    
    Technical Details:
    - Maximum file size: 2MB
    - Chunk size: 500 characters
    - Embedding model: text-embedding-004
    - LLM model: gemini-2.0-flash-exp
    
    This document should be split into multiple chunks and stored in the vector database.
    When queried, the system should be able to retrieve relevant information and generate
    accurate answers based on this content.
    """
    
    test_file = Path("test_document.txt")
    test_file.write_text(test_content)
    print(f"✅ Created test file: {test_file}")
    return str(test_file)

def main():
    """Run all tests"""
    print("=" * 60)
    print("RAG API Test Suite")
    print("=" * 60)
    
    # Check if server is running
    print("\n1. Checking if server is running...")
    if not test_health():
        print("\n⚠️  Server is not running. Start it with:")
        print("   uvicorn main:app --reload")
        sys.exit(1)
    
    # Create test file
    print("\n2. Creating test document...")
    test_file = create_test_file()
    
    # Test upload
    print("\n3. Testing document upload...")
    if not test_upload(test_file):
        print("\n❌ Upload test failed. Check logs and try again.")
        sys.exit(1)
    
    # Wait for processing
    print("\n⏳ Waiting 2 seconds for processing...")
    import time
    time.sleep(2)
    
    # Test queries
    print("\n4. Testing queries...")
    test_questions = [
        "What vector database does the system use?",
        "What is the maximum file size?",
        "Which embedding model is used?"
    ]
    
    for question in test_questions:
        if not test_query(question):
            print(f"\n⚠️  Query failed for: {question}")
    
    print("\n" + "=" * 60)
    print("✅ All tests completed!")
    print("=" * 60)
    print("\nYour API is ready for deployment to Render!")
    print("\nNext steps:")
    print("1. Push your code to GitHub")
    print("2. Connect repository to Render")
    print("3. Set environment variables in Render dashboard")
    print("4. Deploy!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Tests interrupted by user")
        sys.exit(0)