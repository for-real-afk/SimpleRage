"""
Test script to find correct Gemini model names
Run this to see what models are available with your API key
"""

import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

# Configure
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("❌ GEMINI_API_KEY not found in .env file")
    exit(1)

genai.configure(api_key=api_key)

print("=" * 60)
print("Checking Available Gemini Models")
print("=" * 60)
print()

# List embedding models
print("📊 EMBEDDING MODELS (for embedContent):")
print("-" * 60)
embedding_models = []
for model in genai.list_models():
    if 'embedContent' in model.supported_generation_methods:
        embedding_models.append(model.name)
        print(f"  ✓ {model.name}")
        
if not embedding_models:
    print("  ❌ No embedding models found")
else:
    print(f"\n  Total: {len(embedding_models)} models")

print()

# List generation models  
print("💬 TEXT GENERATION MODELS (for generateContent):")
print("-" * 60)
generation_models = []
for model in genai.list_models():
    if 'generateContent' in model.supported_generation_methods:
        generation_models.append(model.name)
        print(f"  ✓ {model.name}")

if not generation_models:
    print("  ❌ No generation models found")
else:
    print(f"\n  Total: {len(generation_models)} models")

print()
print("=" * 60)
print("RECOMMENDED FOR YOUR main.py:")
print("=" * 60)

if embedding_models:
    print(f"\nEmbedding model to use:")
    print(f'  model="{embedding_models[0]}"')
    
if generation_models:
    # Find a flash model
    flash_models = [m for m in generation_models if 'flash' in m.lower()]
    if flash_models:
        print(f"\nGeneration model to use:")
        print(f'  model = genai.GenerativeModel("{flash_models[0]}")')
    else:
        print(f"\nGeneration model to use:")
        print(f'  model = genai.GenerativeModel("{generation_models[0]}")')

print()
print("=" * 60)

# Test embedding
if embedding_models:
    print("\n🧪 Testing embedding generation...")
    try:
        result = genai.embed_content(
            model=embedding_models[0],
            content="This is a test",
            task_type="retrieval_document"
        )
        print(f"✅ Embedding successful!")
        print(f"   Dimensions: {len(result['embedding'])}")
        print(f"   First few values: {result['embedding'][:5]}")
    except Exception as e:
        print(f"❌ Embedding failed: {e}")

# Test generation
if generation_models:
    print("\n🧪 Testing text generation...")
    try:
        flash_models = [m for m in generation_models if 'flash' in m.lower()]
        test_model = flash_models[0] if flash_models else generation_models[0]
        
        model = genai.GenerativeModel(test_model)
        response = model.generate_content("Say hello in one word")
        print(f"✅ Generation successful!")
        print(f"   Response: {response.text}")
    except Exception as e:
        print(f"❌ Generation failed: {e}")

print()
print("=" * 60)
print("Run this script to see the exact model names to use!")
print("=" * 60)