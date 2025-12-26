"""
Test all API connections
"""

import sys
sys.path.append('.')

import os
from dotenv import load_dotenv

load_dotenv()

def check_env_vars():
    """Check if all required environment variables are set"""
    print("\n" + "=" * 80)
    print("CHECKING ENVIRONMENT VARIABLES")
    print("=" * 80)
    
    required_vars = {
        'AZURE_OPENAI_ENDPOINT': 'Azure OpenAI Endpoint',
        'AZURE_OPENAI_API_KEY': 'Azure OpenAI API Key',
        'AZURE_DEPLOYMENT_NAME': 'Azure Deployment Name',
        'AZURE_VISION_DEPLOYMENT_NAME': 'Azure Vision Deployment',
        'AZURE_EMBEDDING_DEPLOYMENT': 'Azure Embedding Deployment',
        'PINECONE_API_KEY': 'Pinecone API Key',
    }
    
    optional_vars = {
        'LANGSMITH_API_KEY': 'LangSmith API Key (Optional)',
    }
    
    all_set = True
    for var, name in required_vars.items():
        value = os.getenv(var)
        if value:
            masked = value[:8] + "..." + value[-4:] if len(value) > 12 else "***"
            print(f"✅ {name}: {masked}")
        else:
            print(f"❌ {name}: NOT SET")
            all_set = False
    
    # Check optional
    for var, name in optional_vars.items():
        value = os.getenv(var)
        if value:
            masked = value[:8] + "..." + value[-4:] if len(value) > 12 else "***"
            print(f"✅ {name}: {masked}")
        else:
            print(f"⏭️  {name}: Not set (optional)")
    
    print("=" * 80)
    
    if not all_set:
        print("\n❌ Some environment variables are missing!")
        print("Please check your .env file")
        return False
    
    print("\n✅ All required environment variables are set")
    return True


def test_azure_openai():
    """Test Azure OpenAI with minimal call"""
    print("\n" + "=" * 80)
    print("TEST 1: AZURE OPENAI (Text Generation)")
    print("=" * 80)
    
    try:
        from openai import AzureOpenAI
        
        client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version="2024-08-01-preview",
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        )
        
        print("\n📡 Sending test request to Azure OpenAI...")
        
        response = client.chat.completions.create(
            model=os.getenv("AZURE_DEPLOYMENT_NAME"),
            messages=[
                {"role": "user", "content": "Say 'Hello'"}
            ],
            max_tokens=5,
            temperature=0
        )
        
        result = response.choices[0].message.content
        tokens_used = response.usage.total_tokens
        
        print(f"\n✅ Azure OpenAI is working!")
        print(f"   Response: {result}")
        print(f"   Tokens used: {tokens_used}")
        print(f"   Cost: ~$0.00001")
        
        return True
    
    except Exception as e:
        print(f"\n❌ Azure OpenAI test failed!")
        print(f"   Error: {str(e)[:200]}")
        return False


def test_azure_vision():
    """Test Azure GPT-4 Vision"""
    print("\n" + "=" * 80)
    print("TEST 2: AZURE GPT-4 VISION (Optional)")
    print("=" * 80)
    
    response = input("\nTest GPT-4 Vision? This will cost ~$0.01 (yes/no): ")
    
    if response.lower() != 'yes':
        print("⏭️  Skipping vision test")
        return True
    
    try:
        from openai import AzureOpenAI
        import base64
        from PIL import Image
        import io
        
        img = Image.new('RGB', (100, 100), color='red')
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        img_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version="2024-08-01-preview",
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        )
        
        print("\n📡 Sending test image to GPT-4 Vision...")
        
        response = client.chat.completions.create(
            model=os.getenv("AZURE_VISION_DEPLOYMENT_NAME"),
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What color is this image?"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{img_data}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=20
        )
        
        result = response.choices[0].message.content
        
        print(f"\n✅ GPT-4 Vision is working!")
        print(f"   Response: {result}")
        print(f"   Cost: ~$0.01")
        
        return True
    
    except Exception as e:
        print(f"\n❌ GPT-4 Vision test failed!")
        print(f"   Error: {str(e)[:200]}")
        return False


def test_azure_embeddings():
    """Test Azure embeddings"""
    print("\n" + "=" * 80)
    print("TEST 3: AZURE EMBEDDINGS")
    print("=" * 80)
    
    try:
        from openai import AzureOpenAI
        
        client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version="2024-08-01-preview",
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        )
        
        print("\n📡 Testing Azure embeddings...")
        
        response = client.embeddings.create(
            input=["This is a test"],
            model=os.getenv("AZURE_EMBEDDING_DEPLOYMENT")
        )
        
        embedding = response.data[0].embedding
        dimension = len(embedding)
        
        print(f"\n✅ Azure Embeddings is working!")
        print(f"   Deployment: {os.getenv('AZURE_EMBEDDING_DEPLOYMENT')}")
        print(f"   Dimension: {dimension}")
        print(f"   Cost: ~$0.00001")
        
        return True
    
    except Exception as e:
        print(f"\n❌ Azure Embeddings test failed!")
        print(f"   Error: {str(e)[:200]}")
        return False


def test_pinecone():
    """Test Pinecone connection"""
    print("\n" + "=" * 80)
    print("TEST 4: PINECONE")
    print("=" * 80)
    
    try:
        from pinecone import Pinecone
        
        print("\n📡 Connecting to Pinecone...")
        
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        indexes = pc.list_indexes()
        
        print(f"\n✅ Pinecone connection successful!")
        print(f"   Existing indexes: {indexes.names()}")
        print(f"   Cost: $0 (free check)")
        
        return True
    
    except Exception as e:
        print(f"\n❌ Pinecone test failed!")
        print(f"   Error: {str(e)[:200]}")
        return False


def test_langsmith():
    """Test LangSmith (optional)"""
    print("\n" + "=" * 80)
    print("TEST 5: LANGSMITH (Optional)")
    print("=" * 80)
    
    api_key = os.getenv("LANGSMITH_API_KEY")
    
    if not api_key:
        print("\n⏭️  LangSmith not configured (optional)")
        return True
    
    try:
        from langsmith import Client
        
        print("\n📡 Testing LangSmith connection...")
        
        client = Client(api_key=api_key)
        list(client.list_projects(limit=1))
        
        print(f"\n✅ LangSmith is working!")
        print(f"   Cost: $0 (free tier)")
        
        return True
    
    except Exception as e:
        print(f"\n❌ LangSmith test failed!")
        print(f"   Error: {str(e)[:200]}")
        print("\n   LangSmith is optional - you can continue without it")
        return True


def main():
    print("\n" + "🚀" * 40)
    print("QUICK RESOURCE CHECK")
    print("This will cost < $0.02 total")
    print("🚀" * 40)
    
    # Check environment variables
    if not check_env_vars():
        return 1
    
    # Run tests
    results = {
        "Azure OpenAI": test_azure_openai(),
        "Azure Vision": test_azure_vision(),
        "Azure Embeddings": test_azure_embeddings(),
        "Pinecone": test_pinecone(),
        "LangSmith": test_langsmith()
    }
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    for service, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {service}")
    
    all_critical_passed = all([
        results["Azure OpenAI"],
        results["Azure Embeddings"],
        results["Pinecone"]
    ])
    
    print("=" * 80)
    
    if all_critical_passed:
        print("\n🎉 ALL CRITICAL TESTS PASSED!")
        print("\nYou're ready to proceed with:")
        print("  1. python scripts/02_setup_pinecone.py")
        print("  2. python scripts/03_ingest_document.py data/raw/YOUR_FILE.docx")
        print("  3. python scripts/04_test_agent.py")
        return 0
    else:
        print("\n❌ SOME CRITICAL TESTS FAILED")
        print("\nPlease fix the issues above before proceeding")
        return 1


if __name__ == "__main__":
    sys.exit(main())