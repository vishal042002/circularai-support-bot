"""
Setup Pinecone index with correct dimensions
"""

import sys
sys.path.append('.')

from pinecone import Pinecone, ServerlessSpec
from config.settings import config
from utils.logging_config import get_logger

logger = get_logger(__name__)


def main():
    print("\n" + "=" * 80)
    print("PINECONE INDEX SETUP")
    print("=" * 80)
    
    pc = Pinecone(api_key=config.pinecone.api_key)
    index_name = config.pinecone.index_name
    dimension = config.embedding.dimension
    
    # Check existing indexes
    existing = pc.list_indexes().names()
    
    if index_name in existing:
        print(f"\n⚠️  Index '{index_name}' already exists")
        
        # Check dimensions
        index_info = pc.describe_index(index_name)
        if index_info.dimension != dimension:
            print(f"\n❌ Dimension mismatch!")
            print(f"   Existing: {index_info.dimension}")
            print(f"   Required: {dimension}")
            
            response = input("\nDelete and recreate index? (yes/no): ")
            if response.lower() == 'yes':
                print(f"\nDeleting index '{index_name}'...")
                pc.delete_index(index_name)
                print("✅ Index deleted")
            else:
                print("\n❌ Cannot proceed with mismatched dimensions")
                return 1
        else:
            print(f"✅ Existing index has correct dimensions ({dimension})")
            return 0
    
    # Create new index
    print(f"\nCreating index '{index_name}'...")
    print(f"  Dimension: {dimension}")
    print(f"  Metric: {config.pinecone.metric}")
    print(f"  Cloud: {config.pinecone.cloud}")
    print(f"  Region: {config.pinecone.region}")
    
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric=config.pinecone.metric,
        spec=ServerlessSpec(
            cloud=config.pinecone.cloud,
            region=config.pinecone.region
        )
    )
    
    print("\n✅ Index created successfully")
    print("=" * 80 + "\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())