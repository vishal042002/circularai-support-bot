"""
Test RAG agent with sample questions
"""

import sys
sys.path.append('.')

from core.embeddings import EmbeddingService
from core.vector_store import VectorStore
from core.llm import LLMService
from core.search import HybridSearch
from agents.rag_agent import RAGAgent
from utils.logging_config import get_logger

logger = get_logger(__name__)


def main():
    print("\n" + "=" * 80)
    print("RAG AGENT TEST")
    print("=" * 80)
    
    # Initialize components
    print("\nInitializing components...")
    embedder = EmbeddingService()
    vector_store = VectorStore()
    llm = LLMService()
    search = HybridSearch(embedder, vector_store)
    
    # Create agent
    agent = RAGAgent(embedder, vector_store, llm, search)
    
    # Test questions
    test_questions = [
        "What is CircularAI?",
        "How do I log in to the platform?",
        "What is Cai?",
        "How do I create a mission?",
        "What are the system requirements?"
    ]
    
    print("\n" + "=" * 80)
    print("TESTING WITH SAMPLE QUESTIONS")
    print("=" * 80)
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n[Question {i}] {question}")
        print("-" * 80)
        
        try:
            response = agent.chat(question, session_id=f"test_session_{i}")
            
            print(f"\nAnswer:\n{response['answer']}")
            print(f"\nConfidence: {response['confidence']:.2%}")
            print(f"Sources: {', '.join(response['sources'])}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            logger.exception(f"Error processing question: {question}")
    
    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80 + "\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())