"""
Interactive chat with RAG agent
"""

import sys
sys.path.append('.')

from core.embeddings import EmbeddingService
from core.vector_store import VectorStore
from core.llm import LLMService
from core.search import HybridSearch
from agents.rag_agent import RAGAgent
from utils.logging_config import get_logger
import uuid

logger = get_logger(__name__)


def print_header():
    """Print chat header"""
    print("\n" + "=" * 80)
    print("INTERACTIVE RAG AGENT CHAT")
    print("=" * 80)
    print("\nInitializing agent...")


def print_ready():
    """Print ready message"""
    print("\n✅ Agent ready!")
    print("\nType your questions below (or 'quit' to exit)")
    print("=" * 80 + "\n")


def main():
    # Print header
    print_header()
    
    # Initialize components
    try:
        embedder = EmbeddingService()
        vector_store = VectorStore()
        llm = LLMService()
        search = HybridSearch(embedder, vector_store)
        agent = RAGAgent(embedder, vector_store, llm, search)
    except Exception as e:
        print(f"\n❌ Failed to initialize agent: {e}")
        logger.exception("Initialization error")
        return 1
    
    # Show greeting
    print_ready()
    greeting = agent.get_greeting()
    print(f"Agent: {greeting}\n")
    
    # Generate unique session ID
    session_id = str(uuid.uuid4())
    
    # Chat loop
    while True:
        try:
            # Get user input
            user_input = input("You: ").strip()
            
            # Check for exit
            if user_input.lower() in ['quit', 'exit', 'bye', 'q']:
                print("\nAgent: Goodbye! Have a great day!\n")
                break
            
            # Skip empty input
            if not user_input:
                continue
            
            # Get response from agent
            print("\nAgent: ", end="", flush=True)
            
            response = agent.chat(user_input, session_id=session_id)
            
            # Print answer
            print(response['answer'])
            
            # Print metadata (smaller, less intrusive)
            print(f"\n[Confidence: {response['confidence']:.0%} | Sources: {', '.join(response['sources'])}]")
            print()
            
        except KeyboardInterrupt:
            print("\n\nAgent: Goodbye!\n")
            break
        
        except Exception as e:
            print(f"\n❌ Error: {e}\n")
            logger.exception("Chat error")
            continue
    
    return 0


if __name__ == "__main__":
    sys.exit(main())