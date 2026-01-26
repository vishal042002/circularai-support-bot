"""
Main RAG Agent with LangGraph orchestration
"""

from typing import Dict, List
import yaml
from pathlib import Path
from langgraph.graph import StateGraph, END

# Try new import first, fall back to old
try:
    from langgraph.checkpoint.sqlite import SqliteSaver
except ImportError:
    try:
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver as SqliteSaver
    except ImportError:
        # Fallback: no checkpointing
        SqliteSaver = None

from agents.graph_nodes import AgentState
from core.embeddings import EmbeddingService
from core.vector_store import VectorStore
from core.llm import LLMService
from core.search import HybridSearch
from config.settings import config
from utils.logging_config import get_logger

logger = get_logger(__name__)


class RAGAgent:
    """
    Production RAG Agent with:
    - Hybrid search (semantic + BM25)
    - LangGraph orchestration
    - Conversation memory
    - Configurable prompts
    """
    
    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: VectorStore,
        llm_service: LLMService,
        search_engine: HybridSearch
    ):
        self.embedder = embedding_service
        self.vector_store = vector_store
        self.llm = llm_service
        self.search = search_engine
        
        # Load prompts
        self.prompts = self._load_prompts()
        
        # Initialize LangGraph
        self.graph = self._build_graph()
        
        logger.info("RAG Agent initialized")
    
    def _load_prompts(self) -> Dict:
        """Load prompts from YAML"""
        prompt_file = Path("config/prompts.yaml")
        if prompt_file.exists():
            with open(prompt_file, 'r') as f:
                return yaml.safe_load(f)
        else:
            # Default prompts
            return {
                'default': {
                    'system_prompt': (
                        "You are a helpful AI assistant for the CircularAI platform. "
                        "Answer questions clearly using the provided context. "
                        "Use bullet points for step-by-step instructions. "
                        "Be concise but complete."
                    )
                },
                'greeting': {
                    'message': "Hello! I'm here to help you with the CircularAI platform. How can I assist you today?"
                }
            }
    
    def _build_graph(self):
        """Build LangGraph workflow"""
        # Create checkpointer if available
        checkpointer = None
        if SqliteSaver is not None:
            try:
                checkpoint_db = config.checkpoint_db
                Path(checkpoint_db).parent.mkdir(parents=True, exist_ok=True)
                
                # Try to create SqliteSaver
                import sqlite3
                conn = sqlite3.connect(checkpoint_db, check_same_thread=False)
                checkpointer = SqliteSaver(conn)
                logger.info("Checkpointing enabled")
            except Exception as e:
                logger.warning(f"Could not enable checkpointing: {e}")
                checkpointer = None
        
        # Define workflow
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("semantic_search", self._semantic_search_node)
        workflow.add_node("bm25_search", self._bm25_search_node)
        workflow.add_node("hybrid_fusion", self._hybrid_fusion_node)
        workflow.add_node("generate_answer", self._generate_answer_node)
        
        # Define edges
        workflow.set_entry_point("semantic_search")
        workflow.add_edge("semantic_search", "bm25_search")
        workflow.add_edge("bm25_search", "hybrid_fusion")
        workflow.add_edge("hybrid_fusion", "generate_answer")
        workflow.add_edge("generate_answer", END)
        
        # Compile with checkpointing if available
        if checkpointer:
            return workflow.compile(checkpointer=checkpointer)
        else:
            return workflow.compile()
    
    # ==================== GRAPH NODES ====================
    
    def _semantic_search_node(self, state: AgentState) -> Dict:
        """Semantic search node"""
        query = state["query"]
        results = self.search.semantic_search(query)
        return {"semantic_results": results}
    
    def _bm25_search_node(self, state: AgentState) -> Dict:
        """BM25 search node"""
        query = state["query"]
        results = self.search.bm25_search(query)
        return {"bm25_results": results}
    
    def _hybrid_fusion_node(self, state: AgentState) -> Dict:
        """Hybrid fusion node"""
        hybrid_results = self.search.hybrid_search(
            query=state["query"],
            semantic_results=state["semantic_results"],
            bm25_results=state["bm25_results"]
        )
        return {
            "hybrid_results": hybrid_results,
            "retrieved_chunks": hybrid_results
        }
    
    def _generate_answer_node(self, state: AgentState) -> Dict:
        """Generate answer node"""
        query = state["query"]
        chunks = state["retrieved_chunks"]

        # DEBUG: Log retrieved chunks scores
        logger.info(f"Query: '{query}'")
        logger.info(f"Retrieved {len(chunks)} chunks:")
        for i, chunk in enumerate(chunks[:3], 1):
            logger.info(
                f"  [{i}] Hybrid={chunk.get('hybrid_score', 0):.3f} "
                f"(Sem={chunk.get('semantic_score', 0):.3f}, "
                f"BM25={chunk.get('bm25_score', 0):.3f}) - "
                f"{chunk['metadata'].get('section_h1', 'Unknown')}"
            )

        # Build context
        context = self._build_context(chunks)

        # Generate answer
        answer = self._generate_with_llm(query, context)

        # Extract sources
        sources = list(set([
            chunk["metadata"].get("section_h1", "Unknown")
            for chunk in chunks[:3]
        ]))

        return {
            "final_answer": answer,
            "sources": sources,
            "messages": [{"role": "assistant", "content": answer}]
        }
    
    # ==================== HELPER METHODS ====================
    
    def _build_context(self, chunks: List[Dict]) -> str:
        """Build context from retrieved chunks"""
        context_parts = []
        
        for i, chunk in enumerate(chunks, 1):
            # Add section info
            section = chunk["metadata"].get("section_h1", "")
            subsection = chunk["metadata"].get("section_h2", "")
            
            header = f"[Source {i}"
            if section:
                header += f" - {section}"
                if subsection:
                    header += f" > {subsection}"
            header += "]"
            
            context_parts.append(f"{header}\n{chunk['text']}")
        
        return "\n\n---\n\n".join(context_parts)
    
    def _generate_with_llm(self, query: str, context: str) -> str:
        """Generate answer using LLM"""
        system_prompt = self.prompts['default']['system_prompt']
        
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
            }
        ]
        
        return self.llm.generate(messages, temperature=0.7, max_tokens=800)

    # ==================== PUBLIC INTERFACE ====================
    
    def chat(
        self,
        message: str,
        session_id: str = "default_session"
    ) -> Dict:
        """
        Main chat interface

        Args:
            message: User's question
            session_id: Session identifier for memory

        Returns:
            Dict with answer and sources
        """
        # Invoke graph
        result = self.graph.invoke(
            {
                "query": message,
                "messages": [{"role": "user", "content": message}],
                "session_id": session_id,
                "iteration": 0
            },
            config={"configurable": {"thread_id": session_id}}
        )

        return {
            "answer": result["final_answer"],
            "sources": result["sources"],
            "retrieved_chunks": result["retrieved_chunks"]
        }
    
    def get_greeting(self) -> str:
        """Get greeting message"""
        return self.prompts.get('greeting', {}).get('message', "Hello! How can I help you?")