"""
LangGraph node definitions
"""

from typing import Dict
from typing_extensions import TypedDict
from typing import Annotated, List
import operator


class AgentState(TypedDict):
    """State for LangGraph agent"""
    messages: Annotated[List[Dict], operator.add]
    query: str
    retrieved_chunks: List[Dict]
    semantic_results: List[Dict]
    bm25_results: List[Dict]
    hybrid_results: List[Dict]
    final_answer: str
    sources: List[str]
    session_id: str
    iteration: int