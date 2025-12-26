"""
CircularAI Support Chatbot - Web Interface
"""

import streamlit as st
from core.embeddings import EmbeddingService
from core.vector_store import VectorStore
from core.llm import LLMService
from core.search import HybridSearch
from agents.rag_agent import RAGAgent
import uuid
import traceback

# Page config
st.set_page_config(
    page_title="CircularAI Support Bot",
    page_icon="🤖",
    layout="centered"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        max-width: 800px;
    }
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .confidence-badge {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-size: 0.875rem;
        margin-top: 0.5rem;
    }
    .confidence-high {
        background-color: #d4edda;
        color: #155724;
    }
    .confidence-medium {
        background-color: #fff3cd;
        color: #856404;
    }
    .confidence-low {
        background-color: #f8d7da;
        color: #721c24;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "agent" not in st.session_state:
    with st.spinner("🔄 Initializing CircularAI Support Bot..."):
        try:
            st.write("🔧 Step 1: Initializing embeddings...")
            embedder = EmbeddingService()
            st.write("✅ Embeddings initialized successfully")
            
            st.write("🔧 Step 2: Initializing vector store...")
            vector_store = VectorStore()
            st.write("✅ Vector store initialized successfully")
            
            st.write("🔧 Step 3: Initializing LLM...")
            llm = LLMService()
            st.write("✅ LLM initialized successfully")
            
            st.write("🔧 Step 4: Initializing hybrid search...")
            search = HybridSearch(embedder, vector_store)
            st.write("✅ Search initialized successfully")
            
            st.write("🔧 Step 5: Initializing RAG agent...")
            st.session_state.agent = RAGAgent(embedder, vector_store, llm, search)
            st.write("✅ Agent initialized successfully")
            
            st.success("🎉 All components initialized! Bot is ready!")
            
        except Exception as e:
            st.error(f"❌ Failed to initialize bot: {e}")
            st.write("**Full error traceback:**")
            st.code(traceback.format_exc())
            st.stop()

# Header
st.title("🤖 CircularAI Support Bot")
st.markdown("Ask me anything about the CircularAI platform!")

# Sidebar
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This chatbot helps you with:
    - Platform navigation
    - Feature explanations
    - Step-by-step guides
    - Troubleshooting
    
    **Powered by:**
    - Azure OpenAI (GPT-4o)
    - Pinecone Vector DB
    - 1,200 indexed documents
    """)
    
    st.divider()
    
    st.header("📊 Stats")
    st.metric("Questions Asked", len([m for m in st.session_state.messages if m["role"] == "user"]))
    
    st.divider()
    
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Show metadata for assistant messages
        if message["role"] == "assistant" and "metadata" in message:
            confidence = message["metadata"]["confidence"]
            sources = message["metadata"]["sources"]
            
            # Confidence badge
            if confidence >= 0.8:
                badge_class = "confidence-high"
                badge_text = "High Confidence"
            elif confidence >= 0.6:
                badge_class = "confidence-medium"
                badge_text = "Medium Confidence"
            else:
                badge_class = "confidence-low"
                badge_text = "Low Confidence"
            
            st.markdown(f"""
            <div class="confidence-badge {badge_class}">
                {badge_text}: {confidence:.0%}
            </div>
            """, unsafe_allow_html=True)
            
            st.caption(f"📚 Sources: {', '.join(sources)}")

# Chat input
if prompt := st.chat_input("Ask a question about CircularAI..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get bot response
    with st.chat_message("assistant"):
        with st.spinner("🤔 Thinking..."):
            try:
                response = st.session_state.agent.chat(
                    prompt,
                    session_id=st.session_state.session_id
                )
                
                # Display answer
                st.markdown(response["answer"])
                
                # Display metadata
                confidence = response["confidence"]
                sources = response["sources"]
                
                if confidence >= 0.8:
                    badge_class = "confidence-high"
                    badge_text = "High Confidence"
                elif confidence >= 0.6:
                    badge_class = "confidence-medium"
                    badge_text = "Medium Confidence"
                else:
                    badge_class = "confidence-low"
                    badge_text = "Low Confidence"
                
                st.markdown(f"""
                <div class="confidence-badge {badge_class}">
                    {badge_text}: {confidence:.0%}
                </div>
                """, unsafe_allow_html=True)
                
                st.caption(f"📚 Sources: {', '.join(sources)}")
                
                # Save to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response["answer"],
                    "metadata": {
                        "confidence": confidence,
                        "sources": sources
                    }
                })
                
            except Exception as e:
                st.error(f"❌ Error: {e}")
                st.code(traceback.format_exc())

# Footer
st.divider()
st.caption("💡 Tip: Ask follow-up questions for more detailed information!")# Force rebuild
