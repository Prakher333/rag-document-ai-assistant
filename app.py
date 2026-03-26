import streamlit as st
import os
from rag import RAGPipeline

st.set_page_config(page_title="Mining Document AI Assistant", page_icon="⛏️", layout="wide")

st.title("⛏️ Mining Document AI Assistant")

# Add footer
st.markdown("""
<style>
.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: transparent;
    color: gray;
    text-align: center;
    padding: 10px;
}
</style>
<div class="footer">Built using Streamlit + LangChain + HuggingFace</div>
""", unsafe_allow_html=True)

# Initialize Session State
if "rag" not in st.session_state:
    st.session_state.rag = None
if "messages" not in st.session_state:
    st.session_state.messages = []

@st.cache_resource
def get_rag_pipeline():
    """Load the HuggingFace models only once using Streamlit cache"""
    return RAGPipeline()

# --- UI Sidebar ---
with st.sidebar:
    st.header("📄 Document Upload")
    pdf_file = st.file_uploader("Upload your Mining Document (PDF)", type=["pdf"])
    
    if st.button("Process Document"):
        if pdf_file is not None:
            with st.spinner("Processing document... Please wait ⏳"):
                try:
                    if st.session_state.rag is None:
                        st.session_state.rag = get_rag_pipeline()
                    st.session_state.rag.load_and_process_pdf(pdf_file)
                    
                    st.success("✅ Document processed successfully! Embeddings are ready.")
                except Exception as e:
                    st.error(f"❌ Error during processing: {e}")
        else:
            st.warning("⚠️ Please upload a PDF file first.")

# --- Chat Interface ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("context"):
            with st.expander("🔍 Show retrieved context"):
                for i, doc in enumerate(message["context"]):
                    st.info(f"**Chunk {i+1}:**\n{doc.page_content}")

# Chat input box
query = st.chat_input("Ask a question from the document...")

if query:
    if st.session_state.rag is None or st.session_state.rag.vector_db is None:
        st.warning("⚠️ Please upload and process a PDF document first before asking questions.")
    else:
        # Add and display user message
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)
            
        # Display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Generating answer... 🤖"):
                try:
                    response = st.session_state.rag.generate_answer(query)
                    answer = response["answer"]
                    context_docs = response["context"]
                    
                    st.markdown(answer)
                    
                    with st.expander("🔍 Show retrieved context"):
                        for i, doc in enumerate(context_docs):
                            st.info(f"**Chunk {i+1}:**\n{doc.page_content}")
                            
                    # Add assistant msg to state
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "context": context_docs
                    })
                except Exception as e:
                    st.error(f"❌ Error generating answer: {e}")
