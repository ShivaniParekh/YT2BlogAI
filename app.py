import streamlit as st
import time
import nltk
from nltk.tokenize import sent_tokenize
from bloggenerator import generate_graph, initialize_model

# Setup NLTK
nltk.download('punkt', quiet=True)

# 1. PAGE CONFIG & STYLING
st.set_page_config(page_title="YouTube Blog Generator", page_icon="🎥", layout="wide")

st.markdown("""
    <style>
        .stTitle { text-align: center; color: #3B82F6; font-weight: bold; }
        .stButton > button { width: 100%; border-radius: 8px; font-weight: bold; }
        .main-container { background-color: #f8f9fa; padding: 20px; border-radius: 15px; }
    </style>
""", unsafe_allow_html=True)

# 2. SESSION STATE INITIALIZATION
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(time.time())
if "graph_state" not in st.session_state:
    st.session_state.graph_state = None
if "finalized" not in st.session_state:
    st.session_state.finalized = False

# 3. SIDEBAR & TOOLS
with st.sidebar:
    st.title("⚙️ Settings")
    model_name = st.selectbox("Select AI Model", ["llama-3.1-8b-instant", "qwen/qwen3-32b"])
    if st.button("🗑️ Reset Application"):
        st.session_state.clear()
        st.rerun()

# Cache the graph so memory persists
@st.cache_resource
def load_graph(_model):
    llm = initialize_model(_model)
    return generate_graph(llm)

graph = load_graph(model_name)
config = {"configurable": {"thread_id": st.session_state.thread_id}}

# 4. MAIN UI
st.markdown("<h1 class='stTitle'>🎥 YouTube Blog Generator</h1>", unsafe_allow_html=True)

# --- MODE A: FINALIZED VIEW ---
if st.session_state.finalized:
    current_state = graph.get_state(config).values
    final_blog = current_state.get("blog")
    
    st.success("🎉 Finalized Blog Content")
    st.markdown(final_blog)
    
    # Simple NLTK summary
    sentences = sent_tokenize(final_blog)
    summary = " ".join(sentences[:2])
    with st.expander("📌 Quick Summary"):
        st.write(summary)
        
    if st.button("➕ Create New Blog"):
        st.session_state.clear()
        st.rerun()

# --- MODE B: DRAFTING / FEEDBACK VIEW ---
elif st.session_state.graph_state:
    current_state = st.session_state.graph_state.values
    blog_draft = current_state.get("blog")

    st.subheader("📝 Blog Draft")
    st.info("Review the draft below and provide feedback if needed.")
    st.markdown(blog_draft)
    
    st.divider()
    
    # Feedback Area
    feedback = st.text_area("What would you like to improve?", placeholder="e.g. Add more bullet points, make it funnier...")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Refine Blog"):
            if feedback:
                graph.update_state(config, {"feedback": feedback})
                with st.spinner("Updating blog..."):
                    for event in graph.stream(None, config): pass
                st.session_state.graph_state = graph.get_state(config)
                st.rerun()
            else:
                st.warning("Please enter feedback text first.")
                
    with col2:
        if st.button("✅ Looks Great! Finalize"):
            graph.update_state(config, {"feedback": "Accepted"})
            with st.spinner("Finalizing..."):
                for event in graph.stream(None, config): pass
            st.session_state.finalized = True
            st.rerun()

# --- MODE C: INITIAL INPUT VIEW ---
else:
    video_url = st.text_input("Paste YouTube URL here:", placeholder="https://www.youtube.com/watch?v=...")
    
    if st.button("🚀 Generate Blog Draft"):
        if video_url:
            initial_input = {"video_url": video_url, "transcript": "", "blog": "", "feedback": "", "final_blog": ""}
            with st.spinner("Analyzing video and writing draft..."):
                # Run graph until it hits the interrupt_before "human_feedback"
                for event in graph.stream(initial_input, config):
                    pass
            st.session_state.graph_state = graph.get_state(config)
            st.rerun()
        else:
            st.error("Please enter a valid URL.")