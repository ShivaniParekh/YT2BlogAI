import streamlit as st
import time
import random
import nltk
from nltk.tokenize import sent_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
from bloggenerator import run_pipeline

# Download required NLTK data
nltk.download('punkt')
nltk.download('vader_lexicon')
nltk.download('punkt_tab')

# Streamlit Theme Configuration
st.set_page_config(
    page_title="YouTube Blog Generator",
    page_icon="📄",
    layout="wide"
)

# Custom CSS for Styling
st.markdown("""
    <style>
        body { background-color: #f5f7fa; }
        .stTitle { text-align: center; font-size: 2.5em; font-weight: bold; color: #3B82F6; }
        .stTextInput > label { font-size: 1.2em; font-weight: bold; }
        .stButton > button { font-size: 1.1em; background: linear-gradient(90deg, #3B82F6, #9333EA); color: white; border-radius: 8px; }
        .stMetric { font-size: 1.3em; font-weight: bold; }
        .stSidebar { background-color: #111827; color: white; padding: 20px; border-radius: 10px; }
        .stSidebar select { font-size: 1.1em; }
        .stExpander { background-color: #f8f9fa; border-radius: 10px; padding: 10px; }
        .stMarkdown { font-size: 1.1em; line-height: 1.6; }
    </style>
""", unsafe_allow_html=True)

# Summary Function
def summarize_text(blog_content):
    """Generate a short summary using NLTK."""
    sentences = sent_tokenize(blog_content)
    return " ".join(sentences[:2])

# Engagement Score Function
def generate_engagement_score(blog_content):
    """Rate blog engagement based on sentiment analysis using NLTK's Vader."""
    sia = SentimentIntensityAnalyzer()
    sentiment = sia.polarity_scores(blog_content)
    score = int((sentiment["compound"] + 1) * 50)  # Convert to 0-100 scale
    return score

def main():
    # Title & Subtitle
    st.markdown("<h1 class='stTitle'>🎥 YouTube Blog Generator</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 18px;'>Transform YouTube videos into structured, SEO-optimized blogs effortlessly.</p>", unsafe_allow_html=True)
    st.divider()

    # Layout: Input fields with columns
    col1, col2 = st.columns([3, 1])

    # Video URL Input
    with col1:
        st.markdown("### 🔗 Enter YouTube Video URL")
        video_url = st.text_input("Paste the YouTube video URL:", placeholder="https://www.youtube.com/watch?v=example")

    # Sidebar for Model Selection
    with st.sidebar:
        st.markdown("## ⚙️ Settings")
        model_name = st.selectbox("Select AI Model", ["qwen-2.5-32b", "gemma2-9b-it", "mixtral-8x7b-32768"])
        st.markdown("---")
        st.markdown("💡 *Tip: Choose a powerful model for better quality blogs.*")

    # Generate Blog Button with Animated Progress Bar
    if st.button("🚀 Generate Blog") and video_url:
        st.markdown("⏳ **Processing...** Please wait while we generate your blog.")

        progress_bar = st.progress(0)
        progress_text = st.empty()

        for percent in range(0, 101, 10):
            time.sleep(0.2)  # Simulating processing time
            progress_bar.progress(percent)
            progress_text.text(f"Processing... {percent}%")

        progress_bar.empty()
        progress_text.empty()

        final_blog = run_pipeline(video_url, model_name)  # Call main pipeline function

        # Generate additional features
        summary = summarize_text(final_blog)
        engagement_score = generate_engagement_score(final_blog)

        # Display Engagement Score
        st.metric(label="📊 Engagement Score", value=f"{engagement_score}/100", delta=random.randint(-5, 5))

        # TL;DR Summary
        with st.expander("📌 **TL;DR (Summary)**", expanded=True):
            st.markdown(f"**{summary}**")

        # Full Generated Blog
        with st.expander("📜 **Generated Blog** (Click to expand)", expanded=True):
            st.markdown(final_blog, unsafe_allow_html=True)

        # Download Button for Markdown File
        st.download_button(
            label="📥 Download as Markdown",
            data=final_blog,
            file_name="generated_blog.md",
            mime="text/markdown"
        )

if __name__ == "__main__":
    main()
