import streamlit as st
from bloggenerator import run_pipeline

def main():
    st.title("🎥 YouTube Blog Generator")
    st.write("Enter a YouTube video URL to extract its transcript and generate a blog post.")
    
    
    # User input for YouTube URL
    video_url = st.text_input("Enter the YouTube video URL:")

    # Model selection
    model_name = st.selectbox("Select Groq Model:", ["qwen-2.5-32b", "gemma-7b", "mixtral-8x7b"])

    if st.button("Generate Blog") and video_url:
        st.write("⏳ Generating blog... Please wait.")

        # Run pipeline
        final_blog, graph_image = run_pipeline(video_url, model_name,)

        # Display the graph output
        st.subheader("LangGraph Workflow:")
        st.image(graph_image, caption="Graph Execution Flow", use_container_width=True)

        # Display the blog output
        st.subheader("Generated Blog:")
        st.write(final_blog)

if __name__ == "__main__":
    main()