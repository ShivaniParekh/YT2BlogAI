import os
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from langchain_groq import ChatGroq
from youtube_transcript_api import YouTubeTranscriptApi
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from typing import TypedDict
from IPython.display import Image

def initialize_model(model_name: str):
    os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
    return ChatGroq(model=model_name)

class State(TypedDict):
    video_url: str
    transcript: str
    blog: str

def extract_transcript(state: State) -> State:
    if "video_url" not in state:
        raise KeyError("Missing 'video_url' in state.")
    
    video_url = state["video_url"].strip()
    # if video_url.startswith("https://www.youtube.com/watch?v="):
    #     video_id = state["video_url"].split("v=")[-1].split("&")[0]
    # elif video_url.startswith("https://youtu.be/"):
    #     video_id = video_url.split("/")[-1].split("?")[0]  # Extract last part after "/"
    video_id=""
    if "youtube.com/watch?v=" in video_url:
        video_id = video_url.split("v=")[-1].split("&")[0]  # Extract ID
        print
    elif "youtu.be/" in video_url:
        video_id = video_url.split("youtu.be/")[-1].split("?")[0]  # Extract ID

    if not video_id:
        raise ValueError("Invalid YouTube URL. Could not extract video ID.")

    
    # video_id = state["video_url"].split("v=")[-1].split("&")[0]
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    text_transcript = " ".join([t["text"] for t in transcript])
    state["transcript"] = text_transcript
    return state

def chunk_text(text: str, max_tokens: int = 500):
    words = text.split()
    return [" ".join(words[i:i + max_tokens]) for i in range(0, len(words), max_tokens)]

def generate_blog_section(chunk: str, llm) -> str:
    prompt_text = f"""
    You are a professional blog writer skilled in writing engaging, informative, and SEO-friendly articles.
    Convert the following YouTube transcript chunk into a well-structured blog section:
    {chunk}
    """
    return llm.invoke(prompt_text).content

def generate_blog(state: State, llm) -> State:
    transcript_chunks = chunk_text(state["transcript"])
    blog_sections = [generate_blog_section(chunk, llm) for chunk in transcript_chunks]
    state["blog"] = "\n\n".join(blog_sections)
    return state

def generate_graph(llm):
    builder = StateGraph(State)
    builder.add_node("extract_transcript", extract_transcript)
    builder.add_node("generate_blog", lambda state: generate_blog(state, llm))

    builder.add_edge(START, "extract_transcript")
    builder.add_edge("extract_transcript", "generate_blog")
    builder.add_edge("generate_blog", END)

    graph = builder.compile()  # Compile the LangGraph object
    graph_image = None

    try:
        graph_image = graph.get_graph().draw_mermaid_png()  # Generate image (if possible)
    except Exception as e:
        print(f"Warning: Could not generate graph image. Error: {e}")

    return graph, graph_image  # Now it always returns two values



def run_pipeline(video_url: str, model_name: str):
    llm = initialize_model(model_name)
    graph, graph_image = generate_graph(llm)  # Get both

    initial_state: State = {"video_url": video_url, "transcript": "", "blog": ""}
    print("✅ Graph Created. Invoking the pipeline...")

    final_state = graph.invoke(initial_state)  # Now graph is correct
    print("✅ Pipeline Execution Complete.")

    return final_state["blog"], graph_image  # Return final blog & graph image



# run_pipeline("https://www.youtube.com/watch?v=UV81LAb3x2g")s






    
