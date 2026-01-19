import os
import heapq
import logging
from typing import TypedDict, List

import nltk
from dotenv import load_dotenv
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from youtube_transcript_api import YouTubeTranscriptApi, RequestBlocked

from langchain_groq import ChatGroq
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph, END

# Load environment variables
load_dotenv()

# Logging Configuration
logger = logging.getLogger("blog_gen")

def setup_nltk():
    """Download required NLTK resources if not already present."""
    resources = [
        ('corpora/stopwords', 'stopwords'),
        ('tokenizers/punkt', 'punkt'),
        ('tokenizers/punkt_tab', 'punkt_tab')
    ]
    for path, name in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(name, quiet=True)

setup_nltk()
memory = MemorySaver()

class State(TypedDict):
    """Represents the state of the blog generation graph."""
    video_url: str
    transcript: str
    blog: str
    feedback: str
    final_blog: str
    tone: str
    seo_meta: str


def initialize_model(model_name:str):
    """
    Initializes the ChatGroq model.
    
    Args:
        model_name (str): The name of the Groq model to use.
    Returns:
        ChatGroq: An instance of the LLM.
    """
    logger.info("Initialize model")
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found in environment variables.")
    logger.info("Model initialized")
    return ChatGroq(model=model_name, temperature=0.7, groq_api_key=api_key)

def nltk_summarizer(text, num_sentences=3):
    """
    Generates a frequency-based summary of the text using NLTK.
    
    Args:
        text (str): The input text to summarize.
        num_sentences (int): Number of sentences for the summary.
    """
    logger.info(" Generating Summary ")
    if not text:
        return ""
    
    # 1. Tokenize and Remove Stopwords
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text.lower())
    
    # 2. Calculate Word Frequencies
    word_frequencies = {}
    for word in words:
        if word not in stop_words and word.isalnum():
            word_frequencies[word] = word_frequencies.get(word, 0) + 1
            
    # 3. Normalize Frequencies
    max_frequency = max(word_frequencies.values())
    for word in word_frequencies.keys():
        word_frequencies[word] = (word_frequencies[word] / max_frequency)
        
    # 4. Score Sentences based on Word Frequency
    sentence_list = sent_tokenize(text)
    sentence_scores = {}
    for sent in sentence_list:
        for word in word_tokenize(sent.lower()):
            if word in word_frequencies:
                if len(sent.split(' ')) < 30:  # Avoid overly long sentences
                    sentence_scores[sent] = sentence_scores.get(sent, 0) + word_frequencies[word]
                    
    # 5. Pick the Top N Sentences
    summary_sentences = heapq.nlargest(num_sentences, sentence_scores, key=sentence_scores.get)
    logger.info(" Summary Generated ")
    return " ".join(summary_sentences)



def extract_transcript(state: State,config: RunnableConfig) -> State:

    """Extracts transcript from a YouTube video URL."""
 
    session_id = config.get("configurable", {}).get("session_id", "UNKNOWN")
    logger.info(f"USER_ID: {session_id} | NODE: extract_transcript | STATUS: Processing")  
    logger.info("🎥 Fetching YouTube transcript...")

    if "video_url" not in state:
        raise KeyError("Missing 'video_url' in state.")

    video_url = state["video_url"].strip()
    video_id=""

    if "youtube.com/watch?v=" in video_url:
        video_id = video_url.split("v=")[-1].split("&")[0]  # Extract ID

    elif "youtu.be/" in video_url:
        video_id = video_url.split("youtu.be/")[-1].split("?")[0]  # Extract ID

    if not video_id:
        raise ValueError("Invalid YouTube URL. Could not extract video ID.")

    try:
        transcript_api = YouTubeTranscriptApi()
        transcript= transcript_api.fetch(video_id=video_id)
        text_transcript = " ".join(snippet.text for snippet in transcript)
        state["transcript"] = text_transcript
        logger.info("✅ Transcript successfully extracted.")
        return state
    except RequestBlocked as e:
            print(f"Error: YouTube blocked the transcript request for video ID '{video_id}'.")
            print(f"Details: {e}") # Consider logging the full error from logs
            return None # Or raise a custom exception
    except Exception as e:
        print(f"An unexpected error occurred while fetching the transcript for '{video_url}': {e}")
        return None

def chunk_text(text: str, max_tokens: int = 500)-> List[str]:

    """Splits text into chunks by word count."""

    words = text.split()
    chunks = []
    for i in range(0, len(words), max_tokens):
        chunks.append(" ".join(words[i:i + max_tokens]))
    return chunks


def generate_blog(state: State,llm,config: RunnableConfig) -> State:

    """Generates a structured blog post from transcript chunks."""
    session_id = config.get("configurable", {}).get("session_id", "UNKNOWN")
    tone = state.get("tone", "Professional")
    
    logger.info(f"USER_ID: {session_id} | NODE: generate_blog | TONE: {state.get('tone')}")
    logger.info("🤖 AI is drafting the blog content...")
    chunks = chunk_text(state["transcript"])
    blog_sections = []

    for chunk in chunks:
        prompt = f"""Extract the key technical or narrative points from this transcript chunk 
        and format them as detailed blog headings and paragraphs:
        {chunk}
        
        Do not include a Title or Introduction yet. Focus only on the facts in this chunk."""

        section = llm.invoke(prompt).content
        blog_sections.append(section)

    combined_body = "\n\n".join(blog_sections)

    # 2. Make ONE final call to wrap it in a blog structure
    tone = state.get("tone", "Professional")
    final_polish_prompt = f"""
        You are a professional blog editor. 
        
        TASK: Convert the following content into a cohesive blog post using a {tone} tone.
        
        STRICT DATA FORMAT:
        The response MUST be divided into two sections by the exact string '|||SEO_SECTION|||'.
        
        SECTION 1: THE BLOG
        - ONE Title (No 'Title:' prefix)
        - ONE Introduction (No Meta Description or Keywords here!)
        - Well-organized subheadings
        - ONE Conclusion
        
        SECTION 2: SEO DATA (Place this ONLY after the separator)
        - Meta Description: [160 chars]
        - Keywords: [5 keywords]

        CONTENT:
        {combined_body}

        Double check: Ensure the string '|||SEO_SECTION|||' is placed between the Conclusion and the SEO data.
        """
    
    state["blog"] = llm.invoke(final_polish_prompt).content
    logger.info("✍️ Draft generation complete.")
    return state

# def human_feedback(state: State) -> dict:
#     return state


# def refine_blog(state: State, llm) -> dict:
    
#     feedback = state.get("feedback", "No human feedback provided.")
    
#     prompt = f"Revise the blog considering following feedback provided from user : {feedback} \n\n Blog Content: {state['blog']}"
#     # Call LLM to refine the blog
#     final_blog = llm.invoke(prompt).content
#     state["final_blog"] = final_blog
#     return {
#         "blog": final_blog,    # Update the blog with the refined version
#         "final_blog": final_blog  # Store the final blog version
#     }

def refine_blog(state: State, llm) -> dict:
    """Refines the existing blog based on user feedback."""
    feedback = state.get("feedback", "")
    prompt = f"Revise this blog based on feedback: {feedback}\n\nOriginal Content: {state['blog']}"
    refined_content = llm.invoke(prompt).content
    return {"blog": refined_content, "final_blog": refined_content}


def route_after_feedback(state: State):
    """Determines whether to end or refine based on feedback status."""
    if state.get("feedback") in ["Accepted", "No feedback provided.", ""]:
        return END
    return "refine_blog"

def generate_graph(llm):
    """Compiles the state graph for the blog generation workflow."""
    builder = StateGraph(State)
       
    builder.add_node("extract_transcript", extract_transcript)
    builder.add_node("generate_blog", lambda state, config: generate_blog(state, llm, config))
    builder.add_node("human_feedback", lambda state: state)
    builder.add_node("refine_blog", lambda state: refine_blog(state, llm)) 

    builder.add_edge(START, "extract_transcript")
    builder.add_edge("extract_transcript", "generate_blog")
    builder.add_edge("generate_blog", "human_feedback")
    # ✅ Loop back to get more feedback if needed
    builder.add_edge("refine_blog", "human_feedback")  
    builder.add_conditional_edges("human_feedback", route_after_feedback, ["refine_blog", END])

    graph = builder.compile(checkpointer=memory, interrupt_before=["human_feedback"])
    return graph

