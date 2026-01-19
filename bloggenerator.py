import os
from dotenv import load_dotenv
load_dotenv()
from youtube_transcript_api import YouTubeTranscriptApi
from IPython.display import Image, display
from youtube_transcript_api import RequestBlocked
from typing import TypedDict, Optional
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import MessagesState
from langgraph.graph import START, StateGraph,END
from langgraph.prebuilt import tools_condition, ToolNode
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq
import nltk
import heapq
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize,word_tokenize

# Setup NLTK
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



memory = MemorySaver()
def initialize_model(model_name):

    os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")
    llm=ChatGroq(model=model_name,temperature=0.7)

    return llm
def nltk_summarizer(text, num_sentences=3):
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
    return " ".join(summary_sentences)

class State(TypedDict):
    video_url: str
    transcript: str
    blog: str
    feedback:str
    final_blog: str
    tone: str           # New: Store the selected tone
    seo_meta: str       # New: Store SEO description/keywords

def extract_transcript(state: State) -> State:

    """Extracts transcript from a YouTube video URL."""

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
        return state
    except RequestBlocked as e:
            print(f"Error: YouTube blocked the transcript request for video ID '{video_id}'.")
            print(f"Details: {e}") # Consider logging the full error from logs
            return None # Or raise a custom exception
    except Exception as e:
        print(f"An unexpected error occurred while fetching the transcript for '{video_url}': {e}")
        return None

def chunk_text(text: str, max_tokens: int = 500):

    """Splits text into chunks of max_tokens words."""

    words = text.split()
    chunks = []
    for i in range(0, len(words), max_tokens):
        chunks.append(" ".join(words[i:i + max_tokens]))
    return chunks

def generate_blog_section(chunk: str,llm) -> str:

    """Generates a blog section for a given transcript chunk."""

    # prompt_text = f"""
    # Generate a structured blog based on the following YouTube transcript chunk:
    # {chunk}

    # Structure:
    # 1. **Title**: A compelling blog title in only 10 words.
    # 2. **Introduction**: A brief introduction in only 40 words.
    # 3. **Headings & Subheadings** 
    # 4. **Conclusion**: A strong closing statement in only 40 words.
    
    # Keep the response concise, and do not exceed 3000 tokens.
    # """

    prompt_text = f"""
        Extract the key technical or narrative points from this transcript chunk 
        and format them as detailed blog headings and paragraphs:
        {chunk}
        
        Do not include a Title or Introduction yet. Focus only on the facts in this chunk.
    """
    return llm.invoke(prompt_text).content  # Invoke LLM for each chunk

def generate_blog(state: State,llm) -> State:

    """Generates a full blog by processing transcript chunks separately."""
    
    transcript_chunks = chunk_text(state["transcript"])  # Split transcript
    # 1. Get the body content from all chunks
    blog_sections = [generate_blog_section(chunk,llm) for chunk in transcript_chunks]  # Process chunks
    # state["blog"] = "\n\n".join(blog_sections)  # Combine sections
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
    return state

def human_feedback(state: State) -> dict:
    # print("\n-------------------------------------------✅ Blog Draft ------------------------------------ \n", state["blog"])
    # print("\n-------------------------------------------✅ End of the Blog------------------------------------ \n")
    # choice = input("\nDo you want to provide feedback to refine the blog? (yes/no/accepted): ").strip().lower()
    
    # if choice == "yes":
    #     state["feedback"] = input("\nEnter your feedback to refine the blog: ")
    #     return {"feedback": state["feedback"], "refine_blog": state["blog"]}
    # elif choice == "accepted":
    #     state["feedback"] = "Accepted"
    #     print("\n------------------------------------ Blog Accepted ----------------------------------------------\n")
    #     return {"feedback": state["feedback"],"final_blog": state["blog"]}  # Ends workflow
    # else:
    #     state["feedback"] = "No feedback provided."
    #     print("\n------------------------------------ No feedback provided.---------------------------------------------- \n")
    #     return {"feedback": state["feedback"],"final_blog": state["blog"]}  # Ends workflow
    return state


def refine_blog(state: State, llm) -> dict:
    
    feedback = state.get("feedback", "No human feedback provided.")
    
    prompt = f"Revise the blog considering following feedback provided from user : {feedback} \n\n Blog Content: {state['blog']}"
    print("\n\n\n 🖍🖍 Refining blog as per the following feedback : ",feedback)
    # Call LLM to refine the blog
    final_blog = llm.invoke(prompt).content
    state["final_blog"] = final_blog

    print("\n✅ Blog Finalized Successfully!\n")
    
    return {
        "blog": final_blog,    # Update the blog with the refined version
        "final_blog": final_blog  # Store the final blog version
    }

# Conditional edges for feedback loop
def route_after_feedback(state):
    if state["feedback"] == "Accepted" or state["feedback"] == "No feedback provided.":
        return END  # ✅ Ends workflow if feedback is accepted or no feedback is given
    return "refine_blog"  # ✅ Loops back to refining


def generate_graph(llm):
    builder = StateGraph(State)
       
    builder.add_node("extract_transcript", extract_transcript)
    builder.add_node("generate_blog", lambda state: generate_blog(state, llm))
    builder.add_node("human_feedback", human_feedback)
    builder.add_node("refine_blog", lambda state: refine_blog(state, llm)) 

    builder.add_edge(START, "extract_transcript")
    builder.add_edge("extract_transcript", "generate_blog")
    builder.add_edge("generate_blog", "human_feedback")

    # ✅ Loop back to get more feedback if needed
    builder.add_edge("refine_blog", "human_feedback")  


    builder.add_conditional_edges("human_feedback", route_after_feedback, ["refine_blog", END])


    graph = builder.compile(checkpointer=memory, interrupt_before=["human_feedback"])
    return graph


def run_pipeline(video_url,model):
    llm = initialize_model(model)
    graph=generate_graph(llm)
    print(graph)

    # url="https://www.youtube.com/watch?v=bxuYDT-BWaI"

    initial_state: State = {
            "video_url": video_url,
            "transcript": "",
            "blog": "",
            "feedback": "",  # No need to store permanently
            "final_blog":""
        }
    # Thread

    print("✅ Graph Created. Invoking the pipeline...")
    messages= graph.invoke(initial_state)
    print("✅ Pipeline Execution Complete.")

    # Ensure the final blog is printed correctly
    if "final_blog" in messages:
        print("\n------------------------------------ ✅ Final Blog Output: ------------------------------------ \n", messages["final_blog"])
        return messages["final_blog"]
    else:
        print("\n❌ Error: 'final_blog' key missing in output state.")
     

# blog = run_pipeline()
# print("\n\n\nFinal Output:" , blog)