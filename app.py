import streamlit as st
import openai
from mem0 import Memory
from config import MEM0_CONFIG
from dotenv import load_dotenv
import time
import json
from datetime import datetime

# Load environment variables
load_dotenv(override=True)

# Page configuration
st.set_page_config(
    page_title="Mem0 + Qdrant Chat",
    page_icon="🧠",
    layout="wide"
)

# Initialize Clients
@st.cache_resource
def get_mem0_client():
    return Memory.from_config(MEM0_CONFIG)

@st.cache_resource
def get_openai_client():
    return openai.OpenAI(
        base_url=MEM0_CONFIG["llm"]["config"]["openai_base_url"],
        api_key=MEM0_CONFIG["llm"]["config"]["api_key"],
    )

mem0_client = get_mem0_client()
openai_client = get_openai_client()

# Session State Initialization
if "messages" not in st.session_state:
    st.session_state.messages = []

if "memory_log" not in st.session_state:
    st.session_state.memory_log = []

# Sidebar placeholder needs to be created inside sidebar context, 
# but functions need to be defined before they are called.
# We'll use a slightly different pattern: define functions first, then create UI.

def render_logs(placeholder):
    """Render logs in the provided placeholder."""
    if placeholder:
        with placeholder.container():
            # Display logs in reverse order (newest first)
            for log in reversed(st.session_state.memory_log):
                with st.expander(f"{log['time']} - {log['type']}"):
                    if isinstance(log['details'], (dict, list)):
                        st.json(log['details'])
                    else:
                        st.markdown(str(log['details']))

def add_log(event_type, details, placeholder):
    """Add an event to the memory log and update display."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.memory_log.append({
        "time": timestamp,
        "type": event_type,
        "details": details
    })
    render_logs(placeholder)

# Sidebar for Configuration
with st.sidebar:
    st.title("⚙️ Configuration")
    user_id = st.text_input("User ID", value="live-chat-user-001")
    
    st.divider()
    
    st.title("🧠 Memory Traces")
    
    # Create placeholder for logs immediately
    log_placeholder = st.empty()
    
    if st.button("🗑️ Clear Memory"):
        try:
            memories = mem0_client.get_all(user_id=user_id)
            if memories and "results" in memories:
                for mem in memories["results"]:
                    mem0_client.delete(mem["id"])
            st.success("Memory cleared successfully!")
            # Clear logs as well to show a fresh state
            st.session_state.memory_log = []
            st.session_state.memory_log.append({
                "time": datetime.now().strftime("%H:%M:%S"),
                "type": "SYSTEM",
                "details": "Memory cleared manually."
            })
            # Force refresh logs after clearing
            render_logs(log_placeholder)
        except Exception as e:
            st.error(f"Error clearing memory: {e}")
            
    st.divider()
    
    # Initial render
    render_logs(log_placeholder)

# Main Chat Interface
st.title("💬 Chat with Mem0 Memory")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("What is on your mind?"):
    # 1. Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Search Memory
    with st.spinner("Searching memory..."):
        search_results = mem0_client.search(query=prompt, user_id=user_id)
        add_log("SEARCH", {
            "query": prompt,
            "results": search_results.get("results", [])
        }, log_placeholder)
    
    # 3. Construct Context
    context_str = ""
    if search_results and "results" in search_results:
        memories_list = [m["memory"] for m in search_results["results"]]
        if memories_list:
            context_str = "\nRelevant memories:\n" + "\n".join(f"- {m}" for m in memories_list)
            
    system_prompt = (
        "You are a helpful AI assistant with long-term memory. "
        "Use the provided relevant memories to personalize your response. "
        "If the user asks about something you remember, refer to it explicitly."
        f"{context_str}"
    )

    # 4. Generate Response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            response = openai_client.chat.completions.create(
                model=MEM0_CONFIG["llm"]["config"]["model"],
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                stream=True
            )
            
            for chunk in response:
                if chunk.choices[0].delta.content:
                    full_response += chunk.choices[0].delta.content
                    message_placeholder.markdown(full_response + "▌")
            
            message_placeholder.markdown(full_response)
            
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            st.error(f"Error generating response: {e}")
            full_response = "I encountered an error."

    # 5. Add to Memory (in background)
    with st.spinner("Saving to memory..."):
        # We store the user input as a memory. 
        # Optionally we could store the interaction pair, but standard mem0 usage is usually user input or specific facts.
        # Let's stick to adding the user input for now as per previous chat.py logic.
        add_result = mem0_client.add(prompt, user_id=user_id)
        add_log("ADD", {
            "input": prompt,
            "result": add_result
        }, log_placeholder)
