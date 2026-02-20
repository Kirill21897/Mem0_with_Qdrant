import os
import sys
import openai
from mem0 import Memory
from config import MEM0_CONFIG
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

# Set up OpenAI client (using OpenRouter)
openai_client = openai.OpenAI(
    base_url=MEM0_CONFIG["llm"]["config"]["openai_base_url"],
    api_key=MEM0_CONFIG["llm"]["config"]["api_key"],
)

# Initialize Mem0 client
print("Инициализация памяти (это может занять несколько секунд)...")
mem0_client = Memory.from_config(MEM0_CONFIG)

def get_ai_response(user_input, user_id, context_memories):
    """
    Generate a response using OpenRouter/OpenAI with injected memory context.
    """
    
    # Format memories into a string context
    context_str = ""
    if context_memories and "results" in context_memories:
        memories_list = [m["memory"] for m in context_memories["results"]]
        if memories_list:
            context_str = "\nRelevant memories:\n" + "\n".join(f"- {m}" for m in memories_list)
    
    system_prompt = (
        "You are a helpful AI assistant with long-term memory. "
        "Use the provided relevant memories to personalize your response. "
        "If the user asks about something you remember, refer to it explicitly."
        f"{context_str}"
    )

    try:
        response = openai_client.chat.completions.create(
            model=MEM0_CONFIG["llm"]["config"]["model"],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ],
            max_tokens=1000  # Limit tokens to avoid quota errors
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating response: {e}"

def clear_user_memory(user_id):
    """
    Safely clear all memories for a specific user.
    """
    try:
        print(f"   (Fetching memories for deletion...)", end="\r")
        memories = mem0_client.get_all(user_id=user_id)
        
        if not memories or "results" not in memories:
            return "No memories found to delete."
            
        memory_list = memories["results"]
        if not memory_list:
            return "Memory is already empty."
            
        count = len(memory_list)
        print(f"   (Deleting {count} memories...)", end="\r")
        
        # Delete memories one by one to avoid wiping the entire collection (which delete_all might do)
        for mem in memory_list:
            mem0_client.delete(mem["id"])
            
        return f"Successfully deleted {count} memories."
    except Exception as e:
        return f"Error clearing memory: {e}"

def main():
    print("\n" + "="*50)
    print("🤖 AI Assistant with Memory (Mem0 + Qdrant)")
    print("Type 'exit' or 'quit' to stop.")
    print("Type '/reset' or '/clear' to clear memory.")
    print("="*50 + "\n")

    user_id = "live-chat-user-001"  # Unique ID for this session/user

    while True:
        try:
            user_input = input("\nYou: ").strip()
            if not user_input:
                continue
                
            if user_input.lower() in ["exit", "quit"]:
                print("Goodbye!")
                break
                
            if user_input.lower() in ["/reset", "/clear"]:
                result = clear_user_memory(user_id)
                print(f"System: {result}")
                continue

            # 1. Search for relevant memories
            print("   (Searching memory...)", end="\r")
            relevant_memories = mem0_client.search(query=user_input, user_id=user_id)
            
            # 2. Generate response using LLM + Memories
            print("   (Thinking...)", end="\r")
            ai_response = get_ai_response(user_input, user_id, relevant_memories)
            
            # Clear status line
            print(" " * 50, end="\r")
            
            print(f"AI: {ai_response}")

            # 3. Add interaction to memory (in background/after response)
            # We store the user input so the AI remembers what the user said.
            # We can optionally store the AI's response too, but storing user input is most critical for facts.
            print("   (Saving memory...)", end="\r")
            mem0_client.add(user_input, user_id=user_id)
            print(" " * 50, end="\r")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    main()
