import os
from dotenv import load_dotenv

from groq import Groq

# Load environment variables from .env file
load_dotenv()

# Example of using Groq's agentic capabilities with the compound-beta model
def agent():
    # Get API key with a helpful error message
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("Please set GROQ_API_KEY environment variable")
    
    
    client = Groq(api_key=api_key)

    completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": "What is the current weather in Tokyo?",
            }
        ],
        # Change model to compound-beta to use agentic tooling
        # model: "llama-3.3-70b-versatile",
        model="compound-beta",
    )

    print(completion.choices[0].message.content)
    
    # Print all tool calls
    print(completion.choices[0].message.executed_tools)

if __name__ == "__main__":
    agent()