import os
from typing import Annotated

# Try loading environment variables (safely)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # dotenv not installed, will rely on system environment variables
    pass

api_key = os.environ.get("GROQ_API_KEY")
if not api_key:
    raise ValueError("Please set GROQ_API_KEY environment variable")
        
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent

# Use a valid Groq model name
model = init_chat_model("llama3-8b-8192", model_provider="groq")

def get_weather(city: str) -> str:  
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

agent = create_react_agent(
    model=model,  
    tools=[get_weather],  
    prompt="You are a helpful assistant that uses tools to answer questions."
)

# Run the agent and capture the result
if __name__ == "__main__":
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "what is the weather in sf"}]}
    )
    print("Final result:")
    print(result)