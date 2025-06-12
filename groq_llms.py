import os
from groq import Groq

# Try loading environment variables (safely)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # dotenv not installed, will rely on system environment variables
    pass


class GroqClient:
    """A wrapper for the Groq API client with sensible defaults and conversation management."""
    
    def __init__(self, api_key=None, default_model="meta-llama/llama-4-scout-17b-16e-instruct"):
        """
        Initialize the Groq client with default settings.
        
        Args:
            api_key (str, optional): API key for Groq. If not provided, will use GROQ_API_KEY env var.
            default_model (str): Default model to use for completions.
        """
        api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("Please set GROQ_API_KEY environment variable")
        
        self.client = Groq(api_key=api_key)
        self.default_model = default_model
        self.conversation_history = []
    
    def ask(self, prompt, model=None, temperature=1.0, max_completion_tokens=1024, 
            top_p=1.0, stop=None, messages=None, return_full=False):
        """
        Send a prompt to the model and get a response.
        
        Args:
            prompt (str): The prompt to send to the model
            model (str, optional): Model to use. Defaults to instance default.
            temperature (float): Sampling temperature (0-1)
            max_completion_tokens (int): Maximum tokens in response
            top_p (float): Top-p sampling parameter
            stop (str or list, optional): Stop sequences
            messages (list, optional): Override conversation history with custom messages
            return_full (bool): If True, returns full completion object. Otherwise just message.
        
        Returns:
            Union[str, ChatCompletion]: Either the response text or full response object
        """
        model = model or self.default_model
        
        if messages is not None:
            # Use provided messages (don't modify conversation history)
            chat_messages = messages + [{"role": "user", "content": prompt}]
        else:
            # Use conversation history
            chat_messages = self.conversation_history + [{"role": "user", "content": prompt}]
            # Update conversation history
            self.conversation_history.append({"role": "user", "content": prompt})
        
        completion = self.client.chat.completions.create(
            model=model,
            messages=chat_messages,
            temperature=temperature,
            max_completion_tokens=max_completion_tokens,
            top_p=top_p,
            stream=False,
            stop=stop,
        )
        
        message_content = completion.choices[0].message.content
        
        # Add assistant response to conversation history if using persistent conversation
        if messages is None:
            self.conversation_history.append({"role": "assistant", "content": message_content})
        
        return completion if return_full else message_content
    
    def agent(self, prompt, messages=None):
        """
        Use Groq's agentic capabilities with the compound-beta model.
        
        Args:
            prompt (str): The prompt to send
            messages (list, optional): Custom message history
            
        Returns:
            str: The assistant's response
        """
        return self.ask(prompt, model="compound-beta", messages=messages)
    
    def reset_conversation(self):
        """Clear the conversation history."""
        self.conversation_history = []
    
    def get_conversation_history(self):
        """Get the current conversation history."""
        return self.conversation_history.copy()

if __name__ == "__main__":
    # Example usage of the new class-based approach
    print("=== Class-based approach ===")
    client = GroqClient()
    
    # Simple usage
    response = client.ask("What is the capital of France?")
    print(f"Response: {response}")
    
    # Conversation with history
    print("\n=== Conversation example ===")
    client.reset_conversation()  # Start fresh
    response1 = client.ask("My name is Alice.")
    print(f"User: My name is Alice.")
    print(f"Assistant: {response1}")
    
    response2 = client.ask("What's my name?")
    print(f"User: What's my name?")
    print(f"Assistant: {response2}")
    
    # Agent example (compound-beta model)
    print("\n=== Agent example ===")
    agent_response = client.agent("What's the current weather in Tokyo?")
    print(f"Agent response: {agent_response}")