import os
from groq import Groq

# Try loading environment variables (safely)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # dotenv not installed, will rely on system environment variables
    pass


# Initialize client at module level
_client = None

def get_groq_client():
    global _client
    if _client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("Please set GROQ_API_KEY environment variable")
        _client = Groq(api_key=api_key)
    return _client


# Example of using Groq's agentic capabilities with the compound-beta model
def agent(prompt, messages = None):
    if messages is None:
        # Start a new messages list if none provided
        messages = [{"role": "user", "content": prompt}]
    else:
        # Append the new prompt to the existing messages
        messages = messages + [{"role": "user", "content": prompt}]

    client = get_groq_client()

    completion = client.chat.completions.create(
        messages=messages,
        # Change model to compound-beta to use agentic tooling
        # model: "llama-3.3-70b-versatile",
        model="compound-beta",
    )

    message = completion.choices[0].message.content

    return message

    # print(completion.choices[0].message.content)
    
    # # Print all tool calls
    # print(completion.choices[0].message.executed_tools)

# if __name__ == "__main__":
#     agent()



def ask_groq_llm(
        prompt, 
        messages = None, 
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        temperature=1.0,
        max_completion_tokens=1024,
        top_p=1.0,
        stop=None,
        return_full = False
        ):

    if messages is None:
        # Start a new messages list if none provided
        messages = [{"role": "user", "content": prompt}]
    else:
        # Append the new prompt to the existing messages
        messages = messages + [{"role": "user", "content": prompt}]

    client = get_groq_client()
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_completion_tokens=max_completion_tokens,
        top_p=top_p,
        stream=False,
        stop=stop,
    )

    if return_full:
        return completion
    else:
        # Extract the message content from the completion
        message = completion.choices[0].message.content
        
        return message
    

if __name__ == "__main__":
    prompt = "What is the capital of France?"
    response = ask_groq_llm(prompt)
    print(response)