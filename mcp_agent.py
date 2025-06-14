from groq_llms import GroqClient as GroqLLM
from external_mcp_server import get_sentiment
import json
import asyncio

# Define the available tools (i.e. functions) for our model to use

tools = [

    {

        "type": "function",

        "function": {

            "name": "get_sentiment",

            "description": "Uses a sentiment analysis model to analyze the sentiment of a given text",

            "parameters": {

                "type": "object",

                "properties": {

                    "text": {

                        "type": "string",

                        "description": "The text to analyze for sentiment",

                    }

                },

                "required": ["text"],

            },

        },

    }

]

async def main(prompt):
    llm = GroqLLM(tools=tools)

    for i in range(20):
        response = llm.ask(prompt, return_full=True)
        # Extract the response and any tool call responses
        response_message = response.choices[0].message
        print(f"Response: {response_message.content}")
        tool_calls = response_message.tool_calls
        print(f"Tool calls: {tool_calls}")
        
        for tool_call in tool_calls:
            if tool_call.function.name == "get_sentiment":
                # Call the sentiment analysis function
                args_dict = json.loads(tool_call.function.arguments)
                text = args_dict.get("text", "")
                
                print(f"Calling sentiment analysis for text: {text}")
                sentiment_result = await get_sentiment(text)
                
                # Create a response message with the sentiment result
                llm.conversation_history.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": tool_call.function.name,
                    "content": sentiment_result
                    }
                )
                # print(f"Sentiment result: {sentiment_result}")
                return sentiment_result

if __name__ == "__main__":
    prompt = "What is the sentiment of this text? 'I love programming!'"
    result = asyncio.run(main(prompt))
    print(f"Sentiment result:{result.content[0].text}")