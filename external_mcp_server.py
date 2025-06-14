from mcp.client.sse import sse_client
from mcp import ClientSession
import asyncio

async def get_sentiment(prompt: str) -> str:
    # Connect to a streamable HTTP server
    async with sse_client("https://techno-1-mcp-sentiment.hf.space/gradio_api/mcp/sse") as (
        read_stream,
        write_stream,
    ):
        # Create a session using the client streams
        async with ClientSession(read_stream, write_stream) as session:
            # Initialize the connection
            await session.initialize()
            # List available tools
            # tools = await session.list_tools()
            # print(f"\n Available tools:{tools}\n")
            # Call a tool
            tool_result = await session.call_tool("mcp_sentimentpredict", {"text": prompt})
            # Print the result of the tool call
            #print(tool_result)
            return tool_result

if __name__ == "__main__":
    # Run the main function using asyncio
    asyncio.run(get_sentiment())