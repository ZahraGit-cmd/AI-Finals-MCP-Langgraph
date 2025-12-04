# mytestmcp.py
import asyncio
import os
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langgraph.prebuilt import create_react_agent
import httpx
from langchain_openai import ChatOpenAI
API_KEY = "sk-QibkR-xrOxD2AsMgdaL0Pg"
client = httpx.Client(verify=False)

llm = ChatOpenAI(
    base_url="https://genailab.tcs.in",
    model="azure_ai/genailab-maas-DeepSeek-V3-0324",
    api_key=API_KEY,
    http_client=client,
)

def mcp_tool_to_openai(tool):
    return {
        "name": tool.name,
        "description": tool.description or "",
        "parameters": tool.inputSchema,
    }

async def main():
    # Path to math_server.py (same directory)
    script_path = os.path.join(os.path.dirname(__file__), "math_server.py")

    if not os.path.exists(script_path):
        raise FileNotFoundError(f"Server script not found: {script_path}")

    # Configure server launch
    server_params = StdioServerParameters(
        command="python",
        args=[script_path],
    )

    # Start the server and connect over stdio
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:

            print("🔗 Initializing MCP session...")
            await session.initialize()
            print("✅ Connected to MCP server!")

            # Call the MCP tool
            result = await session.call_tool("add", {"a": 5, "b": 7})
            number = result.structuredContent["result"]

            print("🔢 Result from MCP tool:", number)
            tools_resp = await session.list_tools()
            print(tools_resp)
            openai_tools = [mcp_tool_to_openai(t) for t in tools_resp.tools]
            print(openai_tools)
            agent = create_react_agent(llm, openai_tools)
            agent_response = agent.invoke({"messages": "what's (3 + 5) * 12?"})
            print("AGENT RESPONSE:", agent_response)
                            


if __name__ == "__main__":
    asyncio.run(main())
