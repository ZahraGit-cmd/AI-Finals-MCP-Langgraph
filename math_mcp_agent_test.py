# run_agent_tools.py
import asyncio
import os, httpx
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain.agents import create_agent

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mc_client import MCPClient


API_KEY = "sk-QibkR-xrOxD2AsMgdaL0Pg"
tiktoken_cache_dir = "./token"
os.environ["TIKTOKEN_CACHE_DIR"] = tiktoken_cache_dir

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


async def run_agent_and_call_tools():
    async with MCPClient() as mcp_session:
        # Get all tools from MCP
        tools_resp = await mcp_session.list_tools()
        openai_tools = [mcp_tool_to_openai(t) for t in tools_resp.tools]

        # Create the agent
        agent = create_agent(llm, openai_tools)

        # Ask the agent
        agent_response = agent.invoke({"messages": "what's (3 + 5) * 12?"})
        print("AGENT RESPONSE:", agent_response)

        # Extract tool_calls from AI messages
        tool_calls = []
        for msg in agent_response["messages"]:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                tool_calls.extend(msg.tool_calls)

        if not tool_calls:
            print("❌ No tools to call.")
            return None

        # Path to your MCP server script
        script_path = os.path.join(os.path.dirname(__file__), "math_server_new.py")
        if not os.path.exists(script_path):
            raise FileNotFoundError(f"Server script not found: {script_path}")

        # Configure MCP server launch
        server_params = StdioServerParameters(
            command="python",
            args=[script_path]
        )

        # Start stdio client session
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                print("✅ Connected to MCP server!")

                last_result = None
                for call in tool_calls:
                    tool_name = call["name"]
                    args = call["args"]

                    # Use previous result if an argument is None
                    for k, v in args.items():
                        if v is None and last_result is not None:
                            args[k] = last_result

                    print(f"🔗 Calling MCP tool: {tool_name} with args: {args}")
                    result = await session.call_tool(tool_name, args)
                    last_result = result.structuredContent.get("result")
                    print(f"🔢 Result from {tool_name}: {last_result}")

                return last_result


if __name__ == "__main__":
    final_result = asyncio.run(run_agent_and_call_tools())
    print("✅ Final result:", final_result)
