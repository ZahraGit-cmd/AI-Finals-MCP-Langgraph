# mytestmcp_run_tools.py
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

            # List tools
            tools_resp = await session.list_tools()
            print("Available tools:", tools_resp)
            openai_tools = [mcp_tool_to_openai(t) for t in tools_resp.tools]
            print("OpenAI-compatible tools:", openai_tools)

            # Create the agent
            agent = create_react_agent(llm, openai_tools)

            # Ask the agent
            agent_response = agent.invoke({"messages": "what's (3 + 5) * 12?"})
            print("AGENT RESPONSE:", agent_response)

            # Extract tool calls from the agent response
            tool_calls = []
            for msg in agent_response["messages"]:
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    tool_calls.extend(msg.tool_calls)

            if not tool_calls:
                print("❌ No tools to call.")
                return

            # 🔹 Print the sequence of tools required
            print("\n🛠 Tools sequence required to solve the problem:")
            for idx, call in enumerate(tool_calls, 1):
                print(f"{idx}. Tool: {call['name']}, Args: {call['args']}")
            print()

            # Execute each tool sequentially
            last_result = None
            for call in tool_calls:
                tool_name = call["name"]
                args = call["args"]

                # Use previous result if any argument is None
                for k, v in args.items():
                    if v is None and last_result is not None:
                        args[k] = last_result

                print(f"🔗 Calling MCP tool: {tool_name} with args: {args}")
                result = await session.call_tool(tool_name, args)
                last_result = result.structuredContent.get("result")
                print(f"🔢 Result from {tool_name}: {last_result}")

            print("✅ Final result:", last_result)

if __name__ == "__main__":
    asyncio.run(main())
