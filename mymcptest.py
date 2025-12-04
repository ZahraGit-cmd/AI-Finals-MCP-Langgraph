# mytestmcp.py
import asyncio
import os
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


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


if __name__ == "__main__":
    asyncio.run(main())
