# ============================================================

import os
import logging
import dotenv
from mcp.types import (
    Tool,
    TextContent,
    ImageContent,
    EmbeddedResource,
)
from mcp.server import Server, NotificationOptions
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from importlib import metadata
import asyncio

# ============================================================
# Logging
# ============================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("maths-search-mcp")

# ============================================================
# Environment Variables
# ============================================================


# ============================================================
# Initialize server
# ============================================================
async def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b

async def multiply(a: int, b: int) -> int:
    """Multiply two numbers"""
    return a * b

server = Server("maths-server")

# ============================================================
# List available tools in the server
# ============================================================
@server.list_tools()
async def handle_list_tools() -> list[Tool]:
    """List available tools in the server."""
    return [
        Tool(
            name="add",
            description="Add 2 numbers",
            inputSchema={
                "type": "object",
                "properties": {
                    "num1": {
                        "type": "integer",
                        "description": "First number to add"
                    },
                    "num2": {
                        "type": "integer",
                        "description": "Second number to add"
                    }
                },
                "required": ["num1", "num2"]
            }
        ),
        Tool(
            name="multiply",
            description="multiply 2 numbers",
            inputSchema={
                "type": "object",
                "properties": {
                    "num1": {
                        "type": "integer",
                        "description": "First number to multiply"
                    },
                    "num2": {
                        "type": "integer",
                        "description": "Second number to multiply"
                    }
                },
                "required": ["num1", "num2"]
            }
        )

    ]

# ============================================================
# Handle tool execution requests
# ============================================================
@server.call_tool()
async def handle_call_tool(name: str, arguments: dict | None) -> list[TextContent | ImageContent | EmbeddedResource]:
    """Handle tool execution requests."""
    
    if name == "add":
        try:    
            return [TextContent(
                type="text",
                text="\n\n---\n\n".join("Called Add")
            )]
        
        except Exception as e:
            error_message = f"Error querying document: {str(e)}"
            logger.error(error_message)
            return [TextContent(type="text", text=error_message)]

    else:
        raise ValueError(f"Unknown tool: {name}")



# ============================================================
# Run the MCP server using stdin/stdout streams
# ============================================================   
async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream,write_stream,InitializationOptions(
            server_name="MATHS_SERVER",
            server_version="1.0.0",
            capabilities=server.get_capabilities(
                notification_options=NotificationOptions(
                    tools_changed=True
                ),
                experimental_capabilities={}
            )
        )
        )
        print("MCP Server is running...")
        await asyncio.Event().wait()
if __name__ == "__main__":
    asyncio.run(main())