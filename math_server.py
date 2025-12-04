# math_server.py
from mcp.server.fastmcp import FastMCP

# Create MCP server instance
mcp = FastMCP("Math Server")

@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers together."""
    return a+b

@mcp.tool()
def subtract(a: int, b: int) -> int:
    """Subtract b from a."""
    return a - b

@mcp.tool()
def multiply(a: int, b: int) -> int:
    """Multiply two numbers together."""
    return a * b

if __name__ == "__main__":
    
    # Important: use stdio transport so the client can connect
    mcp.run(transport="stdio")
