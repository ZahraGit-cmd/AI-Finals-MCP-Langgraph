"""
FastMCP Math Server - Arithmetic Operations Server

Provides mathematical operation tools via FastMCP with SSE transport.
Each operation (add, subtract, multiply, divide) is exposed as a tool.

Run: python fastmcp_math_server.py
Server will start on: http://localhost:8000/sse
"""

from mcp.server.fastmcp import FastMCP

# Initialize FastMCP server
mcp = FastMCP("MathOperations")


@mcp.tool()
async def add(operand1: float, operand2: float) -> str:
    """
    Add two numbers together.
    
    Args:
        operand1: First number to add
        operand2: Second number to add
    
    Returns:
        JSON string with operation result
    """
    result = operand1 + operand2
    return f'{{"operation": "add", "operand1": {operand1}, "operand2": {operand2}, "result": {result}, "expression": "{operand1} + {operand2} = {result}"}}'


@mcp.tool()
async def subtract(operand1: float, operand2: float) -> str:
    """
    Subtract second number from first number.
    
    Args:
        operand1: Number to subtract from
        operand2: Number to subtract
    
    Returns:
        JSON string with operation result
    """
    result = operand1 - operand2
    return f'{{"operation": "subtract", "operand1": {operand1}, "operand2": {operand2}, "result": {result}, "expression": "{operand1} - {operand2} = {result}"}}'


@mcp.tool()
async def multiply(operand1: float, operand2: float) -> str:
    """
    Multiply two numbers together.
    
    Args:
        operand1: First number to multiply
        operand2: Second number to multiply
    
    Returns:
        JSON string with operation result
    """
    result = operand1 * operand2
    return f'{{"operation": "multiply", "operand1": {operand1}, "operand2": {operand2}, "result": {result}, "expression": "{operand1} × {operand2} = {result}"}}'


@mcp.tool()
async def divide(operand1: float, operand2: float) -> str:
    """
    Divide first number by second number.
    
    Args:
        operand1: Number to be divided (numerator)
        operand2: Number to divide by (denominator)
    
    Returns:
        JSON string with operation result or error
    """
    if operand2 == 0:
        return f'{{"operation": "divide", "operand1": {operand1}, "operand2": {operand2}, "error": "Division by zero", "result": null}}'
    
    result = operand1 / operand2
    return f'{{"operation": "divide", "operand1": {operand1}, "operand2": {operand2}, "result": {result}, "expression": "{operand1} ÷ {operand2} = {result}"}}'


if __name__ == "__main__":
    print("=" * 70)
    print("🚀 FastMCP Math Operations Server")
    print("=" * 70)
    print("📡 Starting server on: http://localhost:8000/sse")
    print("🔧 Available tools: add, subtract, multiply, divide")
    print("=" * 70)
    print()
    
    # Run server with SSE transport
    mcp.run(transport="sse")