"""
Math Equation Solver - Complete MCP Client (FIXED)
Multi-Agent System with FastMCP Integration

This version properly handles nested async operations.

Run this after starting the FastMCP server:
    python fastmcp_math_server.py

Then run:
    python mcp_client.py
"""

import asyncio
import json
import os
from typing import TypedDict, Annotated

import httpx
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
import operator

# Apply nest_asyncio to allow nested event loops
import nest_asyncio
nest_asyncio.apply()

# Configuration
API_KEY = "sk-QibkR-xrOxD2AsMgdaL0Pg"

# Create token cache directory if it doesn't exist
token_dir = "./token"
os.makedirs(token_dir, exist_ok=True)
os.environ["TIKTOKEN_CACHE_DIR"] = token_dir

# Initialize LLM
http_client = httpx.Client(verify=False)
llm = ChatOpenAI(
    base_url="https://genailab.tcs.in",
    model="azure_ai/genailab-maas-DeepSeek-V3-0324",
    api_key=API_KEY,
    http_client=http_client
)


# ============= STATE DEFINITION =============

class MathState(TypedDict):
    """State that flows through the workflow"""
    expression: str
    pending_operations: list
    current_operation: dict
    intermediate_results: dict
    result: float
    history: Annotated[list, operator.add]
    next_agent: str
    step_count: int


# ============= AGENT DEFINITIONS =============

class AddAgent:
    """Agent specialized in addition using MCP add tool"""
    
    def __init__(self, mcp_client):
        self.name = "Add Agent"
        self.tool_name = "add"
        self.mcp_client = mcp_client
        self.tools_cache = None
    
    async def get_tool(self):
        """Get the add tool from MCP client"""
        if self.tools_cache is None:
            self.tools_cache = await self.mcp_client.get_tools()
        return next(tool for tool in self.tools_cache if tool.name == self.tool_name)
    
    async def execute_async(self, state: MathState) -> MathState:
        op = state["current_operation"]
        
        print(f"\n{'='*70}")
        print(f"🤖 AGENT INVOKED: {self.name}")
        print(f"🔧 TOOL USED: MCP Tool '{self.tool_name}'")
        print(f"📊 Operation: {op['operand1']} + {op['operand2']}")
        print(f"{'='*70}")
        
        # Get and invoke the tool
        add_tool = await self.get_tool()
        result_str = await add_tool.ainvoke({
            "operand1": op["operand1"],
            "operand2": op["operand2"]
        })
        
        mcp_result = json.loads(result_str)
        result = mcp_result["result"]
        
        history_entry = f"  Step {state['step_count']}: ➕ {self.name} executed MCP tool '{self.tool_name}' → {mcp_result['expression']}"
        
        intermediate_results = state["intermediate_results"].copy()
        if op.get("result_key"):
            intermediate_results[op["result_key"]] = result
        
        return {
            **state,
            "result": result,
            "intermediate_results": intermediate_results,
            "history": [history_entry],
            "next_agent": "supervisor",
            "step_count": state["step_count"] + 1
        }


class SubtractAgent:
    """Agent specialized in subtraction using MCP subtract tool"""
    
    def __init__(self, mcp_client):
        self.name = "Subtract Agent"
        self.tool_name = "subtract"
        self.mcp_client = mcp_client
        self.tools_cache = None
    
    async def get_tool(self):
        """Get the subtract tool from MCP client"""
        if self.tools_cache is None:
            self.tools_cache = await self.mcp_client.get_tools()
        return next(tool for tool in self.tools_cache if tool.name == self.tool_name)
    
    async def execute_async(self, state: MathState) -> MathState:
        op = state["current_operation"]
        
        print(f"\n{'='*70}")
        print(f"🤖 AGENT INVOKED: {self.name}")
        print(f"🔧 TOOL USED: MCP Tool '{self.tool_name}'")
        print(f"📊 Operation: {op['operand1']} - {op['operand2']}")
        print(f"{'='*70}")
        
        subtract_tool = await self.get_tool()
        result_str = await subtract_tool.ainvoke({
            "operand1": op["operand1"],
            "operand2": op["operand2"]
        })
        
        mcp_result = json.loads(result_str)
        result = mcp_result["result"]
        
        history_entry = f"  Step {state['step_count']}: ➖ {self.name} executed MCP tool '{self.tool_name}' → {mcp_result['expression']}"
        
        intermediate_results = state["intermediate_results"].copy()
        if op.get("result_key"):
            intermediate_results[op["result_key"]] = result
        
        return {
            **state,
            "result": result,
            "intermediate_results": intermediate_results,
            "history": [history_entry],
            "next_agent": "supervisor",
            "step_count": state["step_count"] + 1
        }


class MultiplyAgent:
    """Agent specialized in multiplication using MCP multiply tool"""
    
    def __init__(self, mcp_client):
        self.name = "Multiply Agent"
        self.tool_name = "multiply"
        self.mcp_client = mcp_client
        self.tools_cache = None
    
    async def get_tool(self):
        """Get the multiply tool from MCP client"""
        if self.tools_cache is None:
            self.tools_cache = await self.mcp_client.get_tools()
        return next(tool for tool in self.tools_cache if tool.name == self.tool_name)
    
    async def execute_async(self, state: MathState) -> MathState:
        op = state["current_operation"]
        
        print(f"\n{'='*70}")
        print(f"🤖 AGENT INVOKED: {self.name}")
        print(f"🔧 TOOL USED: MCP Tool '{self.tool_name}'")
        print(f"📊 Operation: {op['operand1']} × {op['operand2']}")
        print(f"{'='*70}")
        
        multiply_tool = await self.get_tool()
        result_str = await multiply_tool.ainvoke({
            "operand1": op["operand1"],
            "operand2": op["operand2"]
        })
        
        mcp_result = json.loads(result_str)
        result = mcp_result["result"]
        
        history_entry = f"  Step {state['step_count']}: ✖️ {self.name} executed MCP tool '{self.tool_name}' → {mcp_result['expression']}"
        
        intermediate_results = state["intermediate_results"].copy()
        if op.get("result_key"):
            intermediate_results[op["result_key"]] = result
        
        return {
            **state,
            "result": result,
            "intermediate_results": intermediate_results,
            "history": [history_entry],
            "next_agent": "supervisor",
            "step_count": state["step_count"] + 1
        }


class DivideAgent:
    """Agent specialized in division using MCP divide tool"""
    
    def __init__(self, mcp_client):
        self.name = "Divide Agent"
        self.tool_name = "divide"
        self.mcp_client = mcp_client
        self.tools_cache = None
    
    async def get_tool(self):
        """Get the divide tool from MCP client"""
        if self.tools_cache is None:
            self.tools_cache = await self.mcp_client.get_tools()
        return next(tool for tool in self.tools_cache if tool.name == self.tool_name)
    
    async def execute_async(self, state: MathState) -> MathState:
        op = state["current_operation"]
        
        print(f"\n{'='*70}")
        print(f"🤖 AGENT INVOKED: {self.name}")
        print(f"🔧 TOOL USED: MCP Tool '{self.tool_name}'")
        print(f"📊 Operation: {op['operand1']} ÷ {op['operand2']}")
        print(f"{'='*70}")
        
        divide_tool = await self.get_tool()
        result_str = await divide_tool.ainvoke({
            "operand1": op["operand1"],
            "operand2": op["operand2"]
        })
        
        mcp_result = json.loads(result_str)
        
        if "error" in mcp_result:
            history_entry = f"  Step {state['step_count']}: ➗ {self.name} - Error: {mcp_result['error']}"
            return {
                **state,
                "result": float('inf'),
                "history": [history_entry],
                "next_agent": "end"
            }
        
        result = mcp_result["result"]
        history_entry = f"  Step {state['step_count']}: ➗ {self.name} executed MCP tool '{self.tool_name}' → {mcp_result['expression']}"
        
        intermediate_results = state["intermediate_results"].copy()
        if op.get("result_key"):
            intermediate_results[op["result_key"]] = result
        
        return {
            **state,
            "result": result,
            "intermediate_results": intermediate_results,
            "history": [history_entry],
            "next_agent": "supervisor",
            "step_count": state["step_count"] + 1
        }


class SupervisorAgent:
    """Supervisor agent using LLM for intelligent planning and routing"""
    
    def __init__(self, llm):
        self.name = "Supervisor Agent"
        self.llm = llm
    
    def parse_with_llm(self, expression: str) -> list:
        """Use LLM to parse expression and create operation plan"""
        prompt = f"""You are a math operation planner. Given a math expression or natural language query, 
break it down into a sequence of simple operations (add, subtract, multiply, divide).

Expression: "{expression}"

Return ONLY a valid JSON array of operations in this exact format:
[
  {{"operation": "add", "operand1": 2, "operand2": 5, "result_key": "temp_0"}},
  {{"operation": "multiply", "operand1": "$temp_0$", "operand2": 5, "result_key": "temp_1"}}
]

Rules:
1. Each operation must have: operation (add/subtract/multiply/divide), operand1, operand2, result_key
2. Use "$temp_N$" to reference previous results (where N is the index)
3. For math expressions like "2*(3+4)", evaluate parentheses first
4. For natural language like "add 2 and 5 then multiply by 3", chain operations sequentially
5. Return ONLY the JSON array, no explanations or markdown

JSON array:"""

        try:
            response = self.llm.invoke(prompt)
            response_text = response.content.strip()
            
            # Remove markdown code blocks if present
            if response_text.startswith('```'):
                response_text = response_text.split('```')[1]
                if response_text.startswith('json'):
                    response_text = response_text[4:]
                response_text = response_text.strip()
            
            operations = json.loads(response_text)
            return operations
            
        except Exception as e:
            print(f"❌ Error parsing with LLM: {e}")
            return []
    
    def execute(self, state: MathState) -> MathState:
        pending = state.get("pending_operations", [])
        
        # First call: parse expression
        if not pending and state.get("step_count", 1) == 1:
            expression = state["expression"]
            
            print(f"\n{'='*70}")
            print(f"🤖 AGENT INVOKED: {self.name}")
            print(f"🧠 SUPERVISOR MODE: Parsing & Planning")
            print(f"📝 Expression: '{expression}'")
            print(f"{'='*70}")
            
            history_entry = f"🤖 {self.name}: Using LLM to analyze '{expression}'"
            
            pending = self.parse_with_llm(expression)
            
            if not pending:
                return {
                    **state,
                    "history": [history_entry, "  ❌ LLM could not parse input"],
                    "next_agent": "end"
                }
            
            return {
                **state,
                "pending_operations": pending,
                "history": [history_entry, f"  📋 LLM planned {len(pending)} operation(s)"],
                "next_agent": "supervisor"
            }
        
        # Route to next agent
        if pending:
            next_op = pending[0]
            remaining = pending[1:]
            
            # Resolve operand references
            intermediate = state.get("intermediate_results", {})
            operand1 = next_op["operand1"]
            operand2 = next_op["operand2"]
            
            if isinstance(operand1, str) and operand1.startswith('$'):
                key = operand1.strip('$')
                operand1 = intermediate.get(key, 0.0)
            
            if isinstance(operand2, str) and operand2.startswith('$'):
                key = operand2.strip('$')
                operand2 = intermediate.get(key, 0.0)
            
            current_op = {**next_op, "operand1": operand1, "operand2": operand2}
            
            # Route to agent
            operation = next_op.get("operation", "").lower()
            routing_map = {
                "add": "add_agent",
                "subtract": "subtract_agent",
                "multiply": "multiply_agent",
                "divide": "divide_agent"
            }
            next_agent = routing_map.get(operation, "end")
            
            print(f"\n{'='*70}")
            print(f"🤖 AGENT INVOKED: {self.name}")
            print(f"🧠 SUPERVISOR MODE: Routing Decision")
            print(f"🔀 Routing to: {next_agent.replace('_', ' ').title()}")
            print(f"{'='*70}")
            
            history_entry = f"🔀 {self.name}: Routing to {next_agent.replace('_', ' ').title()} for '{operation}' operation"
            
            return {
                **state,
                "pending_operations": remaining,
                "current_operation": current_op,
                "history": [history_entry],
                "next_agent": next_agent
            }
        else:
            print(f"\n{'='*70}")
            print(f"🤖 AGENT INVOKED: {self.name}")
            print(f"✅ SUPERVISOR MODE: Workflow Complete")
            print(f"{'='*70}")
            
            history_entry = f"✅ {self.name}: All operations complete!"
            return {
                **state,
                "history": [history_entry],
                "next_agent": "end"
            }


# ============= WORKFLOW SETUP =============

def create_math_workflow(mcp_client):
    """Create LangGraph workflow with MCP-enabled agents"""
    
    supervisor = SupervisorAgent(llm)
    add_agent = AddAgent(mcp_client)
    subtract_agent = SubtractAgent(mcp_client)
    multiply_agent = MultiplyAgent(mcp_client)
    divide_agent = DivideAgent(mcp_client)
    
    workflow = StateGraph(MathState)
    
    # Add supervisor node (synchronous)
    workflow.add_node("supervisor", supervisor.execute)
    
    # Wrap async agent execution for LangGraph
    # With nest_asyncio applied, we can now use asyncio.run safely
    def make_async_wrapper(agent):
        def sync_execute(state):
            return asyncio.run(agent.execute_async(state))
        return sync_execute
    
    # Add agent nodes (async wrapped)
    workflow.add_node("add_agent", make_async_wrapper(add_agent))
    workflow.add_node("subtract_agent", make_async_wrapper(subtract_agent))
    workflow.add_node("multiply_agent", make_async_wrapper(multiply_agent))
    workflow.add_node("divide_agent", make_async_wrapper(divide_agent))
    
    # Define routing function
    def route_from_supervisor(state: MathState) -> str:
        return state["next_agent"]
    
    # Add conditional edges from supervisor
    workflow.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {
            "add_agent": "add_agent",
            "subtract_agent": "subtract_agent",
            "multiply_agent": "multiply_agent",
            "divide_agent": "divide_agent",
            "supervisor": "supervisor",
            "end": END
        }
    )
    
    # Add edges back to supervisor from all agents
    for agent_name in ["add_agent", "subtract_agent", "multiply_agent", "divide_agent"]:
        workflow.add_conditional_edges(
            agent_name,
            route_from_supervisor,
            {
                "supervisor": "supervisor",
                "end": END
            }
        )
    
    # Set entry point
    workflow.set_entry_point("supervisor")
    
    return workflow.compile()


# ============= MAIN EXECUTION =============

async def solve_expression_async(expression: str):
    """Solve expression using MCP-enabled multi-agent system"""
    
    print("🔌 Connecting to MCP server...")
    
    # Connect to MCP server
    mcp_client = MultiServerMCPClient({
        "math_server": {
            "url": "http://localhost:8000/sse",
            "transport": "sse"
        }
    })
    
    # Get available tools
    try:
        tools = await mcp_client.get_tools()
        print("\n🔧 MCP Server Connected - Available Tools:")
        for tool in tools:
            print(f"   • {tool.name}: {tool.description}")
        print()
    except Exception as e:
        print(f"❌ Error connecting to MCP server: {e}")
        print("   Make sure the server is running: python fastmcp_math_server.py")
        return None
    
    # Create workflow
    workflow = create_math_workflow(mcp_client)
    
    # Initial state
    initial_state = {
        "expression": expression,
        "pending_operations": [],
        "current_operation": {},
        "intermediate_results": {},
        "result": 0.0,
        "history": [],
        "next_agent": "supervisor",
        "step_count": 1
    }
    
    # Run workflow
    final_state = workflow.invoke(initial_state)
    return final_state


def solve_expression(expression: str):
    """Synchronous wrapper"""
    return asyncio.run(solve_expression_async(expression))


# ============= MAIN =============

if __name__ == "__main__":
    print("=" * 70)
    print("MATH AGENTIC WORKFLOW - FastMCP Integration")
    print("Multi-Agent System with LangChain MCP Client")
    print("=" * 70)
    print()
    print("⚠️  IMPORTANT: Make sure FastMCP server is running first!")
    print("   Terminal 1: python fastmcp_math_server.py")
    print("   Terminal 2: python mcp_client.py")
    print()
    
    # Test expressions
    test_inputs = [
        "add 2 and 5 then multiply by 3",
        "2*(3+4)",
        "(15+5)-(10-2)",
    ]
    
    for expr in test_inputs:
        print(f"\n\n{'#'*70}")
        print(f"📐 INPUT EXPRESSION: {expr}")
        print(f"{'#'*70}")
        
        try:
            result = solve_expression(expr)
            
            if result:
                print(f"\n{'='*70}")
                print("📋 EXECUTION HISTORY:")
                print(f"{'='*70}")
                for step in result["history"]:
                    print(step)
                
                print(f"\n{'='*70}")
                print(f"🎉 FINAL RESULT: {result['result']}")
                print(f"{'='*70}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("✅ All tests completed!")
    print("=" * 70)