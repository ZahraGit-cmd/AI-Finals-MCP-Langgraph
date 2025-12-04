"""
Math Equation Solver - Agentic Workflow using LangGraph with LLM

This implements a supervisor-agent pattern where:
- Supervisor Agent: Uses LLM to intelligently parse and plan operations
- Add Agent: Handles addition operations
- Subtract Agent: Handles subtraction operations
- Multiply Agent: Handles multiplication operations
- Divide Agent: Handles division operations

The supervisor uses an LLM to understand natural language and decide agent routing.
"""

from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, END
import operator
import json
import httpx
from langchain_openai import ChatOpenAI


# Initialize LLM
client = httpx.Client(verify=False)
llm = ChatOpenAI(
    base_url="https://genailab.tcs.in",
    model="azure_ai/genailab-maas-DeepSeek-V3-0324",
    api_key="sk-QibkR-xrOxD2AsMgdaL0Pg",
    http_client=client
)


# Define the state that flows through the workflow
class MathState(TypedDict):
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
    """Agent specialized in addition operations"""
    
    def __init__(self):
        self.name = "Add Agent"
    
    def execute(self, state: MathState) -> MathState:
        op = state["current_operation"]
        result = op["operand1"] + op["operand2"]
        history_entry = f"  Step {state['step_count']}: ➕ {self.name} - {op['operand1']} + {op['operand2']} = {result}"
        
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
    """Agent specialized in subtraction operations"""
    
    def __init__(self):
        self.name = "Subtract Agent"
    
    def execute(self, state: MathState) -> MathState:
        op = state["current_operation"]
        result = op["operand1"] - op["operand2"]
        history_entry = f"  Step {state['step_count']}: ➖ {self.name} - {op['operand1']} - {op['operand2']} = {result}"
        
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
    """Agent specialized in multiplication operations"""
    
    def __init__(self):
        self.name = "Multiply Agent"
    
    def execute(self, state: MathState) -> MathState:
        op = state["current_operation"]
        result = op["operand1"] * op["operand2"]
        history_entry = f"  Step {state['step_count']}: ✖️ {self.name} - {op['operand1']} × {op['operand2']} = {result}"
        
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
    """Agent specialized in division operations"""
    
    def __init__(self):
        self.name = "Divide Agent"
    
    def execute(self, state: MathState) -> MathState:
        op = state["current_operation"]
        
        if op["operand2"] == 0:
            history_entry = f"  Step {state['step_count']}: ➗ {self.name} - Error: Cannot divide by zero"
            return {
                **state,
                "result": float('inf'),
                "history": [history_entry],
                "next_agent": "end"
            }
        
        result = op["operand1"] / op["operand2"]
        history_entry = f"  Step {state['step_count']}: ➗ {self.name} - {op['operand1']} ÷ {op['operand2']} = {result}"
        
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
    """Supervisor agent that uses LLM to parse expressions and intelligently route to agents"""
    
    def __init__(self, llm):
        self.name = "Supervisor Agent"
        self.llm = llm
    
    def parse_with_llm(self, expression: str) -> list:
        """
        Use LLM to parse the expression and create an operation plan
        """
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
4. For natural language like "add 2 and 5 and multiply 5", chain operations sequentially
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
            
            # Parse JSON
            operations = json.loads(response_text)
            return operations
            
        except Exception as e:
            print(f"Error parsing with LLM: {e}")
            return []
    
    def decide_next_agent_with_llm(self, current_state: str, pending_ops: list) -> str:
        """
        Use LLM to decide which agent to call next
        """
        if not pending_ops:
            return "end"
        
        next_op = pending_ops[0]
        operation = next_op.get("operation", "").lower()
        
        # Map operations to agents
        routing_map = {
            "add": "add_agent",
            "subtract": "subtract_agent",
            "multiply": "multiply_agent",
            "divide": "divide_agent"
        }
        
        return routing_map.get(operation, "end")
    
    def execute(self, state: MathState) -> MathState:
        pending = state.get("pending_operations", [])
        
        # First call: parse the expression using LLM
        if not pending and state.get("step_count", 1) == 1:
            expression = state["expression"]
            history_entry = f"🤖 {self.name}: Using LLM to analyze '{expression}'"
            
            # Use LLM to parse
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
        
        # Subsequent calls: route to next agent using LLM decision
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
            
            # Use LLM to decide next agent
            next_agent = self.decide_next_agent_with_llm(
                f"Current result: {state.get('result', 0)}", 
                pending
            )
            
            history_entry = f"🔀 {self.name}: LLM routing to {next_agent.replace('_', ' ').title()}"
            
            return {
                **state,
                "pending_operations": remaining,
                "current_operation": current_op,
                "history": [history_entry],
                "next_agent": next_agent
            }
        else:
            history_entry = f"✅ {self.name}: All operations complete!"
            return {
                **state,
                "history": [history_entry],
                "next_agent": "end"
            }


# ============= WORKFLOW SETUP =============

def create_math_workflow():
    """Create the LangGraph workflow with all agents and LLM-powered supervisor"""
    
    supervisor = SupervisorAgent(llm)
    add_agent = AddAgent()
    subtract_agent = SubtractAgent()
    multiply_agent = MultiplyAgent()
    divide_agent = DivideAgent()
    
    workflow = StateGraph(MathState)
    
    workflow.add_node("supervisor", supervisor.execute)
    workflow.add_node("add_agent", add_agent.execute)
    workflow.add_node("subtract_agent", subtract_agent.execute)
    workflow.add_node("multiply_agent", multiply_agent.execute)
    workflow.add_node("divide_agent", divide_agent.execute)
    
    def route_from_supervisor(state: MathState) -> str:
        return state["next_agent"]
    
    def route_back_to_supervisor(state: MathState) -> str:
        return state["next_agent"]
    
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
    
    for agent_name in ["add_agent", "subtract_agent", "multiply_agent", "divide_agent"]:
        workflow.add_conditional_edges(
            agent_name,
            route_back_to_supervisor,
            {
                "supervisor": "supervisor",
                "end": END
            }
        )
    
    workflow.set_entry_point("supervisor")
    
    return workflow.compile()


# ============= USAGE EXAMPLE =============

def solve_expression(expression: str):
    """
    Solve a math expression or natural language query using LLM-powered routing
    
    Args:
        expression: The expression to solve
    
    Returns:
        Final state with result and execution history
    """
    workflow = create_math_workflow()
    
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
    
    final_state = workflow.invoke(initial_state)
    
    return final_state


if __name__ == "__main__":
    print("=" * 70)
    print("MATH AGENTIC WORKFLOW - LLM-POWERED SUPERVISOR")
    print("=" * 70)
    
    # Test different inputs
    test_inputs = [
        # Natural language
        "add 2 and 5 and multiply 5",
        "multiply 3 and 4 then add 10",
        
        # Math expressions
        "2*(3+4)",
        "(15+5)-(10-2)",
        
        # Complex natural language
        "take 20, subtract 5, then multiply by 3",
    ]
    
    for expr in test_inputs:
        print(f"\n📐 Input: {expr}")
        print("-" * 70)
        
        try:
            result = solve_expression(expr)
            
            for step in result["history"]:
                print(step)
            
            print(f"\n🎉 Final Result: {result['result']}")
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print("=" * 70)