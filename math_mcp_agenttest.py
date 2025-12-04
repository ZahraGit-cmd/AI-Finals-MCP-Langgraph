from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent   # updated import
import os, httpx, asyncio
from mc_client import MCPClient

API_KEY = "sk-QibkR-xrOxD2AsMgdaL0Pg"  # your key
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

async def run_agent():
    async with MCPClient() as session:
        tools_resp = await session.list_tools()
        print(tools_resp)
        openai_tools = [mcp_tool_to_openai(t) for t in tools_resp.tools]
        print(openai_tools)
        llm_with_tools = llm.bind_tools(openai_tools)

        # Option A: use llm_with_tools directly
        response = llm_with_tools.invoke("add 2 and 2 using only provided tools")
        print("Response content:",response)

        # Option B: use create_react_agent with openai_tools
        agent = create_react_agent(llm, openai_tools)
        agent_response = agent.invoke({"messages": "what's (3 + 5) * 12?"})
        print("AGENT RESPONSE:", agent_response)

        return agent_response

if __name__ == "__main__":
    result = asyncio.run(run_agent())
    print(result)
