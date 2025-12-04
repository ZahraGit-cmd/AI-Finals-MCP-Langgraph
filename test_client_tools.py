from langchain_openai import ChatOpenAI
import os 
import httpx 
import tiktoken
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import asyncio
from mc_client import MCPClient
API_KEY="sk-QibkR-xrOxD2AsMgdaL0Pg"
tiktoken_cache_dir = "./token"
os.environ["TIKTOKEN_CACHE_DIR"] = tiktoken_cache_dir 

client = httpx.Client(verify=False) 
# LLM and Embedding setup 
llm = ChatOpenAI( 
base_url="https://genailab.tcs.in", 
model="azure_ai/genailab-maas-DeepSeek-V3-0324", 
api_key=API_KEY, 
http_client=client 
)
def mcp_tool_to_openai(tool):
    return {
        "name": tool.name,
        "description": tool.description or "",
        "parameters": tool.inputSchema,  # MCP already provides JSON schema
    }


async def test_tools():
    async with MCPClient() as session:
        tools_resp = await session.list_tools()
        #print(tools_resp.tools)
        openai_tools = [mcp_tool_to_openai(t) for t in tools_resp.tools]
        llm_with_ttols = llm.bind_tools(openai_tools)
        print(llm_with_ttols.invoke("subtract 2 and 2 by using only provided tools"))
    
if __name__ == "__main__":
    asyncio.run(test_tools())