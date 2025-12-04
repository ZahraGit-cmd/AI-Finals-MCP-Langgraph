"""
LangChain MultiServer MCP Client

1-1 connection between client and MCP server, inside LangChain we have multiple MCP clients to easily connect to multiple MCP servers.
"""
import asyncio

from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()
from langchain_openai import ChatOpenAI
import os, sys
import httpx 
import tiktoken
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import asyncio
from mcp_client import MCPClient
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
from langchain.agents import create_agent

async def createagent():
    client = MultiServerMCPClient({"tcp_server":{"url":"http://localhost:8000/sse", "transport":"sse"}})
    tools = await client.get_tools()
    #print(tools)
    agent =  create_agent(llm,tools)
    return agent
    
async def main():
    agent = await createagent()
    resp = await agent.ainvoke({"messages": [{"role": "user", "content": "What is current Chicago wheather?"}]})
    print(resp["messages"][-1].content)

if __name__ == "__main__":
    asyncio.run(main())

