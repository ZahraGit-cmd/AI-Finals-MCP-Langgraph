from langchain_openai import ChatOpenAI  
import os  
import httpx  
client = httpx.Client(verify=False) 
llm = ChatOpenAI( 
	base_url="https://genailab.tcs.in", 
	model = "azure_ai/genailab-maas-DeepSeek-V3-0324", 
	api_key="sk-E1TXA6DgfwF8pN1m5Fiy4g", 
	http_client = client 
) 
print(llm.invoke("Hi"))
