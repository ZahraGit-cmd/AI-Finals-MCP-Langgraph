from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
import httpx
import os

app = FastAPI()

# Allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # in production, restrict to your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request schema
class ChatRequest(BaseModel):
    message: str

# Configure LLM client
client = httpx.Client(verify=False)
llm = ChatOpenAI(
    base_url="https://genailab.tcs.in",
    model="azure_ai/genailab-maas-DeepSeek-V3-0324",
    api_key="sk-E1TXA6DgfwF8pN1m5Fiy4g",  # ⚠️ use env var in production
    http_client=client
)

@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    user_message = req.message
    response = llm.invoke(user_message)
    return {"reply": str(response.content)}
