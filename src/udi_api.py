from typing import Union
from pydantic import BaseModel
from fastapi import FastAPI
from udi_agent import UDIAgent
from fastapi.middleware.cors import CORSMiddleware



app = FastAPI()

origins = [
    "*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# init agent
agent = UDIAgent(
    model_name="agenticx/UDI-VIS-Beta-v0-Llama-3.1-8B",
    vllm_server_url="http://localhost",
    vllm_server_port=8080
)


@app.get("/")
def read_root():
    return {
        "service": "UDIAgent API",
        "status": "running",
        "endpoints": [
            {"path": "/", "method": "GET", "description": "API status and info"},
            {"path": "/v1/chat/completions", "method": "POST", "description": "Chat completions endpoint (OpenAI compatible)"}
        ]
    }

class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[dict]

# match openai api endpoints
# https://platform.openai.com/docs/api-reference/chat
@app.post("/v1/chat/completions")
def chat_completions(request: ChatCompletionRequest):
    response = agent.chat_completions(messages=request.messages)
    return { "response": response } 


class CompletionRequest(BaseModel):
    model: str
    messages: list[dict]
    tools: list[dict]

@app.post("/v1/completions")
def completions(request: CompletionRequest):
    response = agent.completions(messages=request.messages, tools=request.tools)
    return { "response": response } 