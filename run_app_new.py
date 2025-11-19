from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import requests
import json
from rag_helper import TradingKnowledgeBase

app = FastAPI(title="AI Trading Chatbot with RAG")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Load FAISS
kb = TradingKnowledgeBase()

# Chat memory
chat_history = []

class ChatRequest(BaseModel):
    model: str = "llama3.2:1b"
    messages: list


@app.get("/", response_class=HTMLResponse)
async def homepage(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/v1/chat/completions")
async def chat_completion(request: ChatRequest):
    global chat_history

    # Get user message
    user_message = request.messages[-1]["content"]

    context = kb.retrieve_context(user_message)

    chat_history.append({"role": "user", "content": user_message})

    # System instruction
    system_prompt = (
        "You are a trading education assistant. Use the provided context to answer. "
        "DO NOT give financial advice. Explain concepts clearly."
    )

    # Final prompt
    prompt = (
        f"System: {system_prompt}\n\n"
        f"Retrieved Context:\n{context}\n\n"
        f"Chat History:\n"
    )

    for msg in chat_history:
        prompt += f"{msg['role']}: {msg['content']}\n"

    prompt += "assistant:"

    # Send to Local Llama model
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": request.model, "prompt": prompt},
        stream=True,
    )

    output_text = ""
    for line in response.iter_lines():
        if line:
            data = json.loads(line)
            if "response" in data:
                output_text += data["response"]


    chat_history.append({"role": "assistant", "content": output_text})

    # Final Answer
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": output_text
                }
            }
        ]
    }
