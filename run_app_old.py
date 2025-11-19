from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import requests
import json

app = FastAPI(title="Trading AI Assistant")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Memory for chat
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

    system_prompt = (
        "You are a professional trading education assistant. "
        "You explain swing trading, indicators, price action, risk management, and stock analysis "
        "in a structured and easy-to-understand way. Avoid financial advice."
    )

    prompt = f"System: {system_prompt}\n"
    for msg in chat_history + request.messages:
        prompt += f"{msg['role']}: {msg['content']}\n"
    prompt += "assistant:"

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

    chat_history.extend(request.messages)
    chat_history.append({"role": "assistant", "content": output_text})
    chat_history = chat_history[-20:]

    return {
        "id": "chatcmpl-local",
        "object": "chat.completion",
        "model": request.model,
        "choices": [{"message": {"role": "assistant", "content": output_text}}],
    }
