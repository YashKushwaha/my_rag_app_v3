from fastapi import FastAPI
import uvicorn

from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import uuid
from src.config_loader import get_config
from src.embedder import load_text_embedder
from src.vectorstore import load_vector_store
from src.rag_pipeline import generate_answer, generate_answer_with_history
from src.llms import load_llm

from pathlib import Path
import os
import sys
import warnings
warnings.filterwarnings("ignore")

ROOT_DIR = Path(os.path.dirname(__file__)).resolve()
os.chdir(ROOT_DIR)
print('Working directory set as ', os.getcwd())

# Load configuration
print(Path(__file__).resolve().parents[1] )
config_file = os.path.join(ROOT_DIR , "config", "settings.yaml")
config = get_config(config_file)

# Initialize FastAPI app
app = FastAPI(title="RAG App with History")

# Input schema
class QueryRequest(BaseModel):
    question: str
    #history: List[Dict[str, str]] = []
    session_id: Optional[str] = None  # Optional session_id for tracking user sessions


@app.on_event("startup")
async def startup_event():
    app.state.embedder = load_text_embedder(config['embedding'])
    app.state.vectorstore = load_vector_store(config['vector_store'])
    app.state.llm = load_llm(config['llm'])
    app.state.chat_histories = {}

@app.get("/history")
def get_history(session_id):
    return {'history':app.state.chat_histories[session_id]}

@app.get("/session_list")
def get_session_list():
    return {'session_ids':list(app.state.chat_histories.keys())}

@app.get("/openapi.json")
def get_openapi():
    return app.openapi()

@app.post("/ask")
def ask_rag(request: QueryRequest):
    # If session_id is not provided, generate a new one
    session_id = request.session_id if request.session_id else str(uuid.uuid4())

    # Access or create session history for this session_id
    if session_id not in app.state.chat_histories:
        app.state.chat_histories[session_id] = []

    history = app.state.chat_histories[session_id]

    # Generate answer with history
    answer = generate_answer_with_history(
        question=request.question,
        embedder=app.state.embedder,
        vectorstore=app.state.vectorstore,
        llm=app.state.llm,
        chat_history=history,
    )

    # Append new exchange to history
    history.append({"role": "user", "content": request.question})
    history.append({"role": "assistant", "content": answer})

    return {"answer": answer, "session_id": session_id}

if __name__ == "__main__":
    app_path = Path(__file__).resolve().with_suffix('').name  # gets filename without .py
    uvicorn.run(f"{app_path}:app", host="0.0.0.0", port=8000, reload=True)
