from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

import requests
import base64

from src.config_loader import get_config
from src.embedder import load_embedder, load_image_embedder
from src.vectorstore import load_vector_store
from src.rag_pipeline import generate_answer, explain_image_pipeline
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
config_file = os.path.join(ROOT_DIR , "config", "multi_modal.yaml")
config = get_config(config_file)

# Initialize FastAPI app
app = FastAPI(title="Multimodal RAG App")

# Input schema
class QueryRequest(BaseModel):
    question: str
    image_path:str

@app.on_event("startup")
async def startup_event():
    app.state.image_embedder = load_image_embedder(config['embedding']['image'])    
    app.state.llm = load_llm(config['llm'])
    app.state.last_prompt = None

# API route
@app.get("/show_last_prompt")
def show_last_prompt():
    return app.state.last_prompt


@app.post("/explain_image")
def explain_image(request: QueryRequest):
    image_path = request.image_path
    image_path = os.path.join(os.getcwd(), 'data', 'images', image_path)
    
    answer = explain_image_pipeline(
            question=request.question,
            image_embedder = app.state.image_embedder,
            image_path=image_path,
            llm=app.state.llm
        )
    return {"answer": answer}

@app.post("/explain_image_basic")
def explain_image_basic(request: QueryRequest):
    image_path = request.image_path
    image_path = os.path.join(os.getcwd(), 'data', 'images', image_path)
    with open(image_path, "rb") as img_file:
        image_embeddings = base64.b64encode(img_file.read()).decode("utf-8")

    model = 'llava1.5'
    url = 'http://localhost:11434/api/generate'

    prompt = "Describe what’s happening in this image."
    payload = {"model": model, "prompt": prompt, "stream" : False, "images": [image_embeddings]}          
    response = requests.post(url, json=payload)
    return {"answer": response.json()}

if __name__ == "__main__":
    app_path = Path(__file__).resolve().with_suffix('').name  # gets filename without .py
    uvicorn.run(f"{app_path}:app", host="0.0.0.0", port=8000, reload=True)
