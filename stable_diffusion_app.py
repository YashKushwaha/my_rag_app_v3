from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

import requests
import base64

from src.config_loader import get_config
from src.embedder import load_embedder, load_image_embedder
from src.vectorstore import load_vector_store
from src.rag_pipeline import generate_answer, explain_image_pipeline, image_generation_pipeline
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
config_file = os.path.join(ROOT_DIR , "config", "image_generation.yaml")
config = get_config(config_file)

# Initialize FastAPI app
app = FastAPI(title="Image Generation App")

# Input schema
class QueryRequest(BaseModel):
    question: str


@app.on_event("startup")
async def startup_event():
    #app.state.image_embedder = load_image_embedder(config['embedding']['image'])    
    app.state.llm = load_llm(config['llm'])
    app.state.last_prompt = None

# API route
@app.get("/show_last_prompt")
def show_last_prompt():
    return app.state.last_prompt

@app.post("/generate_image_basic")
def generate_image_basic(request: QueryRequest):
    question=request.question,
    model = 'sd3.5'
    url = 'http://localhost:11434/api/generate'

    prompt = "Generate an image of man walking on moon"
    payload = {"model": model, "prompt": prompt, "stream" : False}          
    response = requests.post(url, json=payload)
    #print(response)
    return {"answer": response.status_code}


@app.post("/generate_image")
def generate_image(request: QueryRequest):
    
    answer = image_generation_pipeline(
            question=request.question,
            llm=app.state.llm
        )
    return {"answer": answer}
### The model is not running locally on ollama will come back to this later

if __name__ == "__main__":
    app_path = Path(__file__).resolve().with_suffix('').name  # gets filename without .py
    uvicorn.run(f"{app_path}:app", host="0.0.0.0", port=8000, reload=True)
