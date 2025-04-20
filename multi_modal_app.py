from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

import requests
import base64

from src.config_loader import get_config
from src.embedder import load_embedder, load_image_embedder
from src.vectorstore import load_vector_store
from src.rag_pipeline import generate_answer
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
    #app.state.text_embedder = load_embedder(config['embedding']['text'])
    #app.state.image_embedder = load_image_embedder(config['embedding']['image'])    
    #app.state.vectorstore = load_vector_store(config['vector_store'])
    app.state.llm = load_llm(config['llm'])

# API route
@app.post("/explain_image")
def explain_image(request: QueryRequest):
    #text_embedder = app.state.text_embedder
    #image_embedder = app.state.image_embedder
    #vectorstore = app.state.vectorstore
    llm = app.state.llm

    image_path = request.image_path
    image_path = os.path.join(os.getcwd(), 'data', 'images', image_path)

    with open(image_path, "rb") as img_file:
        image_b64 = base64.b64encode(img_file.read()).decode("utf-8")
    
    system_prompt = 'You are a helpful assistnat'
    prompt = f'{system_prompt}\nUSER:<images>\{request.question}\nASSISTANT'
    payload = {
        "model": 'llava1.5',
        "prompt": request.question,
        "images": [image_b64],
        "stream": False
    }
    # Send request to Ollama server
    response = requests.post("http://localhost:11434/api/generate", json=payload)

    # Return response text
    if response.ok:
        return response.json()["response"]
    else:
        raise Exception(f"Error from Ollama: {response.text}")


if __name__ == "__main__":
    app_path = Path(__file__).resolve().with_suffix('').name  # gets filename without .py
    uvicorn.run(f"{app_path}:app", host="0.0.0.0", port=8000, reload=True)
