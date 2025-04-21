from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

from src.config_loader import get_config
from src.embedder import load_text_embedder
from src.vectorstore import load_vector_store
from src.rag_pipeline import generate_answer_with_filtering
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
config_file = os.path.join(ROOT_DIR , "config", "rag_with_metadata_filtering.yaml")
config = get_config(config_file)


# Initialize FastAPI app
app = FastAPI(title="Flexible RAG App")

# Input schema
class QueryRequest(BaseModel):
    question: str

@app.on_event("startup")
async def startup_event():
    app.state.embedder = load_text_embedder(config['embedding'])
    app.state.vectorstore = load_vector_store(config['vector_store'])
    app.state.llm = load_llm(config['llm'])

# API route
@app.post("/ask")
def ask_rag(request: QueryRequest):
    embedder = app.state.embedder
    vectorstore = app.state.vectorstore
    llm = app.state.llm

    answer = generate_answer_with_filtering(
        question=request.question,
        embedder=embedder,
        vectorstore=vectorstore,
        llm=llm
    )
    return {"answer": answer}

if __name__ == "__main__":
    app_path = Path(__file__).resolve().with_suffix('').name  # gets filename without .py
    uvicorn.run(f"{app_path}:app", host="0.0.0.0", port=8000, reload=True)
