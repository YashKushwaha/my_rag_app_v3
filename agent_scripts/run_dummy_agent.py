from fastapi import FastAPI
from pydantic import BaseModel
from src.agent.base_agent import Agent
from src.llms import load_llm, DummyLLM
#from src.agent.agent_configs import RAG_AGENT_CONFIG
import uvicorn
from pathlib import Path

llm_client = DummyLLM() #get_llm_client(...)
agent = Agent(**dict(llm=llm_client, system_prompt = '', tools =  {}))

app = FastAPI()

class Query(BaseModel):
    question: str

@app.post("/ask")
def ask_agent(query: Query):
    response = agent.run(query.question)
    return {"response": response}


if __name__ == "__main__":
    app_path = Path(__file__).resolve().with_suffix('').name  # gets filename without .py
    uvicorn.run(f"{app_path}:app", host="0.0.0.0", port=8000, reload=True)