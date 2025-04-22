from fastapi import FastAPI
from pydantic import BaseModel
from src.agent.base_agent import Agent
from src.llms import load_llm, DummyLLM
from src.agent.agent_configs import RAG_AGENT_CONFIG
from src.agent.prompt_template import SYSTEM_PROMPT
from src.agent.tools import TOOLS
import uvicorn
from pathlib import Path
import os
from src.config_loader import get_config

print('Working directory set as ', os.getcwd())

# Load configuration
print(Path(__file__).resolve().parents[1] )
config_file = os.path.join("config", "run_rag_agent.yaml")
config = get_config(config_file)

print('LLM client -> ', config['llm'])
llm_client = load_llm(config['llm'])
agent = Agent(**dict(llm=llm_client, system_prompt = SYSTEM_PROMPT, tools =  TOOLS))

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