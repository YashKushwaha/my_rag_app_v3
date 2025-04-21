from fastapi import FastAPI
from pydantic import BaseModel
from src.agent.base_agent import Agent
from src.llms.factory import get_llm_client
from src.agent.agent_configs import RAG_AGENT_CONFIG

llm_client = get_llm_client(...)
agent = Agent(llm_client, **RAG_AGENT_CONFIG)

app = FastAPI()

class Query(BaseModel):
    question: str

@app.post("/ask")
def ask_agent(query: Query):
    response = agent.run(query.question)
    return {"response": response}
