from .base import BaseLLM
import requests

class OllamaModel(BaseLLM):
    def __init__(self, config):
        self.model = config['model']
        self.url = config['url']

    def generate(self, prompt: str) -> str:
        response = requests.post(self.url, json={"model": self.model, "prompt": prompt, "stream" : False})
        response.raise_for_status()
        return response.json().get("response")

def load_llm(config):
    provider = config["provider"]
    if provider == "ollama":
        model_config = config[provider]
        return OllamaModel(model_config)
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")
