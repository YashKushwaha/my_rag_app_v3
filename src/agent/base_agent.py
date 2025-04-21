class Agent:
    def __init__(self, llm, system_prompt: str, tools: dict):
        self.llm = llm
        self.system_prompt = system_prompt
        self.tools = tools

    def run(self, user_input: str) -> str:
        # Standard chat -> tool invocation -> tool response -> final LLM response loop
        ...
