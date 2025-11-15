from typing import List, Optional

from langchain_core.runnables import Runnable

from ragentools.prompts import get_prompt_and_response_format


class ChatRunnable(Runnable):
    """
    Base on benifits of GoogleGeminiChatAPI/OpenAIGPTChatAPI,
    also allow scalabilty with LangChain.
    """
    def __init__(self, api, prompt_path: Optional[str] = None):
        # api can be GoogleGeminiChatAPI or OpenAIGPTChatAPI
        self.api = api
        if prompt_path:
            self.prompt, self.response_format = get_prompt_and_response_format(prompt_path)

    def run(self, input: dict, config = None) -> dict:
        return self.api.run(
            prompt=input["prompt"],
            response_format=input.get("response_format", None),
            temperature=input.get("temperature", 0.7),
        )

    async def arun(self, input: dict, config= None) -> dict:
        return await self.api.arun(
            prompt=input["prompt"],
            response_format=input.get("response_format", None),
            temperature=input.get("temperature", 0.7),
        )

    def invoke(self, state: dict, config = None) -> dict:
        out = self.run(state)
        return state | out


class EmbRunnable(Runnable):
    def __init__(self, api):
        # api can be GoogleGeminiEmbeddingAPI or OpenAIGPTEmbeddingAPI
        self.api = api

    def run_batches(self, input: dict, config = None) -> List[List[float]]:
        return self.api.run_batches(
            texts=input["texts"],
            dim=input["dim"]
        )

    async def arun_batches(self, input: dict, config= None) -> List[List[float]]:
        return await self.api.arun_batches(
            texts=input["texts"],
            dim=input["dim"]
        )

    def invoke(self, state: dict, config = None) -> dict:
        out = self.run_batches(state)
        return state | out
