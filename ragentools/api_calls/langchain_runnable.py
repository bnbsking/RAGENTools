from typing import List

from langchain_core.runnables import Runnable


class ChatRunnable(Runnable):
    """
    Base on benifits of GoogleGeminiChatAPI/OpenAIGPTChatAPI,
    also allow scalabilty with LangChain.
    """
    def __init__(self, api, **kwargs):
        # api can be GoogleGeminiChatAPI or OpenAIGPTChatAPI
        self.api = api(**kwargs)

    def run(self, input: dict, config = None) -> dict:
        return self.api.run(
            prompt=input["prompt"],
            response_format=input["response_format"],
            temperature=input.get("temperature", 0.7),
            retry_times=input.get("retry_times", 3),
            retry_sec=input.get("retry_sec", 5)
        )

    async def arun(self, input: dict, config= None) -> dict:
        return await self.api.arun(
            prompt=input["prompt"],
            response_format=input["response_format"],
            temperature=input.get("temperature", 0.7),
            retry_times=input.get("retry_times", 3),
            retry_sec=input.get("retry_sec", 5),
        )

    def invoke(self, state: dict, config = None) -> dict:
        out = self.run(state)
        return state | out


class EmbRunnable(Runnable):
    def __init__(self, api, **kwargs):
        # api can be GoogleGeminiEmbeddingAPI or OpenAIGPTEmbeddingAPI
        self.api = api(**kwargs)

    def run_batches(self, input: dict, config = None) -> List[List[float]]:
        return self.api.run_batches(
            texts=input["texts"],
            dim=input["dim"],
            retry_times=input.get("retry_times", 3),
            retry_sec=input.get("retry_sec", 5)
        )

    async def arun_batches(self, input: dict, config= None) -> List[List[float]]:
        return await self.api.arun_batches(
            texts=input["texts"],
            dim=input["dim"],
            retry_times=input.get("retry_times", 3),
            retry_sec=input.get("retry_sec", 5),
        )

    def invoke(self, state: dict, config = None) -> dict:
        out = self.run_batches(state)
        return state | out
