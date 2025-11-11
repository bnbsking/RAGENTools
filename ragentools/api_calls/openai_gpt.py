from typing import List, Optional, Union, Type
from xmlrpc import client

from openai import OpenAI, AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_fixed

from .base_api import BaseAPI


class OpenAIGPTChatAPI(BaseAPI):
    """
    This class wraps OpenAI GPT API calls which has:
    1 async, 2 retry, 3 token count with price, 4 pydantic response, 5 multi-modal input
    """
    def __init__(
            self,
            api_key: str,
            model_name: str,
            base_url: str = "https://api.studio.nebius.com/v1/",
            price_csv_path: str = "",
            retry_times: int = 3,
            retry_sec: int = 5
        ):
        super().__init__(api_key, model_name, price_csv_path)
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.aclient = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self.retry_times = retry_times
        self.retry_sec = retry_sec

    def prompt_to_messages(self, prompt: Union[str, List]) -> List[dict]:
        if isinstance(prompt, str):
            return [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        else:
            return prompt

    def run(
            self,
            prompt: Union[str, List],
            response_format: Union[None, Type] = None,
            temperature: float = 0.7,
            retry_times: Optional[int] = None,
            retry_sec: Optional[int] = None
        ) -> Union[str, dict]:  # process 1 query (prompt) at once
        retry_times = retry_times if retry_times is not None else self.retry_times
        retry_sec = retry_sec if retry_sec is not None else self.retry_sec
        @retry(stop=stop_after_attempt(retry_times), wait=wait_fixed(retry_sec))
        def _call_api() -> Union[str, Type]:
            args = {
                "model": self.model_name,
                "messages": self.prompt_to_messages(prompt),
                "temperature": temperature
            }
            if response_format:
                args["response_format"] = response_format
                response = self.client.chat.completions.parse(**args)
            else:
                response = self.client.chat.completions.create(**args)
            self.update_acc_tokens(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens
            )
            return response.choices[0].message.parsed if response_format else response.choices[0].message.content
        return _call_api()

    async def arun(
            self,
            prompt: Union[str, List],
            response_format: Union[None, Type] = None,
            temperature: float = 0.7,
            retry_times: int = 3,
            retry_sec: int = 5
        ) -> Union[str, Type]:  # process 1 query (prompt) at once
        @retry(stop=stop_after_attempt(retry_times), wait=wait_fixed(retry_sec))
        async def _call_api() -> Union[str, Type]:
            args = {
                "model": self.model_name,
                "messages": self.prompt_to_messages(prompt),
                "temperature": temperature
            }
            if response_format:
                args["response_format"] = response_format
                response = await self.aclient.chat.completions.parse(**args)
            else:
                response = await self.aclient.chat.completions.create(**args)
            self.update_acc_tokens(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens
            )
            return response.choices[0].message.parsed if response_format else response.choices[0].message.content
        return await _call_api()
    

class OpenAIEmbeddingAPI(BaseAPI):
    """
    This class wraps OpenAI Embedding API calls which has:
    1 async, 2 retry, 3 token count with price, 4 batching
    """
    def __init__(
            self,
            api_key: str,
            model_name: str,
            base_url: str = "https://api.studio.nebius.com/v1/",
            price_csv_path: str = "",
            batch_size: int = 64,
            retry_times: int = 3,
            retry_sec: int = 5
        ):
        super().__init__(api_key, model_name, price_csv_path)
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.aclient = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self.batch_size = batch_size
        self.retry_times = retry_times
        self.retry_sec = retry_sec

    def run_batches(
            self,
            texts: List[str],
            retry_times: Optional[int] = None,
            retry_sec: Optional[int] = None
        ) -> List[List[float]]:  # process len(texts) at once
        retry_times = retry_times if retry_times is not None else self.retry_times
        retry_sec = retry_sec if retry_sec is not None else self.retry_sec
        @retry(stop=stop_after_attempt(retry_times), wait=wait_fixed(retry_sec))
        def _call_api(batch: List[str]) -> List[float]:
            response = self.client.embeddings.create(
                model=self.model_name,
                input=batch
            )
            self.update_acc_tokens(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=0
            )
            return [d.embedding for d in response.data]
        out = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i+self.batch_size]
            out.extend(_call_api(batch))
        return out
    
    async def arun_batches(
            self,
            texts: List[str],
            retry_times: Optional[int] = None,
            retry_sec: Optional[int] = None
        ) -> List[List[float]]:  # process len(texts) at once
        retry_times = retry_times if retry_times is not None else self.retry_times
        retry_sec = retry_sec if retry_sec is not None else self.retry_sec
        @retry(stop=stop_after_attempt(retry_times), wait=wait_fixed(retry_sec))
        async def _call_api(batch: List[str]) -> List[float]:
            response = await self.aclient.embeddings.create(
                model=self.model_name,
                input=batch
            )
            self.update_acc_tokens(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=0
            )
            return [d.embedding for d in response.data]
        out = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i+self.batch_size]
            out.extend(await _call_api(batch))
        return out
    