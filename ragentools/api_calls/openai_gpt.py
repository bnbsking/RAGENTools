from typing import List, Union, Type
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
            price_csv_path: str = "/app/ragentools/api_calls/prices.csv",
        ):
        super().__init__(api_key, model_name, price_csv_path)
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.aclient = AsyncOpenAI(base_url=base_url, api_key=api_key)

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
            retry_times: int = 3,
            retry_sec: int = 5
        ) -> Union[str, Type]:  # process 1 query (prompt) at once
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
    