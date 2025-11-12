import base64
from typing import Dict, List, Optional, Union, Type

from openai import OpenAI, AsyncOpenAI
from pydantic import BaseModel

from .base_api import BaseAPI
from ragentools.common.async_funcs import batch_executer_for_func, batch_executer_for_afunc
from ragentools.common.dynamic_retry import dynamic_retry


def img_path_to_openai_url(img_path: str) -> str:
    with open(img_path, "rb") as image_file:
        b64 = base64.b64encode(image_file.read()).decode('utf-8')
    return f"data:image/jpeg;base64,{b64}"


class OpenAIGPTChatAPI(BaseAPI):
    """
    This class wraps OpenAI GPT API calls which has:
    1 async, 2 retry, 3 token count with price, 4 pydantic response, 5 multi-modal input
    """
    def __init__(
            self,
            api_key: str,
            model_name: str,
            base_url: Optional[str] = None,  #https://api.studio.nebius.com/v1/
            price_csv_path: str = "",
            retry_times: int = 3,
            retry_sec: int = 5
        ):
        super().__init__(api_key, model_name, price_csv_path)
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.aclient = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.retry_times = retry_times
        self.retry_sec = retry_sec

    def _prepare_args(self, prompt: Union[str, List], temperature: float) -> Dict:
        if isinstance(prompt, str):
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        else:
            messages = prompt
        return {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature
        }
    
    def _postprocess(self, response, response_format: Optional[Type]) -> Union[str, Dict]:
        self.update_acc_tokens(
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens
        )
        if response_format:
            return response.choices[0].message.parsed
        else:
            return response.choices[0].message.content

    @dynamic_retry
    def run(
            self,
            prompt: Union[str, List],
            response_format: Optional[Type[BaseModel]] = None,
            temperature: float = 0.7,
        ) -> Union[str, BaseModel]:  # process 1 query (prompt)
        args = self._prepare_args(prompt, temperature)
        if response_format:
            args["response_format"] = response_format
            response = self.client.chat.completions.parse(**args)
        else:
            response = self.client.chat.completions.create(**args)
        return self._postprocess(response, response_format)
    
    @dynamic_retry
    async def arun(
            self,
            prompt: Union[str, List],
            response_format: Optional[Type[BaseModel]] = None,
            temperature: float = 0.7
        ) -> Union[str, BaseModel]:  # process 1 query (prompt)
        args = self._prepare_args(prompt, temperature)
        if response_format:
            args["response_format"] = response_format
            response = await self.aclient.chat.completions.parse(**args)
        else:
            response = await self.aclient.chat.completions.create(**args)
        return self._postprocess(response, response_format)
    

class OpenAIEmbeddingAPI(BaseAPI):
    """
    This class wraps OpenAI Embedding API calls which has:
    1 async, 2 retry, 3 token count with price, 4 batching
    """
    def __init__(
            self,
            api_key: str,
            model_name: str,
            base_url: Optional[str] = None,
            batch_size: int = 64,
            price_csv_path: str = "",
            retry_times: int = 3,
            retry_sec: int = 5
        ):
        super().__init__(api_key, model_name, price_csv_path)
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.aclient = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.batch_size = batch_size
        self.retry_times = retry_times
        self.retry_sec = retry_sec

    def _postprocess(self, response) -> List[List[float]]:
        self.update_acc_tokens(
            input_tokens=response.usage.prompt_tokens,
            output_tokens=0
        )
        return [d.embedding for d in response.data]

    @dynamic_retry
    def run_all(self, texts: List[str]) -> List[List[float]]:  # process len(texts)
        response = self.client.embeddings.create(
                model=self.model_name,
                input=texts
        )
        return self._postprocess(response)

    def run_batches(self, texts: List[str]) -> List[List[float]]:  # process len(texts) by batching
        return batch_executer_for_func(inputs=texts, batch_size=self.batch_size, func=self.run_all)
    
    @dynamic_retry
    async def arun_all(self, texts: List[str]) -> List[List[float]]:  # process len(texts)
        response = await self.aclient.embeddings.create(
                model=self.model_name,
                input=texts
        )
        return self._postprocess(response)
    
    async def arun_batches(self, texts: List[str]) -> List[List[float]]:  # process len(texts) at once
        return await batch_executer_for_afunc(inputs=texts, batch_size=self.batch_size, afunc=self.arun_all)