from functools import partial
from typing import List, Union, Optional

from google import genai
from google.genai.types import EmbedContentConfig, GenerateContentConfig
from ragentools.common.async_funcs import batch_executer_for_afunc, batch_executer_for_func
from ragentools.common.dynamic_retry import dynamic_retry

from .base_api import BaseAPI


class GoogleGeminiChatAPI(BaseAPI):
    """
    This class wraps Google Gemini API calls which has:
    1 async, 2 retry, 3 token count with price, 4 pydantic response, 5 multi-modal input
    """
    def __init__(
            self,
            api_key: str,
            model_name: str,
            price_csv_path: str = "",
            retry_times: int = 3,
            retry_sec: int = 5
        ):
        super().__init__(api_key, model_name, price_csv_path)
        self.client = genai.Client(api_key=api_key)
        self.retry_times = retry_times
        self.retry_sec = retry_sec

    def _prepare_args(self, response_format: dict, temperature: float) -> GenerateContentConfig:
        if response_format:
            return GenerateContentConfig(
                    temperature=temperature,
                    response_mime_type="application/json",
                    response_schema={
                        "type": "object",
                        "properties": response_format,
                    }
                )
        else:
            return GenerateContentConfig(temperature=temperature)

    def _postprocess(self, response, response_format: dict) -> Union[str, dict]:
        self.update_acc_tokens(
            input_tokens=response.usage_metadata.prompt_token_count,
            output_tokens=response.usage_metadata.candidates_token_count
        )
        return response.parsed if response_format else response.text

    @dynamic_retry
    def run(
            self,
            prompt: Union[str, List],
            response_format: dict = None,
            temperature: float = 0.7,
        ) -> Union[str, dict]:  # process 1 query (prompt) at once
        cfg = self._prepare_args(response_format, temperature)
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=cfg
        )
        return self._postprocess(response, response_format)

    @dynamic_retry
    async def arun(
            self,
            prompt: Union[str, List],
            response_format: dict = None,
            temperature: float = 0.7,
        ) -> Union[str, dict]:  # process 1 query (prompt) at once
        cfg = self._prepare_args(response_format, temperature)
        response = await self.client.aio.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=cfg
        )
        return self._postprocess(response, response_format)


class GoogleGeminiEmbeddingAPI(BaseAPI):
    """
    This class wraps Google Gemini API calls which has:
    1 async, 2 retry, 3 token count with price, 4 batching
    """
    def __init__(
            self,
            api_key: str,
            model_name: str,
            batch_size: int = 64,
            price_csv_path: str = "",
            retry_times: int = 3,
            retry_sec: int = 5
        ):
        super().__init__(api_key, model_name, price_csv_path)
        self.client = genai.Client(api_key=api_key)
        self.batch_size = batch_size
        self.retry_times = retry_times
        self.retry_sec = retry_sec

    def _prepare_args(self, dim: int) -> EmbedContentConfig:
        return EmbedContentConfig(
            task_type="RETRIEVAL_DOCUMENT",
            output_dimensionality=dim
        )

    def _postprocess(self, result, texts: List[str]) -> List[List[float]]:
        tokens = self.client.models.count_tokens(model=self.model_name, contents=texts)
        self.update_acc_tokens(
            input_tokens=tokens.total_tokens,
            output_tokens=0
        )
        return [x.values for x in result.embeddings]

    @dynamic_retry
    def run_all(self, texts: List[str], dim: int) -> List[List[float]]:  # process len(texts)
        cfg = self._prepare_args(dim)
        result = self.client.models.embed_content(
            model=self.model_name,
            contents=texts,
            config=cfg
        )
        return self._postprocess(result, texts)

    def run_batches(self, texts: List[str], dim: int) -> List[List[float]]:  # process len(texts) at once
        return batch_executer_for_func(
            inputs=texts,
            batch_size=self.batch_size,
            func=partial(self.run_all, dim=dim)
        )

    @dynamic_retry
    async def arun_all(self, texts: List[str], dim: int) -> List[List[float]]:  # process len(texts)
        cfg = self._prepare_args(dim)
        result = await self.client.aio.models.embed_content(
            model=self.model_name,
            contents=texts,
            config=cfg
        )
        return self._postprocess(result, texts)

    async def arun_batches(self, texts: List[str], dim: int) -> List[List[float]]:  # process len(texts) at once
        return await batch_executer_for_afunc(
            inputs=texts,
            batch_size=self.batch_size,
            afunc=partial(self.arun_all, dim=dim)
        )
    