import base64
from pathlib import Path
from typing import List

import numpy as np
import yaml

from ragentools.api_calls.openai_gpt import OpenAIGPTChatAPI, OpenAIEmbeddingAPI, img_path_to_openai_url
from ragentools.common.async_funcs import async_executer, concurrency_wrapper
from ragentools.common.formatting import get_response_model
from ragentools.prompts import get_prompt_and_response_format


class TestOpenAIGPTChatAPI:
    @classmethod
    def setup_class(cls):
        api_key = yaml.safe_load(open("/app/tests/api_keys.yaml"))["OPENAI_API_KEY"]
        cls.api = OpenAIGPTChatAPI(
            api_key=api_key,
            model_name="gpt-5-nano",
            price_csv_path="/app/ragentools/api_calls/prices.csv",
            retry_times=0
        )

    def test_run(self):
        prompt, response_format = get_prompt_and_response_format(
            prompt_path='/app/ragentools/prompts/basic.yaml',
            response_process="model"
        )
        response = self.api.run(prompt=prompt, response_format=response_format, temperature=1)
        #
        assert isinstance(response, response_format)
        print(response, self.api.get_price())

    def test_arun(self):
        args_list = [
            {
                "prompt": "What is the next day of Sunday?",
                "response_format": get_response_model({"ans": {"type": "string"}}),
                "temperature": 1
            },
            {
                "prompt": "What is the capital of France?",
                "response_format": get_response_model({"ans": {"type": "string"}}),
                "temperature": 1
            }
        ]
        limited_api_arun = concurrency_wrapper(self.api.arun, limit=2)
        results = async_executer(limited_api_arun, args_list)
        #
        assert len(results) == len(args_list)
        for result, args in zip(results, args_list):
            assert isinstance(result, args["response_format"])
        print(results, self.api.get_price())
    
    def test_arun_img(self):
        openai_url = img_path_to_openai_url("/app/tests/api_calls/dog.jpg")
        response_format = get_response_model({"description": {"type": "string"}})
        results = async_executer(
            self.api.arun,
            [
                {
                    "prompt": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "What's in this picture?"},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": openai_url
                                    },
                                }
                            ]
                        }
                    ],
                    "response_format": response_format,
                    "temperature": 1
                }
            ]
        )
        #
        assert isinstance(results[0], response_format)
        print(results, self.api.get_price())


class TestOpenAIEmbeddingAPI:
    @classmethod
    def setup_class(cls):
        api_key = yaml.safe_load(open("/app/tests/api_keys.yaml"))["OPENAI_API_KEY"]
        cls.api = OpenAIEmbeddingAPI(
            api_key=api_key,
            model_name="text-embedding-3-small",
            price_csv_path="/app/ragentools/api_calls/prices.csv",
            retry_times=0
        )
        cls.texts = [
            "The dog barked all night.",
            "AI is changing the world."
        ]
        cls.expect_type = List[List[float]]
        cls.expect_shape = (len(cls.texts), 1536)

    def test_run_batches(self):
        embeddings = self.api.run_batches(texts=self.texts)
        assert np.array(embeddings).shape == self.expect_shape
        print(embeddings[0][:3], self.api.get_price())

    def test_arun_batches(self):
        results = async_executer(self.api.arun_batches, [{"texts": self.texts}])
        embeddings = results[0]
        assert np.array(embeddings).shape == self.expect_shape
        print(embeddings[0][:3], self.api.get_price())


if __name__ == "__main__":
    #obj = TestOpenAIGPTChatAPI()
    #obj.setup_class()
    #obj.test_run()
    #obj.test_arun()
    #obj.test_arun_img()

    obj = TestOpenAIEmbeddingAPI()
    obj.setup_class()
    obj.test_run_batches()
    obj.test_arun_batches()
