from typing import List

import numpy as np
import yaml

from ragentools.api_calls.openai_gpt import OpenAIGPTChatAPI, OpenAIEmbeddingAPI
from ragentools.common.async_main import amain_wrapper
from ragentools.common.formatting import get_response_model
from ragentools.prompts import get_prompt_and_response_format


class TestOpenAIGPTChatAPI:
    @classmethod
    def setup_class(cls):
        api_key = yaml.safe_load(open("/app/tests/api_keys.yaml"))["OPENAI_API_KEY"]
        cls.api = OpenAIGPTChatAPI(api_key=api_key, model_name="openai/gpt-oss-20b")

    def test_run(self):
        prompt, response_format = get_prompt_and_response_format(
            prompt_path='/app/ragentools/prompts/basic.yaml',
            response_process="model"
        )
        response = self.api.run(prompt=prompt, response_format=response_format, retry_times=0)
        #
        assert isinstance(response, response_format)
        print(response, self.api.get_price())

    def test_arun(self):
        args_list = [
            {
                "prompt": "What is the next day of Sunday?",
                "response_format": get_response_model({"ans": {"type": "string"}})
            },
            {
                "prompt": "What is the capital of France?",
                "response_format": get_response_model({"ans": {"type": "string"}})
            }
        ]
        results = amain_wrapper(self.api.arun, args_list)
        #
        assert len(results) == len(args_list)
        for result, args in zip(results, args_list):
            assert isinstance(result, args["response_format"])
        print(results, self.api.get_price())
    
    def test_arun_img(self):
        response_format = get_response_model({"description": {"type": "string"}})
        results = amain_wrapper(
            self.api.arun,
            [
                {
                    "prompt": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "What's in this picture?"},
                                {"type": "image_url", "image_url": {"url": "/app/tests/api_calls/dog.jpg"}},
                            ]
                        }
                    ],
                    "response_format": response_format
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
        cls.api = OpenAIEmbeddingAPI(api_key=api_key, model_name="BAAI/bge-multilingual-gemma2")
        cls.texts = [
            "The dog barked all night.",
            "AI is changing the world."
        ]
        cls.expect_type = List[List[float]]

    def test_run_batches(self):
        embeddings = self.api.run_batches(texts=self.texts)
        assert np.array(embeddings).shape == (len(self.texts), 3584)
        print(embeddings[0][:3], self.api.get_price())

    def test_arun_batches(self):
        results = amain_wrapper(self.api.arun_batches, [{"texts": self.texts}])
        embeddings = results[0]
        assert np.array(embeddings).shape == (len(self.texts), 3584)
        print(embeddings[0][:3], self.api.get_price())


if __name__ == "__main__":
    obj = TestOpenAIGPTChatAPI()
    obj.setup_class()
    obj.test_run()
    obj.test_arun()
    obj.test_arun_img()

    # obj = TestOpenAIEmbeddingAPI()
    # obj.setup_class()
    # obj.test_run_batches()
    # obj.test_arun_batches()
