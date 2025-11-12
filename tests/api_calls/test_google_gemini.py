from typing import List

import numpy as np
import yaml

from ragentools.api_calls.google_gemini import GoogleGeminiEmbeddingAPI, GoogleGeminiChatAPI
from ragentools.common.async_funcs import async_executer
from ragentools.common.formatting import get_response_model
from ragentools.prompts import get_prompt_and_response_format


class TestGoogleGeminiChatAPI:
    @classmethod
    def setup_class(cls):
        api_key = yaml.safe_load(open("/app/tests/api_keys.yaml"))["GOOGLE_API_KEY"]
        cls.api = GoogleGeminiChatAPI(api_key=api_key, model_name="gemini-2.0-flash-lite")

    def test_run(self):
        prompt, response_format = get_prompt_and_response_format('/app/ragentools/prompts/basic.yaml')
        response = self.api.run(prompt=prompt, response_format=response_format)
        #
        expect_response_format = get_response_model(response_format)
        expect_response_format(**response)
        print(response, self.api.get_price())

    def test_arun(self):
        args_list = [
            {
                "prompt": "Explain the theory of relativity in simple terms.",
                "response_format": {"ans": {"type": "string"}}
            },
            {
                "prompt": "What are the health benefits of regular exercise?",
                "response_format": {"ans": {"type": "string"}}
            }
        ]
        results = async_executer(self.api.arun, args_list)
        #
        expect_response_format_list = [get_response_model(args["response_format"]) for args in args_list]
        assert len(results) == len(expect_response_format_list)
        for result, expect_response_format in zip(results, expect_response_format_list):
            expect_response_format(**result)
        print(results, self.api.get_price())
    
    def test_arun_img(self):
        response_format = {"description": {"type": "string"}}
        results = async_executer(
            self.api.arun,
            [
                {
                    "prompt": [
                        {
                            "role": "user",
                            "parts": [
                                {"text": "What's in this picture?"},
                                {"inline_data": {
                                    "mime_type": "image/jpeg",
                                    "data": open("/app/tests/api_calls/dog.jpg", "rb").read()
                                }}
                            ]
                        }
                    ],
                    "response_format": response_format
                }
            ]
        )
        #
        expect_response_format = get_response_model(response_format)
        expect_response_format(**results[0])
        print(results, self.api.get_price())


class TestGoogleGeminiEmbedAPI:
    @classmethod
    def setup_class(cls):
        api_key = yaml.safe_load(open("/app/tests/api_keys.yaml"))["GOOGLE_API_KEY"]
        cls.api = GoogleGeminiEmbeddingAPI(api_key=api_key, model_name="gemini-embedding-001")
        cls.texts = [
            "The dog barked all night.",
            "AI is changing the world."
        ]
        cls.dim = 3072
        cls.expect_type = List[List[float]]

    def test_run_batches(self):
        embeddings = self.api.run_batches(texts=self.texts, dim=self.dim)
        assert np.array(embeddings).shape == (len(self.texts), 3072)
        print(embeddings[0][:3], self.api.get_price())

    def test_arun_batches(self):
        results = async_executer(self.api.arun_batches, [{"texts": self.texts, "dim": self.dim}])
        embeddings = results[0]
        assert np.array(embeddings).shape == (len(self.texts), 3072)
        print(embeddings[0][:3], self.api.get_price())


if __name__ == "__main__":
    #obj = TestGoogleGeminiChatAPI()
    #obj.setup_class()
    # obj.test_run()
    # obj.test_arun()
    #obj.test_arun_img()

    obj = TestGoogleGeminiEmbedAPI()
    obj.setup_class()
    obj.test_run_batches()
    obj.test_arun_batches()
