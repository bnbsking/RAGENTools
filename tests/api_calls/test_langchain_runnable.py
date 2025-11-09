import numpy as np
from typing import List
import yaml

from ragentools.api_calls.google_gemini import GoogleGeminiChatAPI, GoogleGeminiEmbeddingAPI
from ragentools.api_calls.langchain_runnable import ChatRunnable, EmbRunnable
from ragentools.common.async_main import amain_wrapper
from ragentools.common.formatting import get_response_model
from ragentools.prompts import get_prompt_and_response_format


class TestChatRunnable:
    @classmethod
    def setup_class(cls):
        api_key = yaml.safe_load(open("/app/tests/api_keys.yaml"))["GOOGLE_API_KEY"]
        cls.runnable = ChatRunnable(
            api=GoogleGeminiChatAPI,
            api_key=api_key,
            model_name="gemini-2.0-flash-lite"
        )

    def test_run(self):
        prompt, response_format = get_prompt_and_response_format('/app/ragentools/prompts/basic.yaml')
        response = self.runnable.run(input={"prompt": prompt, "response_format": response_format})
        #
        expect_response_format = get_response_model(response_format)
        expect_response_format(**response)
        print(response, self.runnable.api.get_price())
    
    def test_arun(self):
        args_list = [
            {
                "input":{
                    "prompt": "Explain the theory of relativity in simple terms.",
                    "response_format": {"ans": {"type": "string"}}
                }
            },
            {
                "input":{
                    "prompt": "What are the health benefits of regular exercise?",
                    "response_format": {"ans": {"type": "string"}}
                }
            }
        ]
        results = amain_wrapper(self.runnable.arun, args_list)
        #
        expect_response_format_list = [get_response_model(args["input"]["response_format"]) for args in args_list]
        assert len(results) == len(expect_response_format_list)
        for result, expect_response_format in zip(results, expect_response_format_list):
            expect_response_format(**result)
        print(results, self.runnable.api.get_price())

    def test_arun_img(self):
        response_format = {"description": {"type": "string"}}
        parts = [
            {"text": "What's in this picture?"},
            {"inline_data": {
                "mime_type": "image/jpeg",
                "data": open("/app/tests/api_calls/dog.jpg", "rb").read()
            }}
        ]
        results = amain_wrapper(
            self.runnable.arun,
            [
                {
                    "input": {
                        "prompt": [{"role": "user", "parts": parts}],
                        "response_format": response_format
                    }
                }
            ]
        )
        #
        expect_response_format = get_response_model(response_format)
        expect_response_format(**results[0])
        print(results, self.runnable.api.get_price())


class TestEmbRunnable:
    @classmethod
    def setup_class(cls):
        api_key = yaml.safe_load(open("/app/tests/api_keys.yaml"))["GOOGLE_API_KEY"]
        cls.runnable = EmbRunnable(
            api=GoogleGeminiEmbeddingAPI,
            api_key=api_key,
            model_name="gemini-embedding-001"
        )
        cls.texts = [
            "The dog barked all night.",
            "AI is changing the world."
        ]
        cls.dim = 3072
        cls.expect_type = List[List[float]]

    def test_run_batches(self):
        embeddings = self.runnable.run_batches(input={"texts": self.texts, "dim": self.dim})
        assert np.array(embeddings).shape == (len(self.texts), 3072)
        print(embeddings[0][:3], self.runnable.api.get_price())

    def test_arun_batches(self):
        results = amain_wrapper(self.runnable.arun_batches, [{"input": {"texts": self.texts, "dim": self.dim}}])
        embeddings = results[0]
        assert np.array(embeddings).shape == (len(self.texts), 3072)
        print(embeddings[0][:3], self.runnable.api.get_price())
    

if __name__ == "__main__":
    # test_instance = TestChatRunnable()
    # test_instance.setup_class()
    # test_instance.test_run()
    # test_instance.test_arun()
    # test_instance.test_arun_img()

    test_instance = TestEmbRunnable()
    test_instance.setup_class()
    test_instance.test_run_batches()
    test_instance.test_arun_batches()