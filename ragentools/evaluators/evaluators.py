import glob
import json
import os

from .base import BaseEvaluator
from ragentools.prompts import get_prompt_and_response_format


class RAGAsEvaluator(BaseEvaluator):
    def __init__(
            self,
            load_path: str,
            save_folder: str,
            api,
            prompt_folder: str = "/app/ragentools/prompts/ragas/ragas"
        ):
        super().__init__(load_path=load_path, save_folder=save_folder)
        self.api = api
        self.prompt_dict = self._init_prompt_dict(prompt_folder)

    def _init_prompt_dict(self, prompt_folder: str):
        prompt_dict = {}
        for prompt_path in glob.glob(f"{prompt_folder}/*.yaml"):
            prompt_name = os.path.basename(prompt_path).replace(".yaml", "")
            prompt, response_format = get_prompt_and_response_format(prompt_path)
            prompt_dict[prompt_name] = {
                "prompt": prompt,
                "response_format": response_format
            }
        return prompt_dict

    def evaluate(self) -> list:
        for i, data_dict in enumerate(self.data):
            metrics = {}
            for metric, prompt_cfg in self.prompt_dict.items():
                prompt = prompt_cfg["prompt"]\
                    .replace("{{ question }}", data_dict["question"])\
                    .replace("{{ answer }}", data_dict["answer"])\
                    .replace("{{ llm_response }}", data_dict.get("llm_response", ""))\
                    .replace("{{ retrieved_chunks }}", data_dict.get("retrieved_chunks", ""))
                metric_dict = self.api.run(prompt=prompt, response_format=prompt_cfg["response_format"])
                metrics[metric] = metric_dict
            self.data[i]["eval"] = metrics
        json.dump(self.data, open(f"{self.save_folder}/eval.json", 'w', encoding='utf-8'), indent=4)

        avg_score = {}
        for metric in self.prompt_dict.keys():
            avg_score[metric] = sum(self.data[i]["eval"][metric]["score"] for i in range(len(self.data))) / len(self.data)
        json.dump(avg_score, open(f"{self.save_folder}/avg_score.json", 'w', encoding='utf-8'), indent=4)
        