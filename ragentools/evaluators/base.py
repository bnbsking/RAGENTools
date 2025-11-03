from abc import ABC, abstractmethod
import json
import os


class BaseEvaluator(ABC):
    def __init__(self, load_path: str, save_folder: str):
        self.data = json.load(open(load_path, 'r', encoding='utf-8'))
        os.makedirs(save_folder, exist_ok=True)
        self.save_folder = save_folder

    @abstractmethod
    def evaluate(self, *args, **kwargs):
        pass
