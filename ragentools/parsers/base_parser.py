from abc import ABC, abstractmethod
from dataclasses import dataclass
import glob
import os
from typing import List

import pandas as pd


@dataclass
class ChunkRecord:
    chunk: str
    source_path: str
    page: int | None = None


class BaseParser(ABC):
    def __init__(
            self,
            input_path_list: List[str],
            output_folder: str,
            chunk_size: int= 800,
            overlap_size: int = 100
        ):
        self.input_path_list = input_path_list
        self.output_folder = output_folder
        os.makedirs(self.output_folder, exist_ok=True)
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size

    def chunk_text(self, text: str) -> List[str]:
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunks.append(text[start:end])
            start += self.chunk_size - self.overlap_size
        return chunks

    @abstractmethod
    def parse_one(self, file_path: str) -> List[ChunkRecord]:
        pass

    def parse(self):
        for input_path in self.input_path_list:
            records = self.parse_one(input_path)
            if not records:
                print(f"No records parsed from {input_path}.")
                continue
            save_csv_path = os.path.join(
                self.output_folder,
                os.path.basename(input_path) + ".csv"
            )
            df = pd.DataFrame([record.__dict__ for record in records])
            df.to_csv(save_csv_path, index=False)
