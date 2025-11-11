import os
from typing import Iterator

import pandas as pd

from . import Document


class BaseSaver:
    def __init__(self, save_folder: str):
        self.save_folder = save_folder
        os.makedirs(save_folder, exist_ok=True)

    def save(self, docs: Iterator[Document]) -> None:
        raise NotImplementedError("Implement your saving logic here.")


class ListSaver(BaseSaver):
    # save all in 1 file
    def save(self, docs: Iterator[Document]) -> None:
        data = []
        for doc in docs:
            data.append({
                "chunk": doc.page_content,
                **doc.metadata
            })
        df = pd.DataFrame(data)
        save_path = f"{self.save_folder}/parsed_lists.csv"
        df.to_csv(save_path, index=False)


class PDFSaver(BaseSaver):
    # save each pdf in separate file. Make sure docs are sorted by source_path
    def save(self, docs: Iterator[Document]) -> None:
        prev_save_path = ""
        acc = {"chunk": [], "source_path": [], "page": []}
        for i, doc in enumerate(docs):
            save_path = os.path.join(
                self.save_folder,
                os.path.basename(doc.metadata["source_path"]) + ".csv"
            )
            # save previous file and reset
            if i != 0 and save_path != prev_save_path:
                pd.DataFrame(acc).to_csv(prev_save_path, index=False)
                acc = {"chunk": [], "source_path": [], "page": []}
            prev_save_path = save_path
            acc["chunk"].append(doc.page_content)
            acc["source_path"].append(doc.metadata["source_path"])
            acc["page"].append(str(doc.metadata.get("page", "")))

        # save remaining docs
        if acc["chunk"]:
            pd.DataFrame(acc).to_csv(prev_save_path, index=False)
            