import glob
import json
import os

import pandas as pd

from ragentools.prompts import get_prompt_and_response_format


def generate_qa_pairs(
        prompt_path: str,
        csv_folder: str,
        sample_each_csv: int,
        api_chat,
        save_path: str
    ):
    prompt, response_format = get_prompt_and_response_format(prompt_path)
    qa_pairs = []  # len = csvs * sample_each_csv * prompt_num
    for csv_path in glob.glob(csv_folder + "/*.csv"):
        df = pd.read_csv(csv_path)
        df = df.sample(n=sample_each_csv, random_state=42)
        for _, row in df.iterrows():
            prompt_rep = prompt.replace("{{ context }}", row["chunk"])
            qa_dict = api_chat.run(prompt_rep, response_format)
            qa_pairs_chunk = []
            for i in range(len(qa_dict) // 2):
                qa_pairs_chunk.append(
                    {
                        "question": qa_dict[f"question-{i + 1}"],
                        "answer": qa_dict[f"answer-{i + 1}"],
                        "source_path": row["source_path"],
                        "page": row["page"],
                    }
                )
            qa_pairs.extend(qa_pairs_chunk)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    json.dump(qa_pairs, open(save_path, "w"), indent=4)
