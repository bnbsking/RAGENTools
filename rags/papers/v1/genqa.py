import glob
import json
import os

import pandas as pd
import yaml

from ragentools.api_calls.google_gemini import GoogleGeminiChatAPI
from ragentools.genqa.genqa import generate_qa_pairs


if __name__ == "__main__":
    cfg = yaml.safe_load(open("/app/rags/papers/v1/rags_papers_v1.yaml"))
    cfg_api = cfg["api"]
    cfg_ind = cfg["indexing"]
    cfg_qa = cfg["gen_qa"]

    api_key = yaml.safe_load(open(cfg_api["api_key_path"]))[cfg_api["api_key_env"]]
    api_chat = GoogleGeminiChatAPI(api_key=api_key, model_name=cfg_api["chat_model_name"])
    
    generate_qa_pairs(
        prompt_path=cfg_qa["prompt_path"],
        csv_folder=cfg_ind["parsed_save_folder"],
        sample_each_csv=cfg_qa["sample_each_csv"],
        api_chat=api_chat,
        save_path=cfg_qa["save_path"],
    )
