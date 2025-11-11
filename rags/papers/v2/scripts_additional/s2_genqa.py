import yaml

from ragentools.api_calls.google_gemini import GoogleGeminiChatAPI
from ragentools.genqa.genqa import generate_qa_pairs


if __name__ == "__main__":
    cfg = yaml.safe_load(open("/app/rags/papers/v2/rags_papers_v2.yaml"))
    cfg_api = cfg["api"]
    cfg_par = cfg["parser"]
    cfg_gqa = cfg["gen_qa"]

    api_key = yaml.safe_load(open(cfg_api["api_key_path"]))[cfg_api["api_key_env"]]
    api_chat = GoogleGeminiChatAPI(api_key=api_key, model_name=cfg_api["chat_model_name"])
    
    generate_qa_pairs(
        prompt_path=cfg_gqa["prompt_path"],
        csv_folder=cfg_par["save_folder"],
        sample_each_csv=cfg_gqa["sample_each_csv"],
        api_chat=api_chat,
        save_path=cfg_gqa["save_path"],
    )
