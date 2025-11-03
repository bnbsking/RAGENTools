import json

import yaml

from ragentools.api_calls.google_gemini import GoogleGeminiChatAPI
from ragentools.prompts import get_prompt_and_response_format


if __name__ == "__main__":
    cfg = yaml.safe_load(open("/app/rags/papers/v1/rags_papers_v1.yaml"))
    cfg_api = cfg["api"]
    cfg_eval = cfg["eval"]

    api_key = yaml.safe_load(open(cfg_api["api_key_path"]))[cfg_api["api_key_env"]]
    api_chat = GoogleGeminiChatAPI(api_key=api_key, model_name=cfg_api["chat_model_name"])
    
    prompt, response_format = get_prompt_and_response_format(
        "/app/ragentools/prompts/ragas/advisor.yaml"
    )
    avg_score_dict = json.load(open("/app/rags/papers/v1/eval/avg_score.json"))
    response = api_chat.run(
        prompt=prompt.replace("{{ avg_score_dict }}", str(avg_score_dict)),
        response_format=response_format
    )
    with open(f"{cfg_eval['save_folder']}/advises.txt", 'w', encoding='utf-8') as f:
        f.write(response)
    print(response)
