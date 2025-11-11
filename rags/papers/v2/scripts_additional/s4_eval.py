import yaml
from ragentools.api_calls.google_gemini import GoogleGeminiChatAPI
from ragentools.evaluators.evaluators import RAGAsEvaluator

if __name__ == "__main__":
    cfg = yaml.safe_load(open("/app/rags/papers/v2/rags_papers_v2.yaml"))
    cfg_api = cfg["api"]
    cfg_ans = cfg["ans"]
    cfg_eval = cfg["eval"]

    api_key = yaml.safe_load(open(cfg_api["api_key_path"]))[cfg_api["api_key_env"]]
    api_chat = GoogleGeminiChatAPI(api_key=api_key, model_name=cfg_api["chat_model_name"], retry_sec=60)
    
    evaluator = RAGAsEvaluator(
        load_path=cfg_ans["save_path"],
        save_folder=cfg_eval["save_folder"],
        api=api_chat,
    )
    evaluator.evaluate()
