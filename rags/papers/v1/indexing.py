import glob
import os

import yaml

from ragentools.api_calls.google_gemini import (
    GoogleGeminiEmbeddingAPI,
    GoogleGeminiChatAPI,
)
from ragentools.indexers.embedding import CustomEmbedding
from ragentools.indexers.indexers import two_level_indexing
from ragentools.parsers.pdf_parser import PDFParser


if __name__ == "__main__":
    cfg = yaml.safe_load(open("/app/rags/papers/v1/rags_papers_v1.yaml"))
    cfg_api = cfg["api"]
    cfg_ind = cfg["indexing"]

    api_key = yaml.safe_load(open(cfg_api["api_key_path"]))[cfg_api["api_key_env"]]
    api_emb = GoogleGeminiEmbeddingAPI(api_key=api_key, model_name=cfg_api["emb_model_name"])
    api_chat = GoogleGeminiChatAPI(api_key=api_key, model_name=cfg_api["chat_model_name"])
    embed_model = CustomEmbedding(api=api_emb, dim=3072)

    parser = PDFParser(
        input_path_list=glob.glob(cfg_ind["data_folder"] + "*.pdf"),
        output_folder=os.path.join(cfg_ind["parsed_save_folder"])
    )
    parser.parse()

    two_level_indexing(
        parsed_csv_folder=cfg_ind["parsed_save_folder"],
        indices_save_folder=cfg_ind["indices_save_folder"],
        embed_model=embed_model,
        api_chat=api_chat
    )
