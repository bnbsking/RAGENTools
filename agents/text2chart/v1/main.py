"""
`blackboard design pattern`
"""
import json
import os
from typing import TypedDict

import matplotlib.pyplot as plt
from langgraph.graph import StateGraph, START, END
from langchain_core.runnables import Runnable
import yaml

from ragentools.api_calls.google_gemini import GoogleGeminiChatAPI
from ragentools.api_calls.langchain_runnable import ChatRunnable


def get_query_generator(json_path: str):
    with open(json_path, "r") as f:
        data_dict = json.load(f)
    for data in data_dict:
        yield data["instruction"], data["id"]


def get_prefix_dict(state: dict, prefix: str) -> dict:
    return {k.removeprefix(prefix): v for k, v in state.items() if k.startswith(prefix)}


class GenNode(ChatRunnable):
    def invoke(self, state: dict, config = None) -> dict:
        prompt = self.prompt.replace("{{ query }}", state["query"])
        input = {"prompt": prompt, "response_format": self.response_format}
        out = self.run(input)
        return state | {"code": out, "code_ver": 1}


class Code2Plot(Runnable):
    def __init__(self, save_folder: str):
        self.save_folder = save_folder

    def invoke(self, state: dict, config = None) -> dict:
        data_id = state["data_id"]
        code = state["code"]
        code_ver = state["code_ver"]

        save_png_path = os.path.join(self.save_folder, f"{data_id}", f"v{code_ver}.png")
        save_py_path = os.path.join(self.save_folder, f"{data_id}", f"v{code_ver}.py")

        code = code.strip().removeprefix("```python").removesuffix("```").strip()
        code = code.replace("plt.show()", "")
        code += f"\nplt.savefig('{save_png_path}'); plt.close()"
        os.makedirs(os.path.dirname(save_py_path), exist_ok=True)
        open(save_py_path, "w").write(code)
        try:
            exec(code)
            plt.imread(save_png_path)  # check if the image is saved correctly
            plt.close()
            return state | {"img_path": save_png_path, "error": None}
        except Exception as e:
            return state | {"img_path": save_png_path, "error": str(e)}


def decide_fix_or_eval(state: dict) -> str:
    if state["error"] and state.get("retry_count", 0) < 2:
        state["retry_count"] = state.get("retry_count", 0) + 1
        return "fix"  # go to fix node
    return "eval"


class FixNode(ChatRunnable):
    def invoke(self, state: dict, config = None) -> dict:
        prompt = self.prompt\
            .replace("{{ query }}", state["query"])\
            .replace("{{ code }}", state["code"])\
            .replace("{{ error_msg }}", state["error"])
        input = {"prompt": prompt, "response_format": self.response_format}
        out = self.run(input)
        return state | {"code": out, "error": None, "code_ver": state["code_ver"] + 1}


class EvalNode(ChatRunnable):
    def invoke(self, state: dict, config = None) -> dict:
        parts = [
            {"text": self.prompt.replace("{{ query }}", state["query"])},
            {"inline_data": {
                "mime_type": "image/jpeg",
                "data": open(state["img_path"], "rb").read()
            }}
        ]
        prompt = [{"role": "user", "parts": parts}]
        input = {"prompt": prompt, "response_format": self.response_format}
        out = self.run(input)
        return state | {f"scores.{k}": v for k, v in out.items()}


def decide_refine_or_save(state: dict) -> str:
    scores = get_prefix_dict(state, "scores.")
    score = sum(v for k, v in scores.items() if k != "explanation")
    if score < 6 and state.get("refine_count", 0) < 2:  # 6 out of 8
        state["refine_count"] = state.get("refine_count", 0) + 1
        return "refine"
    return "save_final_eval"


class RefineNode(ChatRunnable):
    def invoke(self, state: dict, config = None) -> dict:
        prompt = self.prompt\
            .replace("{{ query }}", state["query"])\
            .replace("{{ code }}", state["code"])\
            .replace("{{ suggestion }}", state["scores.explanation"])
        input = {"prompt": prompt, "response_format": self.response_format}
        out = self.run(input)
        return state | {"code": out, "code_ver": state["code_ver"] + 1}


class SaveFinalEval(Runnable):
    def invoke(self, state: dict, config = None) -> dict:
        save_folder = os.path.dirname(state["img_path"])
        scores = get_prefix_dict(state, "scores.")
        with open(os.path.join(save_folder, "final_eval.json"), "w") as f:
            json.dump(scores, f, indent=4)
        return state


if __name__ == "__main__":
    cfg = yaml.safe_load(open("/app/agents/text2chart/v1/agents_text2chart_v1.yaml"))
    cfg_api = cfg["api"]
    cfg_pmt = cfg["prompts"]
    api_key = yaml.safe_load(open(cfg_api["api_key_path"]))[cfg_api["api_key_env"]]
    api = GoogleGeminiChatAPI(api_key=api_key, model_name=cfg_api["model_name"])

    gen_node = GenNode(api=api, prompt_path=cfg_pmt["gen_path"])
    code2plot_node = Code2Plot(cfg["save_folder"])
    fix_node = FixNode(api=api, prompt_path=cfg_pmt["fix_path"])
    eval_node = EvalNode(api=api, prompt_path=cfg_pmt["eval_path"])
    refine_node = RefineNode(api=api, prompt_path=cfg_pmt["refine_path"])
    save_final_eval_node = SaveFinalEval()

    query_generator = get_query_generator(cfg["data_path"])

    graph_builder = StateGraph(TypedDict if cfg.get("mode") == "PLOT" else dict)
    graph_builder.add_node("gen", gen_node)
    graph_builder.add_node("code2plot", code2plot_node)
    graph_builder.add_node("fix", fix_node)
    graph_builder.add_node("eval", eval_node)
    graph_builder.add_node("refine", refine_node)
    graph_builder.add_node("save_final_eval", save_final_eval_node)

    graph_builder.add_edge(START, "gen")
    graph_builder.add_edge("gen", "code2plot")
    graph_builder.add_conditional_edges("code2plot", decide_fix_or_eval,
                                    path_map={"fix": "fix", "eval": "eval"})
    graph_builder.add_edge("fix", "code2plot")
    graph_builder.add_conditional_edges("eval", decide_refine_or_save,
                                    path_map={"refine": "refine", "save_final_eval": "save_final_eval"})
    graph_builder.add_edge("refine", "code2plot")
    graph_builder.add_edge("save_final_eval", END)

    graph = graph_builder.compile()

    if cfg["mode"] == "PLOT":
        graph_image = graph.get_graph().draw_mermaid_png()
        with open(f"{cfg['save_folder']}/graph.png", "wb") as f:
            f.write(graph_image)

    else:  # RUN mode
        for query, data_id in query_generator:
            state = {"query": query, "data_id": data_id}
            share = graph.invoke(state)
    