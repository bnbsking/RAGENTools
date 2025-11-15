import sys
from typing import Dict, TypedDict, Optional

from langchain_core.runnables import Runnable
from langgraph.graph import StateGraph, START, END

from ragentools.api_calls.langchain_runnable import ChatRunnable
from ragentools.rags.rag_engines import BaseRAGEngine
from ragentools.rags.rerankers import BaseReranker
from ragentools.rags.rags import BaseRAG


class NeedRetrievalNode(ChatRunnable):
    def invoke(self, state: Dict, config = None) -> Dict:
        prompt = self.prompt.replace("{{ query }}", state["query"])
        input = {"prompt": prompt, "response_format": self.response_format}
        result = self.run(input)
        return state | {"need_retrieval_score": result["score"]}


def decide_answer_or_query_decomposer(state: Dict) -> str:
    if state["need_retrieval_score"] == 0:
        return "answer"
    else:
        return "query_decomposer"


class AnswerNode(ChatRunnable):
    def invoke(self, state: Dict, config = None) -> Dict:
        prompt = f"""
            You are a helpful assistant.
            Given a user question and some retrieved context, provide a comprehensive answer.
            **Question:** {state['query']}
            **Retrieved:** {state.get('retrieved', '')}
            """
        input = {"prompt": prompt}
        result = self.run(input)
        return state | {"answer": result}


class QueryDecomposerNode(ChatRunnable):
    def invoke(self, state: Dict, config = None) -> Dict:
        prompt = self.prompt.replace("{{ query }}", state["query"])
        input = {"prompt": prompt, "response_format": self.response_format}
        result = self.run(input)
        return state | {"subquestions": result["subquestions"]}


class RetrieveNode(Runnable):
    def __init__(self, rag_engine: BaseRAGEngine, reranker: BaseReranker):
        self.rag_engine = rag_engine
        self.reranker = reranker

    def invoke(self, state: Dict, config = None) -> Dict:
        retrieved_list = []
        for subquestion in state["subquestions"]:
            retrieved = self.rag_engine.retrieve(subquestion)
            retrieved = self.reranker(retrieved)
            retrieved_list.append(retrieved)
        return state | {"retrieved": state.get("retrieved", "") + "\n".join(retrieved_list)}


class IsSufficientNode(ChatRunnable):
    def invoke(self, state: Dict, config = None) -> Dict:
        prompt = self.prompt \
            .replace("{{ query }}", state["query"])\
            .replace("{{ retrieved }}", state["retrieved"])
        input = {"prompt": prompt, "response_format": self.response_format}
        result = self.run(input)
        return state | {"sufficient_score": result["score"], "sufficient_explanation": result["explanation"]}


def decide_answer_or_sufficient(state: Dict) -> str:
    if state["sufficient_score"] == 2 or state["iter_count"] >= 2:
        return "answer"
    else:
        state["iter_count"] += 1
        return "query_fixer"


class QueryFixerNode(ChatRunnable):
    def invoke(self, state: Dict, config = None) -> Dict:
        prompt = self.prompt\
            .replace("{{ query }}", state["query"])\
            .replace("{{ suggestion }}", state["sufficient_explanation"])
        input = {"prompt": prompt, "response_format": self.response_format}
        result = self.run(input)
        return state | {"subquestions": result["subquestions"]}


class IterativeRAG(BaseRAG):
    def __init__(
        self,
        api,
        rag_engine: BaseRAGEngine,
        reranker: BaseReranker,
        need_retrieval_prompt_path: str,
        query_decomposer_prompt_path: str,
        is_sufficient_prompt_path: str,
        query_fixer_prompt_path: str,
        draw: Optional[str] = None
    ):
        self.api = api
        self.rag_engine = rag_engine
        self.reranker = reranker
        self.need_retrieval_prompt_path = need_retrieval_prompt_path
        self.query_decomposer_prompt_path = query_decomposer_prompt_path
        self.is_sufficient_prompt_path = is_sufficient_prompt_path
        self.query_fixer_prompt_path = query_fixer_prompt_path
        self.graph = self._build_graph(draw)

    def _build_graph(self, draw: bool):
        need_retrieval_node = NeedRetrievalNode(self.api, self.need_retrieval_prompt_path)
        query_decomposer_node = QueryDecomposerNode(self.api, self.query_decomposer_prompt_path)
        retrieve_node = RetrieveNode(self.rag_engine, self.reranker)
        is_sufficient_node = IsSufficientNode(self.api, self.is_sufficient_prompt_path)
        query_fixer_node = QueryFixerNode(self.api, self.query_fixer_prompt_path)
        answer_node = AnswerNode(self.api)
        
        graph_builder = StateGraph(TypedDict if draw else dict)
        graph_builder.add_node("need_retrieval", need_retrieval_node)
        graph_builder.add_node("query_decomposer", query_decomposer_node)
        graph_builder.add_node("retrieve", retrieve_node)
        graph_builder.add_node("is_sufficient", is_sufficient_node)
        graph_builder.add_node("query_fixer", query_fixer_node)
        graph_builder.add_node("answer", answer_node)

        graph_builder.add_edge(START, "need_retrieval")
        graph_builder.add_conditional_edges("need_retrieval", decide_answer_or_query_decomposer,
                                    path_map={"answer": "answer", "query_decomposer": "query_decomposer"})
        graph_builder.add_edge("query_decomposer", "retrieve")
        graph_builder.add_edge("retrieve", "is_sufficient")
        graph_builder.add_conditional_edges("is_sufficient", decide_answer_or_sufficient,
                                    path_map={"answer": "answer", "query_fixer": "query_fixer"})
        graph_builder.add_edge("query_fixer", "retrieve")
        graph_builder.add_edge("answer", END)

        graph = graph_builder.compile()

        if draw:
            graph_image = graph.get_graph().draw_mermaid_png()
            with open(draw, "wb") as f:
                f.write(graph_image)
            print("Draw and End the program.")
            sys.exit(0)
        else:
            return graph

    def retrieve(self, query: str) -> str:
        init_state = {"query": query, "iter_count": 0}
        result = self.graph.invoke(init_state)
        return result["answer"]
    
    def index(self):
        raise NotImplementedError
