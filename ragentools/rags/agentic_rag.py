from typing import Dict, Any, List

from langchain_core.runnables import Runnable
from langgraph.graph import StateGraph, START, END

from ragentools.api_calls.langchain_runnable import ChatRunnable
from ragentools.prompts import get_prompt_and_response_format
from ragentools.rags.rag_engines import BaseRAGEngine
from ragentools.rags.rerankers import BaseReranker


class RetrieveNode(Runnable):
    def __init__(self, rag_engine: BaseRAGEngine, reranker: BaseReranker):
        self.rag_engine = rag_engine
        self.reranker = reranker

    def invoke(self, state: Dict) -> Dict:
        query = state["query"]
        retrieved = self.rag_engine.retrieve(query)
        retrieved = self.reranker(retrieved)
        return state | {"retrieved": retrieved}


class SynthesizeNode(ChatRunnable):
    def __init__(self, prompt_path: str):
        self.prompt, self.response_format = get_prompt_and_response_format(prompt_path)

    def invoke(self, state: Dict) -> Dict:
        prompt = self.prompt \
            .replace("{{ query }}", state["query"])\
            .replace("{{ retrieved }}", state["retrieved"])
        input = {"prompt": prompt, "response_format": self.response_format}
        result = self.run(input)
        return state | {"llm_response": result["answer"]}


class CritiqueNode:
    def __init__(self, prompt_path: str):
        self.prompt, self.response_format = get_prompt_and_response_format(prompt_path)

    def invoke(self, state: Dict) -> Dict:
        prompt = self.prompt \
            .replace("{{ query }}", state["query"])\
            .replace("{{ retrieved }}", state["retrieved"])\
            .replace("{{ llm_response }}", state["llm_response"])
        input = {"prompt": prompt, "response_format": self.response_format}
        result = self.run(input)
        return state | {"issues": result.get("issues", [])}


class CorrectNode:
    def __init__(self, correct_prompt):
        self.correct_prompt = correct_prompt

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        result = self.correct_prompt(
            query=state["query"],
            previous_answer=state["answer"],
            issues=state["issues"],
            text=state["reranked_text"]
        )
        return {**state, "answer": result["answer"], "issues": result.get("issues", [])}


class SaveFinalAnswerNode:
    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        # You can add logic here to persist the final answer if desired
        return state


def decide_correct_or_end(state: Dict[str, Any]) -> str:
    if state.get("issues"):
        return "correct"
    return "save_final_answer"


class IterativeRAG:
    def __init__(
        self,
        rag_engine: BaseRAGEngine,
        reranker: BaseReranker,
        synthesize_prompt,
        critique_prompt,
        correct_prompt,
    ):
        self.rag_engine = rag_engine
        self.reranker = reranker
        self.synthesize_prompt = synthesize_prompt
        self.critique_prompt = critique_prompt
        self.correct_prompt = correct_prompt

        self.graph = self._build_graph()

    def _build_graph(self):
        retrieve_node = RetrieveNode(self.rag_engine, self.reranker)
        synthesize_node = SynthesizeNode(self.synthesize_prompt)
        critique_node = CritiqueNode(self.critique_prompt)
        correct_node = CorrectNode(self.correct_prompt)
        save_final_answer_node = SaveFinalAnswerNode()

        graph_builder = StateGraph(dict)
        graph_builder.add_node("retrieve", retrieve_node)
        graph_builder.add_node("synthesize", synthesize_node)
        graph_builder.add_node("critique", critique_node)
        graph_builder.add_node("correct", correct_node)
        graph_builder.add_node("save_final_answer", save_final_answer_node)

        graph_builder.add_edge(START, "retrieve")
        graph_builder.add_edge("retrieve", "synthesize")
        graph_builder.add_edge("synthesize", "critique")
        graph_builder.add_conditional_edges("critique", decide_correct_or_end, path_map={"correct": "correct", "save_final_answer": "save_final_answer"})
        graph_builder.add_edge("correct", "critique")
        graph_builder.add_edge("save_final_answer", END)

        return graph_builder.compile()

    def run(self, query: str) -> Dict[str, Any]:
        init_state = {"query": query}
        return self.graph.invoke(init_state)


# ---------------------------
# Example usage (mocked)
# ---------------------------
if __name__ == "__main__":
    class MockRagEngine:
        def retrieve(self, query):
            return [{"id": 1, "text": f"Mock document for {query}"}]

    class MockReranker:
        def __call__(self, chunks):
            return "\n".join(c["text"] for c in chunks)

    def mock_prompt(**kwargs):
        # Simulate LangGraph-ready prompt callable
        query = kwargs.get("query")
        answer = f"Generated answer for {query}"
        if "previous_answer" in kwargs:
            answer = f"Refined answer for {query}"
        issues = [] if "Refined" in answer else ["missing detail"]
        return {"answer": answer, "issues": issues}

    rag = IterativeRAG(
        rag_engine=MockRagEngine(),
        reranker=MockReranker(),
        synthesize_prompt=mock_prompt,
        critique_prompt=mock_prompt,
        correct_prompt=mock_prompt,
    )

    result = rag.run("Explain LangGraph-based RAG")
    import json
    print(json.dumps(result, indent=2, ensure_ascii=False))
