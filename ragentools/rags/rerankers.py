from abc import ABC
from dataclasses import dataclass
from typing import Any, List

from ragentools.prompts import get_prompt_and_response_format


@dataclass
class RetrievedChunk:
    scores: float
    content: str
    meta: Any


class BaseReranker(ABC):
    def rerank(self, chunks: List[RetrievedChunk], **kwargs) -> List[RetrievedChunk]:
        return sorted(chunks, key=lambda x: x.scores, reverse=kwargs.get("reverse", False))

    def concat(self, chunks: List[RetrievedChunk]) -> str:
        texts = []
        for i, chunk in enumerate(chunks):
            texts.append(f"Chunk {i+1} with score {chunk.scores}:\n{chunk.content}\n")
        return ("\n" + "="*10 + "\n").join(texts)

    def __call__(self, chunks: List[RetrievedChunk], **kwargs) -> str:
        reranked_chunks = self.rerank(chunks, **kwargs)
        return self.concat(reranked_chunks)


class LLMReranker(BaseReranker):
    def __init__(self, api, prompt_path: str):
        self.api = api
        if prompt_path:
            self.prompt, self.response_format = get_prompt_and_response_format(prompt_path)

    def rerank(self, chunks: List[RetrievedChunk], threshold=0.5) -> List[RetrievedChunk]:
        contents = [chunk.content for chunk in chunks]
        prompt = self.prompt\
            .replace("{{ query }}", "")\
            .replace("{{ retrieved }}", "\n\n---\n\n".join(contents))
        result = self.api.run(prompt, self.response_format)
        #assert len(result["scores"]) == len(chunks), "Reranker output length mismatch with input chunks."
        scores_id_chunks = sorted(zip(map(float, result["scores"]), range(len(chunks)), chunks), reverse=True)
        results = []
        for score, _, chunk in scores_id_chunks:
            if score >= threshold:
                chunk.scores = score
                results.append(chunk)
        return results
    