from typing import List


def recursive_summarization(api, texts: List[str], limit_len: int = 25000):
    # 8k * 4 = 32k limit
    if sum(len(t) for t in texts) <= limit_len:
        all_text = "\n".join(texts)
        return api.run(f"Summarize this text in 5 sentences:\n{all_text}")
    else:
        new_texts = []
        acc_chunks = []
        acc_len = 0
        for i, text in enumerate(texts):
            if i == len(texts) - 1 or acc_len + len(text) >= limit_len:
                acc_text = "\n".join(acc_chunks)
                acc_summary = api.run(f"Summarize this text in 3 sentences:\n{acc_text}")
                new_texts.append(acc_summary)
                acc_chunks = [text]
                acc_len = len(text)
            else:
                acc_chunks.append(text)
                acc_len += len(text)
        return recursive_summarization(api, new_texts, limit_len)
