"""Build the prompt from retrieved chunks, ask the LLM, and pull the [S1]-style citations out of the answer."""
import re
import time
from dataclasses import dataclass

from openai import OpenAI

from rag.config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL
from rag.retrieve import Hit

IDK = "I don't know based on the handbook."

PROMPT_V1 = """Answer the question using the context below. Cite your sources like [S1].

Context:
{context}

Question: {question}"""

PROMPT_V2 = """You answer employees' questions using only the handbook excerpts below.

Rules:
1. Use only facts stated in the excerpts. Never use outside knowledge.
2. After every sentence, cite the excerpt it came from, like [S2].
3. Copy numbers, limits and deadlines exactly as written.
4. If the excerpts don't contain the answer, reply with exactly: {idk}
5. Keep it short: two to four sentences.

Handbook excerpts:
{context}

Question: {question}"""

PROMPTS = {"v1": PROMPT_V1, "v2": PROMPT_V2}
client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)


@dataclass
class Answer:
    text: str
    cited: list[Hit]       # the chunks the model actually cited
    sources: list[Hit]     # everything we showed it
    refused: bool
    seconds: float
    prompt_tokens: int


def format_context(hits: list[Hit]) -> str:
    return "\n\n".join(f"[S{n}] {h.chunk.text}" for n, h in enumerate(hits, 1))


def answer(question: str, hits: list[Hit], prompt: str = "v2") -> Answer:
    content = PROMPTS[prompt].format(context=format_context(hits), question=question, idk=IDK)
    t0 = time.time()
    response = client.chat.completions.create(
        model=LLM_MODEL, messages=[{"role": "user", "content": content}],
        temperature=0, seed=42, max_tokens=400,
    )
    text = response.choices[0].message.content.strip()
    numbers = {int(n) for n in re.findall(r"\[S(\d+)\]", text)}
    cited = [hits[n - 1] for n in sorted(numbers) if 0 < n <= len(hits)]
    refused = "don't know" in text.lower().replace("’", "'") or "do not know" in text.lower()
    return Answer(text, cited, hits, refused, time.time() - t0, response.usage.prompt_tokens)


if __name__ == "__main__":
    import sys

    from rag.retrieve import Retriever

    question = " ".join(sys.argv[1:]) or "How many paid sick days do I get?"
    hits = Retriever(mode="rerank").search(question, k=5)
    result = answer(question, hits)
    print(f"Q: {question}\n\n{result.text}\n")
    for n, hit in enumerate(hits, 1):
        mark = "*" if hit in result.cited else " "
        print(f" {mark}[S{n}] {hit.chunk.title[:70]}")
    print(f"\n{result.seconds:.1f}s, {result.prompt_tokens} prompt tokens")
