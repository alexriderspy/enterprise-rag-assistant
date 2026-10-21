"""LLM-as-judge: does the assistant's answer say the same thing as the reference answer?"""
import json

from rag.answer import client
from rag.config import JUDGE_MODEL

JUDGE_PROMPT = """You are grading an HR assistant. Compare the ASSISTANT ANSWER to the REFERENCE ANSWER.

QUESTION: {question}
REFERENCE ANSWER: {reference}
ASSISTANT ANSWER: {candidate}

The assistant answer is "correct" if it gives the same key facts as the reference (numbers, yes/no, who to
contact) and says nothing that contradicts it. Extra correct detail is fine. Missing the main fact,
or getting a number wrong, is "incorrect".

Reply in JSON: {{"verdict": "correct" or "incorrect", "reason": "<one sentence>"}}"""


def judge(question: str, reference: str, candidate: str) -> tuple[bool, str]:
    response = client.chat.completions.create(
        model=JUDGE_MODEL, temperature=0, seed=42, max_tokens=200,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": JUDGE_PROMPT.format(
            question=question, reference=reference, candidate=candidate)}],
    )
    try:
        out = json.loads(response.choices[0].message.content)
        return out.get("verdict") == "correct", out.get("reason", "")
    except json.JSONDecodeError:
        return False, "judge returned invalid JSON"
