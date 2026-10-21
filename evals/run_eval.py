"""Run the eval set through one config and score retrieval, answers, citations and "I don't know"."""
import argparse
import json
import re
from pathlib import Path

from evals.configs import CONFIGS
from evals.judge import judge
from rag.answer import answer
from rag.retrieve import Hit, Retriever

HERE = Path(__file__).parent
K = 5  # chunks we put in the prompt


def load_questions() -> list[dict]:
    return [json.loads(line) for line in (HERE / "questions.jsonl").read_text().splitlines()]


def first_gold_rank(hits: list[Hit], gold: list[str]) -> int | None:
    for rank, hit in enumerate(hits, start=1):
        if set(hit.chunk.section_ids) & set(gold):
            return rank
    return None


def has_facts(text: str, facts: list[list[str]]) -> bool:
    """Exact check: every fact group must appear (any one spelling of it).
    A number can't match inside a bigger number, so "6" doesn't match "16"."""
    def found(alt: str) -> bool:
        pattern = re.escape(alt) if not alt[0].isdigit() else rf"(?<![\d.,]){re.escape(alt)}(?!\d)"
        return re.search(pattern, text, flags=re.IGNORECASE) is not None
    return all(any(found(alt) for alt in group) for group in facts)


def evaluate(name: str, retrieval_only: bool = False) -> dict:
    cfg = CONFIGS[name]
    retriever = Retriever(chunking=cfg["chunking"], mode=cfg["mode"])
    rows = []
    for q in load_questions():
        hits = retriever.search(q["question"], k=10)
        row = {"id": q["id"], "question": q["question"], "answerable": q["answerable"],
               "retrieved": [h.chunk.title for h in hits[:K]]}
        if q["answerable"]:
            row["rank"] = first_gold_rank(hits, q["gold"])
        if not retrieval_only:
            a = answer(q["question"], hits[:K], prompt=cfg["prompt"])
            row.update(answer=a.text, refused=a.refused, seconds=round(a.seconds, 2),
                       prompt_tokens=a.prompt_tokens)
            if q["answerable"] and not a.refused:
                row["correct"], row["judge_reason"] = judge(q["question"], q["answer"], a.text)
                row["facts_ok"] = has_facts(a.text, q["facts"]) if q["facts"] else None
                row["cited_gold"] = any(set(h.chunk.section_ids) & set(q["gold"]) for h in a.cited)
            elif q["answerable"]:
                row.update(correct=False, facts_ok=False if q["facts"] else None, cited_gold=False)
        extra = "" if retrieval_only else f"  correct={row.get('correct', '-')}  refused={row['refused']}"
        print(f"{q['id']}  rank={row.get('rank')}{extra}")
        rows.append(row)
    summary = summarize(rows, retrieval_only)
    if not retrieval_only:                       # only full runs feed the dashboard
        out = HERE / "results" / f"{name}.json"
        out.parent.mkdir(exist_ok=True)
        out.write_text(json.dumps({"config": name, **cfg, "summary": summary, "rows": rows}, indent=2))
    return summary


def summarize(rows: list[dict], retrieval_only: bool) -> dict:
    ans = [r for r in rows if r["answerable"]]
    unans = [r for r in rows if not r["answerable"]]
    ranks = [r["rank"] for r in ans]
    s = {"hit@1": sum(r == 1 for r in ranks) / len(ans),
         "hit@5": sum(r is not None and r <= K for r in ranks) / len(ans),
         "mrr@10": sum(1 / r for r in ranks if r) / len(ans)}
    if retrieval_only:
        return s
    with_facts = [r for r in ans if r["facts_ok"] is not None]
    answered = [r for r in ans if not r["refused"]]
    s.update({
        "answer_correct": sum(r["correct"] for r in ans) / len(ans),
        "facts_exact": sum(r["facts_ok"] for r in with_facts) / len(with_facts),
        "citation_correct": sum(r["cited_gold"] for r in answered) / max(len(answered), 1),
        "idk_when_unanswerable": sum(r["refused"] for r in unans) / len(unans),
        "idk_when_answerable": sum(r["refused"] for r in ans) / len(ans),
        "overall_accuracy": (sum(r["correct"] for r in ans) + sum(r["refused"] for r in unans)) / len(rows),
        "avg_seconds": sum(r["seconds"] for r in rows) / len(rows),
    })
    return s


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("configs", nargs="*", default=list(CONFIGS))
    ap.add_argument("--retrieval-only", action="store_true")
    args = ap.parse_args()
    results = {name: evaluate(name, args.retrieval_only) for name in args.configs}
    print()
    metrics = list(next(iter(results.values())))
    print(f"{'config':<14}" + "".join(f"{m:>13}" for m in metrics))
    for name, s in results.items():
        print(f"{name:<14}" + "".join(f"{s[m]:>13.2f}" for m in metrics))
