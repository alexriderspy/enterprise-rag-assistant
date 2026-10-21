"""Print every saved eval run side by side, or list the answers the judge marked wrong for one config."""
import argparse
import json
from pathlib import Path

RESULTS = Path(__file__).parent / "results"


def table() -> None:
    runs = [json.loads(p.read_text()) for p in sorted(RESULTS.glob("*.json"))]
    names = [r["config"] for r in runs]
    print(f"{'metric':<24}" + "".join(f"{n:>14}" for n in names))
    for metric in runs[0]["summary"]:
        values = "".join(f"{r['summary'][metric]:>14.2f}" for r in runs)
        print(f"{metric:<24}{values}")
    print(f"\n{len(runs[0]['rows'])} questions per config")


def wrong(config: str) -> None:
    rows = json.loads((RESULTS / f"{config}.json").read_text())["rows"]
    for r in rows:
        if r["answerable"] and not r["correct"]:
            print(f"{r['id']}  {r['question']}")
            print(f"     answer: {r['answer'][:110]}")
            reason = r.get("judge_reason") or "refused, so no judge call"
            print(f"     judge:  {reason[:105]}\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--wrong", metavar="CONFIG", help="show the answers the judge marked wrong")
    args = ap.parse_args()
    if args.wrong:
        wrong(args.wrong)
    else:
        table()
