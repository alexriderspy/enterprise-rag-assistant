"""Streamlit app: ask the handbook (with clickable citations) and see the eval dashboard."""
import json
import re
from pathlib import Path

import pandas as pd
import streamlit as st

from rag.answer import answer
from rag.retrieve import Retriever

RESULTS = Path(__file__).parent.parent / "evals" / "results"
st.set_page_config(page_title="Handbook assistant", layout="wide")


@st.cache_resource
def get_retriever() -> Retriever:
    return Retriever(chunking="sections", mode="rerank")


def linkify(text: str, hits) -> str:
    """Turn [S2] in the answer into a link to that section of the handbook."""
    def link(m: re.Match) -> str:
        n = int(m.group(1))
        return f"[[S{n}]]({hits[n - 1].chunk.url})" if 0 < n <= len(hits) else m.group(0)
    return re.sub(r"\[S(\d+)\]", link, text)


def passed(row: dict) -> bool | None:
    """Answerable questions need a correct answer; unanswerable ones need "I don't know"."""
    return row.get("correct") if row["answerable"] else row.get("refused")


ask, evals = st.tabs(["Ask the handbook", "Eval dashboard"])

with ask:
    question = st.chat_input("Ask a question about leave, expenses, travel, relocation...")
    if question:
        st.chat_message("user").write(question)
        hits = get_retriever().search(question, k=5)
        result = answer(question, hits)
        with st.chat_message("assistant"):
            st.markdown(linkify(result.text, hits))
            st.caption(f"{result.seconds:.1f}s · {len(result.cited)} of {len(hits)} sources cited")
        for n, hit in enumerate(hits, 1):
            cited = "cited" if hit in result.cited else "not cited"
            with st.expander(f"[S{n}] {hit.chunk.title}  ({cited})"):
                st.markdown(f"[Open in the handbook]({hit.chunk.url})")
                st.text(hit.chunk.text)

with evals:
    runs = [json.loads(p.read_text()) for p in sorted(RESULTS.glob("*.json"))]
    if not runs:
        st.info("No results yet. Run: python -m evals.run_eval")
        st.stop()
    table = pd.DataFrame({r["config"]: r["summary"] for r in runs}).T
    first, last = table.index[0], table.index[-1]
    tiles = st.columns(4)
    for col, metric in zip(tiles, ["overall_accuracy", "hit@1", "citation_correct", "idk_when_unanswerable"]):
        if metric in table:
            before, after = table.loc[first, metric], table.loc[last, metric]
            col.metric(metric, f"{after:.0%}", f"{after - before:+.0%} vs {first}")
    default = [m for m in ["hit@1", "answer_correct", "citation_correct", "overall_accuracy"] if m in table]
    picked = st.multiselect("Metrics", list(table.columns), default=default)
    st.bar_chart(table[picked], stack=False, height=320)
    st.dataframe(table.style.format("{:.2f}"), use_container_width=True)

    st.subheader("Question by question")
    a, b = st.columns(2)
    left = a.selectbox("Before", table.index, index=0)
    right = b.selectbox("After", table.index, index=len(table) - 1)
    by_config = {r["config"]: {q["id"]: q for q in r["rows"]} for r in runs}
    st.dataframe(pd.DataFrame([{
        "question": q["question"],
        f"{left} right?": passed(q),
        f"{right} right?": passed(by_config[right][qid]),
        "answer after": by_config[right][qid].get("answer", ""),
    } for qid, q in by_config[left].items()]), use_container_width=True, hide_index=True)
