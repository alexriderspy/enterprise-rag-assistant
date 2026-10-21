# Enterprise RAG assistant: answers from a policy handbook, with citations and an eval dashboard

Ask questions about company policy (leave, expenses, travel, relocation, compliance) and get short answers
that cite the exact handbook section, or "I don't know based on the handbook." when the answer isn't there.
The corpus is 14 real policy pages from [GitLab's public handbook](https://handbook.gitlab.com) (CC BY-SA 4.0),
pinned to one commit. A hand-written eval set of 54 questions measures retrieval, answers, citations and
refusals, and every tuning step is re-measured.

Everything runs locally: `bge-small-en-v1.5` embeddings, Chroma, BM25, a MiniLM cross-encoder reranker,
and `qwen2.5:7b` through Ollama.

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
ollama pull qwen2.5:7b                      # and keep `ollama serve` running

python data/fetch_handbook.py               # 14 pages -> data/raw/
python -m rag.parse                         # sections with website-style IDs
python -m rag.chunk                         # fixed-size vs heading-aware chunks
python -m rag.index --chunking fixed        # embeddings -> Chroma
python -m rag.index --chunking sections
python -m rag.answer "How many paid sick days do I get?"
python -m evals.run_eval --retrieval-only   # fast: retrieval metrics for every config
python -m evals.run_eval 1_baseline 4_rerank 5_prompt_v2   # answers, judge, citations, refusals
python -m evals.report                      # all saved runs side by side
python -m streamlit run app/app.py          # chat + eval dashboard
python -m pytest -q
```

## Use a cloud model instead
The LLM is called through the OpenAI-compatible API, so switching is three env vars in `.env`:
`LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL`. `JUDGE_MODEL` lets you grade with a stronger model than the one answering.

## Layout
```
data/fetch_handbook.py   download the 14 pages (pinned commit)
rag/parse.py             markdown -> sections with stable IDs (same anchors as the website)
rag/chunk.py             fixed-size chunks (baseline) and heading-aware chunks
rag/index.py             bge-small embeddings into Chroma
rag/retrieve.py          dense, hybrid (BM25 + dense, reciprocal rank fusion), hybrid + reranker
rag/answer.py            prompt v1 / v2, citations, "I don't know"
evals/questions.jsonl    54 questions: 46 answerable with gold sections, 8 not in the handbook
evals/run_eval.py        hit@1, hit@5, MRR, judge correctness, exact facts, citations, refusals
app/app.py               Streamlit: chat with clickable citations + eval dashboard
```

`media/terminal/` has the exact output of every step from the run in the video (made with `capture.sh`),
and `evals/results/` has the per-question results of every config.
