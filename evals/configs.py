"""The tuning story: each config changes one thing from the one before it."""
CONFIGS = {
    "1_baseline": {"chunking": "fixed", "mode": "dense", "prompt": "v1"},
    "2_sections": {"chunking": "sections", "mode": "dense", "prompt": "v1"},
    "3_hybrid": {"chunking": "sections", "mode": "hybrid", "prompt": "v1"},
    "4_rerank": {"chunking": "sections", "mode": "rerank", "prompt": "v1"},
    "5_prompt_v2": {"chunking": "sections", "mode": "rerank", "prompt": "v2"},
}
