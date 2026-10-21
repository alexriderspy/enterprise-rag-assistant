from evals.run_eval import has_facts
from rag.retrieve import rrf, tokenize


def test_rrf_rewards_chunks_both_retrievers_like():
    scores = rrf([["a", "b", "c"], ["c", "a", "d"]])
    assert sorted(scores, key=scores.get, reverse=True) == ["a", "c", "b", "d"]


def test_tokenize_drops_stopwords():
    assert tokenize("How many paid sick days do I get?") == ["many", "paid", "sick", "days", "get"]


def test_fact_check_does_not_match_inside_bigger_numbers():
    assert has_facts("You get 6 days.", [["6"]])
    assert not has_facts("You get 16 weeks.", [["6"]])
    assert has_facts("Up to $1,500 USD", [["1,500", "1500"]])
    assert has_facts("Use NordLayer instead.", [["nordlayer"]])
