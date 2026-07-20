"""End-to-end test of the graph-beats-vector claim (DESIGN.md §11 step 14, §12).

Thresholds are intentionally loose and DIRECTIONAL: single_hop is roughly a tie,
while the knowledge graph clearly wins on multi_hop and aggregation.
"""

from statistics import mean

import pytest

from agent_memory_mcp import Memory
from agent_memory_mcp.bench import DEFAULT_CORPUS, DEFAULT_QUESTIONS, run_eval
from agent_memory_mcp.models import load_questions, read_corpus_lines


@pytest.fixture(scope="module")
def scorecard():
    corpus = read_corpus_lines(DEFAULT_CORPUS)
    questions = load_questions(DEFAULT_QUESTIONS)
    return run_eval(corpus, questions)


def test_single_hop_is_roughly_a_tie(scorecard):
    graph = scorecard.by_method_category["graph"]["single_hop"]
    vector = scorecard.by_method_category["vector"]["single_hop"]
    # both backends handle single-hop lookups well; neither dominates
    assert graph >= 0.8
    assert vector >= 0.6
    assert abs(graph - vector) <= 0.3


def test_graph_strictly_beats_vector_on_multi_hop_and_aggregation(scorecard):
    g = scorecard.by_method_category["graph"]
    v = scorecard.by_method_category["vector"]
    assert g["multi_hop"] > v["multi_hop"]
    assert g["aggregation"] > v["aggregation"]
    graph_mean = mean([g["multi_hop"], g["aggregation"]])
    vector_mean = mean([v["multi_hop"], v["aggregation"]])
    # a clear, not marginal, win
    assert graph_mean - vector_mean >= 0.4


def test_graph_advantage_is_concentrated_in_hard_questions(scorecard):
    g = scorecard.by_method_category["graph"]
    v = scorecard.by_method_category["vector"]
    single_gap = g["single_hop"] - v["single_hop"]
    hard_gap = mean([g["multi_hop"], g["aggregation"]]) - mean([v["multi_hop"], v["aggregation"]])
    assert hard_gap > single_gap


def test_graph_overall_beats_vector_overall(scorecard):
    assert scorecard.overall["graph"] > scorecard.overall["vector"]
    assert scorecard.n >= 12


def test_live_multi_hop_prefers_graph_in_auto_mode():
    mem = Memory(":memory:")
    for line in read_corpus_lines(DEFAULT_CORPUS):
        mem.remember(line)
    res = mem.recall("Who does Dana's manager report to?", method="auto")
    assert res.method == "graph"
    assert res.answer == "Farah"


def test_compare_shows_graph_winning_a_multi_hop():
    mem = Memory(":memory:")
    for line in read_corpus_lines(DEFAULT_CORPUS):
        mem.remember(line)
    out = mem.compare("Who does Dana's manager report to?")
    assert out["graph"]["answer"] == "Farah"
    # vector cannot assemble the two-hop answer
    assert out["vector"]["answer"] != "Farah"
    assert out["agree"] is False
