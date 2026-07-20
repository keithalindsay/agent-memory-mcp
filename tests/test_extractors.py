"""RulesExtractor tests (DESIGN.md §11 step 7, §12)."""

from agent_memory_mcp.extractors import RulesExtractor, get_extractor

SIX_SENTENCES = [
    "Dana is an Engineer.",
    "Dana is managed by Evan.",
    "Evan is managed by Farah.",
    "Dana is a member of the Search Team.",
    "The Search Team owns the Index Service.",
    "Atlas uses the Index Service.",
]

EXPECTED = {
    ("person:dana", "IS_A", "thing:engineer"),
    ("person:dana", "MANAGED_BY", "person:evan"),
    ("person:evan", "MANAGED_BY", "person:farah"),
    ("person:dana", "MEMBER_OF", "team:search-team"),
    ("team:search-team", "OWNS", "service:index-service"),
    ("thing:atlas", "USES", "service:index-service"),
}


def _all_relations(statements):
    triples = set()
    for st in statements:
        for r in st.relations:
            triples.add((r.src, r.rel, r.dst))
    return triples


def test_six_sentences_yield_six_expected_relations():
    ex = RulesExtractor()
    statements = ex.extract("\n".join(SIX_SENTENCES))
    assert _all_relations(statements) == EXPECTED


def test_each_sentence_extracts_its_relation():
    ex = RulesExtractor()
    for sent in SIX_SENTENCES:
        sts = ex.extract(sent)
        rels = _all_relations(sts)
        assert len(rels) == 1, f"{sent!r} -> {rels}"


def test_garbage_line_yields_no_relations_but_is_retained_raw():
    ex = RulesExtractor()
    sts = ex.extract("Xyzzy plugh frobnicate wibble.")
    assert _all_relations(sts) == set()
    assert len(sts) == 1
    assert sts[0].text.strip() != ""
    assert sts[0].relations == []


def test_extractor_is_deterministic():
    ex = RulesExtractor()
    a = _all_relations(ex.extract("\n".join(SIX_SENTENCES)))
    b = _all_relations(ex.extract("\n".join(SIX_SENTENCES)))
    assert a == b


def test_manages_flips_direction():
    ex = RulesExtractor()
    rels = _all_relations(ex.extract("Evan manages Dana."))
    assert ("person:dana", "MANAGED_BY", "person:evan") in rels


def test_get_extractor_factory_returns_rules():
    ex = get_extractor("rules")
    assert isinstance(ex, RulesExtractor)


def test_conjunction_splits_into_two_relations():
    ex = RulesExtractor()
    rels = _all_relations(ex.extract("Dana is managed by Evan and Evan is managed by Farah."))
    assert ("person:dana", "MANAGED_BY", "person:evan") in rels
    assert ("person:evan", "MANAGED_BY", "person:farah") in rels
