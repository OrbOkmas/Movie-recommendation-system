"""Query processor tests. Gold annotations come from evaluation/dev_queries.json."""
import re

import pytest

from semantic_v2.query import parse_negation, strip_filler


def norm(s):
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


def content(s):
    stop = {"a", "an", "the", "i", "to", "be", "want", "of", "bit", "little"}
    return {w for w in norm(s).split() if w not in stop}


def test_negation_gold(dev_queries, processor):
    checked = 0
    for q in dev_queries:
        if "gold_neg" not in q:
            continue
        pq = processor.process(q["query"])
        assert {norm(t) for t in pq.neg_terms} == {norm(t) for t in q["gold_neg"]}, q["query"]
        for pos in q["gold_pos"]:                               # positive words survive in the whole/clauses
            assert content(pos) <= content(pq.whole_text) | content(" ".join(pq.clauses)), (q["query"], pq.whole_text)
        for neg in q["gold_neg"]:                               # negated words are gone from the positive text
            assert not (content(neg) & content(pq.whole_text)), (q["query"], pq.whole_text)
        checked += 1
    assert checked == 7


def test_clause_gold(dev_queries, processor):
    for q in dev_queries:
        if "gold_clauses" not in q:
            continue
        pq = processor.process(q["query"])
        got = [content(c) for c in pq.clauses]
        want = [content(c) for c in q["gold_clauses"]]
        assert len(got) == len(want), (q["query"], pq.clauses)
        for g, w in zip(got, want):
            assert w <= g | {"humour"} or g <= w | {"film"}, (q["query"], pq.clauses)


def test_plot_like_queries_are_not_split_or_negated(dev_queries, processor):
    for q in dev_queries:
        if q.get("split") is False:
            pq = processor.process(q["query"])
            assert pq.clauses == [] and pq.neg_terms == [], q["query"]


def test_expected_status(dev_queries, processor):
    structural = {"empty", "negation_only", "ok", "ok_with_warning", "low_confidence"}
    for q in dev_queries:
        exp = q.get("expected_status")
        if exp in structural:
            assert processor.process(q["query"]).status == exp, (q["query_id"], q["query"])


def test_long_query_warns_and_still_searchable(processor):
    pq = processor.process("word " * 70)
    assert pq.status == "ok_with_warning"
    assert any(w["code"] == "query_too_long" for w in pq.warnings)


def test_hard_truncation(processor):
    pq = processor.process("scary " * 400)
    codes = {w["code"] for w in pq.warnings}
    assert {"query_too_long", "query_truncated"} <= codes
    assert len(pq.text.split()) == 150


@pytest.mark.parametrize("q", ["", "   ", None, 123, "\n\t"])
def test_empty_and_non_text_never_raise(processor, q):
    assert processor.process(q).status == "empty"


@pytest.mark.parametrize("q", ["12345", "🙂🙂", "!!!???"])
def test_no_letters_is_low_confidence(processor, q):
    assert processor.process(q).status == "low_confidence"


def test_shouting_and_punctuation(processor):
    pq = processor.process("SOMETHING SCARY!!!")
    assert pq.text == "something scary!" and pq.status == "ok"


def test_spelling(processor):
    pq = processor.process("somthing tense on the egde of my seat")
    assert "something" in pq.text and "edge" in pq.text and pq.status == "ok_with_warning"
    assert processor.process("cozzy and a litle bit grim").text.startswith("cozy and a little")
    assert "gloomy" in processor.process("gloooomy thriling movie").text


def test_spelling_leaves_rare_valid_words_and_gibberish(processor):
    assert processor.process("a tearjerker with slapstick").corrections == {}
    assert processor.process("asdfgh qwerty zxcv").corrections == {}
    assert processor.process("dark humor").corrections == {}            # US spelling is not "corrected" to UK


@pytest.mark.parametrize("text,pos,neg", [
    ("No horror, just tension and psychological mystery.", "tension and psychological mystery", ["horror"]),
    ("Exciting but without loud action and explosions.", "exciting", ["loud action", "explosions"]),
    ("A dark mystery, but not something too scary.", "a dark mystery", ["scary"]),
    ("gore-free horror", "horror", ["gore"]),
    ("I don't want violence", "", ["violence"]),
    ("Nothing too scary and nothing too sad.", "", ["scary", "sad"]),
])
def test_parse_negation_cases(text, pos, neg):
    assert parse_negation(text) == (pos, neg)


@pytest.mark.parametrize("text", [
    "I don't know what I want.", "A genre I cannot name, just something different.",
    "Give me a funny, chaotic movie that does not take itself seriously.", "Nothing but fun",
    "A funny romance where two people dislike each other", "not only funny but moving"])
def test_parse_negation_leaves_non_negations_alone(text):
    assert parse_negation(text) == (text, [])


def test_filler():
    assert strip_filler("I would like a calm, slow film.") == "calm, slow film"
    assert strip_filler("Romantic and tense at the same time.") == "romantic and tense"


@pytest.mark.parametrize("q", ["Anything, really.", "Surprise me.", "I don't know what I want.", "A good movie.",
                               "I feel like watching something.", "A genre I cannot name, just something different."])
def test_vague_queries_are_flagged(processor, q):
    pq = processor.process(q)
    assert pq.status == "low_confidence" and pq.warnings[0]["code"] == "too_vague"


def test_gibberish_is_flagged_but_typos_are_not(processor):
    assert processor.process("asdfgh qwerty zxcv").warnings[0]["code"] == "unrecognized_words"
    pq = processor.process("gloooomy thriling mistree")
    assert pq.status == "ok_with_warning" and "gloomy" in pq.text and "thrilling" in pq.text


@pytest.mark.parametrize("q", ["melancholy", "romantic comedy", "something scary", "a lonely city at midnight"])
def test_short_but_specific_queries_are_fine(processor, q):
    assert processor.process(q).status == "ok"
