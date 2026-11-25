import json

import pytest

from semantic_v2.config import SearchConfig, SemanticV2Error


def test_json_round_trip(tmp_path):
    cfg = SearchConfig(alpha=0.3, lam=2.0, hub_fields=("subtitles", "plot"), min_confidence=0.41)
    cfg.to_json(tmp_path / "c.json")
    back = SearchConfig.from_json(tmp_path / "c.json")
    assert back == cfg and isinstance(back.hub_fields, tuple)


def test_partial_config_uses_defaults_and_allows_comments(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"_comment": "tuned on dev", "weights": {"plot": 2.0}, "lam": 0.5}))
    cfg = SearchConfig.from_json(p)
    assert cfg.weights["plot"] == 2.0 and cfg.weights["subtitles"] == SearchConfig().weights["subtitles"]
    assert cfg.lam == 0.5


def test_bad_config_is_rejected(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"lamda": 1}))
    with pytest.raises(SemanticV2Error):
        SearchConfig.from_json(p)
    p.write_text("{not json")
    with pytest.raises(SemanticV2Error):
        SearchConfig.from_json(p)
    p.write_text(json.dumps({"alpha": 3}))
    with pytest.raises(SemanticV2Error):
        SearchConfig.from_json(p)
