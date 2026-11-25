import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


@pytest.fixture(scope="session")
def dev_queries():
    return json.loads((ROOT / "evaluation" / "dev_queries.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def test_queries():
    return json.loads((ROOT / "evaluation" / "mood_queries.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def processor():
    from semantic_v2.config import QueryConfig
    from semantic_v2.query import QueryProcessor
    return QueryProcessor(QueryConfig())
