from __future__ import annotations

import json

from tools.searchsploit_ import _parse


def test_parses_results():
    raw = json.dumps(
        {
            "RESULTS_EXPLOIT": [
                {
                    "EDB-ID": "49962",
                    "Title": "Apache 2.4.49 Path Traversal",
                    "Date_Published": "2021-10-05",
                    "Author": "anonymous",
                    "Platform": "linux",
                    "Type": "webapps",
                    "Verified": "1",
                    "Path": "linux/webapps/49962.py",
                },
                {
                    "EDB-ID": "50383",
                    "Title": "Apache 2.4.49 Auth Bypass",
                    "Date": "2021-10-09",
                    "Type": "remote",
                    "Verified": "0",
                },
            ]
        }
    )
    cands = _parse(raw)
    assert len(cands) == 2
    apache = cands[0]
    assert apache.edb_id == 49962
    assert apache.verified is True
    assert apache.type == "webapps"
    assert str(apache.date_published) == "2021-10-05"


def test_filter_by_type():
    raw = json.dumps(
        {
            "RESULTS_EXPLOIT": [
                {"EDB-ID": "1", "Title": "a", "Type": "remote"},
                {"EDB-ID": "2", "Title": "b", "Type": "webapps"},
            ]
        }
    )
    cands = _parse(raw, filter_type="webapps")
    assert len(cands) == 1
    assert cands[0].edb_id == 2


def test_empty_safe():
    assert _parse("") == []
    assert _parse("not json") == []
