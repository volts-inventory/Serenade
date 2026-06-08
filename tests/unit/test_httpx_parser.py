from __future__ import annotations

import json

from tools.httpx_ import _parse


def test_parses_jsonl():
    docs = [
        {
            "url": "http://10.0.0.5/",
            "host": "10.0.0.5",
            "scheme": "http",
            "port": 80,
            "status_code": 200,
            "title": "Welcome",
            "webserver": "Apache/2.4.49",
            "tech": ["Apache", "PHP/7.4"],
            "content_length": 1024,
            "tls": False,
        },
        {
            "url": "https://10.0.0.5/",
            "host": "10.0.0.5",
            "scheme": "https",
            "port": 443,
            "status_code": 200,
            "tech": {"nginx": {}, "Wordpress": {}},
            "tls": True,
        },
    ]
    probes = _parse("\n".join(json.dumps(d) for d in docs))
    assert len(probes) == 2
    assert probes[0].technologies == ["Apache", "PHP/7.4"]
    assert probes[0].server == "Apache/2.4.49"
    assert probes[1].technologies == ["nginx", "Wordpress"]
    assert probes[1].tls is True


def test_skips_bad_lines():
    docs = '{"url":"http://x","host":"x","scheme":"http","port":80}\n\nnot-json\n'
    probes = _parse(docs)
    assert len(probes) == 1
