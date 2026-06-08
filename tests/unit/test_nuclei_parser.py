from __future__ import annotations

import json

from tools.nuclei_ import _parse


def test_parses_jsonl():
    docs = [
        {
            "template-id": "log4j-rce",
            "info": {
                "name": "Log4Shell",
                "severity": "critical",
                "description": "Log4j2 JNDI",
                "classification": {"cve-id": "CVE-2021-44228", "cwe-id": "CWE-502"},
            },
            "matched-at": "http://10.0.0.5/",
            "host": "10.0.0.5",
            "extracted-results": ["jndi:ldap://x"],
        },
        {
            "template-id": "exposed-env",
            "info": {"name": ".env exposed", "severity": "medium"},
            "matched-at": "http://10.0.0.5/.env",
        },
    ]
    jsonl = "\n".join(json.dumps(d) for d in docs)
    findings = _parse(jsonl)
    assert len(findings) == 2
    log4j = findings[0]
    assert log4j.template_id == "log4j-rce"
    assert log4j.severity == "critical"
    assert log4j.classification.get("cve-id") == ["CVE-2021-44228"]
    assert log4j.classification.get("cwe-id") == ["CWE-502"]
    assert log4j.extracted_results == ["jndi:ldap://x"]


def test_skips_bad_lines():
    jsonl = '{"template-id":"x","info":{"name":"y","severity":"low"}}\nnot-json\n'
    findings = _parse(jsonl)
    assert len(findings) == 1
    assert findings[0].template_id == "x"
