"""Tests for Wave 2 of the pentester fix plan."""

from __future__ import annotations



from agents.loot import _scan_for_loot_hits
from agents.privesc import _remediation_for_section, _section_severity
from worldstate.schema import Severity


# --------------------------------------------------------------------------- #
# F2a — PrivEscAgent helpers                                                  #
# --------------------------------------------------------------------------- #


def test_section_severity_suid_is_high():
    assert _section_severity("suid_files") == Severity.HIGH
    assert _section_severity("Sudo entries") == Severity.HIGH
    assert _section_severity("kernel_exploits") == Severity.HIGH


def test_section_severity_cron_is_medium():
    assert _section_severity("cron_jobs") == Severity.MEDIUM


def test_section_severity_unknown_is_low():
    assert _section_severity("misc_thing") == Severity.LOW


def test_remediation_strings_are_actionable():
    for section, expected_keyword in [
        ("suid", "GTFOBins"),
        ("sudo", "sudoers"),
        ("kernel", "Patch"),
        ("cron", "cron"),
        ("unknown", "manual triage"),
    ]:
        rem = _remediation_for_section(section)
        assert expected_keyword.lower() in rem.lower()


# --------------------------------------------------------------------------- #
# F2c — LootAgent scanner                                                     #
# --------------------------------------------------------------------------- #


def test_loot_scanner_finds_aws_credentials():
    text = """\
/home/admin/.aws/credentials
[default]
aws_access_key_id = AKIAEXAMPLEKEY
aws_secret_access_key = secret
"""
    hits = _scan_for_loot_hits(text)
    assert any(h["kind"] == "aws_credentials" for h in hits)


def test_loot_scanner_finds_ssh_private_key():
    text = "/home/admin/.ssh/id_rsa\n-----BEGIN OPENSSH PRIVATE KEY-----"
    hits = _scan_for_loot_hits(text)
    assert any(h["kind"] == "ssh_private_key" for h in hits)


def test_loot_scanner_finds_env_file():
    text = "/var/www/app/.env\nDB_PASSWORD=letmein"
    hits = _scan_for_loot_hits(text)
    assert any(h["kind"] == "env_file" for h in hits)


def test_loot_scanner_finds_wp_config():
    text = "drwxr-xr-x  /var/www/html/wp-config.php"
    hits = _scan_for_loot_hits(text)
    assert any(h["kind"] == "wp_config" for h in hits)


def test_loot_scanner_finds_kubeconfig():
    text = "ls -la ~/.kube/config\n-rw------- 1 root root 5432 May 19"
    hits = _scan_for_loot_hits(text)
    assert any(h["kind"] == "kubeconfig" for h in hits)


def test_loot_scanner_no_false_positive_on_innocent_text():
    text = "just normal log lines about nothing sensitive here"
    assert _scan_for_loot_hits(text) == []


def test_loot_scanner_returns_path_and_snippet():
    text = "found /root/.aws/credentials in this thing\nrest of context"
    hits = _scan_for_loot_hits(text)
    assert hits
    h = hits[0]
    assert "path" in h
    assert "snippet" in h
    assert ".aws/credentials" in h["path"]


# --------------------------------------------------------------------------- #
# F12 — BloodHound bundling (compose test removed: docker-compose.yml deleted  #
# in native architecture migration Phase 1.3)                                  #
# --------------------------------------------------------------------------- #
