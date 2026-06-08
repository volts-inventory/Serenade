"""Heuristic destructive-operation detector (file removal = destructive)."""

from __future__ import annotations

import pytest

from tools.destructive_scan import looks_destructive, scan_destructive


@pytest.mark.parametrize(
    "snippet",
    [
        "os.system('rm -rf /var/www')",
        "subprocess.run(['rm', '-f', target])",
        "rm /etc/shadow",
        "subprocess.run(['rm', '/var/log/auth.log'])",
        "DROP TABLE users",
        "DROP DATABASE app",
        "TRUNCATE TABLE logs",
        "truncate -s 0 /var/log/auth.log",
        "echo > /var/log/auth.log",
        "vssadmin delete shadows /all /quiet",
        "wmic shadowcopy delete",
        "bcdedit /set {default} recoveryenabled no",
        "reg delete HKLM\\\\SYSTEM\\\\X /f",
        "Clear-EventLog -LogName Security",
        "wevtutil cl Security",
        "shutil.rmtree(path)",
        "os.remove(victim_file)",
        "Path(p).unlink()",
        "powershell Remove-Item -Recurse -Force C:\\\\data",
        "del /f /q C:\\\\Windows\\\\foo",
        "mkfs.ext4 /dev/sdb",
        "dd if=/dev/zero of=/dev/sda bs=1M",
        "echo x > /dev/sda",
        "format C:",
        "cipher /w:C",
        ":(){ :|:& };:",
        "find /tmp -name '*.log' -delete",
    ],
)
def test_detects_destructive(snippet: str) -> None:
    assert looks_destructive(snippet), f"should flag: {snippet!r}"
    assert scan_destructive(snippet)


@pytest.mark.parametrize(
    "snippet",
    [
        "requests.get(url)",
        "subprocess.run(['nmap', '-sV', target])",
        "print('exploit successful, shell opened')",
        "os.path.exists(p)",
        "data = open(path).read()",
        "socket.connect((rhost, rport))",
        "",
    ],
)
def test_ignores_benign(snippet: str) -> None:
    assert not looks_destructive(snippet), f"false positive: {snippet!r}"
