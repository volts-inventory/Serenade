"""Heuristic detector for destructive operations in exploit/PoC code.

"Destructive" = irreversible harm to the *target*: deleting or overwriting
files, wiping disks/filesystems, DoS/fork-bombs. The sandbox isolates our
host, not the client's box, so a script that runs ``rm -rf /`` on the target
is destructive even though it can't hurt us.

This is a best-effort static scan (an obfuscated payload can evade it), used
as defense-in-depth on top of the declarative ``risk.destructive`` manifest
flag and the operator ``destructive_opt_in`` gate — not a substitute for
either. A match means "treat as destructive unless the engagement opted in".
"""

from __future__ import annotations

import re

# Each pattern is (compiled regex, human label). Patterns target operations
# that remove/overwrite data on the target or take it down. Kept conservative
# and high-signal to limit false positives.
_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\brm\b['\"]?\s*,?\s*['\"]?(-[a-zA-Z]|/|~|\$|\w)", re.I), "rm <arg>"),
    (re.compile(r"\b(unlink|shred)\b", re.I), "unlink/shred"),
    (re.compile(r"\brmdir\b", re.I), "rmdir"),
    (re.compile(r"\bfind\b[^\n]*-delete\b", re.I), "find -delete"),
    (re.compile(r"\btruncate\b[^\n]*-s\s*0\b", re.I), "truncate -s 0"),
    (re.compile(r"\bos\.truncate\s*\(", re.I), "os.truncate"),
    (re.compile(r">\s*/var/log/", re.I), "truncate /var/log"),
    (re.compile(r"\bmkfs(\.\w+)?\b", re.I), "mkfs"),
    (re.compile(r"\bdd\b[^\n]*\bof=/dev/", re.I), "dd of=/dev/*"),
    (re.compile(r">\s*/dev/(sd|nvme|hd|vd)", re.I), "overwrite block device"),
    (re.compile(r"\b(fdisk|wipefs|blkdiscard)\b", re.I), "fdisk/wipefs"),
    (re.compile(r"\bdel\s+/[a-zA-Z]", re.I), "del /f|/q|/s"),
    (re.compile(r"\bRemove-Item\b[^\n]*-(Recurse|Force)", re.I), "Remove-Item -Recurse/-Force"),
    (re.compile(r"\b(Format-Volume|Clear-Disk|Clear-Content|Clear-EventLog)\b", re.I), "Format-Volume/Clear-Disk"),
    (re.compile(r"\bformat\s+[a-zA-Z]:", re.I), "format <drive>:"),
    (re.compile(r"\bcipher\s+/w", re.I), "cipher /w"),
    (re.compile(r"\bvssadmin\b[^\n]*\bdelete\b", re.I), "vssadmin delete shadows"),
    (re.compile(r"\bwmic\b[^\n]*shadowcopy[^\n]*delete\b", re.I), "wmic shadowcopy delete"),
    (re.compile(r"\bbcdedit\b[^\n]*recoveryenabled\s+no\b", re.I), "bcdedit recoveryenabled no"),
    (re.compile(r"\bwevtutil\s+cl\b", re.I), "wevtutil cl"),
    (re.compile(r"\breg(\.exe)?\s+delete\b[^\n]*/f\b", re.I), "reg delete /f"),
    (re.compile(r"\b(DROP\s+(TABLE|DATABASE)|TRUNCATE\s+TABLE)\b", re.I), "SQL DROP/TRUNCATE"),
    (re.compile(r"\b(os\.remove|os\.unlink|os\.rmdir|shutil\.rmtree)\s*\(", re.I), "os.remove/shutil.rmtree"),
    (re.compile(r"\.unlink\s*\(", re.I), "Path.unlink"),
    (re.compile(r":\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:", re.I), "fork bomb"),
]


def scan_destructive(text: str) -> list[str]:
    """Return the labels of destructive operations found in ``text``.

    Empty list ⇒ no destructive operation detected.
    """
    if not text:
        return []
    found: list[str] = []
    for pattern, label in _PATTERNS:
        if pattern.search(text):
            found.append(label)
    return found


def looks_destructive(text: str) -> bool:
    """True if ``text`` contains any detectable destructive operation."""
    return bool(scan_destructive(text))
