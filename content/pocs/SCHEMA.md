# Curated PoC Library — Manifest Schema

Every entry in `pocs/<id>/` is a hand-vetted, parameterized exploit that `ExploitAgent` may execute autonomously when `auto_run.allowed: true` and preconditions are satisfied.

EDB scripts from the discovery tier are **never** executed directly. They must be reviewed, re-implemented as a curated entry, CI-tested against a pinned vulnerable lab, and merged via PR before they can run.

## Directory layout

```
pocs/<id>/
├── manifest.yaml         # required; see schema below
├── exploit.py            # required; the parameterized executable
├── README.md             # required; ops notes (intended behavior, side effects, what success looks like)
└── test/
    ├── target_compose.yaml   # optional; vulhub or self-built lab
    └── test_exploit.py       # required; pytest assertion against the lab
```

## manifest.yaml fields

| Field | Type | Required | Notes |
|---|---|---|---|
| `id` | string | yes | Match the directory name; e.g. `CVE-2021-44228-log4shell` |
| `provenance` | list of `{source, ref}` | yes | At least one entry. Sources: `exploitdb`, `metasploit`, `nuclei`, `cve_writeup`, `original_research` |
| `target.service` | string | yes | e.g. `log4j`, `apache_httpd` |
| `target.version_range` | semver-like range | yes | e.g. `>=2.0,<2.15.0` |
| `target.platforms` | list of `linux | windows | multiple` | yes | |
| `parameters.schema_ref` | path | yes | Pydantic-compatible JSON schema for exploit parameters |
| `risk.class` | `recon | enum | exploit | post-exploit | destructive` | yes | Matches `RiskClass` enum |
| `risk.destructive` | bool | yes | Irreversible harm to the target — **file/data removal or overwrite**, disk/filesystem wipe, DoS. If true, requires `engagement.destructive_opt_in`. A PoC that deletes or overwrites target files MUST set this (CI scans `exploit.py` and fails a destructive-looking body that leaves it `false`). |
| `risk.modifies_target` | bool | yes | Spawns process, registry key, writes a file, etc. (reversible side effects — distinct from `destructive`). |
| `risk.modifies_filesystem` | bool | yes | Writes/creates files on the target (disclosure). Note: *removing* files is `destructive`, not just `modifies_filesystem`. |
| `risk.network_egress_from_target` | bool | yes | e.g. JNDI callback |
| `risk.detection_likelihood` | `low | medium | high` | yes | |
| `auto_run.allowed` | bool | yes | If false, always requires operator approval regardless of engagement mode |
| `auto_run.requires_approval` | bool | yes | If true, operator must click approve even in `autonomous` mode |
| `auto_run.preconditions` | list of structured Pydantic predicates (no free-form CEL) | no | Evaluated against `{engagement, target, world_state}`; all must be true |
| `rollback.available` | bool | yes | |
| `rollback.notes` | string | no | What persists after run; what cleanup is possible |
| `license.spdx` | SPDX identifier | yes | `UNKNOWN` will fail CI |
| `license.attribution` | string | yes | Original author / source citation |
| `ci.test_command` | string | yes | Default: `pytest test/test_exploit.py` |
| `ci.lab` | string | no | vulhub directory or "self-built" if test_compose exists |

## Hard rules enforced by `ExploitAgent`

1. Only entries with `auto_run.allowed: true` AND `auto_run.requires_approval: false` AND all preconditions true run without operator click.
2. `risk.destructive: true` always requires explicit per-engagement `destructive_opt_in` (a separate axis from mode).
3. Every execution records the manifest sha256 in the `ExploitAttempt` evidence so the report can prove which version of the PoC was used.
4. `license.spdx == "UNKNOWN"` is a hard fail in CI; the entry is not loaded into the registry.

## Promotion workflow (EDB candidate → curated entry)

1. `VulnTriageAgent` emits an `ExploitCandidate` with `source=exploitdb` and `auto_runnable=false`.
2. UI surfaces it with "Detected: CVE-X (EDB-Y). Not in curated library."
3. Operator runs `serenade-vet --edb-id <Y> --as <new-id>` (CLI in Phase 5).
4. Scaffolds `pocs/<new-id>/` from `pocs/_template/`.
5. Operator implements `exploit.py`, parameter schema, CI test, and sets risk flags.
6. PR review gate before merge.
7. CI runs `pytest pocs/<new-id>/test/` against the pinned lab — must pass.
8. After merge, the entry is available in the curated tier on next engagement.
