# Serenade — Windows worker

A Windows-native worker tier that runs alongside the Linux orchestrator
and consumes Windows-only jobs (mimikatz, Cobalt-Strike-style payloads,
.NET assemblies that need a real Windows runtime).

## Architecture

```
+---------------------+   BLPOP serenade:windows_worker:jobs    +-------------------------+
|  orchestrator       |  ---------------------------------->     |  windows worker (VM)    |
|  (Linux container)  |  RPUSH serenade:windows_worker:results   |  PowerShell agent.ps1   |
|  WindowsJobQueue    |  <----------------------------------     |                         |
+---------------------+                                          +-------------------------+
                                       |                                      |
                                       v                                      v
                                   Redis (jobs)                       Sandbox: per-engagement
                                                                       isolated Hyper-V VM
                                                                       OR Firecracker-Windows
                                                                       OR bare-metal lab box
```

Wire details:

- Job queue: `serenade:windows_worker:jobs` (orchestrator pushes, worker
  BLPOPs).
- Result queue: `serenade:windows_worker:results` (worker pushes,
  orchestrator BRPOPs keyed by `job_id`).
- Heartbeat key: `serenade:windows_worker:heartbeat`; refreshed every
  ~15s with a 60s TTL. Every dispatch checks freshness before pushing
  so a stale worker fails fast (`-102 worker heartbeat stale`).

The Linux orchestrator publishes jobs to a Redis list; the Windows
worker polls the list, runs the requested tool, and pushes a result
back. The worker is **not** the sandbox by itself — it's a job
dispatcher that runs jobs inside a per-engagement Windows isolation
primitive (see "Isolation choices" below).

## What ships in this directory

| File | Role |
|---|---|
| `agent/agent.ps1` | PowerShell agent loop (BLPOP + dispatch + heartbeat) |
| `agent/Run-MimikatzJob.ps1` | Handler for the `mimikatz` job kind |
| `agent/Run-Assembly.ps1` | Handler for the `assembly` job kind (.NET via reflection; HMAC-verified bytes) |
| `agent/agent.config.example.json` | Config schema: Redis URL, queue names, tool dir, auth token, signing key |
| `scripts/install-as-service.ps1` | Register the agent as a Windows service via `sc.exe` |
| `scripts/build-isolated-vm.ps1` | Operator template for a fresh Hyper-V snapshot |
| `docker-compose.windows-worker.yml` | Compose overlay declaring `windows-worker` as an external service (operator stands it up themselves) |
| `serenade_windows_job.schema.json` | JSON schema for the wire format |

## Isolation choices

This is the call the operator has to make. The Linux sandbox uses a
per-engagement Docker network + nftables L3 filter; there is no
equivalent we can ship for Windows because Windows containers don't
have the same network primitives as Linux. Three viable paths:

### A. Hyper-V snapshots (most defensible, most ops)

- Operator builds a clean Windows 11 / Server 2022 VM with the agent
  installed.
- Before each engagement, snapshot the VM. After the engagement, revert
  and delete the snapshot.
- Per-engagement network isolation via Hyper-V virtual switches.
- Tooling: `scripts/build-isolated-vm.ps1` is the template.

### B. Firecracker-Windows (lighter, experimental)

- Firecracker's Windows backend is WIP. If the lab box is bare-metal
  Linux, nested KVM with Windows guests is possible.
- Not recommended for production; documented for completeness.

### C. Disposable bare-metal lab box (simplest, no isolation)

- A dedicated lab machine that gets re-imaged between engagements.
- Fine for solo pentest; **not safe for multi-tenant or MSSP use.**
- The worker config sets `engagement_isolation: "none"` and the
  orchestrator logs a warning on every dispatch so the operator can't
  silently downgrade. The orchestrator additionally refuses high-risk
  jobs (mimikatz, assembly) in this mode unless
  `SERENADE_WINDOWS_WORKER_ALLOW_UNSAFE_ISOLATION=1`.

The wire format and agent code are identical across all three. Only
the lifecycle harness (VM snapshot / Firecracker boot / re-image)
differs.

## Wire format

Jobs are JSON; see `serenade_windows_job.schema.json`. A typical
mimikatz job:

```json
{
  "job_id": "01J6Z2QKQ8...",
  "engagement_id": "1ec3-...",
  "kind": "mimikatz",
  "args": ["sekurlsa::logonpasswords", "exit"],
  "timeout_seconds": 120,
  "submitted_at": "2026-05-20T18:00:00Z"
}
```

The result, pushed onto `serenade:windows_worker:results`:

```json
{
  "job_id": "01J6Z2QKQ8...",
  "exit_code": 0,
  "stdout_b64": "...",
  "stderr_b64": "...",
  "started_at": "2026-05-20T18:00:01Z",
  "ended_at": "2026-05-20T18:00:04Z"
}
```

The orchestrator's `WindowsJobQueue` (see
`src/orchestrator/windows_worker.py`) is the Python side: it serialises
the job, pushes via Redis, and BRPOPs for the result keyed by `job_id`.
Wall-clock timeouts are enforced by the orchestrator side too — a dead
worker doesn't hang the engagement.

`assembly_b64` jobs carry an `assembly_sig` field — hex HMAC-SHA256
over the raw assembly bytes, keyed on
`SERENADE_WINDOWS_WORKER_SIGNING_KEY`. The worker verifies the
signature via `CryptographicOperations::FixedTimeEquals` before
invoking the entry point. A compromised Redis cannot inject arbitrary
.NET into the worker.

## Why this ships as scripts, not a binary

The Windows-side runtime is PowerShell 7. We deliberately do not ship
a compiled .NET binary because:

- Operators have wildly different lab setups (domain-joined vs not,
  Windows Server vs client SKUs, ARM vs x64).
- AV evasion is engagement-specific. The operator stages their own
  mimikatz / Rubeus / SharpHound binary in `C:\Serenade\tools\` — the
  agent invokes whatever's there.
- A signed binary would need either Authenticode or a self-signed cert
  per operator. Out of scope for the platform; operator chooses.

Operator install:

1. Stand up the Windows VM (clean install, no internet during build).
2. Install PowerShell 7: `winget install Microsoft.PowerShell`.
3. Copy this directory's `agent/` and `scripts/` to `C:\Serenade\`.
4. Stage tool binaries in `C:\Serenade\tools\`: `mimikatz.exe`,
   `Rubeus.exe`, `SharpHound.exe`, whatever else.
5. Edit `agent.config.json` with Redis URL, `auth_token`, and
   `signing_key`.
6. Run `scripts/install-as-service.ps1` to register the agent as a
   Windows service.
7. Snapshot the VM. This is your golden image.

Set `SERENADE_WINDOWS_WORKER_ENABLED=1` on the orchestrator. The
matching `SERENADE_WINDOWS_WORKER_AUTH_TOKEN` and
`SERENADE_WINDOWS_WORKER_SIGNING_KEY` must match the worker's
`agent.config.json`.

## Security model

- The Redis queue is shared between the orchestrator and the worker.
  Anyone with Redis access can submit Windows-side jobs. The worker
  validates `auth_token` on every job and rejects mismatches.
- Jobs are **not** subject to OPA gating on the Windows side — gating
  happens at submission time in the orchestrator (same PreLateralCritic
  + OPA path as `windows_via_wmi`).
- The worker logs every job (`job_id`, `kind`, `exit_code`) to
  `C:\Serenade\logs\agent.log`. The orchestrator additionally writes an
  audit row.
- The worker has **no outbound internet**. The operator-side Hyper-V
  switch (or whatever isolation primitive) MUST block egress except to:
  - The orchestrator's Redis (job pull / result push).
  - The engagement scope IPs (target hosts only).
- Mimikatz output is high-value loot. The orchestrator stores it
  encrypted-at-rest in MinIO same as Linux-side loot.
- `parse_mimikatz_output` is bounded against adversary-controlled
  stdout — see [`docs/ARCHITECTURE.md`](../../docs/ARCHITECTURE.md#windows-worker-isolation)
  for the parser hardening.

## Operator health surface

The orchestrator exposes `GET /admin/windows_workers` returning JSON
worker status (heartbeat freshness, queue depth, isolation mode). The
Web UI does not yet render this on a panel; use `curl` or the JSON
output until that lands.
