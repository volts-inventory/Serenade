# Runbook

Solo-operator (laptop) operations. One section per situation. Symptoms
first, exact commands after.

If you are reading this in a panic, jump to the case that matches.

---

## Where data lives

Engagement data is under `~/.serenade/`:

| Path | Contents |
|---|---|
| `serenade.db` | SQLite database: engagements, audit log, tool runs, LLM calls, LangGraph checkpoints (control-flow state), and the world-state snapshot (`world_state_snapshots` -- discovered hosts/services/findings/credentials, with credential secret fields Fernet-encrypted at rest) |
| `evidence/` | Tool output blobs, content-addressed by `{engagement_id}/{blob_id}` |

Configuration lives **outside** `~/.serenade/`: core engine settings are in
the `.env` file at the project root (LLM backend, sandbox settings,
budgets), and TUI preferences are in `~/.config/serenade/config.toml`.

> **Note:** the world-state graph (including credential nodes) is in memory
> *during a run* but is **also** snapshotted to the `world_state_snapshots`
> table in `serenade.db` after every agent step that wrote nodes, and
> rehydrated on resume -- so discovered hosts/services/findings/credentials
> survive a pause/resume. The per-install master key at
> `~/.serenade/.master_key` (mode 0600, generated on first use) backs two
> things that are on disk: the audit-log HMAC chain, and Fernet encryption of
> the credential secret fields inside the world-state snapshot. Back up
> `.master_key` with `serenade.db` and treat it as the most sensitive file in
> the directory -- losing it makes the audit chain unverifiable **and** the
> snapshot's credential secret fields unrecoverable.

---

## Backup

Copy the directory:

```bash
cp -r ~/.serenade/ /path/to/backup/serenade-$(date +%Y%m%d)/
```

Or automate:

```bash
# crontab -e
0 3 * * 0 rsync -a ~/.serenade/ /mnt/nas/serenade-backup/
```

Copy the backup off-laptop weekly (external drive, NAS, cloud storage).
Also back up your `.env` (it holds the LLM backend choice and, if you use
Anthropic, the API key) — it lives at the project root, not under
`~/.serenade/`.

Credential secret fields in the persisted graph are Fernet-encrypted (see
the note above), but evidence blobs and the rest of the SQLite data are
plain text, and the backup is only as protected as the `.master_key` you
store alongside it. Keep backups somewhere you'd be comfortable storing
sensitive engagement data, and guard `.master_key` accordingly.

---

## Reset

### Full reset (start fresh)

```bash
rm -rf ~/.serenade/
serenade --setup
```

This deletes all engagements and evidence. Your `.env` at the project root
is untouched; `--setup` will rewrite it if you re-run the wizard.

### Soft reset (keep config, wipe engagements)

```bash
rm ~/.serenade/serenade.db
rm -rf ~/.serenade/evidence/
serenade    # SQLite is re-created on next launch
```

Your `.env` config (at the project root) survives. New engagements start clean.

### Reset a single engagement

There's no surgical delete yet. Simplest approach: start a new
engagement. Old ones stay in the DB (harmless) until you do a full or
soft reset.

---

## Case 1 -- Laptop swap / fresh install

**Symptoms:** old laptop is gone / dead. You have your
`~/.serenade/` backup.

```bash
# 1. Install serenade on the new machine.
pip install serenade

# 2. Restore data.
cp -r /path/to/backup/serenade-latest/ ~/.serenade/

# 3. Verify Docker is installed and running.
docker info

# 4. Rebuild the sandbox image (it's not in the backup).
serenade --setup    # will detect existing config, only rebuild the image

# 5. Launch.
serenade
```

---

## Case 2 -- Corrupted database

**Symptoms:** serenade crashes on startup with SQLite errors, or
engagement history is garbled.

**Loss model:** SQLite holds engagement metadata, audit log, tool
runs, LangGraph checkpoints, and the world-state snapshot
(`world_state_snapshots`). Deleting `serenade.db` loses the snapshot, so a
resumed engagement starts from an empty graph and re-derives facts by
re-running agents; evidence blobs survive (they're separate files).

```bash
# 1. Back up what you have (in case it's partially recoverable).
cp ~/.serenade/serenade.db ~/.serenade/serenade.db.corrupt

# 2. Delete the corrupted DB.
rm ~/.serenade/serenade.db

# 3. Restart -- a fresh DB is created automatically.
serenade

# 4. If you have a backup, restore from it instead:
cp /path/to/backup/serenade-latest/serenade.db ~/.serenade/serenade.db
serenade --migrate    # apply any schema changes since the backup
serenade
```

---

## Case 3 -- Migration failure

**Symptoms:** `serenade --migrate` exits non-zero, or serenade refuses
to start with an Alembic error.

```bash
# 1. Check current DB version.
serenade --migrate    # shows current head + error

# 2. If the DB is salvageable, try downgrading:
#    (replace with actual alembic command for your install)
python -m alembic downgrade -1
python -m alembic upgrade head

# 3. If that fails, restore from backup (Case 2) or do a soft reset.
```

---

## Troubleshooting

### Docker daemon not running

```
Error: Docker daemon not running.
```

Start Docker:
- macOS: `open -a Docker`
- Linux: `sudo systemctl start docker`

Verify: `docker info`

### Sandbox image not found

```
Error: Sandbox image not found. Run: serenade --setup
```

```bash
serenade --setup    # rebuilds the image
```

Or build manually:

```bash
docker build -t serenade/sandbox:dev -f infra/docker/Dockerfile.sandbox .
```

### Ollama not responding

```
Error: Ollama not responding on localhost:11434.
```

Start Ollama:

```bash
ollama serve
```

Check it's listening:

```bash
curl http://localhost:11434/api/tags
```

If you switched to Anthropic, ignore this -- Ollama is not needed.

### Low RAM warning

```
Warning: RAM: 16 GB detected. Gemma 4 26B needs ~20 GB.
```

Options:
- Switch to Anthropic: `serenade --setup`
- Accept slower performance (models swap between calls instead of
  co-residing)
- Add RAM

### Engagement stuck / not advancing

Check the Events tab in the TUI for the latest event. Common causes:

- **Pending approval**: Approvals tab has unhandled rows. Approve or
  reject them, or switch the engagement to `autonomous` mode (`p` in TUI)
  so exploit gates auto-approve instead of queuing.
- **Stalled-turn watchdog fired**: 4 consecutive steps with zero
  progress. The engagement auto-completes with a `BUDGET STOP` note.
- **Zero-host watchdog**: 3 consecutive recon turns found nothing.
  Usually means the scope is firewalled. Check scope, check network
  access from the sandbox.

### Evidence too large

Evidence blobs are never cleaned up automatically. If `~/.serenade/evidence/`
grows too large:

```bash
# See total size
du -sh ~/.serenade/evidence/

# Remove evidence for a specific engagement
rm -rf ~/.serenade/evidence/<engagement-id>/

# Remove all evidence (engagements still work, but tool output is gone)
rm -rf ~/.serenade/evidence/
```

---

## Operating the opt-in agent patterns

Four opt-in features (all default off) add scoring, memory, debate, and
reflection at the cost of extra LLM calls per step. Architecture details
and config keys are in [ARCHITECTURE.md - Modern agent patterns](./ARCHITECTURE.md#modern-agent-patterns).

| Setting | Config key | Enable |
|---|---|---|
| Step scoring | `enable_step_scoring = true` | 1 extra E4B call/step; ~1-2s on local |
| Reflection memory | `enable_reflection_memory = true` | ~1,500 tokens added to planner context |
| Debate gates | `enable_debate_gates = true` | 2-4 extra E4B calls per exploit/lateral gate |
| Regression reflection | `enable_regression_reflection = true` | LLM-suggested regressions, council-verified |

(Tree-of-Thoughts is documented but not yet implemented — there is no
`enable_thought_tree` setting.)

Set these in your `.env` file via the corresponding `ENABLE_*` environment
variables (e.g. `ENABLE_STEP_SCORING=1`). The `Settings` model has no env
prefix, so it is `ENABLE_STEP_SCORING`, not `SERENADE_ENABLE_STEP_SCORING`.
