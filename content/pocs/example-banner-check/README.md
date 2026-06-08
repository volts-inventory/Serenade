# example-banner-check

A benign reference PoC that demonstrates the curated-tier contract. Connects to
`{host}:{port}` via `nc -nvz` in the sandbox and reports reachability. Not a
real exploit; useful as:
- A smoke test for the `PoCDispatcher` end-to-end
- A reference implementation for `build_invocation` / `parse_result`
- A safe way to verify per-engagement sandbox networks reach scoped targets

This entry is `auto_run.allowed: true` because reachability checks are zero-risk;
real exploits should default to `requires_approval: true` until CI green and
operator review.
