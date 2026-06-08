# destructive-gate-fixture

**Non-functional placeholder.** This is not a real PoC and contains no
destructive (or any other) exploit logic.

## Why it exists

Tracking issue #112, P2 #3. The "destructive action" safety gate
(`PoCDispatcher.can_auto_run` returning `destructive_without_optin`, and the
policy gate's `destructive_not_authorized`) was previously dead code: nothing
shipped set `risk.destructive: true`. This fixture is the minimal artifact that
makes the PoC destructive gate reachable and testable against a real
`PoCLibrary` load.

## Behavior

- `risk.class: destructive` / `risk.destructive: true`.
- `PoCDispatcher.can_auto_run` returns `(False, "destructive_without_optin")`
  for a normal engagement, and `(True, "ok")` only when the engagement sets
  `destructive_opt_in: true` (operator opt-in; PoC path only).
- `build_invocation` raises `NotImplementedError` — a destructive PoC is
  **reported as a finding, never executed**. There is no payload to implement.

## Side effects

None. It never runs anything.
