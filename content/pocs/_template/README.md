# PoC template

Copy this directory to `pocs/<CVE-or-id>/` and fill in:
- `manifest.yaml` — all required fields; set `license.spdx` to a real identifier or CI fails
- `exploit.py` — replace the `NotImplementedError` stub
- `parameters.schema.json` — JSON Schema for the parameters `run()` receives
- `test/test_exploit.py` — pytest assertion against the pinned lab; must pass before merge
- `README.md` — this file; describe intended behavior, side effects, success indicators
