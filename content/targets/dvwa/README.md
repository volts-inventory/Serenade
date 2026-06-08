# DVWA test target

Bundled vulnerable web app for E2E testing the Serenade pipeline against an
intentionally-broken target. **Never** expose this beyond your local workstation;
the compose file binds to `127.0.0.1:8080` for that reason.

## Bring up

```bash
# 1. Bring up the main Serenade stack first (creates the sandbox-net network)
serenade up

# 2. Bring up DVWA on the same network
docker compose -f content/targets/dvwa/docker-compose.yml up -d

# DVWA is now at:
#   - http://127.0.0.1:8080 (from your host)
#   - http://172.30.0.100   (from any Serenade sandbox container)
```

## First-run setup

DVWA needs one-time initialization:

1. Open http://127.0.0.1:8080 in your browser
2. Click "Create / Reset Database"
3. Log in as `admin` / `password`
4. Set security level to `low` (Security tab) for maximum findings

## Seed an engagement against it

```bash
# Via the API:
curl -X POST http://localhost:8000/engagements \
  -H "Content-Type: application/json" \
  -d '{
    "name": "DVWA E2E test",
    "scope": {
      "cidrs": ["172.30.0.0/24"],
      "domains": ["dvwa.local"],
      "urls": ["http://172.30.0.100"]
    },
    "objectives": ["xss_demo", "sqli_demo"],
    "mode": "normal",
    "token_budget": 500000,
    "wallclock_hours": 2,
    "authorization_artifact_sha256": "'$(printf "demo engagement against local DVWA" | sha256sum | cut -d' ' -f1)'"
  }'

# Or use the UI: http://localhost:3000/engagements/new
```

## Expected findings (security level: low)

- Reflected XSS on `vulnerabilities/xss_r/`
- SQL injection on `vulnerabilities/sqli/`
- Command injection on `vulnerabilities/exec/`
- File inclusion on `vulnerabilities/fi/`
- CSRF on `vulnerabilities/csrf/`
- Insecure file upload on `vulnerabilities/upload/`

## Tear down

```bash
docker compose -f content/targets/dvwa/docker-compose.yml down -v
```
