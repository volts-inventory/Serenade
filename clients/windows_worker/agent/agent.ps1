# Serenade Windows worker — agent loop.
#
# Polls Redis via BLPOP for jobs on `windows_worker_q`, dispatches to a
# handler script based on `kind`, pushes the result onto
# `windows_worker_results`. Logs every job to C:\Serenade\logs\agent.log.
#
# Requires PowerShell 7+ and the `Microsoft.PowerShell.Redis` module OR
# the `redis-cli` binary on PATH. We use redis-cli here because it's
# universally available and avoids a NuGet dep that operators have to
# trust.

[CmdletBinding()]
param(
    [string]$ConfigPath = "C:\Serenade\agent.config.json"
)

$ErrorActionPreference = "Stop"

# --- Config ---------------------------------------------------------------
if (-not (Test-Path $ConfigPath)) {
    throw "Config not found at $ConfigPath. Copy agent.config.example.json and edit."
}
$cfg = Get-Content -Raw $ConfigPath | ConvertFrom-Json

$RedisHost  = $cfg.redis_host
$RedisPort  = $cfg.redis_port
$RedisPass  = $cfg.redis_password
$JobQueue   = $cfg.job_queue
$ResultQueue= $cfg.result_queue
$AuthToken  = $cfg.auth_token
$ToolsDir   = $cfg.tools_dir
$LogPath    = $cfg.log_path
$HeartbeatKey = $cfg.heartbeat_key
# R3: isolation mode self-reported on every heartbeat. The orchestrator
# refuses high-risk jobs (mimikatz, assembly) to workers that report 'none'.
# One of: hyperv-snapshot | firecracker | bare-metal | none.
$IsolationMode = if ($cfg.isolation_mode) { $cfg.isolation_mode } else { "none" }
# R3: HMAC-SHA256 key shared with the orchestrator. Required to dispatch
# 'assembly' jobs. The agent verifies $job.assembly_sig against this key
# before invoking the entry point.
$SigningKey = $cfg.signing_key

if (-not (Test-Path (Split-Path $LogPath))) {
    New-Item -ItemType Directory -Path (Split-Path $LogPath) -Force | Out-Null
}

function Write-Log($level, $msg, $extra = $null) {
    $entry = [ordered]@{
        ts    = (Get-Date).ToUniversalTime().ToString("o")
        level = $level
        msg   = $msg
    }
    if ($null -ne $extra) { $entry.extra = $extra }
    $entry | ConvertTo-Json -Compress | Add-Content -Path $LogPath
}

function Invoke-RedisCli {
    param([string[]]$Argv)
    $base = @("-h", $RedisHost, "-p", "$RedisPort")
    if ($RedisPass) { $base += @("-a", $RedisPass, "--no-auth-warning") }
    & redis-cli @base @Argv
}

# --- Heartbeat ------------------------------------------------------------
function Update-Heartbeat {
    # R3: heartbeat payload is JSON with isolation self-report. Orchestrator
    # parses this and refuses to dispatch high-risk jobs to a worker that
    # admits isolation=none unless the operator opts in.
    $payload = [ordered]@{
        ts        = (Get-Date).ToUniversalTime().ToString("o")
        isolation = $IsolationMode
    } | ConvertTo-Json -Compress
    Invoke-RedisCli @("SET", $HeartbeatKey, $payload, "EX", "60") | Out-Null
}

# R5 #4: probe whether the .NET 5+ FixedTimeEquals API is available. PS 7
# uses .NET 5/6/7+ which has it; Windows PowerShell 5.1 uses .NET Framework
# 4.x which does NOT. We pick the path at agent startup so the dispatch
# loop doesn't pay the lookup cost per job.
$Script:HasFixedTimeEquals = $false
try {
    $null = [System.Security.Cryptography.CryptographicOperations]
    $Script:HasFixedTimeEquals = $true
} catch {
    $Script:HasFixedTimeEquals = $false
}

function ConstantTimeEquals-Bytes {
    param([byte[]]$A, [byte[]]$B)
    if ($A.Length -ne $B.Length) { return $false }
    # Real crypto constant-time at the algorithm level (single accumulator,
    # bitwise OR of XORs). Without [byte][char] conversion this loop is
    # bytes-only, which is much cheaper than the prior char-indexed version
    # — still not as good as .NET's FixedTimeEquals (which fences the JIT)
    # but the timing side-channel exposure here isn't network-reachable.
    $diff = 0
    for ($i = 0; $i -lt $A.Length; $i++) {
        $diff = $diff -bor ($A[$i] -bxor $B[$i])
    }
    return $diff -eq 0
}

# --- Assembly HMAC verify (R3 + R4 #5 + R5 #4) ---------------------------
function Verify-AssemblySignature {
    param([string]$AssemblyB64, [string]$Sig)
    if (-not $SigningKey) {
        return $false
    }
    if (-not $Sig) {
        return $false
    }
    try {
        $bytes = [Convert]::FromBase64String($AssemblyB64)
    } catch {
        return $false
    }
    $hmac = New-Object System.Security.Cryptography.HMACSHA256(([Text.Encoding]::UTF8.GetBytes($SigningKey)))
    $computedBytes = $hmac.ComputeHash($bytes)
    $hmac.Dispose()
    # Parse the hex signature to bytes. Sig must be hex; bail on any non-hex.
    if ($Sig.Length -ne ($computedBytes.Length * 2)) { return $false }
    try {
        $sigBytes = New-Object byte[] ($Sig.Length / 2)
        for ($i = 0; $i -lt $sigBytes.Length; $i++) {
            $sigBytes[$i] = [Convert]::ToByte($Sig.Substring($i * 2, 2), 16)
        }
    } catch {
        return $false
    }
    if ($Script:HasFixedTimeEquals) {
        # R4 #5: .NET 5+ primitive — JIT-fenced constant time.
        return [System.Security.Cryptography.CryptographicOperations]::FixedTimeEquals($computedBytes, $sigBytes)
    } else {
        # R5 #4: PS 5.1 / .NET Framework 4.x fallback. Algorithm-level
        # constant-time XOR-OR loop on byte arrays (no char conversion).
        return (ConstantTimeEquals-Bytes $computedBytes $sigBytes)
    }
}

# --- Job dispatch ---------------------------------------------------------
$Handlers = @{
    "mimikatz" = "C:\Serenade\agent\Run-MimikatzJob.ps1"
    "assembly" = "C:\Serenade\agent\Run-Assembly.ps1"
    "shell"    = "C:\Serenade\agent\Run-ShellJob.ps1"
}

function Invoke-Job($job) {
    $jobId = $job.job_id
    $kind  = $job.kind
    Write-Log "info" "job.start" @{ job_id = $jobId; kind = $kind }

    $handler = $Handlers[$kind]
    if (-not $handler) {
        return @{
            job_id    = $jobId
            exit_code = -1
            stdout_b64= ""
            stderr_b64= [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes("unknown job kind: $kind"))
            started_at = (Get-Date).ToUniversalTime().ToString("o")
            ended_at   = (Get-Date).ToUniversalTime().ToString("o")
        }
    }

    # R3: enforce HMAC signature on assembly jobs before dispatch.
    if ($kind -eq "assembly") {
        if (-not (Verify-AssemblySignature $job.assembly_b64 $job.assembly_sig)) {
            Write-Log "warn" "job.bad_assembly_sig" @{ job_id = $jobId }
            return @{
                job_id    = $jobId
                exit_code = -4
                stdout_b64= ""
                stderr_b64= [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes("assembly signature verification failed"))
                started_at = (Get-Date).ToUniversalTime().ToString("o")
                ended_at   = (Get-Date).ToUniversalTime().ToString("o")
            }
        }
    }

    $started = (Get-Date).ToUniversalTime().ToString("o")
    try {
        $result = & $handler -Job $job -ToolsDir $ToolsDir
    } catch {
        $ended = (Get-Date).ToUniversalTime().ToString("o")
        Write-Log "error" "job.handler_threw" @{ job_id = $jobId; error = $_.Exception.Message }
        return @{
            job_id    = $jobId
            exit_code = -2
            stdout_b64= ""
            stderr_b64= [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($_.Exception.Message))
            started_at = $started
            ended_at   = $ended
        }
    }
    $ended = (Get-Date).ToUniversalTime().ToString("o")
    Write-Log "info" "job.done" @{ job_id = $jobId; exit_code = $result.exit_code }
    return @{
        job_id    = $jobId
        exit_code = $result.exit_code
        stdout_b64= [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes(($result.stdout ?? "")))
        stderr_b64= [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes(($result.stderr ?? "")))
        started_at = $started
        ended_at   = $ended
    }
}

# --- Main loop ------------------------------------------------------------
Write-Log "info" "agent.start" @{
    redis_host          = $RedisHost
    job_queue           = $JobQueue
    tools_dir           = $ToolsDir
    ps_version          = $PSVersionTable.PSVersion.ToString()
    has_fixedtimeequals = $Script:HasFixedTimeEquals
    isolation_mode      = $IsolationMode
}

$lastHeartbeat = [DateTime]::MinValue
while ($true) {
    if (((Get-Date) - $lastHeartbeat).TotalSeconds -gt 15) {
        Update-Heartbeat
        $lastHeartbeat = Get-Date
    }
    # BLPOP with 10s timeout so the heartbeat can refresh.
    $raw = Invoke-RedisCli @("BLPOP", $JobQueue, "10")
    if (-not $raw) { continue }
    # redis-cli BLPOP output: first line = key, second line = value.
    $lines = @($raw -split "`n")
    if ($lines.Count -lt 2) { continue }
    $payload = $lines[1]
    try {
        $job = $payload | ConvertFrom-Json
    } catch {
        Write-Log "warn" "job.bad_json" @{ payload_excerpt = $payload.Substring(0, [Math]::Min(200, $payload.Length)) }
        continue
    }
    if ($job.auth_token -ne $AuthToken) {
        Write-Log "warn" "job.bad_auth" @{ job_id = $job.job_id }
        continue
    }
    $result = Invoke-Job $job
    $resultJson = $result | ConvertTo-Json -Compress -Depth 6
    # Push keyed by job_id so the orchestrator can BRPOP the right one.
    $key = "$ResultQueue`:$($job.job_id)"
    Invoke-RedisCli @("LPUSH", $key, $resultJson) | Out-Null
    Invoke-RedisCli @("EXPIRE", $key, "3600") | Out-Null
}
