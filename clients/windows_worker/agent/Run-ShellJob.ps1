# Handler for `kind: shell` jobs.
#
# Runs a tool from C:\Serenade\tools\ (or any allow-listed absolute path
# under ToolsDir). Rejects any path traversal. Captures stdout/stderr,
# enforces the job's timeout_seconds.

[CmdletBinding()]
param(
    [Parameter(Mandatory)] $Job,
    [Parameter(Mandatory)] [string]$ToolsDir
)

if (-not $Job.exe) {
    return @{
        exit_code = -20
        stdout    = ""
        stderr    = "exe missing from job"
    }
}

# Resolve and confirm the exe lives under $ToolsDir. Reject path traversal.
$toolsFull = (Resolve-Path -LiteralPath $ToolsDir).Path
$exePath = Join-Path $ToolsDir $Job.exe
if (-not (Test-Path $exePath)) {
    return @{
        exit_code = -21
        stdout    = ""
        stderr    = "exe not found: $($Job.exe)"
    }
}
$exeFull = (Resolve-Path -LiteralPath $exePath).Path
if (-not $exeFull.StartsWith($toolsFull + [IO.Path]::DirectorySeparatorChar) -and $exeFull -ne $toolsFull) {
    return @{
        exit_code = -22
        stdout    = ""
        stderr    = "exe path escapes tools dir: $($Job.exe)"
    }
}

$argv = @()
foreach ($a in $Job.args) { $argv += [string]$a }

$timeoutSec = if ($Job.timeout_seconds) { [int]$Job.timeout_seconds } else { 120 }

if (-not (Test-Path "C:\Serenade\tmp")) {
    New-Item -ItemType Directory -Path "C:\Serenade\tmp" -Force | Out-Null
}
$outPath = "C:\Serenade\tmp\out.$($Job.job_id).txt"
$errPath = "C:\Serenade\tmp\err.$($Job.job_id).txt"

$proc = Start-Process -FilePath $exeFull -ArgumentList $argv -NoNewWindow -PassThru `
    -RedirectStandardOutput $outPath -RedirectStandardError $errPath

if (-not $proc.WaitForExit($timeoutSec * 1000)) {
    try { $proc.Kill() } catch {}
    Remove-Item $outPath -ErrorAction SilentlyContinue
    Remove-Item $errPath -ErrorAction SilentlyContinue
    return @{
        exit_code = -3
        stdout    = ""
        stderr    = "timeout after $timeoutSec seconds"
    }
}

$out = Get-Content -Raw $outPath -ErrorAction SilentlyContinue
$err = Get-Content -Raw $errPath -ErrorAction SilentlyContinue
Remove-Item $outPath -ErrorAction SilentlyContinue
Remove-Item $errPath -ErrorAction SilentlyContinue

return @{
    exit_code = $proc.ExitCode
    stdout    = ($out ?? "")
    stderr    = ($err ?? "")
}
