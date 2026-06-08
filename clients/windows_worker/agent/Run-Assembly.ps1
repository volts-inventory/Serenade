# Handler for `kind: assembly` jobs.
#
# Loads a .NET assembly via reflection (System.Reflection.Assembly.Load)
# from base64-encoded bytes in the job, calls the entry point with the
# supplied args, captures stdout via redirected Console.Out.
#
# This is the workhorse for "I need to run a custom C# tool" — operator
# compiles their .NET binary, base64s it, submits as a job.

[CmdletBinding()]
param(
    [Parameter(Mandatory)] $Job,
    [Parameter(Mandatory)] [string]$ToolsDir
)

if (-not $Job.assembly_b64) {
    return @{
        exit_code = -11
        stdout    = ""
        stderr    = "assembly_b64 missing from job"
    }
}

try {
    $bytes = [Convert]::FromBase64String($Job.assembly_b64)
    $asm   = [Reflection.Assembly]::Load($bytes)
} catch {
    return @{
        exit_code = -12
        stdout    = ""
        stderr    = "assembly load failed: $($_.Exception.Message)"
    }
}

$entryPoint = $asm.EntryPoint
if (-not $entryPoint) {
    return @{
        exit_code = -13
        stdout    = ""
        stderr    = "assembly has no entry point"
    }
}

$argv = @()
foreach ($a in $Job.args) { $argv += [string]$a }

$origOut = [Console]::Out
$origErr = [Console]::Error
$outBuf = New-Object System.IO.StringWriter
$errBuf = New-Object System.IO.StringWriter
[Console]::SetOut($outBuf)
[Console]::SetError($errBuf)

$exitCode = 0
try {
    $params = if ($entryPoint.GetParameters().Length -eq 0) { @() } else { @(, [string[]]$argv) }
    $rv = $entryPoint.Invoke($null, $params)
    if ($rv -is [int]) { $exitCode = $rv }
} catch {
    $errBuf.WriteLine($_.Exception.Message)
    $exitCode = -14
} finally {
    [Console]::SetOut($origOut)
    [Console]::SetError($origErr)
}

return @{
    exit_code = $exitCode
    stdout    = $outBuf.ToString()
    stderr    = $errBuf.ToString()
}
