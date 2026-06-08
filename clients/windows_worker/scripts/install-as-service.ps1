# Register the Serenade Windows worker agent as a Windows service.
#
# Requires elevation. Run from an admin PowerShell prompt:
#   PS> .\install-as-service.ps1
#
# Uses nssm if present (preferred — handles stdout redirection cleanly),
# else falls back to sc.exe with a wrapper.

[CmdletBinding()]
param(
    [string]$ServiceName = "SerenadeWindowsWorker",
    [string]$AgentScript = "C:\Serenade\agent\agent.ps1",
    [string]$ConfigPath  = "C:\Serenade\agent.config.json"
)

if (-not ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    throw "Must run as Administrator."
}

if (-not (Test-Path $AgentScript)) {
    throw "Agent script not found at $AgentScript."
}
if (-not (Test-Path $ConfigPath)) {
    throw "Config not found at $ConfigPath. Copy agent.config.example.json first."
}

$pwsh = (Get-Command pwsh.exe).Source
$nssm = Get-Command nssm.exe -ErrorAction SilentlyContinue

if ($nssm) {
    Write-Host "Installing via nssm..."
    & $nssm.Source install $ServiceName $pwsh "-NoLogo -NoProfile -ExecutionPolicy Bypass -File `"$AgentScript`" -ConfigPath `"$ConfigPath`""
    & $nssm.Source set $ServiceName AppStdout "C:\Serenade\logs\service.out.log"
    & $nssm.Source set $ServiceName AppStderr "C:\Serenade\logs\service.err.log"
    & $nssm.Source set $ServiceName Start SERVICE_AUTO_START
    & $nssm.Source start $ServiceName
} else {
    Write-Host "nssm not found; falling back to sc.exe (logging will be limited)."
    $binPath = "`"$pwsh`" -NoLogo -NoProfile -ExecutionPolicy Bypass -File `"$AgentScript`" -ConfigPath `"$ConfigPath`""
    sc.exe create $ServiceName binPath= $binPath start= auto DisplayName= "Serenade Windows Worker"
    sc.exe start $ServiceName
}

Write-Host "Service $ServiceName installed. Check status: Get-Service $ServiceName"
