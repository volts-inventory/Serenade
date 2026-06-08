# Template: build a clean Hyper-V VM for use as a Serenade Windows worker.
#
# This is a STARTING POINT, not a turnkey installer. Run on the Hyper-V host
# AFTER you have:
#   - A Windows Server 2022 ISO at $IsoPath
#   - Created a Hyper-V virtual switch in "Internal" mode (no external bridge)
#   - Decided on the VHDX path and VM name
#
# The script:
#   1. Creates the VM with 4 CPUs / 8 GB RAM / 80 GB VHDX
#   2. Attaches the ISO so the operator can do an interactive install
#   3. Wires it to the internal switch (Redis-only egress)
#   4. After install, the operator runs the Configure-Worker.ps1 inside the VM
#
# Snapshot the VM IMMEDIATELY after Configure-Worker finishes. Each engagement
# revert+snapshot from this golden image.

[CmdletBinding()]
param(
    [string]$VmName       = "Serenade-Win-Worker-01",
    [string]$VhdPath      = "C:\HyperV\Serenade\Win-Worker-01.vhdx",
    [string]$IsoPath      = "C:\HyperV\ISO\WindowsServer2022.iso",
    [string]$SwitchName   = "Serenade-Internal",
    [int]$Cpus            = 4,
    [int64]$MemoryBytes   = 8GB,
    [int64]$VhdSizeBytes  = 80GB
)

if (-not (Get-Module -ListAvailable Hyper-V)) {
    throw "Hyper-V module not installed. Enable: Install-WindowsFeature -Name Hyper-V -IncludeManagementTools"
}

if (-not (Get-VMSwitch -Name $SwitchName -ErrorAction SilentlyContinue)) {
    throw "Switch '$SwitchName' not found. Create with: New-VMSwitch -Name '$SwitchName' -SwitchType Internal"
}

if (-not (Test-Path (Split-Path $VhdPath))) {
    New-Item -ItemType Directory -Path (Split-Path $VhdPath) -Force | Out-Null
}

New-VHD -Path $VhdPath -SizeBytes $VhdSizeBytes -Dynamic | Out-Null
New-VM -Name $VmName -MemoryStartupBytes $MemoryBytes -Generation 2 `
    -VHDPath $VhdPath -SwitchName $SwitchName | Out-Null
Set-VMProcessor -VMName $VmName -Count $Cpus
Set-VMMemory -VMName $VmName -DynamicMemoryEnabled $false
Add-VMDvdDrive -VMName $VmName -Path $IsoPath

$dvd = Get-VMDvdDrive -VMName $VmName
Set-VMFirmware -VMName $VmName -FirstBootDevice $dvd -EnableSecureBoot On `
    -SecureBootTemplate "MicrosoftWindows"

Write-Host "VM '$VmName' created. Start it, install Windows, then:"
Write-Host "  1. Disable Windows Defender (engagement-side tools will trip it)"
Write-Host "  2. Disable Windows Update (image is throwaway)"
Write-Host "  3. Install pwsh: winget install Microsoft.PowerShell"
Write-Host "  4. Copy windows_worker\agent and windows_worker\scripts to C:\Serenade\"
Write-Host "  5. Edit C:\Serenade\agent.config.json"
Write-Host "  6. Run scripts\install-as-service.ps1"
Write-Host "  7. Snapshot the VM: Checkpoint-VM -Name '$VmName' -SnapshotName 'golden-v1'"
