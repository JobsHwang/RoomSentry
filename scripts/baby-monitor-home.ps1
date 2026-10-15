param(
    [ValidateSet(
        "menu",
        "status",
        "start",
        "stop",
        "shadow-start",
        "shadow-stop",
        "soak-start",
        "soak-stop",
        "int8-shadow-once",
        "dashboard-start",
        "dashboard-stop"
    )]
    [string]$Action = "menu"
)

$ErrorActionPreference = "Stop"
$Distro = "Ubuntu-24.04"
$WslScriptRoot = $null
foreach ($Prefix in @("\\wsl.localhost\$Distro\", "\\wsl$\$Distro\")) {
    if ($PSScriptRoot.StartsWith($Prefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        $RelativeScriptRoot = $PSScriptRoot.Substring($Prefix.Length).Replace("\", "/")
        $WslScriptRoot = "/$RelativeScriptRoot"
        break
    }
}
if ([string]::IsNullOrWhiteSpace($WslScriptRoot) -or -not $WslScriptRoot.EndsWith("/scripts")) {
    throw "Launch this script from its RoomSentry repository scripts directory."
}
$ProjectRoot = $WslScriptRoot.Substring(0, $WslScriptRoot.Length - "/scripts".Length)
$ProjectMarker = "$ProjectRoot/pyproject.toml"
& wsl.exe -d $Distro --exec test -f $ProjectMarker
if ($LASTEXITCODE -ne 0) {
    throw "The RoomSentry project root is invalid."
}
$WslHomeOutput = & wsl.exe -d $Distro --exec sh -lc 'printf %s "$HOME"' 2>$null
if ($LASTEXITCODE -ne 0) {
    throw "Unable to resolve the WSL home directory."
}
$WslHome = (($WslHomeOutput -join "`n").Trim()).TrimEnd("/")
if (-not $WslHome.StartsWith("/") -or $WslHome.Contains("`n")) {
    throw "The WSL home directory is invalid."
}
$Python = "$ProjectRoot/.venv/bin/python"
$Config = "$ProjectRoot/config/home-shadow.local.toml"
$HAOutbox = "$WslHome/.local/share/baby-monitor/ha-signal-outbox/signals.sqlite3"
$AnalysisDatabase = "$ProjectRoot/data/shadow/state/events.sqlite3"
$CandidateSoakLedger = "$ProjectRoot/data/shadow/state/candidate-soak-ledger.json"
$CandidateHealth = @(
    "$ProjectRoot/data/shadow/buffer/camera_01/candidate-health.json",
    "$ProjectRoot/data/shadow/buffer/camera_02/candidate-health.json",
    "$ProjectRoot/data/shadow/buffer/camera_03/candidate-health.json",
    "$ProjectRoot/data/shadow/buffer/camera_04/candidate-health.json"
)
$DashboardUrl = "http://127.0.0.1:8765/"

$Labels = @{
    "xiaomi_stream_gateway" = "Xiaomi local stream gateway"
    "home_assistant_signals" = "Home Assistant event watcher"
    "local_evidence_buffer" = "Local rolling evidence buffer"
    "ha_signal_buffer_dispatch" = "HA event to local evidence queue"
    "bounded_home_shadow" = "Bounded YOLO/candidate shadow"
    "candidate_soak_24h" = "Four-camera YOLO 24h soak"
    "int8_mage_shadow_once" = "INT8 Mage-VL one-shot shadow"
    "local_dashboard" = "Local dashboard"
}

function Open-LocalDashboard {
    $Result = Invoke-HomeControl -RequestedAction "dashboard-start"
    if ($null -eq $Result) {
        return $false
    }
    for ($Attempt = 0; $Attempt -lt 20; $Attempt++) {
        try {
            $Response = Invoke-WebRequest `
                -Uri $DashboardUrl `
                -UseBasicParsing `
                -TimeoutSec 1
            if ($Response.StatusCode -eq 200) {
                Start-Process $DashboardUrl
                return $true
            }
        }
        catch {
        }
        Start-Sleep -Milliseconds 250
    }
    Write-Host "Dashboard did not become ready." -ForegroundColor Red
    return $false
}

function Invoke-HomeControl {
    param(
        [string]$RequestedAction,
        [bool]$ConfirmBufferEviction = $false
    )

    $Arguments = @(
        "-m", "baby_monitor", "home-control", $RequestedAction,
        "--config", $Config,
        "--ha-outbox-database", $HAOutbox,
        "--analysis-database", $AnalysisDatabase,
        "--candidate-soak-ledger", $CandidateSoakLedger
    )
    foreach ($Health in $CandidateHealth) {
        $Arguments += @("--candidate-health", $Health)
    }
    if ($ConfirmBufferEviction) {
        $Arguments += "--confirm-buffer-eviction"
    }
    $Output = & wsl.exe -d $Distro --cd $ProjectRoot $Python @Arguments 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Operation failed. No private command output was displayed." -ForegroundColor Red
        return $null
    }
    try {
        return (($Output -join "`n") | ConvertFrom-Json)
    }
    catch {
        Write-Host "The local controller returned an invalid status." -ForegroundColor Red
        return $null
    }
}

function Show-HomeStatus {
    param($Status)

    if ($null -eq $Status) {
        return
    }
    Write-Host ""
    Write-Host "BABY MONITOR HOME CONTROL" -ForegroundColor Cyan
    Write-Host "========================="
    foreach ($Component in $Status.components) {
        $Label = $Labels[$Component.component]
        $StateColor = if ($Component.active) { "Green" } elseif ($Component.attention_required) { "Red" } else { "DarkYellow" }
        $Autostart = if ($Component.enabled) { "autostart" } else { "manual" }
        Write-Host (("{0,-36} {1,-12} {2}" -f $Label, $Component.state, $Autostart)) -ForegroundColor $StateColor
    }
    Write-Host ""
    Write-Host ("Daily base monitoring:  " + $(if ($Status.base_operational) { "READY" } else { "STOPPED / DEGRADED" }))
    Write-Host ("HA durable event queue: " + $Status.ha_signal_outbox.status)
    Write-Host ("Local notification queue: " + $Status.notification_queue.status)
    Write-Host ("Local dashboard:        " + $Status.dashboard)
    Write-Host ("Daily digest:           " + $Status.daily_digest)
    Write-Host ("Privacy daily report:   " + $Status.daily_report.status)
    Write-Host ("Web review:             " + $Status.web_review)
    Write-Host "Remote upload:          OFF"
}

function Invoke-AndShow {
    param(
        [string]$RequestedAction,
        [bool]$ConfirmBufferEviction = $false
    )
    $Result = Invoke-HomeControl `
        -RequestedAction $RequestedAction `
        -ConfirmBufferEviction $ConfirmBufferEviction
    Show-HomeStatus -Status $Result
    return $Result
}

if ($Action -ne "menu") {
    if ($Action -eq "dashboard-start") {
        if (Open-LocalDashboard) { exit 0 }
        exit 2
    }
    $Confirmed = $false
    if ($Action -in @("start", "shadow-start", "soak-start", "int8-shadow-once")) {
        $Expected = if ($Action -eq "start") { "START" } elseif ($Action -eq "shadow-start") { "SHADOW" } elseif ($Action -eq "soak-start") { "SOAK24" } else { "INT8" }
        if ((Read-Host "Type $Expected to confirm local rolling-buffer eviction").ToUpperInvariant() -ne $Expected) {
            exit 2
        }
        $Confirmed = $true
    }
    $Result = Invoke-AndShow `
        -RequestedAction $Action `
        -ConfirmBufferEviction $Confirmed
    if ($null -eq $Result) { exit 2 }
    exit 0
}

while ($true) {
    Clear-Host
    Invoke-AndShow -RequestedAction "status" | Out-Null
    Write-Host ""
    Write-Host "1  Start daily monitoring (streams + HA + buffer + event queue)"
    Write-Host "2  Stop all monitoring for this WSL session"
    Write-Host "3  Start bounded shadow stack"
    Write-Host "4  Stop bounded shadow stack"
    Write-Host "5  Run one INT8 Mage-VL shadow job"
    Write-Host "6  Open local dashboard"
    Write-Host "7  Start four-camera YOLO 24-hour soak"
    Write-Host "8  Stop four-camera YOLO soak"
    Write-Host "R  Refresh"
    Write-Host "Q  Quit"
    Write-Host ""
    $Choice = Read-Host "Select"
    switch ($Choice.ToUpperInvariant()) {
        "1" {
            if ((Read-Host "Type START to confirm local rolling-buffer eviction").ToUpperInvariant() -eq "START") {
                Invoke-AndShow -RequestedAction "start" -ConfirmBufferEviction $true | Out-Null
            }
        }
        "2" { Invoke-AndShow -RequestedAction "stop" | Out-Null }
        "3" {
            if ((Read-Host "Type SHADOW to confirm bounded local shadow").ToUpperInvariant() -eq "SHADOW") {
                Invoke-AndShow -RequestedAction "shadow-start" -ConfirmBufferEviction $true | Out-Null
            }
        }
        "4" { Invoke-AndShow -RequestedAction "shadow-stop" | Out-Null }
        "5" {
            if ((Read-Host "Type INT8 to confirm one local GPU shadow job").ToUpperInvariant() -eq "INT8") {
                Invoke-AndShow -RequestedAction "int8-shadow-once" -ConfirmBufferEviction $true | Out-Null
            }
        }
        "6" { Open-LocalDashboard | Out-Null }
        "7" {
            if ((Read-Host "Type SOAK24 to confirm the local 24-hour YOLO soak").ToUpperInvariant() -eq "SOAK24") {
                Invoke-AndShow -RequestedAction "soak-start" -ConfirmBufferEviction $true | Out-Null
            }
        }
        "8" { Invoke-AndShow -RequestedAction "soak-stop" | Out-Null }
        "R" { }
        "Q" { exit 0 }
        default { Write-Host "Unknown selection." -ForegroundColor Yellow }
    }
    if ($Choice.ToUpperInvariant() -notin @("R", "Q")) {
        Read-Host "Press Enter to continue" | Out-Null
    }
}
