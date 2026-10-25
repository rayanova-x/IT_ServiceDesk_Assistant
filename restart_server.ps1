# Reliable server restart -- kills by PORT OWNERSHIP, not by process
# name/path matching. Added 2026-10-24 after two separate incidents where
# a restart left a stray/renamed process still serving the OLD code on
# port 8010, causing live UI tests to show stale behavior that looked
# like a real regression but wasn't (confirmed both times by checking
# Get-NetTCPConnection against the python process list and finding a
# mismatch). Port ownership is the only reliable signal -- a process's
# own Path/name can be wrong or missing by the time it's queried.

$port = 8010

Write-Host "Stopping any process bound to port $port..."
Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue | ForEach-Object {
    Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue
}

# Wait for the port to actually free -- TIME_WAIT sockets can linger
# briefly after Stop-Process; starting too early just creates a new
# stray on a still-occupied port.
$maxWait = 15
$waited = 0
while ((Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue) -and ($waited -lt $maxWait)) {
    Start-Sleep -Seconds 1
    $waited++
}

$stillListening = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
if ($stillListening) {
    Write-Host "WARNING: port $port still has a listener after $maxWait s -- restart may fail."
    exit 1
}

Write-Host "Port $port is free. Starting server..."
