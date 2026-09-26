# PowerShell runner for Stage 5 - Serve the API.
# Automatically clears any stale process on the port before starting.

$ErrorActionPreference = "Stop"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location "$scriptDir\.."

$PORT = 8000

# Kill any process still holding the port from a previous run
$existing = netstat -ano 2>$null | Select-String ":$PORT\s.*LISTENING" | ForEach-Object {
    ($_ -split '\s+')[-1]
} | Select-Object -Unique
foreach ($stalePid in $existing) {
    if ($stalePid -match '^\d+$' -and $stalePid -ne '0') {
        Write-Host "Killing stale process PID $stalePid on port $PORT ..." -ForegroundColor Yellow
        taskkill /F /PID $stalePid 2>$null | Out-Null
        Start-Sleep -Milliseconds 500
    }
}

$pythonExe = ".\venv\Scripts\python.exe"
if (-not (Test-Path $pythonExe)) {
    $pythonExe = "python"
}

Write-Host ""
Write-Host "Starting Nagpur UHI API on http://127.0.0.1:$PORT ..." -ForegroundColor Cyan
Write-Host "Open http://localhost:$PORT in your browser (dashboard loads automatically)." -ForegroundColor Green
Write-Host "Press Ctrl+C to stop." -ForegroundColor DarkGray
Write-Host ""

& $pythonExe -m uvicorn nagpur_uhi.api.main:app --host 127.0.0.1 --port $PORT --reload
