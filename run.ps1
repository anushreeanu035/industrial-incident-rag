# Industrial Incident Investigator - PowerShell Launcher
Write-Host "=====================================================================" -ForegroundColor Cyan
Write-Host " Industrial Incident Investigator - Full Stack Launcher" -ForegroundColor Cyan
Write-Host "=====================================================================" -ForegroundColor Cyan

Set-Location -Path $PSScriptRoot
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

if (Test-Path ".\.venv\Scripts\python.exe") {
    Write-Host "Using project virtual environment: .venv" -ForegroundColor Green
    & ".\.venv\Scripts\python.exe" app.py all
} else {
    Write-Host "Using system Python" -ForegroundColor Yellow
    python app.py all
}
