# Launch local Streamlit UI for audio → SRT
$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $here

$py = Join-Path $here ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $py)) {
    throw "Venv missing. Run .\setup_local.ps1 first."
}

& $py -m pip install -q streamlit
Write-Host "Opening http://localhost:8501 ..."
& $py -m streamlit run app.py --server.headless true --browser.gatherUsageStats false
