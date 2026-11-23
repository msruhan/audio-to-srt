# One-time setup for jalur B (local faster-whisper)
$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $here

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Host "Creating venv..."
    python -m venv .venv
}

Write-Host "Upgrading pip..."
& .\.venv\Scripts\python.exe -m pip install --upgrade pip

Write-Host "Installing faster-whisper..."
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt

Write-Host ""
Write-Host "Checking CUDA via ctranslate2..."
& .\.venv\Scripts\python.exe -c "import ctranslate2; n=ctranslate2.get_cuda_device_count(); print(f'CUDA devices: {n}')"

Write-Host ""
Write-Host "Done. Example:"
Write-Host '  .\transcribe.ps1 -Engine local -Audio "C:\path\to\file.m4a" -Language id -Vad -Txt'
