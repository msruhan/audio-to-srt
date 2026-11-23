# Dual-path audio → SRT helper
#   A (cloud/fast):  .\transcribe.ps1 -Engine ot -Audio path\to\file.m4a
#   B (local):       .\transcribe.ps1 -Engine local -Audio path\to\file.m4a

param(
    [Parameter(Mandatory = $true)]
    [string]$Audio,

    [ValidateSet("ot", "local")]
    [string]$Engine = "local",

    [string]$OutDir = "",

    [string]$Model = "large-v3",

    [string]$Language = "",

    [switch]$Vad,
    [switch]$Txt,
    [switch]$Diarize
)

$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$audioPath = Resolve-Path -LiteralPath $Audio

if (-not (Test-Path -LiteralPath $audioPath)) {
    throw "Audio not found: $Audio"
}

if ($Engine -eq "ot") {
    if (-not (Get-Command ot -ErrorAction SilentlyContinue)) {
        throw "ot CLI not found. Install: npm install -g @opentranscription/cli && ot login"
    }
    $argsList = @("transcribe", "$audioPath")
    if ($Diarize) { $argsList += "--diarize" }
    if ($Language) { $argsList += @("--language", $Language) }
    if ($OutDir) { $argsList += @("--out", $OutDir) }
    Write-Host ">>> ot $($argsList -join ' ')"
    & ot @argsList
    exit $LASTEXITCODE
}

$venvPython = Join-Path $here ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $venvPython)) {
    throw "Local venv missing. Run setup_local.ps1 first."
}

$pyArgs = @((Join-Path $here "transcribe_local.py"), "$audioPath", "--model", $Model)
if ($OutDir) {
    $leaf = [IO.Path]::GetFileNameWithoutExtension($audioPath) + ".srt"
    $pyArgs += @("-o", (Join-Path $OutDir $leaf))
}
if ($Language) { $pyArgs += @("--language", $Language) }
if ($Vad) { $pyArgs += "--vad" }
if ($Txt) { $pyArgs += "--txt" }

Write-Host ">>> local: $venvPython $($pyArgs -join ' ')"
& $venvPython @pyArgs
exit $LASTEXITCODE
