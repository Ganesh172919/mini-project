$ErrorActionPreference = 'Stop'
$submissionRoot = $PSScriptRoot
$projectRoot = Split-Path -Parent $submissionRoot
$interpreter = Join-Path $projectRoot '.submission-venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $interpreter)) {
    throw 'Create the Python environment using the instructions in submission/README.md first.'
}
Write-Host 'Claim Fraud Lab: http://127.0.0.1:8080'
& $interpreter (Join-Path $submissionRoot 'scripts/serve.py')
