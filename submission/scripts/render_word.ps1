param([string]$FileName)
$ErrorActionPreference = 'Stop'
$submissionDir = Split-Path -Parent $PSScriptRoot
$source = Join-Path $submissionDir ('deliverables/' + $FileName + '.docx')
$renderDir = Join-Path $submissionDir ('.build/' + $FileName)
New-Item -ItemType Directory -Force -Path $renderDir | Out-Null
$pdfPath = Join-Path $renderDir ($FileName + '.pdf')
$wordInstance = $null
$documentInstance = $null
try {
    $wordInstance = New-Object -ComObject Word.Application
    $wordInstance.Visible = $false
    $wordInstance.DisplayAlerts = 0
    $wordInstance.AutomationSecurity = 3
    $documentInstance = $wordInstance.Documents.Open($source, $false, $true)
    $documentInstance.Repaginate()
    $pageCount = $documentInstance.ComputeStatistics(2)
    $documentInstance.ExportAsFixedFormat($pdfPath, 17)
    Write-Output "Native Word render: $FileName, $pageCount pages, $pdfPath"
} finally {
    if ($null -ne $documentInstance) { $documentInstance.Close(0) }
    if ($null -ne $wordInstance) { $wordInstance.Quit() }
}
