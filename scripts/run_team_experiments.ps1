param(
    [ValidateRange(1, 100)][int]$RunCount = 4,
    [ValidateRange(0, 3600)][int]$PauseSeconds = 60,
    [ValidateRange(0, 100000)][int]$Warmup = 50,
    [ValidateRange(1, 100000)][int]$Rounds = 10,
    [ValidateRange(1, 100000)][int]$Calls = 100,
    [string]$Operator,
    [ValidateSet('Yes', 'No', 'Not recorded')][string]$PluggedIn,
    [string]$PowerMode,
    [string]$BackgroundLoad,
    [string]$Notes = ''
)

$ErrorActionPreference = 'Stop'
$projectDirectory = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectDirectory
$pythonExecutable = Join-Path $projectDirectory '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonExecutable)) {
    Write-Host 'Run Setup_Environment.cmd first.' -ForegroundColor Red
    exit 1
}
if ([string]::IsNullOrWhiteSpace($Operator)) { $Operator = Read-Host 'Your name or student ID' }
if ([string]::IsNullOrWhiteSpace($PluggedIn)) {
    $PluggedIn = Read-Host 'Is the computer plugged in? Type Yes or No'
    if ($PluggedIn -notin @('Yes', 'No')) { Write-Host 'Enter Yes or No.'; exit 1 }
}
if ([string]::IsNullOrWhiteSpace($PowerMode)) { $PowerMode = Read-Host 'Actual Windows power mode (for example: Best performance)' }
if ([string]::IsNullOrWhiteSpace($BackgroundLoad)) { $BackgroundLoad = Read-Host 'Background workload (for example: idle; other apps closed)' }
if ([string]::IsNullOrWhiteSpace($Operator) -or [string]::IsNullOrWhiteSpace($PowerMode) -or [string]::IsNullOrWhiteSpace($BackgroundLoad)) {
    Write-Host 'Record your name, power mode and background workload before starting.' -ForegroundColor Red
    exit 1
}
$teamLogDirectory = Join-Path $projectDirectory 'logs'
New-Item -ItemType Directory -Path $teamLogDirectory -Force | Out-Null
$batchId = 'team_' + (Get-Date -Format 'yyyyMMdd_HHmmss') + '_' + [guid]::NewGuid().ToString('N').Substring(0, 8)
$batchLog = Join-Path $teamLogDirectory ($batchId + '.log')
$transcriptActive = $false
try {
    Start-Transcript -Path $batchLog | Out-Null
    $transcriptActive = $true
    $isFull = $Warmup -eq 50 -and $Rounds -eq 10 -and $Calls -eq 100
    Write-Host 'Keep the same power and workload conditions. Run only one benchmark at a time.'
    Write-Host 'Close ONNX_Experiment_Tracker.xlsx in Excel before starting.'
    Write-Host "Planned runs: $RunCount. Each successful run is freshly verified and added to the tracker."
    for ($runIndex = 1; $runIndex -le $RunCount; $runIndex++) {
        $purpose = if (-not $isFull) { 'Smoke test' } elseif ($runIndex -eq 1) { 'Independent reproduction' } else { 'Controlled repeat' }
        $runNote = "$Notes Team batch $batchId. Run $runIndex of $RunCount."
        if ($isFull -and $runIndex -gt 1) { $runNote += " Stability repetition $($runIndex - 1); exclude the first independent reproduction from the three-repeat comparison." }
        Write-Host ''
        Write-Host "Starting $purpose ($runIndex of $RunCount)."
        & $pythonExecutable -B -X utf8 -u -m experiment_tools run --operator $Operator --plugged-in $PluggedIn --power-mode $PowerMode --background-load $BackgroundLoad --purpose $purpose --notes $runNote --warmup $Warmup --rounds $Rounds --calls $Calls
        if ($LASTEXITCODE -ne 0) { throw "Run $runIndex stopped. Read the workflow log and recovery command printed above. Do not delete the run folder." }
        if ($runIndex -lt $RunCount -and $PauseSeconds -gt 0) {
            Write-Host "Waiting $PauseSeconds seconds before the next run."
            Start-Sleep -Seconds $PauseSeconds
        }
    }
    Write-Host ''
    Write-Host 'All requested experiments were verified and recorded in ONNX_Experiment_Tracker.xlsx.'
    Write-Host 'Send back the tracker plus the complete output/runs and logs folders.'
    Write-Host "Team session log: $batchLog"
} catch {
    Write-Host $_.Exception.Message -ForegroundColor Red
    Write-Host "Team session log: $batchLog"
    exit 1
} finally {
    if ($transcriptActive) { Stop-Transcript | Out-Null }
}
