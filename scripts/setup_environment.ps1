$ErrorActionPreference = 'Stop'
$projectDirectory = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectDirectory
$setupLogDirectory = Join-Path $projectDirectory 'logs'
New-Item -ItemType Directory -Path $setupLogDirectory -Force | Out-Null
$setupLog = Join-Path $setupLogDirectory ('setup_' + (Get-Date -Format 'yyyyMMdd_HHmmss') + '_' + [guid]::NewGuid().ToString('N').Substring(0, 8) + '.log')
$transcriptActive = $false
try {
    Start-Transcript -Path $setupLog | Out-Null
    $transcriptActive = $true
    $pythonExecutable = Join-Path $projectDirectory '.venv\Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $pythonExecutable)) {
        $pythonLauncher = Get-Command py -ErrorAction SilentlyContinue
        if ($null -ne $pythonLauncher) {
            & $pythonLauncher.Source -3.12 -m venv (Join-Path $projectDirectory '.venv')
        } else {
            $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
            if ($null -eq $pythonCommand) { throw 'Install 64-bit Python 3.12 from python.org, then run Setup_Environment.cmd again.' }
            & $pythonCommand.Source -m venv (Join-Path $projectDirectory '.venv')
        }
        if ($LASTEXITCODE -ne 0) { throw 'Could not create the Python environment. Install 64-bit Python 3.12.' }
    }
    & $pythonExecutable -B -X utf8 -c "import sys,struct; print(sys.version); assert sys.version_info[:2] == (3,12) and struct.calcsize('P') == 8, 'Use 64-bit Python 3.12.'"
    if ($LASTEXITCODE -ne 0) { throw 'The environment must use 64-bit Python 3.12. Use a fresh extracted project folder.' }
    & $pythonExecutable -B -X utf8 -m pip install -r (Join-Path $projectDirectory 'requirements-tracker.txt')
    if ($LASTEXITCODE -ne 0) { throw 'Package installation failed. Internet access is required during setup. Keep the exact error.' }
    & $pythonExecutable -B -X utf8 -m pip check
    if ($LASTEXITCODE -ne 0) { throw 'Dependency validation failed.' }
    & $pythonExecutable -B -X utf8 -m experiment_tools --help
    if ($LASTEXITCODE -ne 0) { throw 'The automatic experiment runner could not load.' }
    Write-Host ''
    Write-Host 'Setup completed. Close the tracker in Excel, then run Run_Experiments.cmd.'
    Write-Host "Setup log: $setupLog"
} catch {
    Write-Host "Setup failed: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "Setup log: $setupLog"
    exit 1
} finally {
    if ($transcriptActive) { Stop-Transcript | Out-Null }
}
