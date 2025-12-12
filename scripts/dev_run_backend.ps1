# dev_run_backend.ps1

# Stop on errors
$ErrorActionPreference = "Stop"

# Get script directory and project root
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path $ScriptDir -Parent

Write-Host "Project root: $ProjectRoot"

# Go to backend directory
Set-Location "$ProjectRoot\backend"

# Ensure backend package is importable (PYTHONPATH)
if ($env:PYTHONPATH) {
    $env:PYTHONPATH = "$ProjectRoot\backend;$env:PYTHONPATH"
}
else {
    $env:PYTHONPATH = "$ProjectRoot\backend"
}
Write-Host "PYTHONPATH = $env:PYTHONPATH"

# Load .env into environment variables if present
$envFile = Join-Path $ProjectRoot ".env"
if (Test-Path $envFile) {
    Write-Host "Loading environment variables from $envFile ..."
    Get-Content $envFile |
    Where-Object { $_ -and -not $_.StartsWith("#") } |
    ForEach-Object {
        # Split on first '=' only
        $parts = $_ -split "=", 2
        if ($parts.Count -eq 2) {
            $name = $parts[0].Trim()
            $value = $parts[1].Trim()
            if ($name) {
                # Set dynamic env var: Env:NAME = VALUE
                Set-Item -Path "Env:$name" -Value $value
                # Uncomment to debug:
                # Write-Host "Set $name=$value"
            }
        }
    }
}
else {
    Write-Host "No .env file found at $envFile. Relying on existing environment variables."
}

Write-Host "Running backend (FastAPI) on http://localhost:8000 ..."
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
