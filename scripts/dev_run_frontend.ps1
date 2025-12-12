# dev_run_frontend.ps1

# Stop on errors
$ErrorActionPreference = "Stop"

# Get script directory and project root
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path $ScriptDir -Parent

Write-Host "Project root: $ProjectRoot"

# GO TO frontend directory
Set-Location "$ProjectRoot\frontend"

# Load .env into environment variables if present (for BACKEND_URL etc.)
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
                # This is the correct way to set a dynamic env var name
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

Write-Host "Running Streamlit frontend on http://localhost:8501 ..."
streamlit run streamlit_app.py --server.port 8501

