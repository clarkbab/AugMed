# publish.ps1 — Build and publish augmed to PyPI
# Usage: .\publish.ps1 [-Prod]
#   -Prod : upload to real PyPI (default is TestPyPI)

param(
    [switch]$Prod
)

$ErrorActionPreference = "Stop"

# 1. Clean previous builds
Write-Host "Cleaning old builds..." -ForegroundColor Cyan
if (Test-Path dist)  { Remove-Item dist  -Recurse -Force }
if (Test-Path build) { Remove-Item build -Recurse -Force }

# 2. Install / upgrade build tools
Write-Host "Installing build tools..." -ForegroundColor Cyan
pip install --upgrade build twine

# 3. Build sdist + wheel
Write-Host "Building package..." -ForegroundColor Cyan
python -m build
if ($LASTEXITCODE -ne 0) { throw "Build failed." }

# 4. Upload
if ($Prod) {
    Write-Host "Uploading to PyPI..." -ForegroundColor Green
    twine upload dist/*
} else {
    Write-Host "Uploading to TestPyPI..." -ForegroundColor Yellow
    twine upload --repository testpypi dist/*
}

Write-Host "Done!" -ForegroundColor Green
