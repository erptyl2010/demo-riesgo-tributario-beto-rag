$ErrorActionPreference = "Stop"

Set-Location -LiteralPath $PSScriptRoot

Write-Host "Iniciando TributIA en http://localhost:8501" -ForegroundColor Cyan
Write-Host "Use Ctrl+C para detener la aplicacion." -ForegroundColor DarkGray

python -m streamlit run app.py
