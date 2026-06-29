@echo off
setlocal

cd /d "%~dp0"

where pnpm >nul 2>nul
if errorlevel 1 (
  echo pnpm was not found. Please install pnpm first, then run this file again.
  echo You can install it with: corepack enable
  pause
  exit /b 1
)

if not exist "node_modules" (
  echo Installing frontend dependencies...
  pnpm install
  if errorlevel 1 (
    echo Dependency installation failed.
    pause
    exit /b 1
  )
)

echo Starting Flashcards Maker WebUI...
echo Close the Tauri window or press Ctrl+C here to stop it.
pnpm tauri dev

if errorlevel 1 (
  echo Flashcards Maker exited with an error.
  pause
  exit /b 1
)

endlocal
