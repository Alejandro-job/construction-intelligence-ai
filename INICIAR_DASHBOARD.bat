@echo off
setlocal
cd /d "%~dp0"
echo ======================================================
echo Construction Intelligence AI - Dashboard PRO
 echo ======================================================
if not exist ".venv\Scripts\python.exe" (
    echo Buscando una version compatible de Python...
    py -3.11 -m venv .venv 2>nul
    if not exist ".venv\Scripts\python.exe" py -3.10 -m venv .venv 2>nul
    if not exist ".venv\Scripts\python.exe" python -m venv .venv
    if not exist ".venv\Scripts\python.exe" (
        echo ERROR: Python 3.10 o 3.11 no disponible. Usa VS Code y comprueba python --version.
        pause
        exit /b 1
    )
)
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto err
".venv\Scripts\python.exe" -m streamlit run app.py
if errorlevel 1 goto err
exit /b 0
:err
echo.
echo Error de instalacion o ejecucion. Comparte la salida de la consola.
pause
exit /b 1
