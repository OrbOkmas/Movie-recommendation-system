@echo off
setlocal
title Movie mood picker
rem Double-click to start. The first run installs everything (15-25 min, needs internet);
rem later runs start in under a minute.
cd /d "%~dp0"

rem --- 1) find a suitable Python (3.11-3.13), install 3.12 if there is none ------------
set "PYEXE="
set "PYARG="
for %%V in (3.12 3.11 3.13) do (
    if not defined PYEXE (
        py -%%V -c "import sys" >nul 2>nul && (set "PYEXE=py" & set "PYARG=-%%V")
    )
)
if not defined PYEXE (
    python -c "import sys; sys.exit(0 if (3,11) <= sys.version_info[:2] <= (3,13) else 1)" >nul 2>nul && set "PYEXE=python"
)
if not defined PYEXE if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" set "PYEXE=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
if not defined PYEXE call :install_python
if not defined PYEXE (
    echo Could not install Python automatically.
    echo Please install Python 3.12 from https://www.python.org/downloads/ and run this file again.
    pause
    exit /b 1
)

rem --- 2) first run: private environment + packages -----------------------------------
rem (set outside the block below: %%VAR%% inside ( ... ) is expanded before the block runs)
set "REQ=requirements-app.txt"
if not exist "requirements-app.txt" set "REQ=requirements.txt"
if not exist ".venv\install_ok.txt" (
    echo First start: installing packages. This takes 10-20 minutes, please wait...
    if not exist ".venv\Scripts\python.exe" "%PYEXE%" %PYARG% -m venv .venv
    if not exist ".venv\Scripts\python.exe" goto :fail
    ".venv\Scripts\python.exe" -m pip install --upgrade pip
    ".venv\Scripts\python.exe" -m pip install -r "%REQ%"
    if errorlevel 1 goto :fail
    echo ok> ".venv\install_ok.txt"
)

rem --- 3) skip Streamlit's e-mail question --------------------------------------------
if not exist "%USERPROFILE%\.streamlit\credentials.toml" (
    mkdir "%USERPROFILE%\.streamlit" 2>nul
    (echo [general]& echo email = "") > "%USERPROFILE%\.streamlit\credentials.toml"
)

rem --- 4) start the app; the browser opens by itself ----------------------------------
echo Starting the app. The browser opens automatically; close this window to stop the app.
".venv\Scripts\python.exe" -m streamlit run app.py --server.headless false --browser.gatherUsageStats false
pause
exit /b 0


:install_python
echo No suitable Python found - installing Python 3.12 for this user (a few minutes)...
where winget >nul 2>nul && winget install -e --id Python.Python.3.12 --scope user --silent --accept-package-agreements --accept-source-agreements
if not exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" (
    echo Downloading the installer from python.org...
    powershell -NoProfile -ExecutionPolicy Bypass -Command "Invoke-WebRequest 'https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe' -OutFile \"$env:TEMP\py312_setup.exe\""
    if exist "%TEMP%\py312_setup.exe" "%TEMP%\py312_setup.exe" /quiet InstallAllUsers=0 PrependPath=1 Include_launcher=1
)
if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" set "PYEXE=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
goto :eof


:fail
echo.
echo Installation failed - see the messages above. Delete the .venv folder and run this file again.
pause
exit /b 1
