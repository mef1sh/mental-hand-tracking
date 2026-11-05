@echo off
rem ===== M.E.N.T.A.L. - double click to start =====
cd /d "%~dp0"
title M.E.N.T.A.L.

rem find Python: first "python", then "py"
set PY=python
%PY% --version >nul 2>nul
if errorlevel 1 set PY=py
%PY% --version >nul 2>nul
if errorlevel 1 (
    echo Python was not found. Install Python from python.org and tick "Add Python to PATH".
    pause
    exit /b 1
)

rem install the libraries only if they are missing
%PY% -c "import cv2, mediapipe" >nul 2>nul
if errorlevel 1 (
    echo Installing the libraries ^(one time only, needs internet^)...
    %PY% -m pip install -r requirements.txt
)

echo Starting M.E.N.T.A.L. ... ^(close the camera window or press q to exit^)
%PY% "M.E.N.T.A.L..py" %*

if errorlevel 1 (
    echo.
    echo The program stopped with an error. Read the text above.
    pause
)
