@echo off
REM =============================================================
REM  Ennoia Environment Setup for Streamlit (with Conda Integration)
REM =============================================================

REM === USER CONFIGURATION ===
set "ENNOIA_ENV_NAME=ennoia"
set "ENNOIA_SSH_HOST=98.84.113.163"
set "ENNOIA_SSH_USER=ec2-user"
set "ENNOIA_SSH_KEY=C:\Users\rices\ennoiaCAT\AWS-Hackathon.pem"
set "ENNOIA_REMOTE_PY=python3"
set "ENNOIA_REMOTE_SCRIPT=/home/ec2-user/ennoiaCAT/operator_table_service.py"
set "ENNOIA_REMOTE_OUT=/tmp/ennoia_tables"
set "ENNOIA_LOCAL_OUT=C:\Users\rices\ennoiaCAT"
set "STREAMLIT_FILE=ennoia_agentic_app.py"

echo.
echo [1/6] Detecting Conda environment path for "%ENNOIA_ENV_NAME%"...

FOR /F "tokens=*" %%i IN ('conda env list ^| findstr "%ENNOIA_ENV_NAME%"') DO set "ENNOIA_ENV_PATH=%%i"
FOR /F "tokens=1" %%a IN ("%ENNOIA_ENV_PATH%") DO set "ENNOIA_ENV_PATH=%%a"
FOR /F "tokens=2 delims= " %%b IN ("%ENNOIA_ENV_PATH%") DO set "ENNOIA_ENV_PATH=%%b"

IF "%ENNOIA_ENV_PATH%"=="" (
    echo ❌ Could not detect Conda environment "%ENNOIA_ENV_NAME%".
    echo Please activate it once with: conda activate %ENNOIA_ENV_NAME%
    pause
    exit /b
)

echo ✅ Conda environment found at: %ENNOIA_ENV_PATH%

echo.
echo [2/6] Creating local output directory: %ENNOIA_LOCAL_OUT%
if not exist "%ENNOIA_LOCAL_OUT%" mkdir "%ENNOIA_LOCAL_OUT%"

echo.
echo [3/6] Writing Conda activate/deactivate scripts...

set "ACTIVATE_DIR=%ENNOIA_ENV_PATH%\etc\conda\activate.d"
set "DEACTIVATE_DIR=%ENNOIA_ENV_PATH%\etc\conda\deactivate.d"
mkdir "%ACTIVATE_DIR%" 2>nul
mkdir "%DEACTIVATE_DIR%" 2>nul

REM --- Activate script ---
(
echo @echo off
echo REM === Ennoia environment variables ===
echo set ENNOIA_SSH_HOST=%ENNOIA_SSH_HOST%
echo set ENNOIA_SSH_USER=%ENNOIA_SSH_USER%
echo set ENNOIA_SSH_KEY=%ENNOIA_SSH_KEY%
echo set ENNOIA_REMOTE_PY=%ENNOIA_REMOTE_PY%
echo set ENNOIA_REMOTE_SCRIPT=%ENNOIA_REMOTE_SCRIPT%
echo set ENNOIA_REMOTE_OUT=%ENNOIA_REMOTE_OUT%
echo set ENNOIA_LOCAL_OUT=%ENNOIA_LOCAL_OUT%
) > "%ACTIVATE_DIR%\ennoia_env.bat"

REM --- Deactivate script ---
(
echo @echo off
echo REM === Clear Ennoia environment variables ===
echo set ENNOIA_SSH_HOST=
echo set ENNOIA_SSH_USER=
echo set ENNOIA_SSH_KEY=
echo set ENNOIA_REMOTE_PY=
echo set ENNOIA_REMOTE_SCRIPT=
echo set ENNOIA_REMOTE_OUT=
echo set ENNOIA_LOCAL_OUT=
) > "%DEACTIVATE_DIR%\ennoia_env.bat"

echo ✅ Environment variable scripts written to:
echo     %ACTIVATE_DIR%
echo     %DEACTIVATE_DIR%

echo.
echo [4/6] Setting variables for this current session...
set ENNOIA_SSH_HOST=%ENNOIA_SSH_HOST%
set ENNOIA_SSH_USER=%ENNOIA_SSH_USER%
set ENNOIA_SSH_KEY=%ENNOIA_SSH_KEY%
set ENNOIA_REMOTE_PY=%ENNOIA_REMOTE_PY%
set ENNOIA_REMOTE_SCRIPT=%ENNOIA_REMOTE_SCRIPT%
set ENNOIA_REMOTE_OUT=%ENNOIA_REMOTE_OUT%
set ENNOIA_LOCAL_OUT=%ENNOIA_LOCAL_OUT%

echo.
echo [5/6] Optional SSH test (press Ctrl+C to skip)
ssh -i "%ENNOIA_SSH_KEY%" %ENNOIA_SSH_USER%@%ENNOIA_SSH_HOST% "echo ✅ SSH OK && %ENNOIA_REMOTE_PY% --version"

echo.
echo [6/6] Done! ✅
echo To apply automatically:
echo     conda deactivate
echo     conda activate %ENNOIA_ENV_NAME%
echo.
echo To launch Streamlit, run:
echo     streamlit run %STREAMLIT_FILE%
echo.
pause
