@echo off
REM Quick Start Script for Hand Gesture Knob Controller (Windows)

echo ================================================
echo Hand Gesture Knob Controller - Quick Start
echo ================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed
    echo Please install Python 3.8+ from python.org
    echo Make sure to check "Add Python to PATH" during installation
    pause
    exit /b 1
)

echo [OK] Python found
python --version
echo.

REM Check if we're in the right directory
if not exist requirements.txt (
    echo ERROR: requirements.txt not found
    echo Please run this script from the project root directory
    pause
    exit /b 1
)

echo Step 1: Creating virtual environment...
if exist venv (
    echo   Virtual environment already exists
) else (
    python -m venv venv
    echo   [OK] Virtual environment created
)
echo.

echo Step 2: Activating virtual environment...
call venv\Scripts\activate.bat
echo   [OK] Virtual environment activated
echo.

echo Step 3: Installing dependencies...
python -m pip install --upgrade pip >nul 2>&1
pip install -r requirements.txt
echo   [OK] Dependencies installed
echo.

echo Step 4: Creating directories...
if not exist config mkdir config
if not exist assets mkdir assets
if not exist docs mkdir docs
echo   [OK] Directories created
echo.

echo ================================================
echo Installation Complete!
echo ================================================
echo.
echo To run the application:
echo   1. Ensure your camera is connected
echo   2. Run: python src\main.py
echo.
echo To run the demo (no FL Studio needed):
echo   python demo.py
echo.
echo To build standalone executable:
echo   python setup.py build
echo.
echo Documentation:
echo   - README.md - Project overview
echo   - docs\FL_STUDIO_GUIDE.md - FL Studio setup
echo   - docs\DEVELOPER_GUIDE.md - Technical docs
echo.
echo Have fun! 🎵
echo.
pause