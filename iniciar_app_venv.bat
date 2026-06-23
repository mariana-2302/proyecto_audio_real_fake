@echo off
cd /d "%~dp0"

if not exist "venv\Scripts\activate.bat" (
    echo No existe el entorno virtual "venv".
    echo Ejecuta primero instalar_y_ejecutar.bat.
    pause
    exit /b 1
)

call "venv\Scripts\activate.bat"
python -m streamlit run app.py

if errorlevel 1 (
    echo.
    echo La aplicacion no pudo iniciarse. Revisa el mensaje anterior.
    pause
)
