@echo off
cd /d "%~dp0"

echo Iniciando la aplicacion Streamlit...
streamlit run app.py

if errorlevel 1 (
    echo.
    echo La aplicacion no pudo iniciarse. Revisa el mensaje anterior.
    pause
)
