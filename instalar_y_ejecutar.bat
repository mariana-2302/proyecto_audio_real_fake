@echo off
setlocal
cd /d "%~dp0"

echo ================================================
echo  Detector de audio REAL / FAKE - Instalacion
echo ================================================

if not exist "venv\Scripts\python.exe" (
    echo Creando entorno virtual...
    py -3.12 -m venv venv 2>nul
    if errorlevel 1 (
        echo No se encontro Python 3.12 mediante el comando py.
        echo Se intentara con el Python predeterminado.
        python -m venv venv
    )
    if errorlevel 1 goto :error
)

call "venv\Scripts\activate.bat"
if errorlevel 1 goto :error

echo Actualizando pip...
python -m pip install --upgrade pip
if errorlevel 1 goto :error

echo Instalando dependencias. La primera vez puede tardar varios minutos...
python -m pip install -r requirements.txt
if errorlevel 1 goto :error

echo Iniciando Streamlit...
python -m streamlit run app.py
if errorlevel 1 goto :error
exit /b 0

:error
echo.
echo Ocurrio un error. Revisa el mensaje mostrado arriba.
pause
exit /b 1
