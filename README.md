# Detector local de audios REAL / FAKE

Aplicación Streamlit adaptada al notebook del proyecto:

- Wav2Vec2: `facebook/wav2vec2-base`.
- Audio mono a 16 kHz.
- Ventanas de 2 segundos con desplazamiento de 1 segundo.
- Mean pooling del último estado oculto.
- Embeddings de 768 dimensiones.
- Clasificador final: Regresión Logística de Spark ML (`lr_model`).
- Etiquetas: REAL = 0 y FAKE = 1.

## Importante antes de iniciar

El notebook contiene el modelo en memoria como `lr_model`, pero no guarda sus coeficientes en los outputs. Por eso este proyecto **no incluye un modelo inventado**.

1. En Databricks, ejecuta el notebook hasta tener `lr_model` y `df_test`.
2. Ejecuta la celda `CELDA_EXPORTAR_MODELO_DATABRICKS.py`.
3. Descarga:

   `/Volumes/workspace/deepfake_bd/data/deployment/modelo_real_fake.json`

4. Colócalo en:

   `models/modelo_real_fake.json`

La celda también guarda un respaldo nativo de Spark y compara 100 predicciones para comprobar que el JSON reproduce el modelo.

## Estructura

```text
proyecto_audio_real_fake/
├── app.py
├── config.py
├── requirements.txt
├── iniciar_app.bat
├── iniciar_app_venv.bat
├── instalar_y_ejecutar.bat
├── CELDA_EXPORTAR_MODELO_DATABRICKS.py
├── exportar_modelo_databricks.py
├── models/
│   ├── LEEME_MODELO.txt
│   └── modelo_real_fake.json        # se obtiene desde Databricks
├── utils/
│   ├── __init__.py
│   ├── audio_processing.py
│   ├── embeddings.py
│   └── classifier.py
└── .streamlit/
    └── config.toml
```

## Opción recomendada: doble clic con entorno virtual

1. Instala Python 3.12 para Windows y marca **Add Python to PATH** durante la instalación.
2. Coloca el JSON del modelo en `models`.
3. Haz doble clic en `instalar_y_ejecutar.bat`.
4. La primera vez se crea `venv`, se instalan dependencias y se descarga Wav2Vec2.
5. El navegador abrirá la aplicación, normalmente en `http://localhost:8501`.
6. En ejecuciones posteriores puedes usar `iniciar_app_venv.bat`.

## Opción simple, sin entorno virtual

Abre una terminal una sola vez en la carpeta y ejecuta:

```bat
pip install -r requirements.txt
streamlit run app.py
```

Luego puedes iniciar con doble clic en `iniciar_app.bat`.

## Crear un archivo .bat desde el Bloc de notas

1. Abre **Bloc de notas**.
2. Pega el contenido del `.bat`.
3. Selecciona **Archivo > Guardar como**.
4. En **Tipo**, elige **Todos los archivos**.
5. Escribe el nombre completo, por ejemplo: `iniciar_app.bat`.
6. Selecciona codificación UTF-8 y guarda en la misma carpeta que `app.py`.
7. Confirma en el Explorador que no se llame `iniciar_app.bat.txt`.
8. Haz doble clic para ejecutarlo.

Para ver extensiones en Windows: Explorador de archivos > Ver > Mostrar > Extensiones de nombre de archivo.

## Flujo completo

```text
audio
  → lectura y validación
  → conversión a mono
  → remuestreo a 16 kHz
  → ventanas de 2 s con salto de 1 s
  → Wav2Vec2 base
  → mean pooling
  → uno o varios embeddings de 768 dimensiones
  → Regresión Logística exportada desde Spark
  → probabilidad REAL / FAKE por ventana
  → promedio de probabilidades para el resultado del audio
```

La agregación por promedio es una regla de despliegue necesaria cuando un audio genera varias ventanas. El entrenamiento y la evaluación originales se hicieron por fila/segmento; por eso esta regla debe declararse como parte del prototipo y validarse posteriormente a nivel de archivo.

## Errores frecuentes

### Falta `models/modelo_real_fake.json`
Ejecuta la celda de exportación en Databricks, descarga el JSON y cópialo a `models`.

### `streamlit` no se reconoce
Usa `instalar_y_ejecutar.bat` o activa el entorno y ejecuta:

```bat
python -m streamlit run app.py
```

### El audio no puede leerse
Convierte el archivo a WAV PCM. La grabación del navegador ya llega como WAV.

### El audio es demasiado corto
Debe durar al menos 2 segundos, igual que la lógica de entrenamiento.

### Wav2Vec2 tarda en cargar
La primera ejecución descarga el modelo y lo guarda en caché. Después se reutiliza.

### Error instalando Torch o Torchaudio
Usa Python 3.12 de 64 bits y vuelve a crear `venv`. Torch y Torchaudio deben resolverse como versiones compatibles.

### La app abre pero no predice
Revisa que el modelo JSON declare 768 coeficientes y que la validación de exportación en Databricks haya terminado sin errores.

## Alcance

Es un prototipo académico local, no un sistema forense ni un servicio productivo. El score depende del dominio de entrenamiento y de la regla de agregación por ventanas.
