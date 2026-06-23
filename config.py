from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "modelo_real_fake.json"

# Debe coincidir con el notebook de entrenamiento.
WAV2VEC_MODEL_ID = "facebook/wav2vec2-base"
TARGET_SAMPLE_RATE = 16_000
WINDOW_SECONDS = 2.0
HOP_SECONDS = 1.0
EMBEDDING_DIM = 768

# Límite práctico para que la demo local no tarde demasiado.
MAX_AUDIO_SECONDS = 60.0

# Regla de agregación para audios que generan más de una ventana.
# Cada ventana se clasifica por separado y luego se promedian las probabilidades FAKE.
AUDIO_AGGREGATION = "mean_probability"
