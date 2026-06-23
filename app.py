from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from config import (
    EMBEDDING_DIM,
    HOP_SECONDS,
    MAX_AUDIO_SECONDS,
    MODEL_PATH,
    TARGET_SAMPLE_RATE,
    WAV2VEC_MODEL_ID,
    WINDOW_SECONDS,
)
from utils.audio_processing import AudioProcessingError, load_audio_bytes, split_into_windows
from utils.classifier import CompactSparkLogisticRegression, ModelLoadError
from utils.embeddings import EmbeddingError, generate_embeddings, load_wav2vec


st.set_page_config(
    page_title="Detector de Audio Real o Fake",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root { --primary:#5b5bd6; --secondary:#8b5cf6; --surface:#ffffff; }
    .stApp { background: linear-gradient(180deg, #f7f8ff 0%, #ffffff 42%); }
    .block-container { max-width: 1120px; padding-top: 2rem; padding-bottom: 3rem; }
    .hero {
        padding: 2rem 2.2rem; border-radius: 24px;
        background: linear-gradient(135deg, #25265e 0%, #5b5bd6 55%, #8b5cf6 100%);
        color: white; box-shadow: 0 18px 50px rgba(53, 45, 150, .20);
        margin-bottom: 1.4rem;
    }
    .hero h1 { margin: 0 0 .45rem 0; font-size: 2.35rem; line-height: 1.1; }
    .hero p { margin: 0; max-width: 820px; color: rgba(255,255,255,.88); font-size: 1.02rem; }
    .info-card {
        background: rgba(255,255,255,.95); border: 1px solid #e7e9f7;
        border-radius: 18px; padding: 1.1rem 1.2rem; min-height: 118px;
        box-shadow: 0 8px 24px rgba(31,35,75,.06);
    }
    .info-card .kicker { color:#6b6f85; font-size:.78rem; text-transform:uppercase; letter-spacing:.08em; }
    .info-card .value { font-size:1.35rem; font-weight:750; color:#22243d; margin-top:.25rem; }
    .result-real, .result-fake {
        border-radius: 22px; padding: 1.6rem; text-align:center; margin-top:1rem;
        box-shadow: 0 12px 32px rgba(31,35,75,.10);
    }
    .result-real { background: linear-gradient(135deg,#e7fbf1,#f5fffa); border:1px solid #9de4bd; }
    .result-fake { background: linear-gradient(135deg,#fff0f1,#fff8f8); border:1px solid #f4a7ad; }
    .result-label { font-size:2.45rem; font-weight:850; letter-spacing:.05em; margin:.15rem 0; }
    .result-real .result-label { color:#087a45; }
    .result-fake .result-label { color:#b4232e; }
    .result-sub { color:#51556b; font-size:.98rem; }
    div[data-testid="stFileUploader"], div[data-testid="stAudioInput"] {
        background:#fff; border:1px solid #e4e6f2; border-radius:16px; padding:.5rem;
    }
    .stButton > button {
        border-radius: 14px; min-height: 3rem; font-weight: 750;
        background: linear-gradient(135deg,#5254cc,#7c4ee8); color:white; border:0;
        box-shadow: 0 8px 20px rgba(91,91,214,.24);
    }
    .stButton > button:hover { color:white; border:0; transform: translateY(-1px); }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Cargando Wav2Vec2 por primera vez…")
def get_wav2vec_resources():
    return load_wav2vec(WAV2VEC_MODEL_ID)


@st.cache_resource(show_spinner=False)
def get_classifier(model_path: str):
    return CompactSparkLogisticRegression.load(model_path)


st.markdown(
    """
    <div class="hero">
      <h1>Detector de audio REAL / FAKE</h1>
      <p>Prototipo local que preprocesa el audio, genera embeddings de 768 dimensiones con Wav2Vec2 y aplica la Regresión Logística entrenada en Spark ML.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown('<div class="info-card"><div class="kicker">Representación</div><div class="value">Wav2Vec2 · 768D</div><div>Mean pooling del último estado oculto.</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="info-card"><div class="kicker">Audio</div><div class="value">Mono · 16 kHz</div><div>Ventanas de 2 s con salto de 1 s.</div></div>', unsafe_allow_html=True)
with col3:
    st.markdown('<div class="info-card"><div class="kicker">Clasificador</div><div class="value">Spark ML · LR</div><div>REAL = 0 y FAKE = 1.</div></div>', unsafe_allow_html=True)

st.write("")
left, right = st.columns([1.45, 1], gap="large")

with left:
    st.subheader("1. Selecciona el audio")
    source = st.radio(
        "Fuente del audio",
        ["Subir un archivo", "Grabar desde el micrófono"],
        horizontal=True,
    )

    selected_audio = None
    selected_name = None
    if source == "Subir un archivo":
        selected_audio = st.file_uploader(
            "Sube un archivo de voz",
            type=["wav", "flac", "ogg", "mp3"],
            help="Para máxima compatibilidad, usa WAV. El audio debe durar al menos 2 segundos.",
        )
        if selected_audio is not None:
            selected_name = selected_audio.name
    else:
        selected_audio = st.audio_input(
            "Graba una muestra de voz",
            sample_rate=TARGET_SAMPLE_RATE,
            help="Habla al menos 2 segundos y luego detén la grabación.",
        )
        if selected_audio is not None:
            selected_name = "grabacion_microfono.wav"

    if selected_audio is not None:
        audio_bytes = selected_audio.getvalue()
        st.caption(f"Audio seleccionado: **{selected_name}**")
        st.audio(audio_bytes)
    else:
        audio_bytes = None
        st.info("Sube o graba un audio para habilitar la predicción.")

with right:
    st.subheader("2. Ejecuta la clasificación")
    st.markdown(
        "El flujo aplicado es: **audio → mono/16 kHz → ventanas de 2 s → Wav2Vec2 → embedding 768D → Regresión Logística → resultado**."
    )

    model_ready = MODEL_PATH.exists()
    if model_ready:
        st.success("Modelo de clasificación encontrado.")
    else:
        st.error("Falta `models/modelo_real_fake.json`.")
        st.caption("Ejecuta la celda de exportación incluida y copia el JSON a la carpeta `models`.")

    predict_clicked = st.button(
        "Predecir",
        type="primary",
        use_container_width=True,
        disabled=audio_bytes is None or not model_ready,
    )

if predict_clicked:
    try:
        with st.status("Procesando el audio…", expanded=True) as status:
            st.write("Leyendo, convirtiendo a mono y remuestreando a 16 kHz…")
            prepared = load_audio_bytes(audio_bytes, TARGET_SAMPLE_RATE)

            if prepared.duration_seconds > MAX_AUDIO_SECONDS:
                raise AudioProcessingError(
                    f"Para esta demo el audio no debe superar {MAX_AUDIO_SECONDS:.0f} segundos. "
                    "Recórtalo y vuelve a intentarlo."
                )

            st.write("Creando ventanas de 2 segundos con desplazamiento de 1 segundo…")
            windows = split_into_windows(
                prepared.waveform,
                prepared.sample_rate,
                WINDOW_SECONDS,
                HOP_SECONDS,
            )

            st.write(f"Generando {len(windows)} embedding(s) de 768 dimensiones con Wav2Vec2…")
            processor, wav2vec_model, device = get_wav2vec_resources()
            embeddings = generate_embeddings(
                windows=windows,
                sample_rate=prepared.sample_rate,
                processor=processor,
                model=wav2vec_model,
                device=device,
                expected_dim=EMBEDDING_DIM,
            )

            st.write("Aplicando el clasificador exportado desde Spark ML…")
            classifier = get_classifier(str(MODEL_PATH))
            if classifier.n_features != EMBEDDING_DIM:
                raise ModelLoadError(
                    f"El modelo espera {classifier.n_features} variables y la app genera {EMBEDDING_DIM}."
                )
            result = classifier.predict_audio(embeddings)
            status.update(label="Predicción completada", state="complete", expanded=False)

        css_class = "result-fake" if result.label.upper() == "FAKE" else "result-real"
        st.markdown(
            f"""
            <div class="{css_class}">
              <div class="result-sub">Resultado del sistema</div>
              <div class="result-label">{result.label.upper()}</div>
              <div class="result-sub">Confianza estimada: <strong>{result.confidence:.1%}</strong></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        metric1, metric2, metric3 = st.columns(3)
        metric1.metric("Probabilidad FAKE", f"{result.fake_probability:.2%}")
        metric2.metric("Probabilidad REAL", f"{result.real_probability:.2%}")
        metric3.metric("Ventanas analizadas", len(windows))

        with st.expander("Ver detalle técnico"):
            st.write(
                {
                    "duracion_segundos": round(prepared.duration_seconds, 3),
                    "sample_rate": prepared.sample_rate,
                    "numero_segmentos": len(windows),
                    "embedding_shape": list(embeddings.shape),
                    "dispositivo_wav2vec": str(device),
                    "umbral_fake": classifier.threshold,
                    "regla_audio": "promedio de probabilidades FAKE por ventana",
                }
            )
            detail = pd.DataFrame(
                {
                    "segmento": range(1, len(result.segment_fake_probabilities) + 1),
                    "probabilidad_FAKE": result.segment_fake_probabilities,
                    "probabilidad_REAL": 1.0 - result.segment_fake_probabilities,
                }
            )
            st.dataframe(detail.style.format({"probabilidad_FAKE": "{:.4f}", "probabilidad_REAL": "{:.4f}"}), use_container_width=True)

        st.warning(
            "Esta salida corresponde a un prototipo académico. La probabilidad no debe interpretarse como garantía forense ni sustituye una verificación especializada."
        )

    except (AudioProcessingError, EmbeddingError, ModelLoadError, ValueError) as exc:
        st.error(str(exc))
    except Exception as exc:
        st.error("Ocurrió un error inesperado durante la predicción.")
        with st.expander("Detalle técnico del error"):
            st.exception(exc)

with st.sidebar:
    st.header("Acerca del prototipo")
    st.markdown(
        """
        **Modelo desplegado:** Regresión Logística de Spark ML  
        **Etiquetas:** REAL = 0, FAKE = 1  
        **Embedding:** `facebook/wav2vec2-base`  
        **Dimensión:** 768  
        **Ventana:** 2 segundos  
        **Salto:** 1 segundo
        """
    )
    st.divider()
    st.caption("La primera carga de Wav2Vec2 puede tardar porque el modelo debe descargarse y almacenarse en caché local.")
