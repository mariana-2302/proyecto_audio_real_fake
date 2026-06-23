from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO

import numpy as np
import soundfile as sf
import torch
import torchaudio


class AudioProcessingError(ValueError):
    """Error entendible para el usuario durante la lectura o preparación del audio."""


@dataclass(frozen=True)
class PreparedAudio:
    waveform: torch.Tensor
    sample_rate: int
    duration_seconds: float


def load_audio_bytes(audio_bytes: bytes, target_sample_rate: int) -> PreparedAudio:
    """Lee bytes de audio, convierte a mono y remuestrea al sample rate objetivo.

    La lógica reproduce el notebook: lectura float32, promedio de canales y
    remuestreo a 16 kHz cuando es necesario. No normaliza ni rellena manualmente.
    """
    if not audio_bytes:
        raise AudioProcessingError("El archivo de audio está vacío.")

    try:
        data, sample_rate = sf.read(
            BytesIO(audio_bytes),
            dtype="float32",
            always_2d=True,
        )
    except Exception as exc:
        raise AudioProcessingError(
            "No se pudo leer el audio. Prueba con un archivo WAV, FLAC, OGG o MP3 válido."
        ) from exc

    if data.size == 0 or data.shape[0] == 0:
        raise AudioProcessingError("El audio no contiene muestras.")
    if sample_rate <= 0:
        raise AudioProcessingError("La frecuencia de muestreo del audio no es válida.")
    if not np.isfinite(data).all():
        raise AudioProcessingError("El audio contiene valores inválidos.")

    # soundfile entrega (muestras, canales). El notebook promedia los canales.
    mono = data.mean(axis=1, dtype=np.float32)
    waveform = torch.from_numpy(mono.copy()).to(torch.float32)

    if sample_rate != target_sample_rate:
        waveform = torchaudio.functional.resample(
            waveform,
            orig_freq=sample_rate,
            new_freq=target_sample_rate,
        )
        sample_rate = target_sample_rate

    if waveform.numel() == 0:
        raise AudioProcessingError("El audio quedó vacío después del preprocesamiento.")

    duration = waveform.numel() / float(sample_rate)
    return PreparedAudio(
        waveform=waveform.contiguous(),
        sample_rate=sample_rate,
        duration_seconds=duration,
    )


def split_into_windows(
    waveform: torch.Tensor,
    sample_rate: int,
    window_seconds: float,
    hop_seconds: float,
) -> list[torch.Tensor]:
    """Divide el audio como en el entrenamiento: 2 s con salto de 1 s.

    Los audios menores a una ventana se rechazan, porque el notebook original
    también los omitía en vez de aplicar padding.
    """
    if waveform.ndim != 1:
        raise AudioProcessingError("La señal preparada debe ser mono.")

    window_size = int(round(window_seconds * sample_rate))
    hop_size = int(round(hop_seconds * sample_rate))
    if window_size <= 0 or hop_size <= 0:
        raise AudioProcessingError("La configuración de segmentación es inválida.")

    if waveform.numel() < window_size:
        raise AudioProcessingError(
            f"El audio debe durar al menos {window_seconds:.0f} segundos. "
            "Los audios más cortos no formaron parte del flujo de entrenamiento."
        )

    return [
        waveform[start : start + window_size]
        for start in range(0, waveform.numel() - window_size + 1, hop_size)
    ]
