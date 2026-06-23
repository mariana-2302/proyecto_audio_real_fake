from __future__ import annotations

import numpy as np
import torch
from transformers import Wav2Vec2Model, Wav2Vec2Processor


class EmbeddingError(RuntimeError):
    """Error al generar o validar embeddings."""


def load_wav2vec(model_id_or_path: str) -> tuple[Wav2Vec2Processor, Wav2Vec2Model, torch.device]:
    """Carga Wav2Vec2 en GPU si está disponible; en caso contrario usa CPU."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = Wav2Vec2Processor.from_pretrained(model_id_or_path)
    model = Wav2Vec2Model.from_pretrained(model_id_or_path)
    model.to(device)
    model.eval()
    return processor, model, device


def generate_embeddings(
    windows: list[torch.Tensor],
    sample_rate: int,
    processor: Wav2Vec2Processor,
    model: Wav2Vec2Model,
    device: torch.device,
    expected_dim: int = 768,
) -> np.ndarray:
    """Genera un vector por ventana usando mean pooling del último estado oculto.

    Esta es la misma operación del notebook:
    outputs.last_hidden_state.squeeze(0).mean(dim=0)
    """
    if not windows:
        raise EmbeddingError("No se generaron ventanas de audio.")

    embeddings: list[np.ndarray] = []
    with torch.inference_mode():
        for segment in windows:
            inputs = processor(
                segment.cpu().numpy(),
                sampling_rate=sample_rate,
                return_tensors="pt",
                padding=False,
            )
            inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
            outputs = model(**inputs)
            embedding = outputs.last_hidden_state.squeeze(0).mean(dim=0)
            vector = embedding.detach().cpu().numpy().astype(np.float64, copy=False)

            if vector.shape != (expected_dim,):
                raise EmbeddingError(
                    f"Se esperaba un embedding de {expected_dim} dimensiones, "
                    f"pero se obtuvo {vector.shape}."
                )
            if not np.isfinite(vector).all():
                raise EmbeddingError("El embedding contiene valores inválidos.")
            embeddings.append(vector)

    matrix = np.vstack(embeddings)
    if matrix.shape[1] != expected_dim:
        raise EmbeddingError("La matriz final de embeddings no tiene 768 columnas.")
    return matrix
