"""Celda/archivo para exportar el modelo final desde Databricks.

Uso dentro del notebook, DESPUÉS de entrenar `lr_model`:

    exec(open('/ruta/exportar_modelo_databricks.py').read())
    exportar_lr_spark_a_json(lr_model, EXPORT_JSON)

También puedes copiar directamente el bloque de uso del final a una celda.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np


def exportar_lr_spark_a_json(lr_model: Any, output_path: str) -> dict:
    """Exporta coeficientes e intercepto de Spark LR binaria a JSON."""
    num_classes = int(lr_model.numClasses)
    num_features = int(lr_model.numFeatures)
    if num_classes != 2:
        raise ValueError(f"Se esperaba clasificación binaria; numClasses={num_classes}")
    if num_features != 768:
        raise ValueError(f"Se esperaban 768 variables; numFeatures={num_features}")

    coefficients = lr_model.coefficients.toArray().astype(float).tolist()
    intercept = float(lr_model.intercept)
    threshold = float(lr_model.getThreshold())

    if len(coefficients) != 768:
        raise ValueError("El vector de coeficientes no tiene 768 posiciones.")
    if not all(math.isfinite(value) for value in coefficients + [intercept, threshold]):
        raise ValueError("El modelo contiene valores no finitos.")

    payload = {
        "format_version": 1,
        "model_type": "spark_ml_binary_logistic_regression",
        "source_variable": "lr_model",
        "n_features": num_features,
        "feature_names": [f"emb_{i}" for i in range(num_features)],
        "coefficients": coefficients,
        "intercept": intercept,
        "threshold": threshold,
        "label_mapping": {"0": "REAL", "1": "FAKE"},
        "wav2vec_model_id": "facebook/wav2vec2-base",
        "target_sample_rate": 16000,
        "window_seconds": 2.0,
        "hop_seconds": 1.0,
        "pooling": "mean_last_hidden_state",
        "audio_level_aggregation": "mean_probability",
        "training_hyperparameters": {
            "maxIter": 100,
            "regParam": 0.0,
        },
        "note": (
            "La agregación por promedio se usa solo cuando un audio entrante genera "
            "más de una ventana; el entrenamiento y la evaluación originales fueron por fila/segmento."
        ),
    }

    # Path de Volume en Databricks o path local normal.
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)

    return payload


def validar_exportacion_con_spark(lr_model: Any, df_test: Any, json_path: str, n: int = 100) -> None:
    """Comprueba que el JSON reproduzca probabilidad y clase del modelo Spark."""
    from pyspark.ml.functions import vector_to_array

    with open(json_path, "r", encoding="utf-8") as file:
        payload = json.load(file)
    coef = np.asarray(payload["coefficients"], dtype=np.float64)
    intercept = float(payload["intercept"])
    threshold = float(payload["threshold"])

    rows = (
        lr_model.transform(df_test.limit(n))
        .select(
            vector_to_array("features").alias("features_array"),
            vector_to_array("probability")[1].alias("spark_probability_fake"),
            "prediction",
        )
        .collect()
    )
    if not rows:
        raise ValueError("No hay filas para validar la exportación.")

    matrix = np.vstack([np.asarray(row["features_array"], dtype=np.float64) for row in rows])
    margins = matrix @ coef + intercept
    local_prob = np.where(
        margins >= 0,
        1.0 / (1.0 + np.exp(-margins)),
        np.exp(margins) / (1.0 + np.exp(margins)),
    )
    spark_prob = np.asarray([row["spark_probability_fake"] for row in rows], dtype=np.float64)
    spark_pred = np.asarray([int(row["prediction"]) for row in rows])
    local_pred = (local_prob >= threshold).astype(int)

    max_diff = float(np.max(np.abs(local_prob - spark_prob)))
    matches = int(np.sum(local_pred == spark_pred))
    print(f"Validación sobre {len(rows)} filas")
    print(f"Diferencia máxima de probabilidad: {max_diff:.12g}")
    print(f"Predicciones coincidentes: {matches}/{len(rows)}")
    if matches != len(rows) or max_diff > 1e-7:
        raise AssertionError("La exportación compacta no coincide con Spark.")


# ---------------------------------------------------------------------------
# BLOQUE PARA COPIAR AL FINAL DEL NOTEBOOK DE DATABRICKS
# ---------------------------------------------------------------------------
# EXPORT_DIR = "/Volumes/workspace/deepfake_bd/data/deployment"
# EXPORT_JSON = f"{EXPORT_DIR}/modelo_real_fake.json"
# payload = exportar_lr_spark_a_json(lr_model, EXPORT_JSON)
# validar_exportacion_con_spark(lr_model, df_test, EXPORT_JSON, n=100)
#
# # Respaldo opcional del modelo Spark original (directorio, no archivo .pkl):
# lr_model.write().overwrite().save(f"{EXPORT_DIR}/modelo_real_fake_spark")
#
# print(f"Modelo compacto exportado en: {EXPORT_JSON}")
# print("Descárgalo y colócalo en: proyecto_audio_real_fake/models/modelo_real_fake.json")
