# EJECUTAR AL FINAL DEL NOTEBOOK, después de crear `lr_model` y `df_test`.
# Esta celda exporta la Regresión Logística seleccionada como modelo final.

import json
import math
import os
import numpy as np
from pyspark.ml.functions import vector_to_array

EXPORT_DIR = "/Volumes/workspace/deepfake_bd/data/deployment"
EXPORT_JSON = f"{EXPORT_DIR}/modelo_real_fake.json"
os.makedirs(EXPORT_DIR, exist_ok=True)

assert int(lr_model.numClasses) == 2, "El modelo debe ser binario."
assert int(lr_model.numFeatures) == 768, "El modelo debe recibir 768 variables."

coefficients = lr_model.coefficients.toArray().astype(float).tolist()
intercept = float(lr_model.intercept)
threshold = float(lr_model.getThreshold())

payload = {
    "format_version": 1,
    "model_type": "spark_ml_binary_logistic_regression",
    "source_variable": "lr_model",
    "n_features": 768,
    "feature_names": [f"emb_{i}" for i in range(768)],
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
    "training_hyperparameters": {"maxIter": 100, "regParam": 0.0},
}

with open(EXPORT_JSON, "w", encoding="utf-8") as file:
    json.dump(payload, file, ensure_ascii=False, indent=2)

# Validación: el JSON debe reproducir las probabilidades y clases de Spark.
rows = (
    lr_model.transform(df_test.limit(100))
    .select(
        vector_to_array("features").alias("features_array"),
        vector_to_array("probability")[1].alias("spark_probability_fake"),
        "prediction",
    )
    .collect()
)
X = np.vstack([np.asarray(r["features_array"], dtype=np.float64) for r in rows])
margins = X @ np.asarray(coefficients, dtype=np.float64) + intercept
p_local = np.where(
    margins >= 0,
    1.0 / (1.0 + np.exp(-margins)),
    np.exp(margins) / (1.0 + np.exp(margins)),
)
p_spark = np.asarray([r["spark_probability_fake"] for r in rows], dtype=np.float64)
y_spark = np.asarray([int(r["prediction"]) for r in rows])
y_local = (p_local >= threshold).astype(int)

max_diff = float(np.max(np.abs(p_local - p_spark)))
coincidencias = int(np.sum(y_local == y_spark))
print(f"Modelo exportado: {EXPORT_JSON}")
print(f"Diferencia máxima de probabilidad: {max_diff:.12g}")
print(f"Predicciones coincidentes: {coincidencias}/{len(rows)}")
assert coincidencias == len(rows) and max_diff <= 1e-7

# Respaldo opcional del modelo Spark completo.
lr_model.write().overwrite().save(f"{EXPORT_DIR}/modelo_real_fake_spark")
print("Respaldo Spark guardado en:", f"{EXPORT_DIR}/modelo_real_fake_spark")
