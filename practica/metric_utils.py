"""F2 compartido por selección, diagnóstico e informes, sin elegir umbrales."""
# numpy: realiza cálculos numéricos y operaciones con vectores y matrices.
import numpy as np


def f2_from_precision_recall(precision, recall):
    """Calcula F2 para escalares o arrays; devuelve cero si ambos son cero.

    Mantiene la protección numérica de las búsquedas existentes y su orden.
    No estima puntuaciones ni consulta etiquetas de prueba.
    """
    precision, recall = np.asarray(precision), np.asarray(recall)
    return 5 * precision * recall / np.maximum(4 * precision + recall, 1e-15)


def f2_from_confusion(matrix):
    """Calcula F2 binario desde una matriz 2×2 ordenada por etiquetas [0, 1].

    Rechaza dimensiones, valores negativos o no finitos; si no hay positivos,
    devuelve cero. No modifica la matriz recibida.
    """
    values = np.asarray(matrix, dtype=float)
    if values.shape != (2, 2) or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('F2 requiere una matriz 2×2 finita y no negativa')
    return float(5 * values[1, 1] / max(5 * values[1, 1] + 4 * values[1, 0] + values[0, 1], 1))
