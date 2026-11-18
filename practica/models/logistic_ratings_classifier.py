"""Modelo público de la tarea; hereda el contrato documentado de BaseModel."""
# .core: comparte BaseModel y auxiliares de preparación, algoritmos y persistencia.
# Se importa: BaseModel.
from .core import BaseModel

class LogisticRatingsClassifier(BaseModel):
    """Estima si las valoraciones están desactivadas y conserva un umbral OOF explícito."""
    # task: identificador de la tarea; determina atributos, métricas y compatibilidad de pesos.
    task = 'ratings_disabled'
