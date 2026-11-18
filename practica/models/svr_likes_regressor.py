"""Modelo público de la tarea; hereda el contrato documentado de BaseModel."""
# .core: comparte BaseModel y auxiliares de preparación, algoritmos y persistencia.
# Se importa: BaseModel.
from .core import BaseModel

class SVRLikesRegressor(BaseModel):
    """Estima likes contemporáneos; usa otros contadores, sin afirmar predicción futura."""
    # task: identificador de la tarea; determina atributos, métricas y compatibilidad de pesos.
    task = 'likes'
