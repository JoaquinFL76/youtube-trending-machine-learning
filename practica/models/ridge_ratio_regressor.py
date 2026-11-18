"""Modelo público de la tarea; hereda el contrato documentado de BaseModel."""
# .core: comparte BaseModel y auxiliares de preparación, algoritmos y persistencia.
# Se importa: BaseModel.
from .core import BaseModel

class RidgeRatioRegressor(BaseModel):
    """Estima likes/dislikes sin recibir ninguno de los dos contadores que definen el objetivo."""
    # task: identificador de la tarea; determina atributos, métricas y compatibilidad de pesos.
    task = 'ratio'
