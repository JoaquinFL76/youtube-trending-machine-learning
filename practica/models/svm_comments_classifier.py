"""Modelo público de la tarea; hereda el contrato documentado de BaseModel."""
# .core: comparte BaseModel y auxiliares de preparación, algoritmos y persistencia.
# Se importa: BaseModel.
from .core import BaseModel

class SVMCommentsClassifier(BaseModel):
    """Estima si los comentarios están desactivados sin usar señales que revelen la etiqueta."""
    # task: identificador de la tarea; determina atributos, métricas y compatibilidad de pesos.
    task = 'comments_disabled'
