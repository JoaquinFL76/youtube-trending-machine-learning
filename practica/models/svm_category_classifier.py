"""Modelo público de la tarea; hereda el contrato documentado de BaseModel."""
# .core: comparte BaseModel y auxiliares de preparación, algoritmos y persistencia.
# Se importa: BaseModel.
from .core import BaseModel

class SVMCategoryClassifier(BaseModel):
    """Clasifica la categoría editorial usando contenido y metadatos sin contadores."""
    # task: identificador de la tarea; determina atributos, métricas y compatibilidad de pesos.
    task = 'category'
