"""Reintentos acotados de convergencia sin cambiar folds ni consultar la prueba."""
# warnings: emite, captura o filtra avisos sin confundirlos con errores.
import warnings
# sklearn.exceptions: identifica avisos específicos, como la falta de convergencia.
# Se importa: ConvergenceWarning.
from sklearn.exceptions import ConvergenceWarning


def fit_search_review(search, x, y):
    """Ajusta una búsqueda y, ante falta de convergencia, duplica una vez las iteraciones.

    Conserva todos los avisos y puntuaciones de ambos intentos. Los hiperparámetros,
    datos y folds son los mismos; el presupuesto final queda explícito en la receta.
    Un aviso restante se declara, nunca se interpreta como convergencia conseguida.
    """
    attempts = []
    for attempt in range(2):
        # iteration_parameters: límites efectivos de cada estimador del Pipeline.
        iteration_parameters = {key: value for key, value in search.estimator.get_params().items()
                                if key.endswith('__max_iter')}
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            search.fit(x, y)
        convergence = sorted({str(w.message) for w in caught if issubclass(w.category, ConvergenceWarning)})
        attempts.append({'attempt': attempt + 1, 'iteration_parameters': iteration_parameters,
                         'best_cv_score': float(search.best_score_),
                         'warnings': sorted({str(w.message) for w in caught}),
                         'convergence_warnings': convergence})
        if not convergence or attempt == 1 or not iteration_parameters:
            break
        # El reintento no cambia candidatos ni usa el resultado de prueba.
        increased = {key: min(int(value) * 2, 60000) for key, value in iteration_parameters.items()
                     if isinstance(value, int) and 0 < value < 60000}
        if not increased:
            break
        search.estimator.set_params(**increased)
    final = attempts[-1]
    return {'attempts': attempts, 'iteration_parameters': final['iteration_parameters'],
            'warnings': final['warnings'], 'remaining_convergence_warnings': final['convergence_warnings'],
            'status': 'warnings_remaining' if final['convergence_warnings'] else 'no_convergence_warning',
            'scope': 'CV agrupada idéntica; como máximo un reintento; ausencia de aviso no prueba óptimo global'}
