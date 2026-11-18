"""Segunda búsqueda reproducible de hiperparámetros sobre la familia ganadora."""
# json: lee y escribe datos estructurados, configuraciones y evidencias en JSON.
import json
# warnings: emite, captura o filtra avisos sin confundirlos con errores.
import warnings
# numpy: realiza cálculos numéricos y operaciones con vectores y matrices.
import numpy as np
# pandas: organiza, limpia y analiza datos tabulares mediante DataFrame y Series.
import pandas as pd
# sklearn.base: clona estimadores sin reutilizar el estado aprendido de otro ajuste.
# Se importa: clone.
from sklearn.base import clone
# sklearn.model_selection: organiza particiones, validación cruzada y búsquedas de parámetros.
# Se importa: ParameterSampler, GridSearchCV.
from sklearn.model_selection import ParameterSampler, GridSearchCV
# models.core: comparte validación, transformaciones, factorías y persistencia de modelos.
# Se importa: features.
from models.core import features
# convergence_review: revisa avisos y realiza reintentos acotados de convergencia.
# Se importa: fit_search_review.
from convergence_review import fit_search_review


def parameter_space(winner):
    """Devuelve alternativas de regularización y representación compatibles con winner."""
    available = winner.get_params()
    space = {'features__text__max_features': [8000, 16000],
             'features__text__min_df': [2, 4],
             'features__text__ngram_range': [(1, 1), (1, 2)],
             'features__text__sublinear_tf': [True, False]}
    for key in ['estimator__C', 'estimator__regressor__C']:
        if key in available:
            space[key] = [.03, .1, .3, 1., 3., 10.]
    for key in ['estimator__alpha', 'estimator__regressor__alpha']:
        if key in available:
            space[key] = [.01, .1, 1., 10., 100.]
    if 'estimator__class_weight' in available:
        space['estimator__class_weight'] = [None, 'balanced']
    for key in ['estimator__gamma', 'estimator__regressor__gamma']:
        if key in available:
            space[key] = ['scale', .01, .1]
    for prefix in ('estimator__', 'estimator__regressor__'):
        if prefix + 'learning_rate' in available and prefix + 'n_estimators' in available and prefix + 'max_depth' in available:
            space[prefix + 'max_depth'] = [3, 6]
            space[prefix + 'n_estimators'] = [80, 140]
            space[prefix + 'learning_rate'] = [0.05, 0.1]
    if 'estimator__shrinkage' in available:
        space['estimator__shrinkage'] = [None, 'auto', 0.1]
    if 'reduce__n_components' in available:
        space['reduce__n_components'] = [30, 60]
    return space


def refine(task, winner, previous, x, y, splits, cache, directory, seed, budget=12):
    """Compara una muestra fija del espacio y la receta anterior en los mismos folds.

    No consulta el holdout. Incluye el punto anterior para evitar perderlo por azar;
    guarda todas las puntuaciones, parámetros y avisos, no sólo el ganador.
    Devuelve Pipeline ajustado, parámetros y puntuación CV para la fase final.
    """
    space = parameter_space(winner)
    sampled = list(ParameterSampler(space, n_iter=budget, random_state=seed))
    # El candidato de referencia reproduce los parámetros de la primera fase.
    reference = {key: winner.get_params()[key] for key in space}
    candidates = [reference] + sampled
    grids = [{key: [value] for key, value in row.items()} for row in candidates]
    scoring = 'f1_macro' if task in ['category', 'comments_disabled', 'ratings_disabled'] else 'neg_mean_absolute_error'
    estimator = clone(winner)
    estimator.memory = cache
    search = GridSearchCV(estimator, grids, cv=splits, scoring=scoring,
                          return_train_score=True, n_jobs=1, error_score='raise')
    print('Refinamiento', task, len(grids), 'configuraciones y', len(splits), 'folds', flush=True)
    review = fit_search_review(search, features(x), y)
    results = search.cv_results_
    rows = []
    for index, params in enumerate(results['params']):
        rows.append({'candidate': index, 'reference': index == 0,
                     'mean_cv_score': float(results['mean_test_score'][index]),
                     'std_cv_score': float(results['std_test_score'][index]),
                     'mean_train_score': float(results['mean_train_score'][index]),
                     'params': params,
                     'fold_scores': [float(results[f'split{k}_test_score'][index]) for k in range(len(splits))]})
    report = {'task': task, 'family': previous['model'], 'space': space,
              'sampling': f'{budget} combinaciones aleatorias con semilla y una referencia explícita',
              'search_rows': len(x), 'folds': len(splits), 'trials': rows,
              'best_index': int(search.best_index_), 'params': dict(search.best_params_, **review['iteration_parameters']),
              'cv_score': float(search.best_score_), 'initial_cv_score': previous['cv_score'],
              'warnings': review['warnings'], 'convergence_review': review,
              'scope': 'Selección interna, no CV anidada; holdout excluido de ambas búsquedas'}
    (directory / f'{task}_refinement.json').write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding='utf-8')
    pd.DataFrame([{**row, 'params': json.dumps(row['params'], ensure_ascii=False)} for row in rows]).to_csv(directory / f'{task}_refinement.csv', index=False)
    # La gráfica muestra todas las configuraciones; el detalle de cada una queda en la tabla.
    # matplotlib.pyplot: crea, guarda y cierra figuras y ejes de los análisis.
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.errorbar(range(len(rows)), [row['mean_cv_score'] for row in rows],
                yerr=[row['std_cv_score'] for row in rows], fmt='o', capsize=3)
    ax.set(xlabel='Configuración (0 = referencia)', ylabel=scoring,
           title=f'Segunda búsqueda: {task}; mejor configuración {search.best_index_}')
    fig.tight_layout()
    fig.savefig(directory / f'{task}_refinement.png', dpi=140)
    plt.close(fig)
    params = dict(previous['params'], **search.best_params_, **review['iteration_parameters'])
    return search.best_estimator_, params, float(search.best_score_), report
