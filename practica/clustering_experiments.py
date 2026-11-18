"""Comparación inductiva de centroides, Mean Shift y mezclas gaussianas."""
# warnings: emite, captura o filtra avisos sin confundirlos con errores.
import warnings
# numpy: realiza cálculos numéricos y operaciones con vectores y matrices.
import numpy as np
# sklearn.base: clona estimadores sin reutilizar el estado aprendido de otro ajuste.
# Se importa: clone.
from sklearn.base import clone
# sklearn.cluster: aporta algoritmos de agrupación de observaciones sin etiquetas.
# Se importa: estimate_bandwidth, AgglomerativeClustering, DBSCAN.
from sklearn.cluster import estimate_bandwidth, AgglomerativeClustering, DBSCAN
# sklearn.decomposition: reduce dimensiones mediante técnicas como PCA y SVD.
# Se importa: TruncatedSVD.
from sklearn.decomposition import TruncatedSVD
# sklearn.metrics: calcula métricas para evaluar predicciones, grupos y similitudes.
# Se importa: silhouette_score, adjusted_rand_score, normalized_mutual_info_score.
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
# sklearn.exceptions: identifica avisos específicos, como la falta de convergencia.
# Se importa: ConvergenceWarning.
from sklearn.exceptions import ConvergenceWarning
# models.core: comparte validación, transformaciones, factorías y persistencia de modelos.
# Se importa: cluster_estimator.
from models.core import cluster_estimator


# Límites añadidos tras observar una solución muy desequilibrada; no son del profesor.
MIN_CLUSTER_FRACTION = 0.05
MIN_STABILITY_ARI = 0.80
SELECTION_POLICY = ('Regla añadida después de observar una segmentación previa del 99,5 %: '
    'cada grupo debe contener al menos el 5 % de los vídeos y tres réplicas válidas '
    'de estabilidad con ARI medio >= 0,80; se maximiza silhouette entre quienes cumplen. '
    'No se usan categorías editoriales ni etiquetas de prueba. Son límites metodológicos '
    'de esta revisión, no requisitos del profesor ni criterios fijados antes del primer estudio.')


def selection_details(labels, eligible, stability):
    """Cuenta grupos sobre todas las filas y aplica tamaño y estabilidad sin categorías.

    Conserva eligible como validez técnica; selectable añade los límites explícitos.
    La estabilidad está condicionada al SVD fijo, no valida utilidad comercial.
    """
    ids, sizes = np.unique(labels, return_counts=True)
    fraction = float(sizes.min() / sizes.sum()) if len(sizes) else 0.0
    valid_stability = len(stability) == 3 and np.isfinite(stability).all()
    selectable = bool(eligible and fraction >= MIN_CLUSTER_FRACTION and valid_stability
                      and np.mean(stability) >= MIN_STABILITY_ARI)
    return {'cluster_sizes': {str(label): int(size) for label, size in zip(ids, sizes)},
            'min_cluster_fraction': fraction, 'selectable': selectable}


def cluster_count(estimator):
    """Devuelve el número configurado de componentes o de centros aprendidos."""
    if hasattr(estimator, 'n_components'):
        return int(estimator.n_components)
    return int(len(estimator.cluster_centers_))


def reseed(estimator, seed):
    """Clona un estimador y fija su semilla cuando la familia admite random_state."""
    result = clone(estimator)
    if 'random_state' in result.get_params():
        result.set_params(random_state=seed)
    return result



def evaluate_stability(estimator, reduced, labels, indices, seed):
    """Calcula tres ARI con remuestreo al 80 %, conservando SVD e índices.

    Devuelve puntuaciones y avisos separados; no cambia la elegibilidad del
    ajuste principal ni utiliza categorías editoriales para elegir el modelo.
    """
    count = len(reduced)
    stability, stability_warnings = [], []
    for offset in [1, 2, 3]:
        subset = np.random.default_rng(seed + offset).choice(count, max(2, int(.8 * count)), replace=False)
        with warnings.catch_warnings(record=True) as replica_warnings:
            warnings.simplefilter('always')
            other = reseed(estimator, seed + offset).fit(reduced[subset])
        stability_warnings.extend(str(w.message) for w in replica_warnings)
        # Los avisos de réplicas se conservan separados de la elegibilidad del ajuste principal.
        if not any(issubclass(w.category, ConvergenceWarning) for w in replica_warnings):
            stability.append(float(adjusted_rand_score(labels[indices], other.predict(reduced[indices]))))
    return stability, stability_warnings


def compare_clusterings(matrix, categories, seed):
    """Compara familias sobre las mismas filas SVD y selecciona sin etiquetas editoriales.

    Todas las familias inductivas ajustan el mismo corpus y se evalúan sobre los
    mismos 1.000 índices. Sólo son elegibles soluciones convergidas, con cobertura
    completa y entre dos y n-1 grupos observados. Las categorías sólo describen.
    DBSCAN y aglomerativo se distinguen como exploraciones transductivas.
    """
    count = matrix.shape[0]
    indices = np.random.default_rng(seed).choice(count, min(1000, count), replace=False)
    trials, winning = [], None
    dimensions = sorted({min(d, count - 1, matrix.shape[1] - 1) for d in [15, 30, 60]})
    for dimension in dimensions:
        reduced = TruncatedSVD(n_components=dimension, random_state=seed).fit_transform(matrix)
        configurations = [(name, k, {}) for k in [2, 3, 4, 5, 6, 8, 10, 12]
                          if k < count for name in ['kmeans', 'minibatch']]
        configurations += [('gmm', k, {'covariance_type': covariance})
                           for k in [2, 3, 5, 8] if k < count for covariance in ['diag', 'tied']]
        # bandwidth se estima sólo con la muestra de ajuste, nunca con categorías.
        for quantile in [.15, .30]:
            bandwidth = float(estimate_bandwidth(reduced, quantile=quantile,
                                                n_samples=min(500, count), random_state=seed))
            if np.isfinite(bandwidth) and bandwidth > 0:
                configurations.append(('meanshift', None, {'bandwidth': bandwidth, 'bandwidth_quantile': quantile}))
        for name, k, options in configurations:
            print('Clustering', name, 'SVD', dimension, 'k', k, options, flush=True)
            params = dict(options, dimensions=dimension, seed=seed)
            estimator = cluster_estimator(name, k, params)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                labels = estimator.fit_predict(reduced)
            mask = labels[indices] != -1
            groups = len(np.unique(labels[indices][mask]))
            score = (float(silhouette_score(reduced[indices][mask], labels[indices][mask]))
                     if 1 < groups < int(mask.sum()) else None)
            convergence = any(issubclass(w.category, ConvergenceWarning) for w in caught)
            converged = bool(getattr(estimator, 'converged_', not convergence)) and not convergence
            if hasattr(estimator, 'n_iter_') and hasattr(estimator, 'max_iter'):
                converged = converged and estimator.n_iter_ < estimator.max_iter
            eligible = score is not None and bool(mask.all()) and converged
            stability, stability_warnings = evaluate_stability(estimator, reduced, labels, indices, seed) if eligible else ([], [])
            trial = {'name': name, 'k': cluster_count(estimator), 'dimensions': dimension,
                     'params': options, 'silhouette': score, 'coverage': float(mask.mean()),
                     'n': count, 'evaluation_n': len(indices), 'observed_evaluation_clusters': groups,
                     'eligible': eligible, 'converged': converged,
                     'warnings': sorted({str(w.message) for w in caught}),
                     'stability_ARI': stability,
                     'stability_warnings': sorted(set(stability_warnings)),
                     'stability_ARI_mean': float(np.mean(stability)) if stability else None,
                     'ARI': float(adjusted_rand_score(categories, labels)),
                     'NMI': float(normalized_mutual_info_score(categories, labels))}
            trial.update(selection_details(labels, eligible, stability))
            if name == 'gmm':
                trial.update(AIC=float(estimator.aic(reduced)), BIC=float(estimator.bic(reduced)))
            trials.append(trial)
            if trial['selectable'] and (winning is None or score > winning[0]):
                winning = (score, name, estimator, dimension, reduced, labels, params)
    if winning is None:
        raise ValueError('Ninguna segmentación cumple tamaño mínimo 5 %, tres réplicas válidas y ARI medio >= 0,80; no se relajan los límites')
    reduced = winning[4]
    exploration = [('agglomerative', AgglomerativeClustering(n_clusters=k), {'k': k})
                   for k in [2, 3, 5, 8, 12] if k < len(indices)]
    exploration += [('dbscan', DBSCAN(eps=eps, min_samples=minimum), {'eps': eps, 'min_samples': minimum})
                    for eps in [.7, 1.5, 3.] for minimum in [5, 10]]
    for name, estimator, parameters in exploration:
        labels = estimator.fit_predict(reduced[indices])
        mask = labels != -1
        groups = len(np.unique(labels[mask]))
        score = (float(silhouette_score(reduced[indices][mask], labels[mask]))
                 if 1 < groups < int(mask.sum()) else None)
        trials.append({'name': name, **parameters, 'dimensions': winning[3],
                       'scope': 'Exploración transductiva; ajuste sobre los índices comunes, sin predict nativo',
                       'eligible': False, 'selectable': False, 'n': len(indices), 'evaluation_n': len(indices),
                       'coverage': float(mask.mean()), 'noise_fraction': float((~mask).mean()),
                       'silhouette': score, 'ARI': float(adjusted_rand_score(categories.iloc[indices], labels)),
                       'NMI': float(normalized_mutual_info_score(categories.iloc[indices], labels))})
    return trials, winning
