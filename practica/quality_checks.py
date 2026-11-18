"""Validaciones adicionales internas por canal y de estabilidad del pipeline completo."""
# metric_utils: comparte el cálculo de F2 y validaciones de matrices de confusión.
# Se importa: f2_from_precision_recall.
from metric_utils import f2_from_precision_recall
# configparser: lee y escribe las secciones de configuración de train.conf.
import configparser
# datetime: representa fechas, horas y zonas horarias para registrar la ejecución.
# Se importa: datetime, timezone.
from datetime import datetime, timezone
# hashlib: calcula huellas digitales para comprobar integridad y procedencia.
import hashlib
# json: lee y escribe datos estructurados, configuraciones y evidencias en JSON.
import json
# pathlib: maneja rutas de archivos y carpetas mediante objetos, como Path.
# Se importa: Path.
from pathlib import Path
# time: mide duraciones y consulta referencias de tiempo del proceso.
import time
# warnings: emite, captura o filtra avisos sin confundirlos con errores.
import warnings
# numpy: realiza cálculos numéricos y operaciones con vectores y matrices.
import numpy as np
# pandas: organiza, limpia y analiza datos tabulares mediante DataFrame y Series.
import pandas as pd
# sklearn.base: clona estimadores sin reutilizar el estado aprendido de otro ajuste.
# Se importa: clone.
from sklearn.base import clone
# sklearn.metrics: calcula métricas para evaluar predicciones, grupos y similitudes.
# Se importa: adjusted_rand_score, precision_recall_curve, fbeta_score.
from sklearn.metrics import adjusted_rand_score, precision_recall_curve, fbeta_score
# sklearn.model_selection: organiza particiones, validación cruzada y búsquedas de parámetros.
# Se importa: GroupShuffleSplit.
from sklearn.model_selection import GroupShuffleSplit
# models: expone las siete clases finales y el registro que relaciona tareas con clases.
# Se importa: REGISTRY.
from models import REGISTRY
# models.core: comparte validación, transformaciones, factorías y persistencia de modelos.
# Se importa: estimator_from_name, features, positive_scores.
from models.core import estimator_from_name, features, positive_scores
# model_selection: compara modelos y conserva métricas, particiones y huellas del estudio.
# Se importa: group_sample, evaluate, ci_by_video.
from model_selection import group_sample, evaluate, ci_by_video


def channel_components(data):
    """Agrupa vídeos y alias observados de canal en componentes conexos.

    data necesita video_id/channel_title. Normaliza títulos con Unicode NFC,
    espacios y casefold. Si un vídeo aparece con varios nombres, une sus alias.
    Devuelve una Series alineada de componentes. No inventa IDs de creador:
    channel_title no es un identificador canónico y puede contener homónimos.
    """
    # parents: bosque unión-búsqueda; los prefijos separan canales de vídeos.
    parents = {}

    def find(node):
        """Obtiene la raíz y comprime caminos para mantener eficiente la agrupación."""
        parents.setdefault(node, node)
        while parents[node] != node:
            parents[node] = parents[parents[node]]
            node = parents[node]
        return node

    def union(left, right):
        """Une un vídeo con un nombre de canal sin consultar categorías u objetivos."""
        left, right = find(left), find(right)
        if left != right:
            parents[max(left, right)] = min(left, right)

    # pairs: sólo las relaciones distintas; repetir apariciones no cambia componentes.
    pairs = data[['video_id', 'channel_title']].copy()
    pairs['channel_title'] = pairs.channel_title.fillna('').astype(str).str.normalize('NFC').str.casefold().str.strip()
    for video, channel in pairs.drop_duplicates().itertuples(index=False, name=None):
        if channel:
            union('v:' + str(video), 'c:' + channel)
        else:
            find('v:' + str(video))
    mapping = {video: find('v:' + str(video)) for video in data.video_id.unique()}
    return data.video_id.map(mapping).rename('channel_component')


def partition_channels(data, seed):
    """Reserva 60/20/20 % aproximado de componentes para ajuste/calibración/prueba.

    Divide ANTES de muestrear. Devuelve tres DataFrames y evidencia de ausencia
    de solapamiento por componente, vídeo y nombre normalizado de canal.
    No garantiza identidad real de creadores cuando falta su ID canónico.
    """
    data = data.copy()
    data['channel_title'] = data.channel_title.fillna('').astype(str).str.normalize('NFC').str.casefold().str.strip()
    if data.channel_title.eq('').any():
        raise ValueError('Excluir y auditar filas sin nombre de canal antes de partir')
    data['channel_component'] = channel_components(data)
    outer_train, test = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=seed).split(
        data, groups=data.channel_component))
    development = data.iloc[outer_train]
    train, calibration = next(GroupShuffleSplit(n_splits=1, test_size=.25, random_state=seed + 1).split(
        development, groups=development.channel_component))
    parts = [development.iloc[train].copy(), development.iloc[calibration].copy(), data.iloc[test].copy()]
    overlap = {}
    for field in ['channel_component', 'video_id', 'channel_title']:
        sets = [set(part[field]) for part in parts]
        overlap[field] = [len(sets[a] & sets[b]) for a, b in [(0, 1), (0, 2), (1, 2)]]
        if any(overlap[field]):
            raise AssertionError(f'Solapamiento de {field} entre particiones')
    return parts, {'overlap_counts_train_cal_train_test_cal_test': overlap,
                   'full_partition_rows': [len(part) for part in parts],
                   'components': [part.channel_component.nunique() for part in parts]}


def select_calibration_threshold(y, scores):
    """Elige F2 usando exclusivamente etiquetas de la partición de calibración.

    Devuelve (umbral, diagnóstico). Si falta alguna clase, deja el umbral en None;
    nunca utiliza prueba para resolver empates o cambiar la política elegida.
    """
    if set(np.unique(y)) != {0, 1}:
        return None, {'status': 'insufficient_classes', 'rows': len(y)}
    precision, recall, thresholds = precision_recall_curve(y, scores)
    f2 = f2_from_precision_recall(precision[:-1], recall[:-1])
    best = int(np.argmax(f2))
    return float(thresholds[best]), {'status': 'selected_on_calibration_only', 'rows': len(y),
        'positive_rows': int(np.sum(y)), 'f2': float(f2[best]),
        'precision': float(precision[best]), 'recall': float(recall[best])}


def cost_scenarios(default_matrix, calibrated_matrix):
    """Contrasta costes hipotéticos por 1.000 casos, sin elegir política en prueba.

    FP vale una unidad y FN vale 1/2/5/10 unidades según escenario. Devuelve una
    lista de costes; no representa costes reales acordados con un negocio.
    """
    rows = []
    for fn_cost in [1, 2, 5, 10]:
        row = {'false_positive_unit_cost': 1, 'false_negative_unit_cost': fn_cost}
        for name, matrix in [('default', default_matrix), ('calibrated', calibrated_matrix)]:
            matrix = np.asarray(matrix)
            row[name + '_cost_per_1000'] = float(1000 * (matrix[0, 1] + fn_cost * matrix[1, 0]) / matrix.sum())
        rows.append(row)
    return rows


def validate_supervised(train, calibration, test, recipes, seed):
    """Reajusta recetas congeladas y evalúa cinco tareas en canales disjuntos.

    Devuelve métricas, referencias, intervalos y calibración separada. No vuelve
    a buscar hiperparámetros. Estos proceden del experimento base de la misma
    fuente: es un diagnóstico adicional interno, no validación externa/anidada.
    """
    results = {}
    for task in ['category', 'comments_disabled', 'ratings_disabled', 'likes', 'ratio']:
        print('Validación por canal:', task, flush=True)
        # blocks: se excluyen objetivos ausentes sin alterar las particiones previas.
        blocks = [part.dropna(subset=[task]).reset_index(drop=True) for part in [train, calibration, test]]
        ys = [part[task] if task == 'category' else pd.to_numeric(part[task]).astype(
            int if task.endswith('_disabled') else float) for part in blocks]
        classification = task in ['category', 'comments_disabled', 'ratings_disabled']
        if classification and ys[0].nunique() < 2:
            results[task] = {'status': 'insufficient_training_classes'}
            continue
        spec = recipes[task]
        model = REGISTRY[task](estimator=estimator_from_name(spec['model']), params=spec['params'])
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter('always')
            model.fit(blocks[0], ys[0])
        # views: se pasa sólo el esquema que consume el modelo; no se descartan columnas del usuario a ciegas.
        views = [part.reindex(columns=model.columns_) for part in blocks]
        raw = model.predict(views[2])
        threshold, selection = None, None
        if task.endswith('_disabled'):
            threshold, selection = select_calibration_threshold(
                ys[1], positive_scores(model.pipeline_, features(views[1])))
            model.decision_threshold_ = threshold
        prediction = model.predict(views[2])
        baseline = estimator_from_name('baseline_classifier' if classification else 'baseline_regressor')
        baseline.fit(np.zeros((len(ys[0]), 1)), ys[0])
        baseline_prediction = baseline.predict(np.zeros((len(ys[2]), 1)))
        result = {'status': 'passed', 'recipe': spec, 'partition_rows': [len(part) for part in blocks],
            'train': evaluate(ys[0], ys[0], model.predict(views[0]), classification),
            'default_threshold_test': evaluate(ys[0], ys[2], raw, classification),
            'test': evaluate(ys[0], ys[2], prediction, classification),
            'baseline_test': evaluate(ys[0], ys[2], baseline_prediction, classification),
            'confidence_interval': ci_by_video(ys[2], prediction, blocks[2].video_id, classification, seed,
                                               sorted(set(ys[0]) | set(ys[2]))),
            'threshold': threshold, 'calibration': selection,
            'warnings': sorted({str(item.message) for item in recorded})}
        if task.endswith('_disabled'):
            result['positive_f2'] = float(fbeta_score(ys[2], prediction, beta=2, zero_division=0))
            result['default_positive_f2'] = float(fbeta_score(ys[2], raw, beta=2, zero_division=0))
            result['cost_scenarios'] = cost_scenarios(
                result['default_threshold_test']['confusion_matrix'], result['test']['confusion_matrix'])
        results[task] = result
    return results


def full_pipeline_stability(root, conf, seed):
    """Reajusta vocabulario, imputación, escala, SVD y centroides en tres remuestreos.

    Separa 1.000 vídeos de auditoría que ninguno de los cuatro ajustes utiliza.
    Compara asignaciones sobre esos mismos vídeos mediante ARI. La receta de
    clustering ya está seleccionada: no mide incertidumbre de elegir k/dimensión.
    """
    source = pd.read_csv(root / conf['data']['input']).drop_duplicates('video_id')
    # Limita memoria/tiempo conservando un conjunto común y reproducible de vídeos.
    source = source.sample(n=min(6000, len(source)), random_state=seed).reset_index(drop=True)
    audit_indices = np.random.default_rng(seed).choice(len(source), min(1000, len(source) // 3), replace=False)
    audit = source.iloc[audit_indices]
    training = source.drop(index=audit_indices)
    template = REGISTRY['cluster']().load(root / conf['output']['cluster'])
    params = dict(template.params, seed=seed)
    reference = REGISTRY['cluster'](estimator=clone(template.pipeline_.named_steps['estimator']), params=params).fit(training)
    reference_labels = reference.predict(audit.reindex(columns=reference.columns_))
    values = []
    for offset in [1, 2, 3]:
        subset = training.sample(frac=.8, random_state=seed + offset)
        # clustering_experiments: compara familias de clustering y calcula estabilidad.
        # Se importa: reseed.
        from clustering_experiments import reseed
        estimator = reseed(template.pipeline_.named_steps['estimator'], seed + offset)
        model = REGISTRY['cluster'](estimator=estimator, params=dict(params, seed=seed + offset)).fit(subset)
        labels = model.predict(audit.reindex(columns=model.columns_))
        values.append(float(adjusted_rand_score(reference_labels, labels)))
        if not (not set(subset.video_id) & set(audit.video_id)):
            raise ValueError('Comprobación incumplida en quality_checks.py: not set(subset.video_id) & set(audit.video_id)')
    return {'replicate_ARI': values, 'mean_ARI': float(np.mean(values)), 'training_videos': len(training),
            'audit_videos': len(audit), 'audit_training_video_overlap': 0,
            'all_preprocessing_refit': True, 'selected_recipe_fixed': True,
            'scope': 'Estabilidad de todo el pipeline con receta fija; auditoría interna, no nuevos géneros validados'}


def report_markdown(report):
    """Delega la presentación española de los diagnósticos sin alterar sus cálculos."""
    # results_report: comparte formato y conclusiones desde evidencias ya calculadas.
    from results_report import quality_report_text
    return quality_report_text(report)


def run(config_path, resume=False):
    """Ejecuta o reutiliza únicamente diagnósticos completos con huellas verificadas.

    config_path identifica train.conf. Devuelve el informe y guarda plan, métricas,
    texto interpretativo y hashes de salidas. resume no acepta resultados parciales
    ni cambios de código/base/calibración. No modifica los siete modelos finales.
    """
    path = Path(config_path).resolve()
    root = path.parent
    conf = configparser.ConfigParser()
    conf.read(path, encoding='utf-8')
    directory = root / conf['output']['reports']
    base = json.loads((directory / 'evaluation.json').read_text(encoding='utf-8'))
    signature = hashlib.sha256(Path(__file__).read_bytes() + base['fingerprint'].encode()
                               + base['calibration']['fingerprint'].encode()).hexdigest()
    output = directory / 'quality_checks.json'
    if resume and output.exists():
        previous = json.loads(output.read_text(encoding='utf-8'))
        if (previous.get('complete') and previous.get('fingerprint') == signature
                and set(previous.get('output_hashes', {})) == {'quality_plan.json', 'quality_report.md'} and all(
                (directory / name).exists() and hashlib.sha256((directory / name).read_bytes()).hexdigest() == digest
                for name, digest in previous.get('output_hashes', {}).items())):
            print('Validaciones adicionales reutilizadas tras verificar sus huellas.', flush=True)
            return previous
        raise ValueError('Diagnósticos de calidad incompatibles o incompletos; ejecutar sin reutilización')
    started = time.monotonic()
    # El plan se escribe antes de ajustar; el seed/presupuestos no dependen de la nueva prueba.
    seed = conf.getint('training', 'seed') + 1000
    plan = {'seed': seed, 'limits_train_calibration_test': [12000, 4000, 6000],
            'created_utc': datetime.now(timezone.utc).isoformat(), 'base_fingerprint': base['fingerprint'],
            'selection': 'Recetas congeladas; umbral F2 sólo en calibración; prueba descriptiva sin selección',
            'scope': 'Validación adicional interna por componentes vídeo/channel_title, no externa ni anidada'}
    (directory / 'quality_plan.json').write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding='utf-8')
    data = pd.read_csv(root / conf['data']['input'])
    missing_channels = data.channel_title.fillna('').astype(str).str.strip().eq('')
    excluded = int(missing_channels.sum())
    data = data.loc[~missing_channels].copy()
    data['ratio'] = (data.likes / data.dislikes.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
    partitions, evidence = partition_channels(data, seed)
    samples = [group_sample(part, limit, seed + index).reset_index(drop=True)
               for index, (part, limit) in enumerate(zip(partitions, plan['limits_train_calibration_test']))]
    del data, partitions
    recipes = json.loads((root / 'models/recipes.json').read_text(encoding='utf-8'))
    tasks = validate_supervised(*samples, recipes, seed)
    report = {'complete': True, 'fingerprint': signature, 'plan': plan, 'partitions': evidence,
              'missing_channel_rows_excluded': excluded, 'sample_rows': [len(part) for part in samples],
              'tasks': tasks, 'full_pipeline_stability': full_pipeline_stability(root, conf, seed),
              'seconds': time.monotonic() - started, 'finished_utc': datetime.now(timezone.utc).isoformat()}
    (directory / 'quality_report.md').write_text(report_markdown(report), encoding='utf-8')
    report['output_hashes'] = {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
                             for name in ['quality_plan.json', 'quality_report.md']}
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    return report
