"""Evaluación proxy, controles y preparación de revisión humana de la recuperación."""
# json: lee y escribe datos estructurados, configuraciones y evidencias en JSON.
import json
# unicodedata: normaliza caracteres y consulta sus propiedades Unicode.
import unicodedata
# numpy: realiza cálculos numéricos y operaciones con vectores y matrices.
import numpy as np
# pandas: organiza, limpia y analiza datos tabulares mediante DataFrame y Series.
import pandas as pd


def script_group(text):
    """Identifica el sistema de escritura dominante, sin inferir idioma o país.

    Examina letras Unicode del título. Requiere un 60 % para asignar un sistema;
    devuelve mixed u other si no hay predominio o letras reconocibles.
    """
    # counts: letras de cada familia; marcas y números no se interpretan como idioma.
    counts = dict.fromkeys(['LATIN', 'CYRILLIC', 'HANGUL', 'HIRAGANA', 'KATAKANA', 'CJK'], 0)
    for char in str(text):
        if char.isalpha():
            name = unicodedata.name(char, '')
            for family in counts:
                if family in name:
                    counts[family] += 1
                    break
    total = sum(counts.values())
    if not total:
        return 'other'
    winner = max(counts, key=counts.get)
    return winner.lower() if counts[winner] / total >= .6 else 'mixed'


def query_diagnostics(queries, catalog, predictions):
    """Construye una fila diagnóstica por consulta, respetando su orden original.

    Incluye coincidencia editorial /5 y su esperanza bajo recuperación aleatoria.
    El control aleatorio se calcula analíticamente con la distribución del catálogo;
    no es una medición de satisfacción. Devuelve un DataFrame.
    """
    if len(queries) != len(predictions):
        raise ValueError('Debe existir una lista de candidatos por consulta')
    # lookup: catálogo sin IDs repetidos; frequencies: referencia marginal de categoría.
    lookup = catalog.set_index('video_id')
    frequencies = catalog.category.value_counts(normalize=True)
    records = []
    for (_, row), items in zip(queries.iterrows(), predictions):
        ids = [item['video_id'] for item in items]
        records.append({
            'query_id': row.video_id, 'region': row.get('region', 'unknown'),
            'script': script_group(row.title), 'category': row.category,
            'category_agreement_at_5_PROXY': sum(lookup.loc[i, 'category'] == row.category for i in ids) / 5,
            'random_category_expectation': float(frequencies.get(row.category, 0)),
            'returned': len(ids), 'empty': not bool(ids),
        })
    return pd.DataFrame(records)


def evaluate_retrieval(model, queries, catalog):
    """Mide coincidencia editorial, cobertura y diversidad en consultas disjuntas.

    Devuelve (métricas, listas de vecinos). Las consultas OOV puntúan cero y el
    denominador top-5 siempre es cinco. Rechaza solapamiento de IDs o tablas vacías.
    No convierte categorías en etiquetas humanas de relevancia.
    """
    if queries.empty or len(catalog) < 5:
        raise ValueError('Se necesitan consultas y un catálogo de al menos cinco vídeos')
    if not catalog.video_id.is_unique or not queries.video_id.is_unique:
        raise ValueError('Catálogo y consultas deben tener identidades únicas')
    if set(queries.video_id) & set(catalog.video_id):
        raise ValueError('Catálogo y consultas deben tener identidades disjuntas')
    # vectors: exactamente la representación que aplica predict, incluido focused.
    vectors = model._encode_queries(queries)
    valid = np.flatnonzero(np.asarray(vectors.getnnz(axis=1)) > 0)
    batches = model.predict(queries.iloc[valid].reindex(columns=model.columns_)) if len(valid) else []
    lookup_predictions = dict(zip(valid, batches))
    predictions = [lookup_predictions.get(i, []) for i in range(len(queries))]
    lookup = catalog.set_index('video_id')
    covered = {item['video_id'] for items in predictions for item in items}
    diversity = [lookup.loc[[item['video_id'] for item in items], 'channel_title'].nunique() / len(items)
                 if items else 0 for items in predictions]
    diagnostics = query_diagnostics(queries, catalog, predictions)
    metrics = {
        'category_agreement_at_5_PROXY': float(diagnostics.category_agreement_at_5_PROXY.mean()),
        'random_category_expectation': float(diagnostics.random_category_expectation.mean()),
        'catalog_coverage': len(covered) / len(catalog),
        'unique_channel_fraction_at_5': float(np.mean(diversity)),
        'empty_query_fraction': float(diagnostics['empty'].mean()), 'query_videos': len(queries),
    }
    return metrics, predictions


def export_diagnostics(queries, catalog, predictions, directory, seed):
    """Guarda errores por región/escritura e IC bootstrap por consulta única.

    Devuelve el resumen del control y un IC percentil 95 % con 500 réplicas.
    Es condicional a catálogo/modelo fijos; no incorpora selección de representación.
    """
    diagnostics = query_diagnostics(queries, catalog, predictions)
    diagnostics.to_csv(directory / 'recommendation_query_diagnostics.csv', index=False)
    for field in ['region', 'script']:
        grouped = diagnostics.groupby(field).agg(
            query_videos=('query_id', 'size'),
            category_agreement_at_5_PROXY=('category_agreement_at_5_PROXY', 'mean'),
            random_category_expectation=('random_category_expectation', 'mean'),
            empty_query_fraction=('empty', 'mean'))
        grouped.to_csv(directory / f'recommendation_by_{field}.csv')
    # Bootstrap sobre consultas únicas: no se considera cada vecino una observación independiente.
    values = diagnostics.category_agreement_at_5_PROXY.to_numpy()
    rng = np.random.default_rng(seed)
    boot = values[rng.integers(0, len(values), size=(500, len(values)))].mean(axis=1)
    return {'proxy_ci95': np.quantile(boot, [.025, .975]).tolist(),
            'random_category_expectation': float(diagnostics.random_category_expectation.mean()),
            'proxy_gain_over_random_expectation': float(np.mean(values) - diagnostics.random_category_expectation.mean()),
            'method': '500 réplicas por consulta; catálogo y modelo fijos; categoría sigue siendo sólo un proxy',
            'script_warning': 'Sistema de escritura Unicode aproximado; no detección de idioma ni nacionalidad'}


def preserve_human_labels(path, fresh, keys, label_columns):
    """Conserva anotaciones reales al regenerar una hoja con los mismos pares.

    Si cambian los pares y hay anotaciones, aborta antes de sobrescribir. Las
    etiquetas vacías no se rellenan con inferencias ni valoraciones automáticas.
    """
    if path.exists():
        previous = pd.read_csv(path, keep_default_na=False)
        annotated = [name for name in label_columns if name in previous
                     and previous[name].astype(str).str.strip().ne('').any()]
        if annotated:
            if set(map(tuple, previous[keys].to_numpy())) != set(map(tuple, fresh[keys].to_numpy())):
                raise FileExistsError(f'Hay anotaciones humanas: conservar y versionar antes de reemplazar {path}')
            fresh = fresh.drop(columns=label_columns).merge(previous[keys + label_columns], on=keys, validate='one_to_one')
    fresh.to_csv(path, index=False)


def export_human_review(queries, catalog, predictions, directory, seed):
    """Prepara revisión ciega de 50 consultas con vecinos y dos controles aleatorios.

    La hoja ciega oculta categoría, rango, similitud y origen del candidato; la
    clave separada permite analizar controles después. Dos revisores puntúan 0/1/2
    con motivo y adjudicación. Devuelve recuentos, sin fabricar resultados humanos.
    """
    rng = np.random.default_rng(seed)
    lookup = catalog.set_index('video_id')
    positions = sorted(rng.choice(len(queries), min(50, len(queries)), replace=False))
    visible, answer_key, conventional = [], [], []
    for position in positions:
        row = queries.iloc[position]
        items = predictions[position]
        returned = {item['video_id'] for item in items}
        pool = catalog.loc[~catalog.video_id.isin(returned | {row.video_id})]
        controls = pool.sample(n=min(2, len(pool)), random_state=seed + int(position))
        pairs = [(item['video_id'], 'retrieved', rank, item['similarity'])
                 for rank, item in enumerate(items, 1)]
        pairs += [(item.video_id, 'random_control', None, None) for item in controls.itertuples()]
        for pair in rng.permutation(len(pairs)):
            candidate_id, origin, rank, similarity = pairs[int(pair)]
            candidate = lookup.loc[candidate_id]
            # pair_id: clave estable; el orden aleatorio no codifica origen ni posición.
            pair_id = f'{row.video_id}__{candidate_id}'
            visible.append({'pair_id': pair_id, 'query_id': row.video_id, 'query_title': row.title,
                'query_description': str(row.get('description', ''))[:250],
                'candidate_id': candidate_id, 'candidate_title': candidate.title,
                'candidate_description': str(candidate.get('description', ''))[:250],
                'query_url': f'https://www.youtube.com/watch?v={row.video_id}',
                'candidate_url': f'https://www.youtube.com/watch?v={candidate_id}',
                'reviewer_1_0_1_2': '', 'reviewer_1_reason': '',
                'reviewer_2_0_1_2': '', 'reviewer_2_reason': '', 'adjudicated_0_1_2': ''})
            answer_key.append({'pair_id': pair_id, 'origin': origin, 'rank': rank, 'similarity': similarity,
                               'query_category': row.category, 'candidate_category': candidate.category})
            if origin == 'retrieved':
                conventional.append({'query_id': row.video_id, 'query_title': row.title, 'query_category': row.category,
                    'rank': rank, 'video_id': candidate_id, 'title': candidate.title, 'similarity': similarity,
                    'candidate_category': candidate.category, 'human_relevance_0_1_2': '', 'human_reason': ''})
    sheet = pd.DataFrame(visible)
    labels = ['reviewer_1_0_1_2', 'reviewer_1_reason', 'reviewer_2_0_1_2', 'reviewer_2_reason', 'adjudicated_0_1_2']
    preserve_human_labels(directory / 'recommendation_blind_review.csv', sheet, ['pair_id'], labels)
    pd.DataFrame(answer_key).to_csv(directory / 'recommendation_blind_key.csv', index=False)
    conventional = pd.DataFrame(conventional, columns=[
        'query_id', 'query_title', 'query_category', 'rank', 'video_id', 'title', 'similarity',
        'candidate_category', 'human_relevance_0_1_2', 'human_reason'])
    preserve_human_labels(directory / 'recommendation_human_review.csv', conventional,
                          ['query_id', 'video_id'], ['human_relevance_0_1_2', 'human_reason'])
    protocol = '''# Protocolo de revisión humana pendiente

Dos personas revisan por separado recommendation_blind_review.csv sin abrir la clave.
Se usa título, descripción y, cuando sea accesible, el contenido enlazado. Puntuar:
0 = no relacionado; 1 = relación parcial; 2 = contenido útil para la misma intención.
Registrar motivo y no confundir idioma/escritura o categoría editorial con relevancia.
No deducir veracidad de una afirmación por similitud temática. Si el vídeo no se puede
valorar, dejar la nota vacía y explicar el motivo; no convertirlo automáticamente en 0.
Resolver discrepancias en adjudicated_0_1_2 tras las valoraciones independientes.
Sólo entonces abrir recommendation_blind_key.csv para comparar vecinos y controles.
La selección de representación ya está fijada: no volver a optimizar con esta hoja.

Las puntuaciones permanecen vacías hasta que dos evaluadores humanos completen la revisión independiente.
La hoja no se sobrescribe si contiene anotaciones y cambian sus pares. Aún no hay
Precision@5, NDCG ni acuerdo entre revisores humanos que se puedan publicar.
'''
    (directory / 'human_review_protocol.md').write_text(protocol, encoding='utf-8')
    return {'queries': len(positions), 'pairs': len(sheet), 'reviewers_required': 2,
            'human_evaluation_status': 'pending', 'labels_fabricated': False}
