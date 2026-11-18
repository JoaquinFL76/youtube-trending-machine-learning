"""Partición por vídeo y muestreo comunes a búsqueda, calibración y ajuste final."""
# sklearn.model_selection: organiza particiones, validación cruzada y búsquedas de parámetros.
# Se importa: GroupShuffleSplit.
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold, GroupKFold


def split_video_pools(data, seed):
    """Reserva el 20 % de grupos de vídeo y devuelve pools sin muestrear.

    Conserva filas, índices y orden de GroupShuffleSplit con una única partición.
    No consulta objetivos ni transforma datos; rechaza solapamientos explícitamente.
    """
    train, test = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=seed).split(
        data, groups=data.video_id))
    development, holdout = data.iloc[train], data.iloc[test]
    if set(development.video_id) & set(holdout.video_id):
        raise ValueError('Solapamiento de video_id entre entrenamiento y prueba')
    return development, holdout


def group_sample(df, limit, seed):
    """Selecciona vídeos completos con presupuesto aproximado por categoría.

    limit limita filas aproximadamente, no vídeos: conservar grupos puede superar
    el presupuesto mínimo por estrato. seed fija el orden; devuelve una copia.
    Con la misma semilla, presupuestos crecientes producen muestras anidadas."""
    if not limit or len(df) <= limit:
        return df.copy()
    # groups: identificadores o tabla de grupos completos de vídeo.
    groups = df[['video_id','category']].drop_duplicates('video_id').copy()
    groups['rows'] = groups.video_id.map(df.video_id.value_counts())
    # selected: IDs de vídeo seleccionados sin partir sus apariciones.
    selected = []
    for category, block in groups.groupby('category', sort=True):
        # block: tabla de vídeos de una categoría, reordenada con la semilla fija.
        block = block.sample(frac=1, random_state=seed)
        # budget: presupuesto aproximado de filas asignado a una categoría.
        budget = max(1, int(limit * block.rows.sum() / len(df)))
        # keep: IDs cuyo acumulado cabe en el presupuesto del estrato.
        keep = block.loc[block.rows.cumsum() <= budget, 'video_id'].tolist()
        selected.extend(keep or [block.video_id.iloc[0]])
    return df.loc[df.video_id.isin(selected)].copy()


def build_group_folds(data, target, folds, classification, seed):
    """Construye folds idénticos para búsqueda, calibración y diagnóstico.

    Clasificación aproxima proporciones de clases sin separar vídeos. Regresión
    utiliza GroupKFold sin shuffle. Rechaza solapamiento y cobertura incompleta;
    la escasez de clases sigue generando el aviso del particionador, sin ocultarlo.
    """
    splitter = (StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
                if classification else GroupKFold(n_splits=folds))
    splits = list(splitter.split(data, target, groups=data.video_id))
    coverage = [0] * len(data)
    for train, valid in splits:
        if set(data.video_id.iloc[train]) & set(data.video_id.iloc[valid]):
            raise ValueError('Un vídeo aparece en ajuste y validación del mismo fold')
        for index in valid:
            coverage[index] += 1
    if any(count != 1 for count in coverage):
        raise ValueError('Cada fila debe aparecer exactamente una vez en validación')
    return splits
