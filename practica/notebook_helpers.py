"""Presentación del cuaderno: imágenes, detalle y diagnóstico de particiones.

No elige modelos ni modifica pesos. Las decisiones científicas permanecen en
los módulos del experimento; las cifras se calculan desde los datos recibidos.
"""
# pathlib y os: construyen destinos y publican imágenes preparadas.
from pathlib import Path
import os
# numpy y pandas: resumen datos y contabilizan cobertura de los folds.
import numpy as np
import pandas as pd
# matplotlib.pyplot: maneja las figuras ya construidas por las celdas.
import matplotlib.pyplot as plt
# IPython.display: muestra imágenes, textos y tablas enriquecidas.
from IPython.display import Image, HTML, Markdown, display as _native_display
# results_report: mantiene una convención numérica española para lectura humana.
from results_report import format_number, highlight_number, human_value, conclusion_paragraphs, clarify_narrative, warning_groups, task_description

def show_figure(name, root, display):
    """Exporta y muestra la figura activa, cerrándola incluso si falla.

    name debe ser un identificador simple sin extensión ni separadores. Prepara
    el PNG en memoria y lo publica mediante temporal y reemplazo atómico en reports.
    No modifica datos, devuelve None y propaga errores con contexto. El reemplazo
    no garantiza durabilidad frente a cortes eléctricos ni escritores concurrentes.
    """
    # io: prepara los bytes del PNG antes de modificar el archivo de destino.
    from io import BytesIO
    # tempfile: crea un archivo exclusivo en la carpeta de salida.
    import tempfile
    # re: limita name a un identificador simple, sin rutas externas.
    import re
    # figure: figura exacta que se exporta y se cerrará; no se crea una vacía.
    figure = plt.gcf() if plt.get_fignums() else None
    # temporary y target: publicación preparada y ruta definitiva del PNG.
    temporary, target = None, root/'reports'
    try:
        if figure is None:
            raise ValueError('No hay una figura activa para exportar')
        if not isinstance(name, str) or re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', name) is None:
            raise ValueError('name debe contener sólo letras ASCII, números, guion o guion bajo, sin extensión')
        target = root/'reports'/f'{name}.png'
        target.parent.mkdir(parents=True, exist_ok=True)
        localize_figure(figure)
        # Respetar el motor explícito de la figura; evita mezclar dos maquetaciones.
        if figure.get_layout_engine() is None:
            figure.tight_layout()
        # content: imagen completa; su preparación no toca un PNG anterior.
        with BytesIO() as buffer:
            figure.savefig(buffer, format='png', dpi=140, bbox_inches='tight')
            content = buffer.getvalue()
        with tempfile.NamedTemporaryFile(dir=target.parent, prefix='.'+name+'_',
                                         suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(content)
        os.replace(temporary, target)
        display(Image(data=content))
    except Exception as exc:
        raise RuntimeError(f'No se pudo guardar o mostrar el gráfico {name!r} en {target}: {exc}') from exc
    finally:
        try:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        finally:
            if figure is not None:
                plt.close(figure)

def show_details(title, *tables):
    """Muestra tablas técnicas desplegables sin repetirlas en la lectura principal.

    title identifica el bloque; tables contiene DataFrames con la evidencia completa.
    Escapa el título y las celdas para conservar el texto como contenido, no código HTML.
    """
    # html: escapa caracteres especiales para mostrar texto como contenido HTML.
    # Se importa: escape.
    from html import escape
    # body: tablas completas conservadas en el HTML del cuaderno ejecutado.
    body = ''.join(spanish_table_html(table) for table in tables)
    display(HTML('<details><summary>' + escape(title) + '</summary>' + body + '</details>'))

def _trial_review_note(trial):
    """Resume avisos e intentos de búsqueda; conserva la revisión íntegra en el detalle.

    Reutiliza la clasificación de warning_groups y no deduce convergencia a
    partir de una puntuación. Sin avisos no implica haber alcanzado un óptimo.
    """
    # review: registro original del ensayo, que nunca se modifica para presentarlo.
    review = trial.get('convergence_review')
    if not isinstance(review, dict):
        return 'Sin revisión registrada; consultar el detalle'
    # groups: distingue avisos pendientes, resueltos y ajenos a convergencia.
    groups = warning_groups({'trials': [trial]})
    if groups['remaining_convergence']:
        status = 'Persisten avisos de convergencia'
    elif groups['resolved_convergence']:
        status = 'Avisos de convergencia resueltos'
    else:
        status = 'Sin avisos de convergencia'
    # attempts: búsquedas ejecutadas, no número de modelos ajustados en las particiones.
    attempts = review.get('attempts', [])
    count = len(attempts) if isinstance(attempts, list) else 0
    note = status + '; ' + format_number(count, 0) + (' intento de búsqueda' if count == 1 else ' intentos de búsqueda')
    if groups['other_warnings']:
        note += '; otros avisos disponibles en el detalle'
    return note


def show_model_comparison(task, comparison, root, *, regression=False):
    """Presenta comparación, refinamiento y evidencia sin recalcular la selección.

    Las puntuaciones guardadas se maximizan. Para leer la regresión se invierte
    el signo de −MAE y se conservan las dispersiones de los folds.
    """
    trials = comparison['trials']
    valid = [trial for trial in trials if 'cv_score' in trial and np.isfinite(trial['cv_score'])]
    if not valid:
        raise ValueError('No hay familias con puntuación CV finita para explicar la comparación')
    winner = max(valid, key=lambda trial: trial['cv_score'])
    metric, sign, decimals = ('MAE', -1, 2) if regression else ('Macro-F1', 1, 4)
    metric_context = (
        'La tabla técnica conserva −MAE. Una puntuación mayor corresponde a un error menor '
        'en la escala original del objetivo, incluso con transformación logarítmica.'
        if regression else
        'El macro-F1 promedia las clases presentes en las etiquetas reales o predichas '
        'de cada fold. Revisar el soporte de las clases raras ayuda a interpretar la comparación.'
    )
    display(Markdown(
        f'**Comparación de familias: {task_description(task)}.** {highlight_number(len(trials))} entradas, '
        f'{highlight_number(len(valid))} con puntuación finita y '
        f'{highlight_number(len(trials)-len(valid))} sin ella. '
        f'La mejor es **{winner["model"]}**, con {metric} medio '
        f'{highlight_number(sign*winner["cv_score"], decimals)} y desviación entre folds '
        f'{highlight_number(winner["cv_std"], decimals)}. '
        'Los candidatos usan los mismos folds. Una diferencia pequeña requiere cautela. '
        + metric_context
    ))

    refinement = comparison['refinement']
    selected = refinement['trials'][refinement['best_index']]
    reference = next(trial for trial in refinement['trials'] if trial['reference'])
    gain = refinement['cv_score'] - reference['mean_cv_score']
    if not np.isfinite(gain) or gain < -1e-10:
        raise ValueError('El refinamiento no conserva una puntuación válida al menos igual a su referencia')
    decision = ('Se conserva la configuración de referencia.' if selected['reference']
                else 'Se selecciona otra configuración. Puede haber empate en la puntuación.')
    gain_label = 'Reducción de MAE' if regression else 'Variación de macro-F1'
    display(Markdown(
        f'**Refinamiento para {task_description(task)}: {comparison["selected"]}.** '
        f'{highlight_number(len(refinement["trials"]))} configuraciones, '
        f'{highlight_number(refinement["folds"])} folds y '
        f'{highlight_number(refinement["search_rows"])} filas elegibles. '
        f'{metric} de la búsqueda inicial {highlight_number(sign*refinement["initial_cv_score"], decimals)}. '
        f'Referencia reevaluada {highlight_number(sign*reference["mean_cv_score"], decimals)}. '
        f'Configuración seleccionada {highlight_number(refinement["best_index"])} con '
        f'{metric} de {highlight_number(sign*refinement["cv_score"], decimals)} y desviación entre folds '
        f'{highlight_number(selected["std_cv_score"], decimals)}. '
        f'{gain_label} respecto a la referencia reevaluada {highlight_number(gain, decimals)}. '
        + decision + ' Los índices identifican ensayos. Las barras muestran dispersión entre folds. '
        'El refinamiento usa los folds de desarrollo y la prueba reservada se mantiene aparte.'
    ))
    # comparison_table: copia de presentación; las métricas y revisiones originales se conservan.
    comparison_table = pd.DataFrame(trials).drop(columns=['params','warnings'], errors='ignore')
    if 'convergence_review' in comparison_table:
        comparison_table['convergence_review'] = [_trial_review_note(trial) for trial in trials]
    display(comparison_table)
    display(pd.Series(refinement['params']).to_frame(f'parámetros seleccionados · {task_description(task)}'))
    # La figura se reconstruye desde puntuaciones guardadas para localizar sus ejes.
    figure, axis = plt.subplots(figsize=(9,4))
    axis.errorbar(range(len(refinement['trials'])),
                  [row['mean_cv_score'] for row in refinement['trials']],
                  yerr=[row['std_cv_score'] for row in refinement['trials']], fmt='o', capsize=3)
    axis.set(xlabel='Configuración (0 = referencia)', ylabel='−MAE' if regression else 'macro-F1',
             title=f'Refinamiento de parámetros · {task_description(task)}\nConfiguración elegida: {refinement["best_index"]}')
    show_figure(f'{task}_refinement', root, display)
    display(Markdown(comparison_conclusion(comparison, regression=regression)))
    show_details('Espacio de búsqueda, configuraciones ensayadas y avisos de familias',
                 pd.Series(refinement['space']).to_frame('valores explorados'),
                 pd.DataFrame(refinement['trials']), pd.DataFrame(trials))


def _protocol_group_folds(data, target, folds, classification, seed, task, display):
    """Traduce sólo el aviso conocido de clases escasas en este diagnóstico visual.

    Conserva los índices de build_group_folds y no cambia datos ni modelos. Los
    demás avisos se vuelven a emitir con su categoría, origen y línea originales,
    incluso si falla la construcción. La ruta de entrenamiento no se modifica.
    """
    # warnings y re: identifican un aviso concreto; no filtran familias de avisos.
    import warnings
    import re
    # training_data: conserva la construcción canónica y sus controles por vídeo.
    from training_data import build_group_folds
    # pattern: texto exacto del aviso de scikit-learn que se explica en español.
    pattern = r'The least populated class in y has only [0-9]+ members, which is less than n_splits=[0-9]+\.'
    captured, scarce = [], False
    try:
        with warnings.catch_warnings(record=True) as captured:
            warnings.filterwarnings('always', message=pattern, category=UserWarning,
                                    module=r'sklearn\.model_selection\._split')
            splits = build_group_folds(data, target, folds, classification, seed)
    finally:
        for warning in captured:
            known = (warning.category is UserWarning
                     and re.fullmatch(pattern, str(warning.message)) is not None
                     and Path(warning.filename).name == '_split.py'
                     and 'sklearn' in Path(warning.filename).parts)
            if known:
                scarce = True
            else:
                warnings.warn_explicit(warning.message, warning.category,
                                       warning.filename, warning.lineno,
                                       source=warning.source)
    if scarce:
        # counts: apariciones por clase, sin inventar ejemplos ni cambiar las particiones.
        counts = target.value_counts()
        # rare_count y agreement: concordancia del texto según el número real de clases.
        rare_count = int((counts < folds).sum())
        agreement = 'clase tiene' if rare_count == 1 else 'clases tienen'
        # html: conserva el identificador de tarea como texto en el mensaje Markdown.
        from html import escape
        display(Markdown(
            f'**Aviso sobre clases poco frecuentes ({escape(task_description(task))}):** '
            f'**{format_number(rare_count, 0)}** {agreement} menos de '
            f'**{format_number(folds, 0)}** apariciones en la muestra de búsqueda. '
            f'La menor tiene **{format_number(int(counts.min()), 0)}**. '
            'No es posible que todas aparezcan en cada parte de validación. '
            'Se conserva la separación por vídeo y el reparto previsto; la tabla siguiente '
            'indica qué particiones tienen clases ausentes. Sin ejemplos de una clase en ajuste, '
            'el modelo no puede aprenderla; sin ejemplos en validación, no se puede medir '
            'su detección en esa partición. **Conclusión:** la validación cruzada sigue ejecutándose, '
            'pero la calidad en las clases escasas no puede estimarse con fiabilidad.'
        ))
    return splits


def describe_protocol(df, ROOT, display):
    """Muestra tamaños y controles calculados con el mismo reparto y las mismas particiones del experimento."""
    # configparser: lee la semilla y los presupuestos vigentes, sin fijarlos en la explicación.
    import configparser
    # training_data: aplica la misma reserva por vídeo utilizada por los experimentos.
    from training_data import split_video_pools
    # protocol_config: configuración actual del experimento.
    protocol_config = configparser.ConfigParser()
    if not protocol_config.read(ROOT/'train.conf', encoding='utf-8'):
        raise FileNotFoundError(f'No se pudo leer la configuración: {ROOT / "train.conf"}')
    # protocol_train y protocol_test: identidades del reparto, antes del muestreo por presupuestos.
    protocol_train, protocol_test = split_video_pools(
        df[['video_id']], protocol_config.getint('training', 'seed'))
    # protocol_overlap: comprobación de separación de identidades, no de canales.
    protocol_overlap = len(set(protocol_train.video_id) & set(protocol_test.video_id))
    display(Markdown(
        f'**Reserva inicial por vídeo, semilla {protocol_config.getint("training", "seed")}:** '
        f'desarrollo: **{format_number(len(protocol_train), 0)} apariciones / {format_number(protocol_train.video_id.nunique(), 0)} vídeos**; '
        f'prueba: **{format_number(len(protocol_test), 0)} apariciones / {format_number(protocol_test.video_id.nunique(), 0)} vídeos**. '
        f'Vídeos compartidos: **{format_number(protocol_overlap, 0)}**. '
        'Se reserva el 20 % de los grupos de vídeo; el porcentaje de filas puede diferir. '
        f'Presupuestos configurados de filas: búsqueda **{format_number(protocol_config.getint("training", "max_rows"), 0)}**, '
        f'ajuste final **{format_number(protocol_config.getint("training", "final_rows"), 0)}**, '
        f'prueba **{format_number(protocol_config.getint("training", "test_rows"), 0)}**; '
        f'validación cruzada: **{protocol_config.getint("training", "folds")} particiones**. '
        'Son presupuestos aproximados: se conservan vídeos completos y se mostrarán '
        'los tamaños efectivos en los resultados. Desarrollo contiene los datos '
        'para búsqueda, validación interna y ajuste final; prueba queda separada de esas decisiones. '
        'Esta separación no garantiza canales distintos ni generalización temporal, '
        'y la prueba interna ya se ha consultado durante el desarrollo.'
    ))
    # Particiones: explica el procedimiento con la cantidad configurada, sin fijarla a tres.
    display(Markdown(
        '**Cómo funciona la validación cruzada.** Para cada tarea, la muestra de búsqueda se reparte en **{parts} partes**, manteniendo juntas las apariciones de un mismo vídeo. Se hacen **{parts} rondas**: en cada una se deja una parte para validar (comprobar el modelo) y se usa el resto para ajustar (aprender de los datos). Cada parte se usa una vez para validar. Las partes pueden tener tamaños diferentes. La reserva de prueba permanece fuera de este proceso.'.format(
            parts=format_number(protocol_config.getint('training', 'folds'), 0))
    ))
    # training_data: reproduce el muestreo de búsqueda con vídeos completos.
    from training_data import group_sample
    # cv_pool: columnas mínimas para describir grupos y objetivos, sin ajustar transformadores.
    cv_pool = df[['video_id', 'category', 'comments_disabled', 'ratings_disabled', 'likes', 'dislikes']].copy()
    cv_pool['ratio'] = (cv_pool.likes / cv_pool.dislikes.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
    cv_development, _ = split_video_pools(cv_pool, protocol_config.getint('training', 'seed'))
    cv_search = group_sample(cv_development, protocol_config.getint('training', 'max_rows'),
                             protocol_config.getint('training', 'seed')).reset_index(drop=True)
    # cv_rows: controles calculados por tarea y partición; no son métricas de modelos.
    cv_rows = []
    for cv_task in ['category', 'comments_disabled', 'ratings_disabled', 'likes', 'ratio']:
        cv_data = cv_search.dropna(subset=[cv_task]).reset_index(drop=True)
        cv_classification = cv_task in ['category', 'comments_disabled', 'ratings_disabled']
        cv_target = cv_data[cv_task] if cv_task == 'category' else pd.to_numeric(cv_data[cv_task]).astype(int if cv_classification else float)
        cv_splits = _protocol_group_folds(
            cv_data, cv_target, protocol_config.getint('training', 'folds'),
            cv_classification, protocol_config.getint('training', 'seed'), cv_task, display)
        cv_seen = np.zeros(len(cv_data), dtype=int)
        for cv_number, (cv_fit, cv_valid) in enumerate(cv_splits, 1):
            cv_overlap = len(set(cv_data.video_id.iloc[cv_fit]) & set(cv_data.video_id.iloc[cv_valid]))
            if cv_overlap:
                raise ValueError(f'Fuga por vídeo en la validación cruzada: {cv_task}, partición {cv_number}')
            cv_seen[cv_valid] += 1
            cv_rows.append({'tarea': cv_task, 'fold': cv_number,
                            'filas ajuste': len(cv_fit), 'filas validación': len(cv_valid),
                            'vídeos compartidos': cv_overlap,
                            'clases ausentes en ajuste': len(set(cv_target)-set(cv_target.iloc[cv_fit])) if cv_classification else None,
                            'clases ausentes en validación': len(set(cv_target)-set(cv_target.iloc[cv_valid])) if cv_classification else None})
        if not np.all(cv_seen == 1):
            raise ValueError(f'La validación debe cubrir cada fila exactamente una vez: {cv_task}')
    cv_summary = pd.DataFrame(cv_rows)
    display(cv_summary.rename(columns={'fold': 'partición'}))
    display(Markdown(
        f'**Validación cruzada:** {format_number(len(cv_summary), 0)} combinaciones de tarea y partición comprobadas '
        f'sobre una muestra inicial de búsqueda de {format_number(len(cv_search), 0)} apariciones. '
        f'Solapamientos de vídeo entre ajuste y validación: **{format_number(int(cv_summary["vídeos compartidos"].sum()), 0)}**. '
        f'Particiones de clasificación con alguna clase ausente en validación: '
        f'**{format_number(int(cv_summary["clases ausentes en validación"].fillna(0).gt(0).sum()), 0)}**. '
        f'Particiones de clasificación con alguna clase ausente en ajuste: '
        f'**{format_number(int(cv_summary["clases ausentes en ajuste"].fillna(0).gt(0).sum()), 0)}**. '
        'Cada fila elegible de una tarea valida exactamente una vez. La estratificación intenta '
        'aproximar proporciones de clases conservando vídeos completos; no garantiza todas '
        'las clases en cada partición. TF-IDF, imputación y escala se aprenden dentro del ajuste '
        'de cada partición, mediante el Pipeline. Esta tabla comprueba el reparto; no entrena '
        'modelos ni mide su calidad. La reserva de prueba no participa en estas particiones.'
    ))
    # Liberar las copias del diagnóstico antes de ejecutar los experimentos.
    del cv_pool, cv_development, _, cv_search, cv_data, cv_target, cv_seen

    # Liberar el reparto explicativo: el experimento lo reconstruye con la misma función y semilla.
    del protocol_train, protocol_test



def prepare_numeric_projections(sample, seed=42):
    """Calcula PCA y t-SNE sobre cuatro contadores log-escalados, sin entrenar modelos finales.

    Exige tres vídeos distintos, contadores numéricos no negativos y alguna
    variación. Mantiene la política descriptiva de rellenar ausentes con cero y
    registra cuántos; no reutiliza el imputador del modelo ni demuestra separación.
    """
    # sklearn.preprocessing: iguala las escalas de los cuatro contadores tras log1p.
    from sklearn.preprocessing import StandardScaler
    # sklearn.decomposition: obtiene dos direcciones lineales de máxima varianza.
    from sklearn.decomposition import PCA
    # sklearn.manifold: explora vecindarios mediante una representación no lineal.
    from sklearn.manifold import TSNE
    columns = ['views','likes','dislikes','comment_count']
    if len(sample)<3 or not set(columns+['video_id','cluster']).issubset(sample.columns):
        raise ValueError('PCA/t-SNE requieren al menos tres vídeos y las columnas de contadores, identidad y grupo')
    if sample.video_id.isna().any() or sample.video_id.astype(str).str.strip().eq('').any() or sample.video_id.duplicated().any():
        raise ValueError('La muestra de proyecciones debe contener identidades utilizables y únicas')
    labels = pd.to_numeric(sample.cluster,errors='coerce').to_numpy(dtype=float)
    if not np.isfinite(labels).all() or (labels<0).any() or not np.equal(labels,np.floor(labels)).all():
        raise ValueError('Los grupos de las proyecciones deben ser identificadores enteros no negativos')
    original = sample[columns]
    numeric = original.apply(pd.to_numeric,errors='coerce')
    if (original.notna() & numeric.isna()).any().any() or np.isinf(numeric.to_numpy(dtype=float)).any() or (numeric<0).any().any():
        raise ValueError('Los contadores de PCA/t-SNE deben ser numéricos, finitos y no negativos; sólo se admiten ausentes explícitos')
    missing = int(numeric.isna().sum().sum())
    numeric = numeric.fillna(0)
    scaled = StandardScaler().fit_transform(np.log1p(numeric))
    constant = [column for column,values in zip(columns,scaled.T) if np.ptp(values)==0]
    if len(constant)==len(columns):
        raise ValueError('No hay variación en los contadores: PCA/t-SNE no producirían una proyección informativa')
    pca = PCA(n_components=2,random_state=seed).fit(scaled)
    projection = pca.transform(scaled)
    perplexity = min(30,len(sample)-1)
    tsne = TSNE(n_components=2,perplexity=perplexity,max_iter=500,init='pca',random_state=seed)
    embedding = tsne.fit_transform(scaled)
    if not np.isfinite(projection).all() or not np.isfinite(embedding).all() or not np.isfinite(pca.explained_variance_ratio_).all():
        raise ValueError('PCA/t-SNE no han producido coordenadas y varianzas finitas')
    stats = dict(n=len(sample),columns=columns,missing_filled_with_zero=missing,
                 constant_columns=constant,seed=seed,perplexity=perplexity,max_iter=500,
                 explained_variance_ratio=pca.explained_variance_ratio_.tolist(),
                 loadings=pca.components_.T.tolist(),
                 tsne_kl_divergence=float(tsne.kl_divergence_))
    return dict(numeric=numeric,pca=pca,projection=projection,embedding=embedding,stats=stats)


def project_model_coordinates(reduced, labels, categories, seed=42):
    """Proyecta las coordenadas SVD y conserva una única geometría para ambos colores.

    Valida dimensiones, categorías y grupos. El PCA descriptivo se ajusta aquí;
    no se reajustan preprocesador, SVD ni clustering y no se selecciona con colores.
    """
    # sklearn.decomposition: obtiene dos ejes descriptivos de las coordenadas SVD.
    from sklearn.decomposition import PCA
    reduced = np.asarray(reduced,dtype=float)
    labels = np.asarray(labels)
    categories = pd.Series(categories).reset_index(drop=True)
    if reduced.ndim!=2 or min(reduced.shape)<2 or not np.isfinite(reduced).all():
        raise ValueError('La proyección del modelo requiere al menos dos vídeos y dos componentes SVD finitos')
    if labels.ndim!=1 or len(labels)!=len(reduced) or len(categories)!=len(reduced):
        raise ValueError('Las etiquetas y categorías no coinciden con las filas del SVD')
    numeric_labels = pd.to_numeric(pd.Series(labels),errors='coerce').to_numpy(dtype=float)
    if not np.isfinite(numeric_labels).all() or (numeric_labels<0).any() or not np.equal(numeric_labels,np.floor(numeric_labels)).all():
        raise ValueError('Los grupos del modelo deben ser enteros no negativos')
    if categories.isna().any() or categories.astype(str).str.strip().eq('').any():
        raise ValueError('Faltan categorías para colorear el espacio del modelo')
    if not np.any(np.ptp(reduced,axis=0)>0):
        raise ValueError('El espacio SVD no tiene variación que permita un PCA informativo')
    pca = PCA(n_components=2,random_state=seed).fit(reduced)
    projection = pca.transform(reduced)
    if not np.isfinite(projection).all() or not np.isfinite(pca.explained_variance_ratio_).all():
        raise ValueError('La proyección del modelo no produjo coordenadas y varianzas finitas')
    return dict(projection=projection,labels=numeric_labels.astype(int),categories=pd.Categorical(categories),
                stats=dict(n=len(reduced),svd_dimensions=reduced.shape[1],
                           groups=len(np.unique(numeric_labels)),categories=categories.nunique(),
                           explained_variance_ratio=pca.explained_variance_ratio_.tolist(),seed=seed))


def prepare_model_space_view(root):
    """Reconstruye el corpus canónico y contrasta sus etiquetas con el pipeline guardado.

    Reutiliza partición y muestreo del entrenamiento. Sólo usa transform/predict
    del pipeline final; el único fit es el PCA descriptivo posterior al SVD.
    """
    # configparser: recupera rutas, presupuestos y semilla declarados por la práctica.
    import configparser
    # models y training_data: cargan pesos y reutilizan el protocolo de muestra canónico.
    from models import REGISTRY
    from models.core import features
    from training_data import split_video_pools, group_sample
    config = configparser.ConfigParser()
    config.read(root/'train.conf',encoding='utf-8')
    seed = config.getint('training','seed')
    data = pd.read_csv(root/config['data']['input'])
    development,_ = split_video_pools(data,seed)
    final = group_sample(development,config.getint('training','final_rows'),seed)
    unique = final.drop_duplicates('video_id').reset_index(drop=True)
    sample = unique.sample(n=min(config.getint('training','cluster_rows'),len(unique)),random_state=seed).reset_index(drop=True)
    model = REGISTRY['cluster']().load(root/config['output']['cluster'])
    pipeline = model.pipeline_
    matrix = pipeline.named_steps['features'].transform(features(sample))
    reduced = pipeline.named_steps['reduce'].transform(matrix)
    labels = pipeline.named_steps['estimator'].predict(reduced)
    saved = pd.read_csv(root/'reports/cluster_sample.csv')
    expected = sample[['video_id','category','views','likes','dislikes','comment_count']].copy()
    expected.insert(2,'cluster',labels)
    try:
        pd.testing.assert_frame_equal(expected,saved,check_dtype=False,rtol=1e-9,atol=1e-8)
    except AssertionError as exc:
        raise ValueError('La muestra o los grupos publicados no corresponden al modelo guardado y al muestreo configurado') from exc
    view = project_model_coordinates(reduced,labels,sample.category,seed)
    # Cruce descriptivo del mismo corpus contrastado, sin nuevo ajuste.
    view['region_counts']=pd.crosstab(labels,sample.region).rename_axis('cluster')
    view['stats'].update(transformed_features=matrix.shape[1],
                         svd_explained_variance=float(pipeline.named_steps['reduce'].explained_variance_ratio_.sum()),
                         sample_and_labels_verified=True,pipeline_refit=False)
    return view


def draw_model_space_view(view, *, category_label=str):
    """Dibuja las mismas coordenadas con colores discretos por grupo y por categoría.

    Comparte límites de ejes. La leyenda editorial enumera sus categorías sin
    interpolar colores como si sus códigos representaran magnitudes continuas.
    """
    # matplotlib.colors y lines: asignan un color discreto a cada etiqueta y su leyenda.
    from matplotlib.colors import BoundaryNorm
    from matplotlib.lines import Line2D
    coordinates = view['projection']
    labels = view['labels']
    categories = view['categories']
    groups, codes = np.unique(labels,return_inverse=True)
    fig, axes = plt.subplots(1,2,figsize=(16,6),sharex=True,sharey=True)
    palette = plt.get_cmap('tab20').resampled(len(groups))
    bounds = np.arange(len(groups)+1)-.5
    first = axes[0].scatter(*coordinates.T,c=codes,cmap=palette,norm=BoundaryNorm(bounds,palette.N),s=9,alpha=.6)
    bar = fig.colorbar(first,ax=axes[0],ticks=np.arange(len(groups)),boundaries=bounds,label='Grupo')
    bar.ax.set_yticklabels([str(value) for value in groups])
    count = len(categories.categories)
    categorical_palette = plt.get_cmap('tab20').resampled(count)
    axes[1].scatter(*coordinates.T,c=categories.codes,cmap=categorical_palette,
                    norm=BoundaryNorm(np.arange(count+1)-.5,count),s=9,alpha=.5)
    handles = [Line2D([],[],marker='o',linestyle='',color=categorical_palette(i),label=category_label(str(category)),markersize=5)
               for i,category in enumerate(categories.categories)]
    axes[1].legend(handles=handles,loc='center left',bbox_to_anchor=(1,.5),fontsize=7)
    ratios = view['stats']['explained_variance_ratio']
    for axis,title in zip(axes,['Grupos en el espacio SVD del modelo','Categorías en las mismas coordenadas']):
        axis.set(title=title,xlabel=f'PCA 1 del SVD ({format_number(ratios[0]*100,1)} %)',
                 ylabel=f'PCA 2 del SVD ({format_number(ratios[1]*100,1)} %)')
    return fig


def draw_cluster_selection(report):
    """Dibuja ensayos por familia y distingue selección sin unir puntos heterogéneos.

    Valida el ganador con el narrador canónico. Los colores son dimensiones SVD;
    forma y estrella identifican admisibilidad y solución guardada, sin ajustar.
    """
    # results_report: contrasta criterios y ganador antes de presentar la figura.
    from results_report import cluster_selection_plot_summary
    # matplotlib.lines: crea las claves de la leyenda sin añadir curvas de datos.
    from matplotlib.lines import Line2D
    cluster_selection_plot_summary(report)
    rows = [row for row in report['clustering'] if row['name'] not in ['agglomerative','dbscan'] and row['silhouette'] is not None]
    if any(not isinstance(row['k'],int) or row['k']<1 or not isinstance(row['dimensions'],int) or row['dimensions']<1 for row in rows):
        raise ValueError('El gráfico requiere números positivos de grupos y dimensiones SVD')
    families = sorted({row['name'] for row in rows})
    dimensions = sorted({row['dimensions'] for row in rows})
    columns = min(2,len(families))
    height = (len(families)+columns-1)//columns
    fig,axes = plt.subplots(height,columns,figsize=(12,4*height),sharex=True,sharey=True,squeeze=False)
    palette = plt.get_cmap('tab10').resampled(len(dimensions))
    selected = report['cluster_selected']
    for axis,family in zip(axes.flat,families):
        trials = [row for row in rows if row['name']==family]
        for index,dimension in enumerate(dimensions):
            for admitted,marker in [(True,'o'),(False,'x')]:
                subset = [row for row in trials if row['dimensions']==dimension and bool(row['selectable'])==admitted]
                if subset:
                    axis.scatter([row['k'] for row in subset],[row['silhouette'] for row in subset],
                                 color=palette(index),marker=marker,s=55,alpha=.8)
        if family==selected['name']:
            axis.scatter([selected['k']],[selected['silhouette']],marker='*',s=260,
                         color='gold',edgecolors='black',zorder=5)
            axis.annotate('Seleccionado',(selected['k'],selected['silhouette']),xytext=(9,10),textcoords='offset points',fontsize=9)
        axis.set(title=f'{family}: {len(trials)} ensayos',xlabel='Número de grupos (k observado)',ylabel='Silueta en índices comunes')
        axis.set_xticks(sorted({row['k'] for row in rows}))
        axis.grid(alpha=.2)
    for axis in list(axes.flat)[len(families):]:
        axis.set_visible(False)
    handles = [Line2D([],[],marker='o',linestyle='',color=palette(index),label=f'SVD={dimension}') for index,dimension in enumerate(dimensions)]
    handles += [Line2D([],[],marker='o',linestyle='',color='black',label='Cumple la regla'),
                Line2D([],[],marker='x',linestyle='',color='black',label='No cumple la regla'),
                Line2D([],[],marker='*',linestyle='',color='gold',markeredgecolor='black',markersize=12,label='Seleccionado')]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.955),ncol=min(3,len(handles)),fontsize=9)
    # El título reserva espacio para la leyenda incluso después de tight_layout.
    fig.suptitle('Comparación de familias inductivas\n\n\n',fontsize=12)
    return fig


def recommendation_protocol(root, result, return_frames=False):
    """Reconstruye particiones de recuperación y contrasta catálogo final sin fit.

    Reutiliza el muestreo del pool de desarrollo; comprueba identidades disjuntas
    entre catálogo y consultas. Contrasta el orden de consultas de prueba y el
    catálogo ampliado guardado. No reproduce métricas ni reajusta las variantes. return_frames permite devolver
    también catálogo comparativo y consultas de prueba para una auditoría separada.
    """
    # configparser: recupera presupuestos y semilla de entrenamiento.
    import configparser
    # sklearn.model_selection: reproduce exactamente las divisiones de índices.
    from sklearn.model_selection import train_test_split
    # training_data y models: reutilizan muestreo canónico y carga de pesos.
    from training_data import split_video_pools, group_sample
    from models import REGISTRY
    config = configparser.ConfigParser()
    config.read(root/'train.conf',encoding='utf-8')
    seed = config.getint('training','seed')
    data = pd.read_csv(root/config['data']['input'])
    development,_ = split_video_pools(data,seed)
    final = group_sample(development,config.getint('training','final_rows'),seed)
    unique = final.drop_duplicates('video_id').reset_index(drop=True)
    all_catalog = unique.sample(n=min(config.getint('training','recommend_rows'),len(unique)),random_state=seed).reset_index(drop=True)
    if len(all_catalog)<9 or all_catalog.video_id.isna().any() or all_catalog.video_id.astype(str).str.strip().eq('').any() or all_catalog.video_id.duplicated().any():
        raise ValueError('El protocolo de recomendación necesita vídeos utilizables únicos y tamaño suficiente para sus tres particiones')
    catalog_idx, query_idx = train_test_split(np.arange(len(all_catalog)),test_size=.4,random_state=seed)
    validation_idx,test_idx = train_test_split(query_idx,test_size=.5,random_state=seed)
    catalog = all_catalog.iloc[catalog_idx]
    validation = all_catalog.iloc[validation_idx].sample(n=min(300,len(validation_idx)),random_state=seed)
    test = all_catalog.iloc[test_idx].sample(n=min(300,len(test_idx)),random_state=seed)
    pools = [set(frame.video_id) for frame in [catalog,validation,test]]
    overlap = [len(pools[0]&pools[1]),len(pools[0]&pools[2]),len(pools[1]&pools[2])]
    if any(overlap):
        raise ValueError('Catálogo, validación y prueba de recomendación comparten identidades')
    diagnostics = pd.read_csv(root/'reports/recommendation_query_diagnostics.csv')
    if 'query_id' not in diagnostics or list(diagnostics.query_id)!=list(test.video_id):
        raise ValueError('Las consultas de prueba publicadas no coinciden con el muestreo configurado')
    model = REGISTRY['recommend']().load(root/config['output']['recommend'])
    if list(model.catalog_.video_id)!=list(all_catalog.video_id):
        raise ValueError('El catálogo del recomendador guardado no corresponde al catálogo ampliado reconstruido')
    selected = model.manifest_['training'].get('selected')
    if selected!=result['selected']:
        raise ValueError('La variante guardada del recomendador no coincide con el informe')
    stats = dict(seed=seed,unique_development_videos=len(unique),final_catalog=len(all_catalog),
                 selection_catalog=len(catalog),validation_pool=len(validation_idx),test_pool=len(test_idx),
                 validation_queries=len(validation),test_queries=len(test),pairwise_overlap=overlap,
                 unused_in_comparison=len(all_catalog)-len(catalog)-len(validation)-len(test),
                 test_query_order_verified=True,final_catalog_order_verified=True,selected=selected,
                 final_analyzer=model.vectorizer_.analyzer,final_ngram_range=list(model.vectorizer_.ngram_range),
                 final_content_mode=model.content_options_['mode'],model_refit=False)
    return (stats,catalog,test) if return_frames else stats


def recommendation_control_audit(result, diagnostics, catalog, seed=42):
    """Reproduce referencia marginal e intervalo percentil del protocolo original.

    Mantiene catálogo y modelo fijos, con 500 remuestreos de consultas completas.
    No recalibra, no selecciona variantes ni trata cinco vecinos como observaciones
    independientes. Devuelve evidencia y medias bootstrap para una figura descriptiva.
    """
    # results_report: valida primero denominadores y medias de las consultas.
    from results_report import recommendation_metrics_summary
    recommendation_metrics_summary(result,diagnostics)
    if type(seed) is not int or seed<0:
        raise ValueError('La semilla del bootstrap debe ser un entero no negativo')
    if not {'video_id','category'}.issubset(catalog.columns) or len(catalog)!=result['catalog_videos']:
        raise ValueError('El catálogo de la referencia aleatoria no tiene columnas o tamaño compatibles')
    for frame,column in [(catalog,'video_id'),(catalog,'category'),(diagnostics,'category')]:
        if column not in frame or frame[column].isna().any() or frame[column].astype(str).str.strip().eq('').any():
            raise ValueError('Faltan identidades o categorías para la referencia aleatoria')
    if catalog.video_id.duplicated().any() or set(catalog.video_id)&set(diagnostics.query_id):
        raise ValueError('La referencia requiere catálogo único y consultas disjuntas')
    frequencies=catalog.category.value_counts(normalize=True)
    expected=diagnostics.category.map(frequencies).fillna(0).to_numpy(dtype=float)
    stored=pd.to_numeric(diagnostics.random_category_expectation,errors='coerce').to_numpy()
    if not np.allclose(expected,stored,rtol=1e-9,atol=1e-10):
        raise ValueError('Las probabilidades aleatorias publicadas no coinciden con las frecuencias del catálogo')
    values=diagnostics.category_agreement_at_5_PROXY.to_numpy(dtype=float)
    baseline=float(expected.mean());observed=float(values.mean());gain=observed-baseline
    controls=result['controls']
    for field,value in [('random_category_expectation',baseline),('proxy_gain_over_random_expectation',gain)]:
        saved=controls.get(field)
        if not isinstance(saved,(float,int)) or not np.isfinite(saved) or not np.isclose(saved,value,rtol=1e-9,atol=1e-10):
            raise ValueError(f'El control {field} no coincide con las consultas y el catálogo')
    saved=np.asarray(controls.get('proxy_ci95'),dtype=float)
    if saved.shape!=(2,) or not np.isfinite(saved).all() or not 0<=saved[0]<=saved[1]<=1:
        raise ValueError('El intervalo proxy debe tener dos límites finitos ordenados entre cero y uno')
    rng=np.random.default_rng(seed)
    bootstrap=values[rng.integers(0,len(values),size=(500,len(values)))].mean(axis=1)
    interval=np.quantile(bootstrap,[.025,.975])
    if not np.allclose(saved,interval,rtol=1e-9,atol=1e-10):
        raise ValueError('El intervalo publicado no coincide con las 500 réplicas del protocolo declarado')
    return dict(status='passed',queries=len(values),catalog=len(catalog),seed=seed,replicas=500,
                observed_proxy=observed,random_expectation=baseline,gain=gain,ci95=interval.tolist(),
                bootstrap_means=bootstrap.tolist(),constant_query_scores=bool(np.ptp(values)==0),
                catalog_and_model_fixed=True,model_fits=0)


def draw_recommendation_bootstrap(audit):
    """Muestra medias remuestreadas y percentiles, sin interpretarlos como predicciones.

    La distribución es condicional a consultas y modelo; no incluye selección.
    """
    # matplotlib.ticker: presenta proporciones como porcentajes en el eje horizontal.
    from matplotlib.ticker import PercentFormatter
    if audit.get('status')!='passed':
        raise ValueError('No se dibuja un bootstrap que no haya superado la auditoría')
    fig,axis=plt.subplots(figsize=(10,4))
    lower,upper=audit['ci95']
    axis.hist(audit['bootstrap_means'],bins=20,color='#4386a8',edgecolor='white')
    axis.axvspan(lower,upper,color='#dda740',alpha=.2,label='Intervalo percentil nominal 95 %')
    axis.axvline(lower,color='#9a6200',linestyle='--')
    axis.axvline(upper,color='#9a6200',linestyle='--')
    axis.axvline(audit['observed_proxy'],color='black',label='Media proxy observada')
    axis.xaxis.set_major_formatter(PercentFormatter(1))
    axis.set(title=f'Bootstrap de consultas: {audit["replicas"]} réplicas, catálogo/modelo fijos',
             xlabel='Media de coincidencia editorial proxy',ylabel='Número de réplicas')
    axis.legend(fontsize=9)
    return fig


def show_narrative_details(title, text, shown=None):
    """Muestra conclusiones a la vista y conserva la explicación técnica desplegable.

    El texto se calcula y valida antes de presentarlo; no oculta errores ni
    elimina evidencia. La etiqueta se escapa para evitar interpretarla como HTML.
    shown, si se indica, reúne los párrafos de detalle ya mostrados en la misma
    serie de tareas: una cautela idéntica sólo se despliega la primera vez. Las
    conclusiones visibles y los párrafos con cifras propios de cada tarea se conservan.
    """
    # html: mantiene el título como texto aunque contenga caracteres especiales.
    from html import escape
    # conclusions: párrafos decisivos a la vista; el resto permanece desplegable.
    conclusions = conclusion_paragraphs(text)
    for conclusion in conclusions:
        display(Markdown(conclusion))
    detail = '\n\n'.join(paragraph for paragraph in text.split('\n\n') if paragraph not in conclusions)
    if shown is not None:
        detail = drop_repeated(detail, shown)
    if not detail.strip():
        return
    display(Markdown('<details><summary>'+escape(title)+'</summary>\n\n'+detail+'\n\n</details>'))


def drop_repeated(text, shown):
    """Elimina párrafos idénticos a otros ya mostrados y registra los nuevos en shown.

    Sólo se omiten repeticiones exactas: cualquier párrafo con cifras o matices
    propios de la tarea es distinto y se conserva. La primera aparición queda completa.
    """
    paragraphs = [paragraph for paragraph in text.split('\n\n') if paragraph.strip() not in shown]
    shown.update(paragraph.strip() for paragraph in paragraphs if paragraph.strip())
    return '\n\n'.join(paragraphs)


# matplotlib.ticker: conserva el cálculo original de marcas y localiza sus etiquetas.
from matplotlib.ticker import Formatter


class _SpanishFormatter(Formatter):
    """Envuelve un formateador de ejes conservando escalas, posiciones y offset."""
    def __init__(self, original):
        """Conserva el formateador científico original y su estado de cálculo."""
        self.original = original

    def __call__(self, value, pos=None):
        """Localiza la etiqueta producida por el formateador original."""
        return spanish_tick(self.original(value, pos))

    def set_locs(self, locs):
        """Actualiza posiciones en ambos formateadores sin recalcular los datos."""
        super().set_locs(locs)
        self.original.set_locs(locs)

    def get_offset(self):
        """Conserva y localiza el factor de escala mostrado junto al eje."""
        return spanish_tick(self.original.get_offset())


def spanish_tick(label):
    """Localiza etiquetas numéricas; conserva fechas, nombres y fórmulas de escala."""
    # re: reconoce números completos o decimales dentro del MathText de matplotlib.
    import re
    text = str(label)
    numeric = re.fullmatch(r'([−+\-]?)(\d+)(?:\.(\d+))?(%)?', text)
    if numeric:
        sign, integer, fraction, percent = numeric.groups()
        result = sign + format_number(int(integer), 0)
        if fraction is not None:
            result += ',' + fraction
        return result + (' %' if percent else '')
    # MathText puede contener decimales; los rótulos de categorías ya están localizados.
    if '$' in text:
        return re.sub(r'(\d+)\.(\d+)', lambda match: match[1] + ',' + match[2], text)
    return text


def localize_figure(figure):
    """Aplica el formato español a ejes y barras de color sin alterar sus escalas."""
    for legend in figure.legends:
        for label in legend.get_texts():
            label.set_text(clarify_narrative(label.get_text()))
    for axis in figure.axes:
        # Los rótulos usan la misma terminología que las explicaciones del cuaderno.
        axis.set_title(clarify_narrative(axis.get_title()))
        axis.set_xlabel(clarify_narrative(axis.get_xlabel()))
        axis.set_ylabel(clarify_narrative(axis.get_ylabel()))
        legend = axis.get_legend()
        if legend is not None:
            for label in legend.get_texts():
                label.set_text(clarify_narrative(label.get_text()))
        for coordinate in [axis.xaxis, axis.yaxis]:
            formatter = coordinate.get_major_formatter()
            if not isinstance(formatter, _SpanishFormatter):
                coordinate.set_major_formatter(_SpanishFormatter(formatter))
            minor = coordinate.get_minor_formatter()
            if not isinstance(minor, _SpanishFormatter):
                coordinate.set_minor_formatter(_SpanishFormatter(minor))


def spanish_table_html(table, *, index=True):
    """Genera HTML escapado y números españoles; index=False oculta el índice auxiliar.

    Conserva la tabla recibida y el comportamiento anterior cuando index=True.
    """
    frame = table.to_frame() if isinstance(table, pd.Series) else table
    # presentation: copia textual exclusiva para mostrar; no modifica tipos de entrada.
    presentation = frame.map(human_value)
    # Sólo los índices sintéticos del informe de clasificación reciben nombres legibles.
    # Las etiquetas de las clases y los datos originales se conservan.
    if {'precision', 'recall', 'f1-score', 'support'}.issubset(frame.columns):
        summary_names = {'accuracy': 'exactitud global', 'macro avg': 'media por clase',
                         'weighted avg': 'media ponderada por casos'}
        presentation.index = presentation.index.map(lambda value: summary_names.get(value, value))
    if isinstance(presentation.columns, pd.MultiIndex):
        presentation.columns = pd.MultiIndex.from_tuples(
            [tuple(clarify_narrative(human_value(value)) for value in key) for key in presentation.columns],
            names=presentation.columns.names)
    else:
        presentation.columns = presentation.columns.map(lambda value: clarify_narrative(human_value(value)))
    if isinstance(presentation.index, pd.MultiIndex):
        presentation.index = pd.MultiIndex.from_tuples(
            [tuple(human_value(value) for value in key) for key in presentation.index],
            names=presentation.index.names)
    else:
        presentation.index = presentation.index.map(human_value)
    # La auditoría usa tres cabeceras alineadas y columnas que permiten envolver texto.
    if list(presentation.columns) == ['Descripción', 'Valor'] and presentation.index.name == 'Campo técnico':
        presentation = presentation.reset_index()
        html = presentation.to_html(index=False, escape=True, border=0)
        html = html.replace('<table ', '<table style="width:100%;min-width:760px;table-layout:fixed;" ', 1)
        html = html.replace('<thead>', '<colgroup><col style="width:25%"><col style="width:32%"><col style="width:43%"></colgroup><thead>', 1)
        cell_style = 'text-align:left;vertical-align:top;white-space:normal;overflow-wrap:anywhere;padding:8px 10px;line-height:1.4;'
        html = html.replace('<th>', '<th style="'+cell_style+'">').replace('<td>', '<td style="'+cell_style+'">')
        return '<div style="width:100%;max-width:100%;overflow-x:auto;">'+html+'</div>'
    # column_descriptions: significado de los identificadores usados por las tablas técnicas.
    column_descriptions = {
        'model': 'modelo o referencia', 'cv_score': 'puntuación media en validación cruzada',
        'cv_std': 'desviación entre particiones', 'cv_train_score': 'puntuación media en ajuste',
        'fold_scores': 'puntuación en cada partición', 'seconds': 'duración en segundos',
        'convergence_review': 'revisión del ajuste y sus avisos', 'params': 'parámetros utilizados',
        'warnings': 'avisos registrados', 'error': 'mensaje de error',
        'candidate': 'identificador del ensayo', 'reference': 'configuración de referencia',
        'mean_cv_score': 'puntuación media de validación', 'std_cv_score': 'desviación de validación',
        'mean_train_score': 'puntuación media de ajuste', 'mean_fit_time': 'tiempo medio de ajuste',
        'mean_score_time': 'tiempo medio de evaluación', 'train_score': 'puntuación de ajuste',
        'iteration_parameters': 'presupuesto de iteraciones', 'search_rows': 'filas de búsqueda',
        'folds': 'número de particiones', 'precision': 'precisión de positivos',
        'recall': 'recuperación de casos reales', 'f1-score': 'F1', 'support': 'casos reales',
        'macro_f1': 'F1 medio con igual peso por clase', 'balanced_accuracy': 'exactitud equilibrada',
        'mae': 'error absoluto medio', 'rmse': 'raíz del error cuadrático medio',
        'r2': 'coeficiente de determinación',
    }
    # Una segunda fila aclara el encabezado sin renombrar columnas del objeto original.
    if not isinstance(frame.columns, pd.MultiIndex) and any(key in column_descriptions for key in frame.columns):
        presentation.columns = pd.MultiIndex.from_tuples([
            (human_value(original), column_descriptions.get(original, human_value(readable)))
            for original, readable in zip(frame.columns, presentation.columns)
        ])
    return presentation.to_html(escape=True, index=index)


def display_spanish(*objects, **kwargs):
    """Muestra tablas localizadas y delega Markdown, imágenes y otros objetos a IPython."""
    for item in objects:
        if isinstance(item, Markdown):
            item = Markdown(clarify_narrative(item.data))
        if isinstance(item, (pd.DataFrame, pd.Series)):
            _native_display(HTML(spanish_table_html(item)), **kwargs)
        else:
            _native_display(item, **kwargs)


# display: mismo punto de presentación en todas las funciones de este módulo.
display = display_spanish


# _EXPLAINED_FAMILIES: explicaciones de familia ya mostradas en esta sesión del cuaderno.
_EXPLAINED_FAMILIES = set()


def comparison_conclusion(comparison, *, regression=False):
    """Justifica la comparación observada y explica cambios de parámetros, sin usar test."""
    valid = sorted([row for row in comparison['trials'] if np.isfinite(row.get('cv_score', np.nan))],
                   key=lambda row: row['cv_score'], reverse=True)
    winner = valid[0]
    lines = ['**Conclusión de la selección.** Se elige **' + winner['model'] + '** por la mejor '
             + ('media de −MAE' if regression else 'media de macro-F1') + ' en los mismos folds de desarrollo.']
    if len(valid) > 1:
        gap = winner['cv_score'] - valid[1]['cv_score']
        lines.append(f'La ventaja sobre **{valid[1]["model"]}** es **{format_number(gap,4)}** '
                     + ('unidades de MAE.' if regression else 'puntos de macro-F1.'))
    family = winner['model']
    if family in ['svm', 'svr_log_target', 'ridge_log_target', 'logistic']:
        explanation = ('La familia seleccionada combina una respuesta lineal regularizada con atributos '
                       'TF-IDF: muchos atributos textuales dispersos, de los que cada vídeo activa pocos. '
                       'Esto permite acumular señales de términos sin estimar todas sus interacciones. '
                       + ('El objetivo log1p reduce la escala de los extremos durante el ajuste y la '
                          'predicción vuelve a la escala original.' if family.endswith('_log_target') else ''))
        # La explicación completa se da la primera vez; después se remite a ella sin repetirla.
        if explanation in _EXPLAINED_FAMILIES:
            lines.append('Es también una familia lineal regularizada con atributos TF-IDF; '
                         'su funcionamiento se explica en el primer reto donde se selecciona.')
        else:
            _EXPLAINED_FAMILIES.add(explanation)
            lines.append(explanation)
    alternatives = [row for row in valid[1:] if any(token in row['model'] for token in ['rbf','forest','xgb'])]
    if alternatives:
        lines.append('Alternativas no lineales contrastadas: ' + '; '.join(
            row['model'] + ' = ' + format_number((-1 if regression else 1)*row['cv_score'],4)
            for row in alternatives) + '. Aquí no superan la media ganadora con las rejillas y '
            'presupuestos utilizados; no demuestra inferioridad general ni aísla la causa de la diferencia.')
    reference = next(row for row in comparison['refinement']['trials'] if row['reference'])['params']
    selected = comparison['refinement']['params']
    descriptions = {'C':'inverso de la regularización; mayor C reduce la regularización relativa y penaliza más los errores',
                    'alpha':'intensidad de la regularización; mayor alpha reduce más los coeficientes',
                    'class_weight':'ponderación de clases; balanced compensa su frecuencia',
                    'max_features':'límite de términos del vocabulario',
                    'min_df':'mínimo de documentos en los que debe aparecer un término',
                    'ngram_range':'longitud mínima y máxima de secuencias de palabras',
                    'sublinear_tf':'sustituye el recuento por una frecuencia logarítmica',
                    'max_iter':'límite de iteraciones para permitir la convergencia'}
    changes = [key for key in selected if selected[key] != reference.get(key)]
    if changes:
        lines.append('**Qué cambia en el refinamiento:** ' + '; '.join(
            key + ': ' + (human_value(reference[key]) if key in reference else 'no registrado en la referencia') + ' → ' + human_value(selected[key])
            + ' (' + descriptions.get(key.split('__')[-1], 'parámetro de la configuración') + ')'
            for key in changes) + '. Los límites de iteraciones pueden proceder de la revisión de '
            'convergencia. Al variar varios parámetros no se atribuye la mejora a uno de ellos por separado.')
    else:
        lines.append('El refinamiento conserva los parámetros de referencia: las alternativas '
                     'ensayadas no mejoran su media de validación.')
    return '\n\n'.join(lines)


def draw_date_views(delay, calendar, cutoff=20, *, publication_hours=None):
    """Dibuja una vista principal por calendario y su cola, sin recortar datos.

    La diferencia con hora sólo se resume como diagnóstico opcional; no genera
    otro histograma. publication_hours aporta las franjas UTC (0 a 23) de las
    barras, conservando su altura en apariciones. Devuelve los recuentos de ambas
    medidas para contrastar la precisión temporal, sin cambiar las Series.
    """
    if cutoff not in (20, 30):
        raise ValueError('El límite de la vista temporal debe ser 20 o 30 días')
    if not all(isinstance(values, pd.Series) for values in (delay, calendar)) or not delay.index.equals(calendar.index):
        raise ValueError('Las dos medidas temporales deben ser Series con los mismos índices')
    if not delay.isna().equals(calendar.isna()):
        raise ValueError('Las medidas temporales deben compartir los mismos pares válidos')
    # hour_bands: cuatro franjas UTC, asignadas sólo a pares temporales válidos.
    hour_labels = ['00:00–06:00', '06:00–12:00', '12:00–18:00', '18:00–24:00']
    hour_bands = None
    if publication_hours is not None:
        if not isinstance(publication_hours, pd.Series) or not publication_hours.index.equals(delay.index):
            raise ValueError('Las horas UTC deben tener los mismos índices que las fechas')
        valid_hours = publication_hours.loc[delay.notna()]
        try:
            numeric_hours = valid_hours.to_numpy(dtype=float)
        except (ValueError, TypeError) as exc:
            raise ValueError('Las horas UTC deben ser enteros entre 0 y 23') from exc
        if (not np.isfinite(numeric_hours).all() or (numeric_hours < 0).any()
                or (numeric_hours >= 24).any() or not np.equal(numeric_hours,np.floor(numeric_hours)).all()):
            raise ValueError('Las horas UTC deben ser enteros entre 0 y 23 en cada par válido')
        hour_bands = pd.Series(pd.NA, index=delay.index, dtype='Int64')
        hour_bands.loc[delay.notna()] = (valid_hours // 6).astype(int)
    # bounds: extremos abiertos a la izquierda y cerrados a la derecha de la cola.
    bounds = [bound for bound in (30, 60, 365) if bound > cutoff]
    labels = [f'({left}, {right}] días' for left, right in zip([cutoff]+bounds[:-1], bounds)] + [f'Más de {bounds[-1]} días']
    panels = []
    for name, values in [('A', delay), ('B', calendar)]:
        finite = values.dropna()
        if finite.empty or not np.isfinite(finite.to_numpy(dtype=float)).all():
            raise ValueError('No hay valores temporales finitos suficientes para dibujar')
        if name == 'B' and not np.equal(finite, np.floor(finite)).all():
            raise ValueError('La diferencia por calendario debe contener días enteros')
        lower = -1 if name == 'A' else 0
        window = finite.loc[finite.between(lower, cutoff, inclusive='both')]
        tail = [int(((finite > left) & (finite <= right)).sum()) for left, right in zip([cutoff]+bounds[:-1], bounds)]
        tail.append(int(finite.gt(bounds[-1]).sum()))
        underflow = int(finite.lt(lower).sum())
        if len(window)+sum(tail)+underflow != len(finite):
            raise ValueError('La vista temporal y sus colas no suman los pares válidos')
        panels.append({'name': name, 'lower': lower, 'valid': len(finite), 'visible': len(window),
                       'negative': int(finite.lt(0).sum()), 'underflow': underflow,
                       'tail': tail, 'max': float(finite.max())})
    # calendar_panel: única distribución principal; la medida A queda como diagnóstico.
    panel = panels[1]
    window = calendar.loc[calendar.between(0,cutoff,inclusive='both')]
    positions = np.arange(0,cutoff+1)
    figure = plt.figure(figsize=(11,7.5),layout='constrained')
    grid = figure.add_gridspec(2,1,height_ratios=[1.6,1.15])
    axis = figure.add_subplot(grid[0,0])
    hour_colors = plt.get_cmap('Blues')(np.array([.28,.48,.68,.88]))
    legend_handles, pending_percentages = [], []
    if hour_bands is None:
        counts=window.astype(int).value_counts().reindex(positions,fill_value=0)
        axis.bar(positions,counts.values,width=.85,color='#297aa3')
    else:
        # counts: cada columna representa un día y cada fila una franja UTC.
        selected_bands=hour_bands.loc[calendar.between(0,cutoff,inclusive='both')]
        counts=np.asarray([window.loc[selected_bands.eq(band)].astype(int).value_counts()
                           .reindex(positions,fill_value=0).to_numpy() for band in range(4)],dtype=int)
        totals=counts.sum(axis=0)
        if int(totals.sum())!=panel['visible']:
            raise ValueError('Las franjas horarias no suman las apariciones del calendario')
        panel['hour_totals']=counts.sum(axis=1).tolist()
        panel['hour_rows']=[]
        bottom=np.zeros(len(positions))
        for band in range(4):
            bars=axis.bar(positions,counts[band],bottom=bottom,width=.85,
                          color=hour_colors[band],edgecolor='white',linewidth=.25)
            legend_handles.append(bars)
            for column,bar in enumerate(bars):
                if totals[column]:
                    percent=float(counts[band,column]/totals[column]*100)
                    panel['hour_rows'].append({'Intervalo de días':str(positions[column]),
                                              'Franja UTC':hour_labels[band],
                                              'Apariciones':int(counts[band,column]),
                                              'Porcentaje de la barra':percent,
                                              'Total de la barra':int(totals[column])})
                    if counts[band,column]:pending_percentages.append((bar,percent,band))
            bottom+=counts[band]
        panel['hour_rows'].sort(key=lambda row:(int(row['Intervalo de días']),hour_labels.index(row['Franja UTC'])))
    axis.axvline(0,color='#953e27',linestyle='--',linewidth=1,zorder=0)
    axis.set(title=f'Días entre publicación y aparición en tendencias\nDetalle de 0 a {cutoff} días de calendario',
             xlim=(-.5,cutoff+.7),xlabel='Días del calendario',ylabel='Apariciones',
             xticks=[0]+list(range(5,cutoff+1,5)))
    axis.text(.98,.96,format_number(panel['visible'],0)+' apariciones\n'+format_number(panel['visible']/panel['valid']*100,2)+' % de los pares válidos',
              transform=axis.transAxes,ha='right',va='top',fontsize=9,
              bbox={'facecolor':'white','alpha':.9,'edgecolor':'none'})
    # La cola pertenece al mismo calendario; no se duplica con otro panel horario.
    tail_axis=figure.add_subplot(grid[1,0])
    display_labels=(['Días anteriores a 0'] if panel['underflow'] else [])+labels
    tail_values=([panel['underflow']] if panel['underflow'] else [])+panel['tail']
    bars=tail_axis.bar(np.arange(len(display_labels)),tail_values,width=.6,color='#996230')
    for bar,count in zip(bars,tail_values):
        tail_axis.annotate(format_number(count,0)+'\n'+format_number(count/panel['valid']*100,2)+' %',
                           (bar.get_x()+bar.get_width()/2,bar.get_height()),xytext=(0,4),
                           textcoords='offset points',ha='center',va='bottom',fontsize=9,fontweight='bold')
    tail_axis.set_xticks(np.arange(len(display_labels)),display_labels)
    tail_axis.set(title='Fuera de la vista principal: intervalos sin recortar valores',ylabel='Apariciones')
    tail_axis.set_ylim(0,max(1,max(tail_values,default=0)*1.4))
    if hour_bands is not None:
        figure.legend(legend_handles,hour_labels,loc='outside upper center',ncol=4,
                      title='Hora de publicación UTC (inicio incluido; fin excluido)',fontsize=9,title_fontsize=10)
        figure.canvas.draw()
        for bar,percent,band in pending_percentages:
            pixels=bar.get_height()/max(np.diff(axis.get_ylim())[0],1)*axis.bbox.height
            width_pixels=bar.get_width()/np.diff(axis.get_xlim())[0]*axis.bbox.width
            if pixels>=16 and width_pixels>=22:
                text=axis.text(bar.get_x()+bar.get_width()/2,bar.get_y()+bar.get_height()/2,
                               format_number(percent,0)+' %',ha='center',va='center',fontsize=8.5,
                               fontweight='bold',color='black' if band<2 else 'white',clip_on=True)
                box=text.get_window_extent(renderer=figure.canvas.get_renderer())
                if box.width>width_pixels-2 and box.height<=width_pixels-2:
                    text.set_rotation(90)
                    box=text.get_window_extent(renderer=figure.canvas.get_renderer())
                if box.width>width_pixels-2 or box.height>pixels-2:text.remove()
    return {'cutoff':cutoff,'lower':0,'main':'B','tail_labels':labels,'panels':panels,
            'shared_tail':panels[0]['tail']==panel['tail'],
            'hour_labels':hour_labels if hour_bands is not None else []}

