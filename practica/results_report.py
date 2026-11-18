"""Conclusiones calculadas a partir de las evidencias, sin métricas prefijadas."""
# metric_utils: comparte el cálculo de F2 y validaciones de matrices de confusión.
# Se importa: f2_from_precision_recall, f2_from_confusion.
from metric_utils import f2_from_precision_recall, f2_from_confusion
# pathlib: maneja rutas de archivos y carpetas mediante objetos, como Path.
# Se importa: Path.
from pathlib import Path
# json: lee y escribe datos estructurados, configuraciones y evidencias en JSON.
import json
# pandas: organiza, limpia y analiza datos tabulares mediante DataFrame y Series.
import pandas as pd



# TASK_DESCRIPTIONS: preguntas de las siete tareas, para rótulos comunes y coherentes.
TASK_DESCRIPTIONS = {
    'category': 'categoría del vídeo', 'comments_disabled': 'comentarios desactivados',
    'ratings_disabled': 'valoraciones desactivadas', 'likes': 'número de «me gusta»',
    'ratio': 'proporción «me gusta/no me gusta»', 'cluster': 'segmentación de vídeos',
    'recommend': 'recomendación de vídeos similares',
}


def task_description(task):
    """Devuelve una etiqueta de lectura; conserva el código de una tarea desconocida."""
    return TASK_DESCRIPTIONS.get(task, str(task))


def clarify_narrative(text):
    """Expresa la prosa en español sin modificar código, enlaces o cifras.

    Cada fragmento se traduce en una sola pasada con prioridad para las frases
    completas. Así una sustitución no rompe la concordancia de otra. Después
    se ajustan las concordancias conocidas. Se conservan bloques de código,
    identificadores técnicos y destinos de enlaces; la operación es idempotente.
    """
    # re: reconoce frases completas y protege los fragmentos técnicos.
    import re
    # replacements y agreements: traducciones y concordancias revisadas.
    replacements = [('los folds comparten parte del entrenamiento y son pocos',
      'las particiones comparten parte del entrenamiento y su número es reducido'),
     ('de la segmentación ya calculado', 'de la segmentación ya calculada'),
     ('todos los folds', 'todas las particiones'),
     ('los mismos folds', 'las mismas particiones'),
     ('estos folds', 'estas particiones'),
     ('los folds', 'las particiones'),
     ('cada fold', 'cada partición'),
     ('del fold', 'de la partición'),
     ('el fold', 'la partición'),
     ('un fold', 'una partición'),
     ('folds agrupados', 'particiones agrupadas'),
     ('folds efectivos', 'particiones efectivas'),
     ('folds compartidos', 'particiones compartidas'),
     ('folds', 'particiones'),
     ('fold', 'partición'),
     ('ese pipeline', 'esa cadena de procesamiento'),
     ('todo el pipeline', 'toda la cadena de procesamiento'),
     ('del pipeline', 'de la cadena de procesamiento'),
     ('el pipeline', 'la cadena de procesamiento'),
     ('un pipeline', 'una cadena de procesamiento'),
     ('pipeline', 'cadena de procesamiento'),
     ('del holdout', 'de la reserva de prueba'),
     ('el holdout', 'la reserva de prueba'),
     ('un holdout', 'una reserva de prueba'),
     ('holdout', 'reserva de prueba'),
     ('hold-out', 'reserva de prueba'),
     ('pool', 'conjunto'),
     ('un recall alto', 'una recuperación alta'),
     ('el recall positivo', 'la recuperación de positivos'),
     ('recall positivo', 'recuperación de positivos'),
     ('del recall', 'de la recuperación'),
     ('el recall', 'la recuperación'),
     ('recall', 'recuperación'),
     ('balanced accuracy', 'exactitud equilibrada'),
     ('accuracy', 'exactitud'),
     ('precision', 'precisión'),
     ('support', 'casos reales'),
     ('f1-score', 'F1'),
     ('scores negativos', 'puntuaciones negativas'),
     ('score negativo', 'puntuación negativa'),
     ('los scores', 'las puntuaciones'),
     ('del score', 'de la puntuación'),
     ('el score', 'la puntuación'),
     ('un score', 'una puntuación'),
     ('scores', 'puntuaciones'),
     ('score', 'puntuación'),
     ('el baseline entrenado', 'la referencia sencilla entrenada'),
     ('el baseline', 'la referencia sencilla'),
     ('del baseline', 'de la referencia sencilla'),
     ('un test interno', 'una prueba interna'),
     ('del clustering', 'de la segmentación'),
     ('el clustering', 'la segmentación'),
     ('un clustering', 'una segmentación'),
     ('baseline', 'referencia sencilla'),
     ('bootstrap', 'remuestreo con reemplazo'),
     ('ranking', 'listado ordenado'),
     ('clustering', 'segmentación'),
     ('cluster', 'grupo'),
     ('silhouette', 'silueta'),
     ('bandwidth', 'ancho de banda'),
     ('features', 'atributos'),
     ('feature', 'atributo'),
     ('target', 'objetivo'),
     ('test', 'prueba'),
     ('del clustering ya calculado', 'de la segmentación ya calculada')]
    agreements = [('cadena de procesamiento guardado', 'cadena de procesamiento guardada'),
     ('cadena de procesamiento completo', 'cadena de procesamiento completa'),
     ('cadena de procesamiento seleccionado', 'cadena de procesamiento seleccionada'),
     ('cadena de procesamiento ajustado', 'cadena de procesamiento ajustada'),
     ('cadena de procesamiento congelado', 'cadena de procesamiento congelada'),
     ('particiones agrupados', 'particiones agrupadas'),
     ('particiones ampliados', 'particiones ampliadas'),
     ('particiones distintos', 'particiones distintas'),
     ('particiones separados', 'particiones separadas'),
     ('puntuaciones finitos', 'puntuaciones finitas'),
     ('puntuaciones positivos', 'puntuaciones positivas'),
     ('precisión/recall', 'precisión/recuperación'),
     ('precision/recall', 'precisión/recuperación'),
     ('a recuperación', 'a la recuperación'),
     ('el recuperación', 'la recuperación'),
     ('del recuperación', 'de la recuperación'),
     ('del segmentación', 'de la segmentación'),
     ('el segmentación', 'la segmentación'),
     ('el prueba', 'la prueba'),
     ('del prueba', 'de la prueba'),
     ('segmentación ya calculado', 'segmentación ya calculada'),
     ('un prueba interno', 'una prueba interna'),
     ('un prueba interno examinado', 'una prueba interna ya examinada'),
     ('el referencia sencilla entrenado', 'la referencia sencilla entrenada'),
     ('ese cadena de procesamiento', 'esa cadena de procesamiento'),
     ('los puntuaciones', 'las puntuaciones')]
    # contexts y prefixes: conservan artículos y demostrativos al traducir sustantivos.
    contexts={'holdout':'reserva de prueba','pipeline':'cadena de procesamiento',
              'clustering':'segmentación','baseline':'referencia sencilla','recall':'recuperación'}
    prefixes={'el':'la','del':'de la','al':'a la','un':'una','este':'esta','ese':'esa'}
    for english,spanish in contexts.items():
        for prefix,agreement in prefixes.items():
            replacements.append((prefix+' '+english,agreement+' '+spanish))
    # translators: cada expresión elige primero la frase más larga compatible.
    translators=[]
    for pairs in (replacements,agreements):
        lookup={old.casefold():new for old,new in pairs}
        alternatives='|'.join(re.escape(key) for key in sorted(lookup,key=len,reverse=True))
        pattern=re.compile(r'(?<![\w/])(?:'+alternatives+r')(?![\w/])',re.IGNORECASE)
        translators.append((lookup,pattern))
    # protected: el contenido ejecutable y las rutas conservan su texto exacto.
    protected=r'(```[\s\S]*?```|~~~[\s\S]*?~~~|`[^`\n]*`|!?\[[^\]]*\]\([^)]+\)|<[A-Za-z/!][^>]*>)'
    fragments=re.split(protected,str(text))
    for index in range(0,len(fragments),2):
        fragment=fragments[index]
        for lookup,pattern in translators:
            def translated(match):
                """Conserva la mayúscula inicial de la frase localizada."""
                value=lookup[match[0].casefold()]
                return value[0].upper()+value[1:] if match[0][0].isupper() else value
            fragment=pattern.sub(translated,fragment)
        fragments[index]=fragment
    return ''.join(fragments)


def format_number(value, decimals=3):
    """Formatea cifras de lectura humana con coma decimal y punto de millares.

    No modifica datos, JSON ni CSV destinados al procesamiento numérico.
    """
    return f'{value:,.{decimals}f}'.translate(str.maketrans({',': '.', '.': ','}))


def format_signed_number(value, decimals=3):
    """Añade signo explícito a una diferencia con formato español."""
    return ('+' if value >= 0 else '−') + format_number(abs(value), decimals)


def human_value(value):
    """Representa números y estructuras anidadas para lectura, sin mutar entradas.

    Conserva textos, identificadores y versiones; los formatos CSV/JSON siguen
    siendo técnicos. Los números muy pequeños usan notación científica con coma.
    """
    # numbers y math: distinguen escalares reales y conservan valores no finitos.
    from numbers import Integral, Real
    import math
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, Integral):
        return format_number(value, 0)
    if isinstance(value, Real):
        if not math.isfinite(value):
            return '—' if math.isnan(value) else ('−∞' if value < 0 else '∞')
        if value and abs(value) < 0.000001:
            return f'{value:.6g}'.replace('.', ',')
        return format_number(value, 6).rstrip('0').rstrip(',')
    if isinstance(value, dict):
        return '; '.join(str(key) + ': ' + human_value(item) for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return '[' + '; '.join(human_value(item) for item in value) + ']'
    if value is None:
        return 'None'
    return str(value)


def highlight_number(value, decimals=0, suffix=''):
    """Destaca una cifra ya formateada y su unidad en una salida Markdown."""
    return f'**{format_number(value, decimals)}{suffix}**'


def warning_groups(result):
    """Distingue convergencia pendiente, resuelta y otros avisos de las búsquedas.

    Usa los tipos ya clasificados en convergence_review; no infiere el tipo por
    el nombre del modelo ni atribuye avisos de toda una rejilla al ganador.
    """
    groups = {'remaining_convergence': [], 'resolved_convergence': [], 'other_warnings': []}
    for trial in result.get('trials', []):
        review = trial.get('convergence_review', {})
        remaining = set(review.get('remaining_convergence_warnings', []))
        convergence = set(remaining)
        warnings = set(trial.get('warnings', []))
        for attempt in review.get('attempts', []):
            convergence.update(attempt.get('convergence_warnings', []))
            warnings.update(attempt.get('warnings', []))
        if remaining:
            groups['remaining_convergence'].append(trial['model'])
        elif convergence:
            groups['resolved_convergence'].append(trial['model'])
        if warnings - convergence:
            groups['other_warnings'].append(trial['model'])
    return groups


def cluster_balance_note(profiles):
    """Interpreta tamaños observados sin cambiar la regla de selección geométrica."""
    total = int(profiles.n.sum())
    largest = float(profiles.n.max() / total)
    smallest = int(profiles.n.min())
    return (f'Distribución de tamaños: el grupo mayor reúne {format_number((largest)*100, 1)} % de los {total} vídeos; el menor contiene {smallest}. '
            + ('La solución es muy desequilibrada: silhouette favorable no acredita una segmentación temática o comercial útil. Debe leerse como exploración y contrastarse con alternativas y requisitos de tamaño/estabilidad definidos para el negocio.'
               if largest > .95 else 'Los tamaños deben valorarse junto con perfiles, estabilidad y el objetivo de negocio; silhouette no valida utilidad por sí solo.'))


def cluster_stability_text(report, quality=None):
    """Explica réplicas de la solución elegida y, si existe, del pipeline completo.

    No selecciona modelos. Separa SVD fijo del reajuste del preprocesamiento y
    muestra la dispersión sin calificarla automáticamente como estabilidad alta.
    quality debe corresponder al mismo experimento; se omite antes del diagnóstico.
    """
    selected = report['cluster_selected']
    trial = next((trial for trial in report['clustering']
                  if trial.get('selectable', trial.get('eligible')) and trial['name'] == selected['name']
                  and trial['dimensions'] == selected['dimensions']
                  and trial.get('k') == selected['k'] and trial.get('silhouette') == selected['silhouette']), None)
    if trial is None:
        raise ValueError('No hay un ensayo elegible que respalde la segmentación seleccionada')
    replicas = trial.get('stability_ARI', [])
    text = f"Silhouette de la solución seleccionada: **{format_number(selected['silhouette'], 3)}**. "
    if replicas:
        text += (f"Estabilidad con SVD fijo: ARI medio **{format_number(sum(replicas) / len(replicas), 3)}**; "
                 f"{len(replicas)} réplicas [{'; '.join(format_number(value, 3) for value in replicas)}], "
                 f"rango {format_number(min(replicas), 3)}–{format_number(max(replicas), 3)}. "
                 'La media debe leerse junto con las réplicas: no acredita reproducción uniforme ni utilidad de los grupos. ')
    else:
        text += 'No hay réplicas válidas de estabilidad con SVD fijo. '
    full = (quality or {}).get('full_pipeline_stability', {})
    replicas = full.get('replicate_ARI', [])
    if replicas:
        text += (f"Al reajustar todo el preprocesamiento con la receta congelada, ARI medio "
                 f"**{format_number(sum(replicas) / len(replicas), 3)}**; réplicas "
                 f"[{'; '.join(format_number(value, 3) for value in replicas)}]. "
                 f"El diagnóstico utiliza {full['training_videos']} vídeos de entrenamiento y "
                 f"{full['audit_videos']} de auditoría interna, con "
                 f"{full['audit_training_video_overlap']} IDs compartidos. No es validación externa. ")
    return text + 'Pocas réplicas limitan la interpretación; el balance se describe con los tamaños reales de los perfiles. Estabilidad, silueta y utilidad son conceptos distintos.'


def cluster_interpretation_text(report, profiles, quality=None):
    """Compone la interpretación académica desde resultados actuales, sin archivos.

    La misma fuente se muestra junto a las gráficas y se publica en el informe.
    quality se añade después del diagnóstico para evitar cifras de otro estudio.
    """
    interpretation = report['cluster_interpretation']
    ari, nmi = interpretation['ARI'], interpretation['NMI']
    association = ('La coincidencia con categorías editoriales es débil: los grupos no recuperan géneros de forma fiable.'
                   if ari < .1 else 'Existe asociación con categorías, pero no equivalencia entre particiones.')
    lines = ['# Interpretación de la segmentación', '',
             f'ARI frente a categoría: **{format_number(ari, 4)}**; NMI: **{format_number(nmi, 4)}**. {association}', '',
             'ARI=0 corresponde aproximadamente al acuerdo esperado al azar; NMI mide información compartida. Las categorías no eligen el modelo ni el número de grupos.', '',
             'Las proyecciones comparten coordenadas del espacio TF-IDF + metadatos + SVD del modelo. La proyección 2D pierde información y no sustituye silhouette.', '',
             'cluster_category_proportions.csv guarda fracciones entre 0 y 1 por grupo; el cuaderno las muestra también como porcentajes. cluster_terms.csv contiene términos reconstruidos desde centros o medias de la familia; son aproximaciones descriptivas y pueden reflejar plantillas o sistemas de escritura.', '',
             '## Criterio de selección', '', localize_policy(report.get('clustering_selection_policy', 'Criterio histórico de silueta.')), '',
             '## Estabilidad medida', '', cluster_stability_text(report, quality), '',
             '## Tamaños y utilidad', '', cluster_balance_note(profiles), '']
    return '\n'.join(lines)



def measured_limitations(root, r):
    """Compone límites por canal y F2 desde evidencias sin recalibrar en prueba."""
    quality = json.loads((root/'reports/quality_checks.json').read_text(encoding='utf-8'))
    comments = quality['tasks']['comments_disabled']
    ratings = r['tasks']['ratings_disabled']
    # F2 del umbral original se obtiene de su matriz, sin recalibrar el modelo.
    matrix = ratings['default_threshold_holdout']['confusion_matrix']
    ratings_default_f2 = f2_from_confusion(matrix)
    return [
        '## Resultados que condicionan la interpretación', '',
        f"En el diagnóstico interno por canales disjuntos, comments_disabled obtiene macro-F1 {format_number(comments['test']['macro_f1'],3)}, frente a la referencia {format_number(comments['baseline_test']['macro_f1'],3)}. Se reajusta la receta con otra partición y calibración: no es una evaluación directa de los pesos finales. La evaluación por vídeo y el diagnóstico por canal evalúan condiciones distintas; ninguno constituye validación externa prospectiva.", '',
        f"En ratings_disabled, F2 pasa de {format_number(ratings_default_f2,3)} con el umbral predeterminado a {format_number(ratings['positive_f2'],3)} con el umbral OOF. Se conserva la política elegida en entrenamiento y se informa de su resultado; no se cambia el umbral mirando el test.", '',
        'La estrategia de calibración se revisó durante el desarrollo tras examinar resultados internos; se conserva la trazabilidad de ambas etapas y no constituye validación externa prospectiva.', '',
        'La selección de familias y el refinamiento usan macro-F1; el umbral posterior optimiza F2 en OOF. Son objetivos distintos. Comparar con AP dentro de CV es una mejora experimental pendiente.', '']



# RETOS_OPCIONALES reúne las tareas que el enunciado ofrece para subir nota.
RETOS_OPCIONALES = frozenset({'comments_disabled', 'ratings_disabled', 'ratio'})
ALCANCE_RETOS = (
    'La pregunta guía del cuaderno se formuló a partir del enunciado. '
    'Los retos pedidos abarcan categoría, número de likes, segmentación y recomendación. '
    'Los retos opcionales para subir nota abarcan la deshabilitación de comentarios o valoraciones '
    'y la proporción de likes y dislikes. La práctica aborda comentarios y valoraciones por separado. '
    'El [glosario de la práctica](../GUIA_DE_CONSULTA.md#glosario) explica las familias, las métricas y los términos del protocolo.'
)


def main():
    """Interpreta métricas existentes sin volver a seleccionar modelos con el test.

    Escribe results.md y devuelve su texto; informa también resultados negativos."""
    # root: directorio de la práctica, independiente del directorio de invocación.
    root = Path(__file__).resolve().parent
    # r: informe evaluation.json del que se derivan todas las conclusiones.
    r = json.loads((root/'reports/evaluation.json').read_text(encoding='utf-8'))
    # regional_supports: casos reales por clase para evitar conclusiones sobre clases ausentes.
    regional_supports = regional_class_supports(root, r)
    lines = ['# Informe de resultados y decisiones', '', ALCANCE_RETOS, '',
        f"Datos limpios: **{format_number(r['full_rows'],0)} filas**. Búsqueda: **{format_number(r['sampling_rows'],0)}**; ajuste final ampliado: **{format_number(r['final_train_rows'],0)}**; prueba: **{format_number(r['holdout_rows'],0)}**. La muestra de búsqueda queda incluida en el ajuste final. Ningún video_id cruza entrenamiento y prueba.", '',
        'La muestra se estratifica aproximadamente por categoría conservando grupos completos; representativeness.csv cuantifica diferencias por región, categoría y etiquetas minoritarias. La prueba representa otros vídeos de tendencias del mismo periodo, no vídeos futuros ni canales nuevos.', '',
        'La prueba interna se ha examinado durante el desarrollo y no constituye una validación externa prospectiva.', '',
        '| Objetivo | Enunciado | Modelo | Entrenamiento | Prueba | Referencia en la misma prueba | IC 95 % por vídeo |',
        '|---|---|---|---:|---:|---:|---|']
    for task,result in r['tasks'].items():
        # classification: indicador que determina métrica y familia de estimadores.
        classification = 'macro_f1' in result['holdout']
        # metric: nombre de métrica y convención de comparación de la tarea.
        metric = 'macro_f1' if classification else 'mae'
        # ci: intervalo bootstrap por vídeo, condicional al modelo ajustado.
        ci = result['confidence_interval']
        scope = 'Opcional para subir nota' if task in RETOS_OPCIONALES else 'Pedido'
        lines.append(f"| {task} | {scope} | {result['selected']} | {format_number(result['train'][metric],3)} | {format_number(result['holdout'][metric],3)} | {format_number(result['baseline_holdout'][metric],3)} | [{format_number(ci['low'],3)}, {format_number(ci['high'],3)}] |")
    lines += ['', 'Clasificación: macro-F1, mayor es mejor. Regresión: MAE en unidades originales, menor es mejor. Los intervalos remuestrean vídeos y son condicionales al ajuste; no incluyen incertidumbre de selección ni cambio temporal.', '',
        'El macro-F1 de train/test y su intervalo usan la unión de etiquetas de ajuste y prueba, con zero_division=0. Una clase sin soporte de prueba aporta cero por esa convención y no permite medir su fiabilidad. La CV usa el scorer sklearn f1_macro con la unión de clases presentes en y_true o y_pred de cada fold. Los candidatos comparten folds, pero no necesariamente las clases promediadas: predecir una clase ausente añade un F1 cero. La escasez de clases también afecta a la variación. No se ha cambiado esta convención para conservar el experimento registrado.', '', '## Decisiones por problema', '']
    for task,result in r['tasks'].items():
        lines.append(f'### {task}')
        reduced = [f"{trial['model']}: SVD {trial['params']['reduce__n_components']}"
                   for trial in result['trials'] if 'reduce__n_components' in trial.get('params', {})]
        if reduced:
            lines.append('Comparación condicionada por la representación: ' + '; '.join(reduced)
                         + '. Los candidatos sin SVD usan la representación TF-IDF y sus metadatos. Se comparan recetas completas, por lo que las diferencias combinan algoritmo y representación. El refinamiento explora configuraciones de la familia ganadora dentro del presupuesto descrito.')
        # best: tupla con puntuación, nombre, estimador y diagnóstico de la mejor familia.
        best = next(t for t in result['trials'] if t['model']==result['selected'])
        if result.get('refinement'):
            refined = result['refinement']
            row = refined['trials'][refined['best_index']]
            best = dict(best, cv_score=refined['cv_score'], cv_std=row['std_cv_score'], fold_scores=row['fold_scores'], params=refined['params'])
            lines.append(f"Segunda fase: {len(refined['trials'])} configuraciones, incluida la referencia; CV inicial {format_number(refined['initial_cv_score'],4)}, refinada {format_number(refined['cv_score'],4)}. Las rejillas y los avisos están en evaluation.json y se muestran en el cuaderno. La selección no consulta la prueba.")
        lines.append(f"CV seleccionada: {format_number(best['cv_score'],4)} ± {format_number(best['cv_std'],4)}; folds {human_value(best['fold_scores'])}. Parámetros: `{best['params']}`. El signo de la MAE en CV es negativo por convención de sklearn.")
        if task in ['comments_disabled','ratings_disabled']:
            # report: informe del experimento con métricas, protocolo y procedencia.
            report = result['holdout']['classification_report']
            # positive: métricas de la clase positiva o coeficientes positivos del análisis local.
            positive = report.get('1',report.get('1.0',{}))
            # base_positive: precisión y recall positivos con el umbral predeterminado.
            base_positive = result['default_threshold_holdout']['classification_report'].get('1',{})
            # selection: diagnóstico del conjunto OOF utilizado para elegir el umbral.
            selection=result.get('threshold_selection') or {}
            lines.append(f"Umbral elegido únicamente con OOF en {selection.get('pool','muestra de búsqueda')} ({format_number(selection.get('oof_rows',result['search_rows']),0)} filas) para maximizar F2 (prioridad académica a la recuperación). Recall positivo en prueba: {format_number((positive.get('recall',0))*100,1)} %, frente a {format_number((base_positive.get('recall',0))*100,1)} % con el umbral predeterminado; precisión {format_number((positive.get('precision',0))*100,1)} %, soporte {format_number(positive.get('support',0),0)}. AP {format_number(result['average_precision'],3)}, referencia aleatoria por prevalencia {format_number(result['positive_prevalence'],3)}.")
            if selection.get('warnings'):
                lines.append(f"Avisos de los ajustes OOF de calibración: {selection['warnings']}. Se mantienen como límite del experimento.")
            # p0: precisión positiva con el umbral predeterminado; r0: recall positivo con el umbral
            # predeterminado.
            p0,r0=base_positive.get('precision',0),base_positive.get('recall',0)
            # default_f2: F2 calculado con el umbral original para contrastar el resultado OOF.
            default_f2=float(f2_from_precision_recall(p0, r0))
            # tuned_cm: matriz de prueba con el umbral elegido en entrenamiento.
            # default_cm: matriz de prueba con el umbral predeterminado.
            tuned_cm=result['holdout']['confusion_matrix']
            default_cm=result['default_threshold_holdout']['confusion_matrix']
            lines.append(f"F2 de prueba: {format_number(result['positive_f2'],3)} con el umbral OOF y {format_number(default_f2,3)} con el predeterminado. Falsos positivos: {tuned_cm[0][1]} frente a {default_cm[0][1]}; verdaderos positivos: {tuned_cm[1][1]} frente a {default_cm[1][1]}. El cambio de recall debe leerse junto con la precisión y F2; por sí solo no acredita una mejora de la utilidad.")
            if result['positive_f2'] < default_f2:
                lines.append('El objetivo F2 optimizado en OOF no mejora en esta reserva de prueba. La variabilidad de la minoría y el traslado entre modelos OOF y ajuste completo limitan la calibración. Un siguiente experimento debería separar una partición de calibración dentro del entrenamiento o usar validación anidada, acordar costes y validarse externamente. Se conserva la política predefinida y se informa el fallo; no se elige de nuevo mirando esta prueba.')
            if 'search_threshold_positive_f2' in result:
                lines.append(f"La calibración inicial sobre la muestra de búsqueda obtuvo F2 de prueba {format_number(result['search_threshold_positive_f2'],3)}; se conserva como diagnóstico. La fase final utiliza OOF del pool ampliado, sin seleccionar el umbral con las etiquetas de prueba.")
            lines.append('El balance entre falsas alarmas y omisiones depende de la tarea: no se propone automatizar decisiones sobre creadores. Una aplicación de cribado necesitaría revisión humana, costes explícitos y nueva validación. Las curvas PR y ambas matrices permiten valorar el intercambio sin seleccionar de nuevo en test.')
        elif task == 'category':
            lines.append(f"Macro-F1 de prueba {format_number(result['holdout']['macro_f1'],3)}, frente a {format_number(result['baseline_holdout']['macro_f1'],3)} del clasificador mayoritario. La diferencia con entrenamiento es {format_signed_number(result['train']['macro_f1']-result['holdout']['macro_f1'], 3)}: debe leerse junto con la curva de aprendizaje y el soporte por clase. No todas las categorías tienen la misma fiabilidad.")
            if result['unseen_test_labels']:
                lines.append(f"Categorías ausentes del ajuste final: {result['unseen_test_labels']}. No cabe esperar acierto en una categoría no aprendida.")
            # supported: clases con observaciones reales en el conjunto de prueba.
            supported=[label for label in result['holdout']['labels'] if result['holdout']['classification_report'].get(label,{}).get('support',0)>0]
            # unsupported: clases sin soporte de prueba; su F1 cero es una convención explícita.
            unsupported=sorted(set(result['holdout']['labels'])-set(supported))
            if unsupported:
                # mean_supported: macro-F1 complementario restringido a las clases observadas.
                mean_supported=sum(result['holdout']['classification_report'][label]['f1-score'] for label in supported)/len(supported)
                # baseline_supported: macro-F1 de referencia sobre esas mismas clases observadas.
                baseline_supported=sum(result['baseline_holdout']['classification_report'][label]['f1-score'] for label in supported)/len(supported)
                lines.append(f"Clases sin soporte en esta prueba: {unsupported}. Como lectura complementaria, macro-F1 sólo sobre las {len(supported)} clases observadas: {format_number(mean_supported,3)}, referencia {format_number(baseline_supported,3)}. Esta alternativa no reemplaza la definición global ni su intervalo bootstrap.")
            # classes: clases con soporte suficiente para interpretar errores por categoría.
            classes = [(name,value) for name,value in result['holdout']['classification_report'].items() if isinstance(value,dict) and value.get('support',0)>=20 and name not in ['macro avg','weighted avg']]
            # weakest: categorías con menor F1 entre las que tienen soporte mínimo.
            weakest = sorted(classes,key=lambda item:item[1]['f1-score'])[:3]
            lines.append('Categorías con menor F1 y al menos 20 ejemplos: '+ '; '.join(f"{name}: F1 {format_number(value['f1-score'],3)}, recall {format_number((value['recall'])*100,1)} %, n={format_number(value['support'],0)}" for name,value in weakest)+'. Priorizar estos errores para revisión asistida; un promedio global no acredita clasificación fiable de todas las categorías.')
        else:
            # regression_summary: comparación segura de errores y lectura en escala original.
            lines.append(regression_summary(task, result))
            lines.append('La diferencia RMSE–MAE refleja sensibilidad a errores extremos; los 100 mayores errores se entregan con título, región y objetivo real. Un R² bajo o negativo sigue siendo un límite aunque la MAE mejore respecto a la mediana.')
            if task == 'ratio':
                lines.append(ratio_ablation_summary(result))
        # regions: tabla de métricas regionales con sus tamaños.
        regions=pd.read_csv(root/f'reports/{task}_by_region.csv')
        lines.append(regional_summary(task, result, regions, regional_supports))
        # coeff_path: CSV de coeficientes asociados a atributos del modelo seleccionado.
        coeff_path=root/f'reports/{task}_coefficients.csv'
        if coeff_path.exists():
            # coefficients: extracto por salida, con etiquetas textuales y escalas explícitas.
            coefficients=pd.read_csv(coeff_path,dtype={'class':str,'feature':str})
            lines.append(coefficient_summary(task, result, coefficients))
        # learning_curve_summary: valida folds y explicita el sentido favorable de la métrica.
        lines.append(learning_curve_summary(task, result))
        groups = warning_groups(result)
        failures = [trial['model'] for trial in result['trials'] if 'error' in trial]
        lines.append(f"Convergencia pendiente en búsquedas por familias: {groups['remaining_convergence']}; "
                     f"avisos de convergencia resueltos tras reintento: {groups['resolved_convergence']}; "
                     f"otros avisos registrados: {groups['other_warnings']}; candidatos fallidos: {failures}. "
                     'Los otros avisos no se interpretan como falta de convergencia. Cada aviso agrupa configuraciones de una búsqueda y no identifica necesariamente al ganador.')
        remaining = result.get('refinement', {}).get('convergence_review', {}).get('remaining_convergence_warnings', [])
        if remaining:
            lines.append(f'Avisos de convergencia restantes en el refinamiento: {remaining}. Su atribución a una configuración concreta no queda demostrada por el resumen agregado.')
        lines.append('')
    # winner: mejor candidato según validación, sin seleccionar por la prueba.
    winner = r['cluster_selected']
    # profiles: resúmenes por cluster; describen actividad relativa, sin validar segmentos comerciales.
    profiles = pd.read_csv(root/'reports/cluster_profiles.csv')
    lines += ['## Segmentación y recuperación', '',
        f"Clustering: {winner['name']}, k={winner['k']}, SVD={winner['dimensions']}, silhouette={format_number(winner['silhouette'],3)}. Se probaron dimensiones 15/30/60; KMeans/MiniBatch con k=2/3/4/5/6/8/10/12; Mean Shift con dos estimaciones de bandwidth y GMM con componentes 2/3/5/8 y covarianzas diag/tied. Aglomerativo y DBSCAN se conservan como exploración transductiva separada. La selección inductiva utiliza silhouette en los mismos 1.000 índices; se informa estabilidad por remuestreo condicionada a una representación SVD fija, sin elegir con las categorías editoriales."]
    quality = json.loads((root/'reports/quality_checks.json').read_text(encoding='utf-8'))
    balance_note = cluster_balance_note(profiles)
    lines += ['', localize_policy(r.get('clustering_selection_policy', 'Criterio histórico de silueta, anterior a la regla de tamaño y estabilidad.')), '', balance_note, '', cluster_stability_text(r, quality), '']
    (root/'reports/cluster_interpretation.md').write_text(
        clarify_narrative(cluster_interpretation_text(r, profiles, quality)), encoding='utf-8')
    lines += [cluster_profiles_summary(profiles, pd.read_csv(root/'reports/cluster_sample.csv')), '']
    lines += [cluster_category_summary(r, pd.read_csv(root/'reports/cluster_sample.csv'), pd.read_csv(root/'reports/cluster_categories.csv',index_col=0), pd.read_csv(root/'reports/cluster_category_proportions.csv',index_col=0)), '']
    lines += [cluster_terms_summary(pd.read_csv(root/'reports/cluster_terms.csv'), *cluster_term_representation(root)), '']
    lines += ['', 'Los rangos de audiencia son relativos a la muestra y no segmentos comerciales validados. Comparar contingencia, ARI y NMI evita confundir una separación por popularidad con géneros editoriales. En DBSCAN silhouette excluye ruido: consultar siempre cobertura, sin premiar una solución que descarta casi todo.', '',
        f"Recomendador: {r['recommendation']['selected']}. Coincidencia de categoría top-5 en consultas de prueba disjuntas: {format_number((r['recommendation']['independent_proxy_test']['category_agreement_at_5_PROXY'])*100,1)} %; cobertura del catálogo {format_number((r['recommendation']['independent_proxy_test']['catalog_coverage'])*100,1)} %; fracción media de canales distintos en cada lista {format_number((r['recommendation']['independent_proxy_test']['unique_channel_fraction_at_5'])*100,1)} %.", '',
        'La categoría es un proxy imperfecto: dos vídeos de la misma categoría pueden ser irrelevantes y dos de distinta categoría resultar útiles. La evaluación humana de la utilidad queda fuera de lo solicitado en el ejercicio y se propone como ampliación con valoraciones independientes de personas.', '',
        '## Alcance y siguientes decisiones', '',
        'Likes y ratio son estimaciones contemporáneas; no predicciones antes de publicar. Moderación excluye contadores que revelarían la respuesta. Las asociaciones de coeficientes y correlaciones no son causales. La repetición de canales puede favorecer el rendimiento; falta una validación externa por tiempo/canal.', '',
        'Para una aplicación operativa: acordar costes de errores, obtener relevancia humana, ampliar la validación temporal y monitorizar cambios de idioma y región. Estas acciones se derivan de las limitaciones observadas.', '',
        'Las comprobaciones por canal y de estabilidad se explican en `reports/quality_report.md`. Los diagnósticos de consultas y de escritura se conservan en los CSV. La interpretación de los grupos está en `reports/cluster_interpretation.md`.']
    limitations = measured_limitations(root, r)
    position = lines.index('## Decisiones por problema')
    lines[position:position] = limitations
    # text: informe interpretativo calculado a partir de las métricas.
    text = clarify_narrative('\n'.join(lines)+'\n')
    (root/'reports/results_details.md').write_text(editorial_detail(text),encoding='utf-8')
    brief=clarify_narrative(executive_summary(root,r))
    (root/'reports/results.md').write_text(brief,encoding='utf-8')
    return brief

# Punto de entrada: al ejecutar este archivo directamente, __name__ vale "__main__".
# El bloque siguiente inicia su ejecución; al importarlo desde otro archivo, no se ejecuta.



# _EVALUATION_NOTES: criterios compartidos; las cifras permanecen en cada tarea.
_EVALUATION_NOTES = {'curve_uncertainty': 'Las barras de error de las curvas de aprendizaje muestran la desviación entre particiones. No son intervalos de confianza ni intervalos de predicción individual para un '
                      'vídeo. Las particiones comparten parte del entrenamiento y su número '
                      'es reducido.',
 'curve_protocol': 'Para cada tamaño se ajustan clones sin estado aprendido, usando vídeos '
                   'completos de la parte de ajuste; la validación del fold permanece fija. '
                   'El preprocesamiento se aprende dentro del Pipeline. La receta ya fue '
                   'seleccionada con estos datos de desarrollo: esta curva no es validación '
                   'anidada independiente ni utiliza el holdout.',
 'curve_labels': 'Macro-F1 se calcula con las clases presentes en respuestas o predicciones '
                 'de cada evaluación; entrenamiento y validación pueden promediar conjuntos '
                 'de clases distintos. La brecha no es una comparación perfectamente '
                 'homogénea de etiquetas.',
 'curve_threshold': 'Las predicciones de los clones usan el umbral predeterminado del '
                    'estimador; no incorporan el umbral OOF final. No se equipara esta línea '
                    'con el rendimiento del modelo final calibrado.',
 'regional_labels': 'evaluate fija las etiquetas con la unión de las clases del ajuste y las '
                    'respuestas de cada región; macro-F1 incluye los ceros convencionales '
                    'para etiquetas de esa unión sin casos ni predicciones en la región. No '
                    'se interpreta como una media sólo de clases presentes. La tabla de '
                    'soportes no contiene errores ni predicciones por clase.',
 'regional_composition': 'La composición de objetivos, idiomas, canales y vídeos puede variar '
                         'entre regiones; no se atribuye la diferencia al país. Un mismo '
                         'vídeo puede aparecer en varias regiones dentro de prueba. Se '
                         'describe una prueba interna, sin cambiar recetas ni elegir otro '
                         'umbral con estos resultados.',
 'fixed_test': 'Se describe el modelo ya fijado. Estas métricas no justifican cambiar su '
               'umbral con el test; los costes, los soportes y una evaluación externa siguen '
               'siendo necesarios.'}


def evaluation_reading_guide():
    """Explica una sola vez los criterios comunes a las cinco tareas del cuaderno.

    Los informes autónomos mantienen estas notas por defecto. La vista conjunta
    las presenta aquí y solicita a cada narración sólo su contexto específico.
    """
    lines=['**Lectura común de la evaluación.** Las cifras y conclusiones se leen por tarea. Las notas sobre etiquetas y macro-F1 se aplican a clasificación; las del umbral OOF, a las dos tareas binarias. El resto explica el alcance de las curvas, las regiones y la prueba.']
    lines.extend(_EVALUATION_NOTES.values())
    lines.append('Los coeficientes dependen de la escala transformada y no demuestran causalidad. Las curvas de la muestra de búsqueda no demuestran saturación ni garantizan lo que ocurrirá con más datos. Los soportes, regiones y mayores errores completos se conservan en los desplegables.')
    return '\n\n'.join(lines)


def binary_score_explanation(result):
    """Describe la escala del umbral según la familia, sin inferir calibración.

    positive_scores prioriza predict_proba en la logística y utiliza la función
    de decisión en la SVM lineal. Una familia no reconocida recibe una explicación
    neutra: no se afirma que cualquier puntuación sea una probabilidad.
    """
    family=result.get('selected')
    if family=='svm':
        return 'La SVM lineal devuelve una puntuación de decisión, no una probabilidad; un umbral negativo no representa una probabilidad negativa.'
    if family=='logistic':
        return 'La regresión logística devuelve la probabilidad estimada de la clase 1, que se compara con el umbral elegido. Eso no garantiza que las probabilidades estén bien calibradas.'
    return 'La puntuación y el umbral se comparan en la escala del estimador seleccionado; no se presupone que esa puntuación sea una probabilidad calibrada.'


def classification_summary(task, metrics, *, include_shared_notes=True):
    """Explica la matriz y las métricas de prueba sin entrenar ni elegir umbrales.

    Valida etiquetas, recuentos, soportes y concordancia de las medias. Las filas
    son clases reales y las columnas predichas. Un denominador cero se explica
    como no estimable, aunque el informe numérico use zero_division=0.
    """
    # numpy: comprueba matrices y calcula recuentos por fila y columna.
    import numpy as np
    try:
        labels = metrics['labels']
        matrix = np.asarray(metrics['confusion_matrix'], dtype=float)
        report = metrics['classification_report']
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f'{task}: faltan etiquetas, matriz o informe de clasificación válido') from exc
    if (not isinstance(labels, list) or not labels or not all(isinstance(x, str) and x.strip() for x in labels)
            or len(set(labels)) != len(labels)):
        raise ValueError(f'{task}: las etiquetas deben ser textos únicos y no vacíos')
    if (matrix.shape != (len(labels), len(labels)) or not np.isfinite(matrix).all()
            or (matrix < 0).any() or not np.equal(matrix, np.floor(matrix)).all()):
        raise ValueError(f'{task}: la matriz debe ser cuadrada, finita y contener recuentos enteros no negativos')
    # actual/predicted/correct: soportes reales, totales predichos y diagonal.
    actual, predicted, correct = matrix.sum(axis=1), matrix.sum(axis=0), matrix.diagonal()
    total = float(actual.sum())
    if total <= 0:
        raise ValueError(f'{task}: no hay observaciones de prueba para interpretar la matriz')
    # precision/recall/f1: convención numérica cero para denominadores vacíos.
    precision = np.divide(correct, predicted, out=np.zeros_like(correct), where=predicted > 0)
    recall = np.divide(correct, actual, out=np.zeros_like(correct), where=actual > 0)
    f1 = np.divide(2*precision*recall, precision+recall, out=np.zeros_like(correct), where=precision+recall > 0)
    for index, label in enumerate(labels):
        row = report.get(label, {})
        expected = {'support': actual[index], 'precision': precision[index],
                    'recall': recall[index], 'f1-score': f1[index]}
        for field, value in expected.items():
            recorded = row.get(field)
            if not isinstance(recorded, (float, int)) or not np.isfinite(recorded) or not np.isclose(recorded, value, rtol=1e-9, atol=1e-12):
                raise ValueError(f'{task}: {field} de la clase {label} no concuerda con la matriz')
    # macro: incluye etiquetas declaradas; balanced: sólo clases con soporte real.
    macro, balanced = float(f1.mean()), float(recall[actual > 0].mean())
    for field, value in [('macro_f1', macro), ('balanced_accuracy', balanced)]:
        recorded = metrics.get(field)
        if not isinstance(recorded, (float, int)) or not np.isfinite(recorded) or not np.isclose(recorded, value, rtol=1e-9, atol=1e-12):
            raise ValueError(f'{task}: {field} no concuerda con la matriz')
    def n(value):
        """Presenta un recuento con separador español de millares."""
        return format_number(value, 0)

    def pct(value):
        """Presenta una proporción como porcentaje con dos decimales."""
        return format_number(value*100, 2) + ' %'
    lines = [f'**Lectura de clasificación: {task_description(task)}.** Filas = clase real; columnas = predicción, en el orden {labels}.',
             f'Diagonal: {n(correct.sum())} aciertos de {n(total)} apariciones; fuera de la diagonal: {n(total-correct.sum())} errores. Accuracy {pct(correct.sum()/total)}. No son necesariamente vídeos distintos.',
             f'Macro-F1 {format_number(macro,4)}: media de F1 sobre {n(len(labels))} etiquetas declaradas. Balanced accuracy {format_number(balanced,4)}: media de la recuperación de las {n((actual>0).sum())} clases con casos reales. No equivale al porcentaje global de aciertos.']
    absent = [label for label, count in zip(labels, actual) if count == 0]
    if absent:
        lines.append('Sin casos reales en prueba: ' + ', '.join(absent) + '. Su recall no se puede estimar; el informe usa cero por convención, no como prueba de fracaso. No hay que equiparar recall macro y balanced accuracy si usan conjuntos de clases distintos.')
    if labels == ['0', '1']:
        tn, fp, fn, tp = matrix.ravel()
        lines.append(f'Clase positiva 1 = función desactivada. VN={n(tn)}, FP={n(fp)}, FN={n(fn)}, VP={n(tp)}. Positivos reales: {n(tp+fn)}; positivos predichos: {n(tp+fp)}.')
        ptext = pct(tp/(tp+fp)) if tp+fp else 'no estimable: no se predijo ningún positivo (informe numérico: 0)'
        rtext = pct(tp/(tp+fn)) if tp+fn else 'no estimable: no hay positivos reales (informe numérico: 0)'
        lines.append(f'Precisión positiva = VP/(VP+FP): {ptext}. Recall positivo = VP/(VP+FN): {rtext}. F1 positivo {format_number(f1[1],4)}; macro-F1 también incluye F1 de la clase 0.')
        lines.append(f'Predecir siempre 0 acertaría {pct(actual[0]/total)} en esta prueba, pero detectaría 0 positivos. Es una ilustración del desequilibrio; no sustituye la referencia sencilla entrenada ni elige otro umbral.')
    lines.append((_EVALUATION_NOTES['fixed_test'] if include_shared_notes else ''))
    return '\n\n'.join(line for line in lines if line)


def category_confusion_summary(metrics, minimum_support=20, top=5):
    """Interpreta clases y confusiones dirigidas sin modificar decisiones.

    Valida primero el informe con classification_summary. Muestra todas las clases,
    separa recuentos absolutos de porcentajes sobre la fila real y explicita el
    soporte mínimo del listado proporcional. Los desempates usan las etiquetas.
    """
    # numpy: calcula soportes y ordena las celdas fuera de la diagonal.
    import numpy as np
    if (isinstance(minimum_support, bool) or not isinstance(minimum_support, int)
            or minimum_support < 1 or isinstance(top, bool) or not isinstance(top, int) or top < 1):
        raise ValueError('Categorías: soporte mínimo y número de ejemplos deben ser enteros positivos')
    classification_summary('category', metrics)
    # labels/matrix/report: orden compartido de clases, recuentos e informe por clase.
    labels = metrics['labels']
    matrix = np.asarray(metrics['confusion_matrix'], dtype=float)
    report = metrics['classification_report']
    # actual/predicted: denominadores reales por fila y predichos por columna.
    actual, predicted = matrix.sum(axis=1), matrix.sum(axis=0)

    def number(value, decimals=0):
        """Presenta cifras de lectura en español, sin cambiar su valor de cálculo."""
        return format_number(value, decimals)

    def escape_label(label):
        """Evita que una etiqueta rompa una fila de la tabla Markdown."""
        return label.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('|', '&#124;').replace('\n', ' ').replace('\r', ' ')

    lines = ['**Lectura por categoría.** Soporte = casos reales; precisión = aciertos / predicciones de esa categoría; recall = aciertos / casos reales. La tabla conserva todas las categorías, incluso las ausentes.', '',
             '| Categoría | Soporte real | Predicciones | Aciertos | Precisión | Recall | F1 |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for i, label in enumerate(labels):
        row = report[label]
        precision = number(row['precision']*100, 2)+' %' if predicted[i] else 'no estimable'
        recall = number(row['recall']*100, 2)+' %' if actual[i] else 'sin soporte'
        lines.append(f"| {escape_label(label)} | {number(actual[i])} | {number(predicted[i])} | {number(matrix[i,i])} | {precision} | {recall} | {number(row['f1-score'],4)} |")
    lines.append('')
    lines.append('F1 muestra el cero convencional del informe cuando no es estimable; no implica fracaso observado si falta soporte. Precisión puede ser cero estimable cuando se predice una clase que nunca es la real.')
    # errors: confusiones dirigidas con recuento estrictamente positivo.
    errors = [(float(matrix[i,j]), labels[i], labels[j], float(actual[i]))
              for i in range(len(labels)) for j in range(len(labels)) if i != j and matrix[i,j] > 0]
    if not errors:
        lines.append('No hay confusiones fuera de la diagonal en estas observaciones. Esto no demuestra generalización ni calidad para categorías sin soporte.')
    else:
        counts = sorted(errors, key=lambda x: (-x[0], x[1], x[2]))[:top]
        lines.append(f'\n**Hasta {number(top)} confusiones con más casos** (real → predicha; porcentaje sobre la fila real):')
        for count, real, pred, support in counts:
            lines.append(f'- {escape_label(real)} → {escape_label(pred)}: {number(count)} de {number(support)}, {number(count/support*100,2)} %.')
        rates = sorted((x for x in errors if x[3] >= minimum_support), key=lambda x: (-x[0]/x[3], -x[0], x[1], x[2]))[:top]
        eligible = int((actual >= minimum_support).sum())
        lines.append(f'\n**Hasta {number(top)} confusiones con mayor fracción de la clase real**, entre las {number(eligible)} clases con soporte de al menos {number(minimum_support)} apariciones. Este filtro es descriptivo, no selecciona modelos; las demás clases siguen en la tabla.')
        if not rates:
            lines.append('No hay confusiones en clases que alcancen ese soporte; no se rebaja automáticamente el límite.')
        for count, real, pred, support in rates:
            lines.append(f'- {escape_label(real)} → {escape_label(pred)}: {number(count/support*100,2)} %, {number(count)} de {number(support)}.')
    lines.append('')
    lines.append('Las flechas tienen dirección: A → B no equivale a B → A y sus denominadores pueden diferir. La matriz normalizada por fila divide por el soporte real; una fila sin soporte se marca sin datos, no como 0 % de errores. No se suman porcentajes de filas distintas.')
    if errors:
        count, real, pred, support = sorted(errors, key=lambda item: (-item[0], item[1], item[2]))[0]
        lines.append(f'**Conclusión.** La confusión con más casos es {escape_label(real)} → '
                     f'{escape_label(pred)}: {number(count)} errores. Revisar ejemplos de esa pareja '
                     'es una prioridad concreta para estudiar el texto y la consistencia de las etiquetas.')
        # shows_sports: solapamiento editorial plausible, sin adjudicar contenido no revisado.
        shows_sports = next((item for item in errors if item[1:3] == ('Shows', 'Sports')), None)
        if shows_sports:
            lines.append('Shows → Sports es compatible con programas deportivos que encajan en ambas '
                         'etiquetas; es una hipótesis editorial para revisar ejemplos, no contenido confirmado.')
    lines.append('Estos resultados describen confusiones, no explican su causa. La repetición de vídeos, canales, idiomas y el etiquetado editorial limitan la interpretación. No se cambian familias, parámetros ni umbrales tras mirar el test.')
    return '\n'.join(lines)


def regression_summary(task, result):
    """Explica MAE, RMSE y R² de prueba con comparación segura de la referencia.

    No ajusta ni selecciona modelos. Valida finitud, signos y RMSE >= MAE. La
    reducción relativa de MAE sólo se calcula si la referencia tiene MAE positivo;
    una referencia perfecta no admite una reducción porcentual definida.
    """
    # math y numbers: validan magnitudes numéricas finitas sin aceptar booleanos.
    from math import isfinite, isclose
    from numbers import Real, Integral
    if not isinstance(result, dict):
        raise ValueError(f'{task}: se esperaba un informe de regresión')
    rows = result.get('test_rows')
    if isinstance(rows, bool) or not isinstance(rows, Integral) or rows < 1:
        raise ValueError(f'{task}: test_rows debe ser un entero positivo')
    for name in ['holdout', 'baseline_holdout']:
        metrics = result.get(name)
        if not isinstance(metrics, dict):
            raise ValueError(f'{task}: falta el bloque {name} de regresión')
        for field in ['mae', 'rmse', 'r2']:
            value = metrics.get(field)
            if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
                raise ValueError(f'{task}: {name}.{field} debe ser un número finito')
        if metrics['mae'] < 0 or metrics['rmse'] < 0 or metrics['r2'] > 1+1e-12:
            raise ValueError(f'{task}: {name} contiene métricas fuera de sus límites')
        if metrics['rmse'] < metrics['mae'] and not isclose(metrics['rmse'], metrics['mae'], rel_tol=1e-10, abs_tol=1e-12):
            raise ValueError(f'{task}: {name}.rmse no puede ser menor que MAE en las mismas observaciones')
    # metrics/reference: errores del modelo y de la mediana ajustada en entrenamiento.
    metrics, reference = result['holdout'], result['baseline_holdout']
    delta = reference['mae']-metrics['mae']
    unit = 'likes' if task == 'likes' else 'unidades del cociente likes/dislikes (no puntos porcentuales)' if task == 'ratio' else 'unidades del objetivo'
    lines = [f'**Lectura de regresión: {task_description(task)}.** {format_number(rows,0)} apariciones de prueba. MAE y RMSE están en la escala original: {unit}; no son errores en logaritmos ni porcentajes.',
             f'MAE {format_number(metrics["mae"],2)}: media de las distancias absolutas entre valor real y predicción. No es la mediana del error ni garantiza que cada caso quede dentro de ese margen.',
             f'RMSE {format_number(metrics["rmse"],2)}: raíz de la media de los errores al cuadrado. Da más peso a errores grandes; su distancia respecto al MAE no permite contar cuántos extremos hay sin mirar los casos.']
    if reference['mae'] > 0:
        relative = delta/reference['mae']*100
        direction = 'disminuye' if delta > 0 else 'aumenta' if delta < 0 else 'no cambia'
        lines.append(f'Frente a la referencia de mediana del ajuste, evaluada sobre las mismas filas (MAE {format_number(reference["mae"],2)}), el MAE {direction}. Reducción absoluta referencia − modelo: {format_number(delta,2)}; reducción relativa: {format_number(relative,2)} %. Un valor negativo indica empeoramiento; no es un porcentaje de aciertos.')
    else:
        lines.append(f'La referencia tiene MAE cero: la reducción relativa no está definida y no se divide por cero. Diferencia absoluta referencia − modelo: {format_number(delta,2)}. '+('El modelo también tiene MAE cero: empate en esta métrica.' if metrics['mae'] == 0 else 'El modelo empeora frente a esa referencia perfecta.'))
    lines.append(f'R² {format_number(metrics["r2"],4)}: 1 − SSE/SST, respecto al predictor constante igual a la media de los valores reales de esta prueba. Es un denominador matemático de evaluación; no se usa esa media para entrenar el modelo ni el baseline de mediana.')
    if metrics['r2'] < 0:
        lines.append('R² negativo: la suma de errores cuadrados supera la de esa referencia de media de prueba. Puede ocurrir aunque el MAE mejore frente a la mediana de ajuste; son referencias y criterios distintos.')
    else:
        lines.append('R² no es el porcentaje de predicciones correctas ni una garantía de error pequeño. Con un objetivo constante SST=0 y con menos de dos casos la definición habitual requiere tratamiento especial; estos valores agregados no permiten diagnosticar por sí solos esas situaciones.')
    lines.append('Estas magnitudes no se comparan directamente entre likes y ratio porque sus unidades y escalas son distintas. Describen modelos ya fijados sobre una prueba interna ya examinada; no justifican reoptimizar con sus etiquetas.')
    return '\n\n'.join(lines)


# Punto de entrada: las funciones auxiliares ya están definidas antes de main.



def largest_errors_summary(task, result, errors):
    """Valida la cola ordenada y describe su peso sin extrapolar al conjunto.

    errors contiene apariciones, no vídeos deduplicados. Contrasta error absoluto,
    finitud, orden y compatibilidad con MAE/RMSE globales; no acredita por sí solo
    que el CSV provenga de los pesos actuales. Las participaciones usan totales
    reconstruidos de las métricas, con tolerancia de redondeo.
    """
    # numpy: calcula residuos, tolerancias y acumulaciones de errores.
    import numpy as np
    regression_summary(task, result)
    required = {'video_id','region','actual','predicted','absolute_error'}
    if not isinstance(errors, pd.DataFrame) or not required.issubset(errors.columns):
        raise ValueError(f'{task}: el CSV de errores requiere '+', '.join(sorted(required)))
    if errors.empty or len(errors) > min(100, result['test_rows']):
        raise ValueError(f'{task}: cantidad de filas de errores vacía o superior al máximo exportado')
    for column in ['video_id','region']:
        if errors[column].isna().any() or not errors[column].map(lambda value: isinstance(value,str) and bool(value.strip())).all():
            raise ValueError(f'{task}: {column} debe identificar cada aparición')
    try:
        values = errors[['actual','predicted','absolute_error']].apply(pd.to_numeric, errors='raise').to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{task}: real, predicción y error deben ser numéricos') from exc
    if not np.isfinite(values).all() or (values[:,2] < 0).any():
        raise ValueError(f'{task}: valores no finitos o errores negativos en el CSV')
    actual, predicted, absolute = values.T
    # residual: positivo = exceso de predicción; negativo = defecto.
    residual = predicted-actual
    if not np.allclose(absolute,np.abs(residual),rtol=1e-9,atol=1e-8):
        raise ValueError(f'{task}: absolute_error no coincide con |predicted − actual|')
    if any(right > left and not np.isclose(right,left,rtol=1e-10,atol=1e-8) for left,right in zip(absolute,absolute[1:])):
        raise ValueError(f'{task}: el CSV de errores no está ordenado de mayor a menor')
    # Totales agregados de prueba: suma absoluta y suma cuadrática de todas sus filas.
    total_abs = result['holdout']['mae']*result['test_rows']
    total_square = result['holdout']['rmse']**2*result['test_rows']
    tail_abs, tail_square = float(absolute.sum()), float(np.square(absolute).sum())
    for partial, total in [(tail_abs,total_abs),(tail_square,total_square)]:
        if not np.isfinite(partial) or not np.isfinite(total) or partial > total+max(1e-8,abs(total)*1e-8):
            raise ValueError(f'{task}: la cola de errores supera el total compatible con las métricas')
    count = len(errors)
    lines = [f'**Casos con errores grandes: {task_description(task)}.** El CSV contiene {format_number(count,0)} apariciones ordenadas por error absoluto, de {format_number(result["test_rows"],0)} apariciones de prueba ({format_number(count/result["test_rows"]*100,2)} %). Vídeos distintos representados: {format_number(errors.video_id.nunique(),0)}; repetir un video_id no convierte dos apariciones en duplicados inválidos.',
             f'Distribución en estas filas: sobreestimaciones = {format_number((residual>0).sum(),0)}; subestimaciones = {format_number((residual<0).sum(),0)}; coincidencias exactas = {format_number((residual==0).sum(),0)}. No es un balance del conjunto de prueba: la lista está seleccionada por magnitud de error.']
    worst = errors.iloc[0]
    lines.append(f'Mayor error de la lista: vídeo {worst.video_id}, región {worst.region}; real {format_number(actual[0],2)}, predicción {format_number(predicted[0],2)}, diferencia predicción − real {format_number(residual[0],2)} y error absoluto {format_number(absolute[0],2)}.')
    if total_abs > 0:
        lines.append(f'Estas filas reúnen el {format_number(tail_abs/total_abs*100,2)} % de la suma de errores absolutos de toda la prueba. El denominador es MAE × número de filas, no sólo el error de esta lista.')
    else:
        lines.append('El error absoluto total es cero: su participación porcentual no está definida, no se divide por cero.')
    if total_square > 0:
        lines.append(f'Reúnen el {format_number(tail_square/total_square*100,2)} % de la suma de errores cuadrados total (RMSE² × número de filas). Es una participación en esa suma, no en el RMSE ni en la varianza del objetivo.')
    else:
        lines.append('El error cuadrático total es cero: su participación porcentual no está definida.')
    if task == 'ratio' and total_abs > 0 and total_square > 0:
        under, over = int((residual < 0).sum()), int((residual > 0).sum())
        direction = ('todas son subestimaciones' if under == count else
                     'todas son sobreestimaciones' if over == count else
                     'predominan las subestimaciones' if under > over else
                     'predominan las sobreestimaciones' if over > under else
                     'hay el mismo número de subestimaciones y sobreestimaciones')
        conclusion = (
            f'**Conclusión para la proporción «me gusta/no me gusta».** En las '
            f'{highlight_number(count)} apariciones con mayor error, {direction}: '
            f'{highlight_number(under)} predicciones por debajo del valor real y '
            f'{highlight_number(over)} por encima. Corresponden a '
            f'{highlight_number(errors.video_id.nunique())} vídeos y aportan '
            f'{highlight_number(tail_abs/total_abs*100,2," %")} del error absoluto total y '
            f'{highlight_number(tail_square/total_square*100,2," %")} del cuadrático. '
            'El patrón describe esta cola seleccionada, no el signo de los errores de toda la prueba. '
        )
        if under > over and result.get('selected') in ('ridge_log_target','svr_log_target'):
            conclusion += ('La transformación logarítmica del objetivo y la regularización podrían '
                           'contribuir a estimaciones menos extremas: es una hipótesis compatible '
                           'con la receta, no una causa comprobada por esta tabla. ')
        conclusion += ('Para confirmar un sesgo global haría falta examinar todos los residuos; '
                       'estos casos no justifican elegir otra receta usando la prueba ni retirar sus errores difíciles.')
        lines.append(conclusion)
    elif total_abs > 0 and total_square > 0:
        # concentration: evita atribuir concentración dominante a cualquier cola.
        concentration = tail_abs / total_abs >= .4 or tail_square / total_square >= .5
        if concentration:
            lines.append(
                f'**Conclusión.** Las {highlight_number(count)} apariciones de '
                f'{highlight_number(errors.video_id.nunique())} vídeos concentran '
                f'{highlight_number(tail_abs/total_abs*100,2," %")} del error absoluto y '
                f'{highlight_number(tail_square/total_square*100,2," %")} del cuadrático. '
                'Un conjunto reducido de identidades pesa mucho en las métricas por aparición; '
                'el error cuadrático es especialmente sensible a esos casos. '
                'Conviene comprobar también métricas por vídeo en una futura evaluación, '
                'sin retirar ahora los errores difíciles del test.')
        else:
            lines.append('**Conclusión.** Esta cola exportada no concentra la mayor parte del error cuadrático; '
                         'su lectura debe complementarse con las métricas de toda la prueba.')
    lines.append('La tabla visible presenta hasta ocho filas y añade diferencia predicción − real. El signo distingue exceso y defecto; absolute_error mide la distancia. La lista completa sigue en el CSV. No se eliminan estos casos ni se cambia el modelo para mejorar esta prueba; tampoco se atribuye una causa sin revisar los datos originales.')
    return '\n\n'.join(lines)


# Punto de entrada: todos los auxiliares están definidos antes de llamar a main.



def learning_curve_summary(task, result, *, include_shared_notes=True):
    """Valida e interpreta tamaños, medias y dispersión de las curvas guardadas.

    include_shared_notes=False omite sólo las notas que ya aparecen en la guía común.
    No vuelve a ajustar modelos. Verifica coherencia de medias y desviaciones de
    validación con los folds registrados; los scores individuales de entrenamiento
    no están guardados, por lo que no se afirma haberlos recalculado.
    """
    # numpy: valida finitud y reproduce media y desviación poblacional entre folds.
    import numpy as np
    curve = result.get('learning_curve') if isinstance(result,dict) else None
    if not isinstance(curve,list) or len(curve)<2:
        raise ValueError(f'{task}: se requieren al menos dos puntos de curva')
    holdout = result.get('holdout',{})
    classification = 'macro_f1' in holdout
    if not classification and 'mae' not in holdout:
        raise ValueError(f'{task}: falta el tipo de métrica de la curva')
    previous_fraction, previous_size, fold_count = 0.,0.,None
    for point in curve:
        if not isinstance(point,dict):
            raise ValueError(f'{task}: punto de curva no válido')
        fields = ['fraction','mean_train_rows','validation_mean','validation_std','train_mean']
        if any(isinstance(point.get(key),bool) or not isinstance(point.get(key),(float,int)) or not np.isfinite(point[key]) for key in fields):
            raise ValueError(f'{task}: faltan magnitudes finitas en la curva')
        fraction,size = point['fraction'],point['mean_train_rows']
        if not previous_fraction < fraction <= 1 or size<=0 or size<previous_size or point['validation_std']<0:
            raise ValueError(f'{task}: fracciones, tamaños o dispersión incoherentes en la curva')
        raw_scores=point.get('fold_scores')
        if not isinstance(raw_scores,list) or len(raw_scores)<2:
            raise ValueError(f'{task}: faltan puntuaciones de al menos dos folds')
        if any(isinstance(value,bool) or not isinstance(value,(float,int)) for value in raw_scores):
            raise ValueError(f'{task}: puntuaciones de folds no numéricas')
        scores=np.asarray(raw_scores,dtype=float)
        if not np.isfinite(scores).all() or (scores<0).any() or point['train_mean']<0 or (classification and ((scores>1).any() or point['train_mean']>1)):
            raise ValueError(f'{task}: puntuaciones fuera del dominio de la métrica')
        if fold_count is not None and len(scores)!=fold_count:
            raise ValueError(f'{task}: cambia el número de folds entre puntos')
        if not np.isclose(scores.mean(),point['validation_mean'],rtol=1e-9,atol=1e-12) or not np.isclose(scores.std(ddof=0),point['validation_std'],rtol=1e-9,atol=1e-12):
            raise ValueError(f'{task}: media o desviación no coincide con los folds')
        previous_fraction, previous_size, fold_count = fraction,size,len(scores)
    first,last=curve[0],curve[-1]
    metric='macro-F1' if classification else 'MAE'
    delta=last['validation_mean']-first['validation_mean']
    favorable=delta if classification else -delta
    trend='aumenta' if delta>0 else 'disminuye' if delta<0 else 'no cambia'
    conclusion='mejora numérica entre extremos' if favorable>0 else 'empeoramiento numérico entre extremos' if favorable<0 else 'empate entre extremos'
    # gap: positivo indica ventaja del ajuste en su propia métrica; no prueba sobreajuste por sí solo.
    gap=last['train_mean']-last['validation_mean'] if classification else last['validation_mean']-last['train_mean']
    favorable_steps=sum((right['validation_mean']-left['validation_mean'])*(1 if classification else -1)>0 for left,right in zip(curve,curve[1:]))
    lines=[f'**Curva de aprendizaje: {task_description(task)}.** {format_number(len(curve),0)} tamaños y {format_number(fold_count,0)} folds. Eje X: media de filas efectivas de ajuste por fold, de {format_number(first["mean_train_rows"],1)} a {format_number(last["mean_train_rows"],1)}; no es el número de vídeos ni el tamaño del ajuste final.',
           f'Validación {metric}: {format_number(first["validation_mean"],4)} → {format_number(last["validation_mean"],4)}; {trend}. '+('Mayor es mejor' if classification else 'Menor es mejor')+f': {conclusion}. Pasos consecutivos favorables: {format_number(favorable_steps,0)} de {format_number(len(curve)-1,0)}; no se presupone una evolución monótona.',
           f'En el último tamaño, entrenamiento {format_number(last["train_mean"],4)} y validación {format_number(last["validation_mean"],4)}. Brecha orientada para que un valor positivo indique ventaja del entrenamiento: {format_number(gap,4)}. Es un diagnóstico condicionado a los datos y al protocolo, no una prueba aislada de sobreajuste.',
           f'Barras: media de validación ± desviación entre folds, calculada con ddof=0; última desviación {format_number(last["validation_std"],4)}. ' + (_EVALUATION_NOTES['curve_uncertainty'] if include_shared_notes else ''),
           (_EVALUATION_NOTES['curve_protocol'] if include_shared_notes else '')]
    # conclusion_evidence: cada cierre muestra el rango y la métrica de su propia tarea.
    conclusion_evidence = (
        f'**Conclusión para {task_description(task)}.** '
        f'Con {highlight_number(first["mean_train_rows"], 1)} → '
        f'{highlight_number(last["mean_train_rows"], 1)} filas medias de ajuste por partición, '
        f'el {metric} de validación pasa de '
        f'{highlight_number(first["validation_mean"], 4 if classification else 2)} a '
        f'{highlight_number(last["validation_mean"], 4 if classification else 2)}. '
    )
    if favorable > 0:
        lines.append(conclusion_evidence + 'Se observa mejora dentro del rango medido; '
                     'merece contrastar una muestra mayor mediante validación de desarrollo, '
                     'sin extrapolar una ganancia fuera de ese rango.')
    else:
        lines.append(conclusion_evidence + 'No se observa mejora entre los extremos; '
                     'antes de ampliar la muestra conviene revisar representación, regularización '
                     'y dispersión entre particiones.')
    if classification:
        lines.append((_EVALUATION_NOTES['curve_labels'] if include_shared_notes else ''))
    if task in ['comments_disabled','ratings_disabled']:
        lines.append((_EVALUATION_NOTES['curve_threshold'] if include_shared_notes else ''))
    if any(right['mean_train_rows']==left['mean_train_rows'] for left,right in zip(curve,curve[1:])):
        lines.append('Hay tamaños efectivos repetidos: las fracciones solicitadas no garantizan más filas al conservar grupos completos.')
    if include_shared_notes:
        lines.append(f'Estos {format_number(len(curve),0)} tamaños en la muestra de búsqueda no demuestran saturación ni predicen qué ocurriría al ampliar el corpus. Una mejora observada puede motivar otro experimento; no garantiza que más datos resuelvan el error ni justifica reajustar decisiones con el test.')
    return '\n\n'.join(line for line in lines if line)


# Punto de entrada: todos los auxiliares están definidos antes de main.



def regional_class_supports(root, report):
    """Reconstruye soportes de clases en el holdout canónico sin predecir ni ajustar.

    Lee sólo las columnas necesarias; conserva orden, partición por video_id y
    muestreo. Los soportes permiten distinguir clase ausente de rendimiento bajo.
    """
    # configparser: recupera la ruta, semilla y presupuesto reales de la práctica.
    import configparser
    # training_data: reutiliza la partición y muestreo canónicos sin duplicarlos.
    from training_data import split_video_pools, group_sample
    root=Path(root)
    conf=configparser.ConfigParser()
    if not conf.read(root/'train.conf',encoding='utf-8'):
        raise FileNotFoundError(f'No existe la configuración regional: {root/"train.conf"}')
    data=pd.read_csv(root/conf['data']['input'],usecols=['video_id','region','category','comments_disabled','ratings_disabled'])
    seed=conf.getint('training','seed')
    _,test_pool=split_video_pools(data,seed)
    test=group_sample(test_pool,conf.getint('training','test_rows'),seed+1)
    rows=[]
    for task,result in report['tasks'].items():
        if 'macro_f1' not in result['holdout']:
            continue
        block=test.dropna(subset=[task]).copy()
        if len(block)!=result['test_rows']:
            raise ValueError(f'{task}: el soporte de prueba reconstruido no coincide con el informe')
        # target: representación textual de etiquetas, igual a la del informe.
        target=block[task].astype(str) if task=='category' else pd.to_numeric(block[task]).astype(int).astype(str)
        labels=result['holdout']['labels']
        if not set(target).issubset(labels):
            raise ValueError(f'{task}: hay clases reales fuera de las etiquetas declaradas')
        for region,indices in block.groupby('region').groups.items():
            counts=target.loc[indices].value_counts()
            for label in labels:
                rows.append({'task':task,'region':region,'label':label,'rows':int(counts.get(label,0))})
    return pd.DataFrame(rows,columns=['task','region','label','rows'])


def regional_summary(task, result, regions, class_supports=None, minimum_rows=100, *, include_shared_notes=True):
    """Con include_shared_notes=False, la guía conjunta conserva las cautelas comunes.
    Valida soportes regionales y describe extremos sin atribuir causas.

    MAE global es media ponderada por filas; RMSE requiere ponderar sus cuadrados.
    No se promedian macro-F1, balanced accuracy o R² regionales para obtener global.
    El filtro de soporte es descriptivo, nunca una selección del modelo.
    """
    # numpy: comprueba finitud, dominios, agregación y empates numéricos.
    import numpy as np
    if isinstance(minimum_rows,bool) or not isinstance(minimum_rows,int) or minimum_rows<1:
        raise ValueError(f'{task}: el soporte mínimo debe ser un entero positivo')
    classification='macro_f1' in result['holdout']
    fields=['macro_f1','balanced_accuracy'] if classification else ['mae','rmse','r2']
    required={'region','rows',*fields}
    if not isinstance(regions,pd.DataFrame) or regions.empty or not required.issubset(regions.columns):
        raise ValueError(f'{task}: tabla regional vacía o sin columnas requeridas')
    if not regions.region.map(lambda value:isinstance(value,str) and bool(value.strip()) and value==value.strip()).all() or regions.region.duplicated().any():
        raise ValueError(f'{task}: regiones vacías, duplicadas o con espacios ambiguos')
    data=regions.copy()
    try:
        data[['rows']+fields]=data[['rows']+fields].apply(pd.to_numeric,errors='raise')
    except (ValueError,TypeError) as exc:
        raise ValueError(f'{task}: soportes o métricas regionales no numéricos') from exc
    if not np.isfinite(data.rows).all() or (data.rows<=0).any() or (data.rows!=np.floor(data.rows)).any() or data.rows.sum()!=result['test_rows']:
        raise ValueError(f'{task}: los soportes regionales deben ser enteros positivos y sumar test_rows')
    for field in fields:
        undefined_r2=(field=='r2') & (data.rows<2) & data[field].isna()
        if not np.isfinite(data.loc[~undefined_r2,field]).all():
            raise ValueError(f'{task}: {field} regional no finito')
    if classification:
        if ((data[fields]<0)|(data[fields]>1)).any().any():
            raise ValueError(f'{task}: métricas de clasificación regional fuera de [0,1]')
    else:
        regression_summary(task,result)
        if (data.mae<0).any() or (data.rmse<0).any() or (data.r2.dropna()>1+1e-12).any() or (data.rmse+1e-8<data.mae).any():
            raise ValueError(f'{task}: errores regionales fuera de sus límites')
        global_mae=float(np.average(data.mae,weights=data.rows))
        global_rmse=float(np.sqrt(np.average(data.rmse**2,weights=data.rows)))
        if not np.isclose(global_mae,result['holdout']['mae'],rtol=1e-8,atol=1e-8) or not np.isclose(global_rmse,result['holdout']['rmse'],rtol=1e-8,atol=1e-8):
            raise ValueError(f'{task}: MAE/RMSE regionales no reconstruyen las métricas globales')
    metric='balanced_accuracy' if classification else 'mae'
    eligible=data.loc[data.rows>=minimum_rows]
    lines=[f'**Diagnóstico regional: {task_description(task)}.** {format_number(len(data),0)} regiones y {format_number(data.rows.sum(),0)} apariciones; soportes entre {format_number(data.rows.min(),0)} y {format_number(data.rows.max(),0)}. Son filas de prueba, no vídeos únicos ni personas de esas nacionalidades.',
           f'Comparación descriptiva por {metric}: '+('mayor es favorable' if classification else 'menor es favorable')+f'. Regiones con al menos {format_number(minimum_rows,0)} filas: {format_number(len(eligible),0)}. El filtro no garantiza soporte de cada clase, independencia ni significación estadística.']
    if eligible.empty:
        lines.append('Ninguna región alcanza el soporte mínimo; no se rebaja el límite ni se inventa una región peor.')
    else:
        worst_value=eligible[metric].min() if classification else eligible[metric].max()
        worst=eligible.loc[np.isclose(eligible[metric],worst_value,rtol=1e-10,atol=1e-12)].sort_values('region')
        for row in worst.itertuples():
            lines.append(f'Valor menos favorable entre las elegibles: {row.region}, {metric} {format_number(getattr(row,metric),4)}, n={format_number(row.rows,0)}. '+('Empate numérico entre regiones.' if len(worst)>1 else ''))
    if classification:
        if class_supports is None or not isinstance(class_supports,pd.DataFrame) or not {'task','region','label','rows'}.issubset(class_supports.columns):
            raise ValueError(f'{task}: faltan soportes por clase para interpretar las regiones')
        local=class_supports.loc[class_supports.task==task].copy()
        labels=result['holdout']['labels']
        expected={(region,label) for region in data.region for label in labels}
        if local[['region','label']].duplicated().any() or set(zip(local.region,local.label))!=expected:
            raise ValueError(f'{task}: soportes por clase incompletos o duplicados')
        counts=pd.to_numeric(local.rows,errors='coerce')
        if not np.isfinite(counts).all() or (counts<0).any() or (counts!=np.floor(counts)).any():
            raise ValueError(f'{task}: soportes por clase no válidos')
        local['rows']=counts
        totals=local.groupby('region').rows.sum()
        if any(totals[row.region]!=row.rows for row in data.itertuples()):
            raise ValueError(f'{task}: soportes por clase no suman las filas de su región')
        present=local.loc[local.rows>0].groupby('region').size()
        lines.append(f'Clases con casos reales por región: entre {format_number(present.min(),0)} y {format_number(present.max(),0)}, de {format_number(len(labels),0)} etiquetas declaradas. Balanced accuracy promedia recall sólo de clases presentes; esos conjuntos pueden diferir entre regiones.')
        if task in ['comments_disabled','ratings_disabled']:
            positives=local.loc[local.label=='1'].set_index('region').rows
            lines.append('Soporte positivo real por región: '+ '; '.join(f'{region}={format_number(positives[region],0)}' for region in sorted(data.region))+'.')
            absent=sorted(positives.loc[positives==0].index)
            if absent:
                lines.append('Sin positivos reales: '+', '.join(absent)+'. Una balanced accuracy alta o perfecta en esas regiones no demuestra detección de la clase 1; la recuperación de positivos no se puede estimar.')
        lines.append((_EVALUATION_NOTES['regional_labels'] if include_shared_notes else ''))
    else:
        lines.append('MAE global se reconstruye ponderando los MAE regionales por filas; RMSE global requiere raíz de la media ponderada de RMSE². R² global no es el promedio de los R² regionales: las medias y dispersiones locales difieren. Un MAE regional alto puede coexistir con R² alto.')
    lines.append((_EVALUATION_NOTES['regional_composition'] if include_shared_notes else ''))
    return '\n\n'.join(line for line in lines if line)


# Punto de entrada: todos los auxiliares están definidos antes de main.



def coefficient_summary(task, result, coefficients):
    """Valida y contextualiza el extracto exportado de coeficientes lineales.

    Exige pares clase/atributo únicos, pesos finitos y orden por magnitud dentro
    de cada salida. No atribuye causalidad, importancia universal ni contribución
    individual; la procedencia se contrasta por separado con el estimador guardado.
    """
    # numpy y html: validan números y escapan nombres externos en Markdown.
    import numpy as np
    from html import escape
    if not isinstance(coefficients,pd.DataFrame) or coefficients.empty or not {'class','feature','coefficient'}.issubset(coefficients.columns):
        raise ValueError(f'{task}: falta una tabla de coeficientes con clase, atributo y peso')
    data=coefficients.copy()
    for field in ['class','feature']:
        if not data[field].map(lambda value:isinstance(value,str) and bool(value.strip())).all():
            raise ValueError(f'{task}: nombres de clase/atributo vacíos o no textuales')
    if data[['class','feature']].duplicated().any():
        raise ValueError(f'{task}: atributo repetido dentro de la misma clase/salida')
    try:
        data['coefficient']=pd.to_numeric(data.coefficient,errors='raise')
    except (ValueError,TypeError) as exc:
        raise ValueError(f'{task}: coeficientes no numéricos') from exc
    if not np.isfinite(data.coefficient).all():
        raise ValueError(f'{task}: coeficientes no finitos')
    for label,block in data.groupby('class',sort=False):
        if len(block)>15:
            raise ValueError(f'{task}: el extracto excede quince coeficientes para {label}')
        magnitude=block.coefficient.abs().to_numpy()
        if any(right>left and not np.isclose(right,left,rtol=1e-10,atol=1e-12) for left,right in zip(magnitude,magnitude[1:])):
            raise ValueError(f'{task}: coeficientes sin orden descendente por magnitud en {label}')
    if task=='category':
        expected=set(result['holdout']['labels'])
    elif task in ['comments_disabled','ratings_disabled']:
        expected={'1'}
    else:
        expected={task}
    if not set(data['class']).issubset(expected):
        raise ValueError(f'{task}: etiqueta de coeficientes incompatible con la tarea')

    def safe(value):
        """Conserva nombres legibles sin romper el formato Markdown."""
        return escape(value).replace('&amp;','&').replace('`','&#96;').replace('|','&#124;')

    lines=[f'**Coeficientes: {task_description(task)}.** {format_number(len(data),0)} pesos exportados para {format_number(data["class"].nunique(),0)} clases/salidas. Se conserva hasta 15 por clase/salida, ordenados por valor absoluto; no es el vector completo y la ausencia de un atributo no prueba peso cero.',
           'text__ identifica TF-IDF del contenido combinado, no necesariamente una palabra del título; num__ usa variables imputadas, transformadas con log1p y escaladas; cat__ identifica indicadores de región. Los pesos no están en la escala de las columnas originales.']
    family=result.get('selected','')
    if family=='lda_classifier':
        lines.append('LDA supervisado: los pesos svd__ se aplican a componentes SVD escalados. Cada componente combina atributos y su peso corresponde a una puntuación de decisión.')
    elif family=='svm':
        lines.append('SVM lineal: el peso modifica la puntuación de decisión de la salida correspondiente. Un peso positivo eleva esa puntuación al aumentar ese atributo transformado, manteniendo los demás fijos; no es una probabilidad.')
    elif family=='logistic':
        lines.append('Regresión logística: los pesos modifican los log-odds de la clase positiva; la función logística los convierte en probabilidad estimada. El coeficiente no es un cambio directo de probabilidad. El umbral OOF decide después y no altera estos pesos.')
    elif family.endswith('_log_target'):
        lines.append('Regresión con objetivo log1p: los pesos lineales actúan sobre log1p(y); expm1 devuelve la predicción a la escala original. Un coeficiente de 1 no significa sumar un like ni una unidad de ratio, y no se aplica un multiplicador sin considerar el cambio real del atributo transformado.')
    else:
        lines.append('La interpretación exacta depende del estimador y de sus transformaciones; no se atribuye una escala a partir del nombre del atributo.')
    if task=='category':
        report=result['holdout']['classification_report']
        labels=sorted(data['class'].unique(),key=lambda label:(-report.get(label,{}).get('support',0),label))[:3]
        lines.append('Ejemplos de hasta tres categorías con mayor soporte real entre las exportadas; el extracto completo queda en el detalle. Cada salida tiene su propio vector, no una importancia global común.')
    else:
        labels=list(data['class'].unique())
    for label in labels:
        block=data.loc[data['class']==label]
        strongest=block.reindex(block.coefficient.abs().sort_values(ascending=False,kind='stable').index).head(3)
        lines.append('Salida '+safe(label)+': '+ '; '.join(f'`{safe(row.feature)}` = {format_number(row.coefficient,4)}' for row in strongest.itertuples())+'.')
        lines.append(f'En el extracto de esta salida: pesos positivos {format_number((block.coefficient>0).sum(),0)}, negativos {format_number((block.coefficient<0).sum(),0)}, ceros {format_number((block.coefficient==0).sum(),0)}. Estos recuentos no describen todos los pesos del modelo.')
    lines.append('Un peso negativo reduce esa puntuación o respuesta transformada, no implica efecto perjudicial. La contribución de un caso requiere peso × valor transformado, más intercepto y el resto de atributos. TF-IDF se normaliza: añadir una palabra puede cambiar otras coordenadas; los atributos están relacionados.')
    lines.append('No se comparan magnitudes entre escalas o modelos como importancia universal, ni se concluye causalidad por términos, canales o regiones.')
    if task=='category':
        lines.append('La clasificación multiclase compara puntuaciones de varias salidas; elevar una no garantiza cambiar la etiqueta final.')
    elif task in ['comments_disabled','ratings_disabled']:
        lines.append('En clasificación binaria, la etiqueta depende de la puntuación o probabilidad y del umbral de decisión; cambiar una contribución no garantiza cruzar ese umbral.')
    else:
        lines.append('En regresión se estima una respuesta numérica, no una etiqueta de clase; su interpretación debe respetar las transformaciones del objetivo y la escala de salida.')
    if task in ['category', 'comments_disabled', 'ratings_disabled']:
        lines.append('**Lectura conjunta.** Los términos también pueden recoger nombres propios y '
                     'plantillas recurrentes de creadores. El diagnóstico por canales disjuntos '
                     '(apartado de calidad) comprueba si el rendimiento se mantiene al cambiar ese contexto; '
                     'un coeficiente aislado no demuestra memorización.')
    lines.append('No se reajusta el modelo mirando estas asociaciones.')
    return '\n\n'.join(lines)


# Punto de entrada: todos los auxiliares se definen antes de main.



def precision_recall_summary(task, result, curve):
    """Contrasta curva PR, AP y punto operativo con los recuentos de prueba.

    AP usa sumas de incrementos de recall ponderados por precisión, no trapecios.
    El extremo final sin umbral tiene precisión convencional 1 y recall 0. El
    umbral OOF se describe sin optimizar otro punto con las etiquetas de prueba.
    """
    # numpy: valida arrays, reproduce AP y localiza el corte operativo en la curva.
    import numpy as np
    classification_summary(task,result['holdout'])
    classification_summary(task,result['default_threshold_holdout'])
    if result['holdout']['labels']!=['0','1'] or result['default_threshold_holdout']['labels']!=['0','1']:
        raise ValueError(f'{task}: la curva PR requiere el orden binario 0, 1')
    if not isinstance(curve,pd.DataFrame) or len(curve)<2 or not {'precision','recall','threshold'}.issubset(curve.columns):
        raise ValueError(f'{task}: curva PR vacía o incompleta')
    try:
        p=curve.precision.to_numpy(dtype=float);rec=curve.recall.to_numpy(dtype=float);thresholds=curve.threshold.to_numpy(dtype=float)
    except (ValueError,TypeError) as exc:
        raise ValueError(f'{task}: curva PR no numérica') from exc
    if not np.isfinite(p).all() or not np.isfinite(rec).all() or ((p<0)|(p>1)|(rec<0)|(rec>1)).any() or (np.diff(rec)>1e-12).any():
        raise ValueError(f'{task}: precisión/recall fuera de dominio o recall desordenado')
    if not np.isnan(thresholds[-1]) or not np.isfinite(thresholds[:-1]).all() or (np.diff(thresholds[:-1])<=0).any() or not np.isclose(p[-1],1.) or not np.isclose(rec[-1],0.) or not np.isclose(rec[0],1.):
        raise ValueError(f'{task}: umbrales o extremos de la curva PR incoherentes')
    for field in ['average_precision','positive_prevalence','positive_precision','positive_recall','positive_f2','threshold']:
        value=result.get(field)
        if isinstance(value,bool) or not isinstance(value,(float,int)) or not np.isfinite(value):
            raise ValueError(f'{task}: {field} no es finito')
        if field!='threshold' and not 0<=value<=1:
            raise ValueError(f'{task}: {field} fuera de [0,1]')
    matrix=np.asarray(result['holdout']['confusion_matrix'],dtype=float)
    default=np.asarray(result['default_threshold_holdout']['confusion_matrix'],dtype=float)
    tn,fp,fn,tp=matrix.ravel();total=matrix.sum();positive=tp+fn
    if total!=result['test_rows'] or default.sum()!=total or not np.array_equal(matrix.sum(axis=1),default.sum(axis=1)):
        raise ValueError(f'{task}: las matrices no representan los mismos casos de prueba')
    if positive==0:
        raise ValueError(f'{task}: sin positivos reales no se interpreta recall ni AP como detección; revisar el soporte')
    precision=tp/(tp+fp) if tp+fp else 0.
    recall=tp/positive
    f2=f2_from_precision_recall(precision,recall)
    ap=float(np.sum((rec[:-1]-rec[1:])*p[:-1]))
    expected={'positive_prevalence':positive/total,'positive_precision':precision,'positive_recall':recall,'positive_f2':f2,'average_precision':ap}
    if any(not np.isclose(result[field],value,rtol=1e-8,atol=1e-10) for field,value in expected.items()) or not np.isclose(p[0],positive/total,rtol=1e-8,atol=1e-10):
        raise ValueError(f'{task}: curva PR, AP, prevalencia o punto operativo no concuerdan con el informe')
    index=int(np.searchsorted(thresholds[:-1],result['threshold'],side='left'))
    if not np.isclose(rec[index],recall,rtol=1e-8,atol=1e-10) or (tp+fp>0 and not np.isclose(p[index],precision,rtol=1e-8,atol=1e-10)):
        raise ValueError(f'{task}: el umbral OOF no coincide con el punto operativo de la curva')
    dtn,dfp,dfn,dtp=default.ravel()
    dp=dtp/(dtp+dfp) if dtp+dfp else 0.
    dr=dtp/(dtp+dfn) if dtp+dfn else 0.
    default_f2=f2_from_precision_recall(dp,dr)
    direction='mejora' if f2>default_f2 else 'empeora' if f2<default_f2 else 'empata'
    lines=[f'**Curva precisión–recuperación: {task_description(task)}.** {format_number(positive,0)} positivos reales de {format_number(total,0)} apariciones de prueba; prevalencia {format_number(positive/total*100,2)} %. Eje X = recall de la clase 1; eje Y = precisión de la clase 1.',
           f'La línea horizontal muestra la prevalencia, referencia descriptiva de precisión para una selección aleatoria independiente del objetivo. No es un modelo entrenado ni una garantía exacta de AP aleatoria en esta muestra finita. AP observada {format_number(ap,4)} resume el ranking en todos los cortes; no es accuracy, precisión en el punto marcado ni una probabilidad calibrada.',
           'AP se reproduce como suma de (recall anterior − recall siguiente) × precisión del tramo. No se usa integración trapezoidal; dibujar segmentos o escalones no cambia la definición de la métrica.',
           f'Umbral OOF ya fijado: {format_number(result["threshold"],6)}. Punto en prueba: precisión {format_number(precision*100,2)} %, recall {format_number(recall*100,2)} %, F2 {format_number(f2,4)}. '+binary_score_explanation(result),
           f'Con umbral predeterminado: precisión {format_number(dp*100,2)} %, recall {format_number(dr*100,2)} %, F2 {format_number(default_f2,4)}. El umbral OOF {direction} F2 en esta prueba. FP {format_number(fp,0)} frente a {format_number(dfp,0)}; FN {format_number(fn,0)} frente a {format_number(dfn,0)}. Se conserva la decisión de entrenamiento aunque el test resulte desfavorable.',
           'La última fila tiene precisión 1, recall 0 y carece de umbral: es un extremo convencional sin positivos seleccionados, no detección perfecta ni un punto operativo con precisión estimable.',
           'La curva de prueba describe compromisos y no se usa para escoger otro umbral. AP resume ordenación; el punto operativo debe valorarse con costes, soportes y evaluación externa. Las apariciones pueden repetir vídeos.']
    if tp+fp==0:
        lines.append('Sin positivos predichos, la precisión del punto operativo no es estimable; el informe registra cero por convención y el gráfico omite ese marcador.')
    if tn+fp==0:
        lines.append('No hay negativos reales: la curva no permite evaluar discriminación frente a esa clase, aunque AP sea alta.')
    return '\n\n'.join(lines)


# Punto de entrada: los auxiliares se definen antes de main.



def ratio_ablation_summary(result):
    """Valida y explica la ablación condicional de contadores, fold a fold.

    Recupera los MAE con contadores del candidato seleccionado, invirtiendo el
    signo de neg_mean_absolute_error. No reoptimiza ninguna variante ni consulta
    el holdout para decidir. El soporte estadístico no se infiere de tres folds.
    """
    # numpy: comprueba finitud, medias y diferencias emparejadas entre folds.
    import numpy as np
    ablation=result.get('ratio_ablation') if isinstance(result,dict) else None
    if not isinstance(ablation,dict):
        raise ValueError('ratio: falta el informe de ablación')
    for field in ['without_views_comment_count_mae','with_counters_mae']:
        value=ablation.get(field)
        if isinstance(value,bool) or not isinstance(value,(float,int)) or not np.isfinite(value) or value<0:
            raise ValueError(f'ratio: {field} debe ser un MAE finito no negativo')
    refinement=result.get('refinement')
    if refinement:
        index=refinement.get('best_index');trials=refinement.get('trials',[])
        if isinstance(index,bool) or not isinstance(index,int) or not 0<=index<len(trials):
            raise ValueError('ratio: índice del refinamiento no válido para la ablación')
        with_raw=trials[index].get('fold_scores')
        cv_score=refinement.get('cv_score')
    else:
        trial=next((row for row in result.get('trials',[]) if row.get('model')==result.get('selected') and 'error' not in row),None)
        if trial is None:raise ValueError('ratio: falta el candidato seleccionado con contadores')
        with_raw=trial.get('fold_scores');cv_score=trial.get('cv_score')
    without_raw=ablation.get('without_fold_mae')
    for values in [with_raw,without_raw]:
        if not isinstance(values,list) or len(values)<2 or any(isinstance(value,bool) or not isinstance(value,(float,int)) or not np.isfinite(value) for value in values):
            raise ValueError('ratio: faltan puntuaciones finitas de al menos dos folds de ablación')
    with_mae=-np.asarray(with_raw,dtype=float);without_mae=np.asarray(without_raw,dtype=float)
    if len(with_mae)!=len(without_mae) or (with_mae<0).any() or (without_mae<0).any():
        raise ValueError('ratio: folds no alineados o signos de MAE inválidos en la ablación')
    if isinstance(cv_score,bool) or not isinstance(cv_score,(float,int)) or not np.isfinite(cv_score) or not np.isclose(-cv_score,with_mae.mean(),rtol=1e-9,atol=1e-10):
        raise ValueError('ratio: puntuación CV no coincide con los folds con contadores')
    if not np.isclose(with_mae.mean(),ablation['with_counters_mae'],rtol=1e-9,atol=1e-10) or not np.isclose(without_mae.mean(),ablation['without_views_comment_count_mae'],rtol=1e-9,atol=1e-10):
        raise ValueError('ratio: las medias de ablación no coinciden con sus folds')
    # difference: positivo indica menor MAE al incluir los contadores.
    difference=without_mae-with_mae
    mean_difference=float(difference.mean())
    benefit=int((difference>0).sum());worse=int((difference<0).sum());ties=int((difference==0).sum())
    direction='menor' if mean_difference>0 else 'mayor' if mean_difference<0 else 'igual'
    lines=['**Ablación en la proporción «me gusta/no me gusta».** Se retiran conjuntamente views y comment_count, manteniendo la familia, la receta seleccionada y los mismos folds por vídeo. No se retiran likes y dislikes: ya están excluidos como predictores para evitar revelar directamente el cociente.',
           f'MAE medio con contadores {format_number(with_mae.mean(),4)}; sin ellos {format_number(without_mae.mean(),4)}. Con contadores el MAE es {direction}. Diferencia sin − con: {format_number(mean_difference,4)} unidades de ratio; positivo favorece incluirlos.']
    if without_mae.mean()>0:
        lines.append(f'Reducción relativa sobre la variante sin contadores: {format_number(mean_difference/without_mae.mean()*100,2)} %. Es una diferencia de error, no porcentaje de aciertos ni prueba de significación.')
    else:
        lines.append('El MAE sin contadores es cero: la reducción relativa no está definida y no se divide por cero.')
    lines.append(f'Comparación emparejada: menor MAE con contadores en {format_number(benefit,0)} de {format_number(len(with_mae),0)} folds; mayor en {format_number(worse,0)}; empate en {format_number(ties,0)}. No se supone un beneficio uniforme.')
    lines.extend(['','| Fold | MAE con contadores | MAE sin contadores | Sin − con |','|---|---:|---:|---:|'])
    for i,(with_value,without_value,delta) in enumerate(zip(with_mae,without_mae,difference),1):
        lines.append(f'| {i} | {format_number(with_value,4)} | {format_number(without_value,4)} | {format_number(delta,4)} |')
    lines.extend(['','Los hiperparámetros se seleccionaron con contadores; la variante sin ellos no recibió otra búsqueda. La ablación es condicional a esa receta y no demuestra que todo modelo sin contadores sea peor. Las dos variables se retiran juntas: no se identifica su aportación individual.',
                  'Las particiones comparten parte del entrenamiento y su número es reducido; una diferencia media pequeña no acredita significación ni causalidad. Ambos ajustes reaprenden sus transformadores dentro del fold; no se supone que todos los pesos o escalas permanezcan idénticos al quitar variables.',
                  'views y comment_count son señales contemporáneas. Esta comparación no convierte el ejercicio en predicción antes de publicar, no garantiza disponibilidad futura y no autoriza cambiar atributos tras observar el holdout. Se conserva el resultado del estudio de desarrollo.'])
    return '\n'.join(lines)


# Punto de entrada: todos los auxiliares se definen antes de main.



def clustering_representation(root):
    """Lee el estado de representación de la segmentación guardado, sin fit ni predict.

    Devuelve dimensiones, atributos de entrada y varianza explicada del SVD real.
    La varianza corresponde a ese corpus y representación, no al PCA numérico.
    """
    # configparser y models: localizan y recuperan el pipeline final existente.
    import configparser
    from models import REGISTRY
    root=Path(root);conf=configparser.ConfigParser()
    if not conf.read(root/'train.conf',encoding='utf-8'):
        raise FileNotFoundError(f'No existe la configuración: {root/"train.conf"}')
    model=REGISTRY['cluster']().load(root/conf['output']['cluster'])
    prep=model.pipeline_.named_steps['features'];reducer=model.pipeline_.named_steps['reduce']
    names=prep.get_feature_names_out()
    return {'input_columns':list(model.columns_), 'transformed_features':len(names),
            'svd_dimensions':int(reducer.components_.shape[0]),'svd_input_features':int(reducer.components_.shape[1]),
            'explained_variance_fraction':float(reducer.explained_variance_ratio_.sum()),
            'fit_rows':model.training_manifest_.get('fit_rows'),
            'numeric_columns':list(next(columns for name,transform,columns in prep.transformers_ if name=='num'))}


def clustering_overview(report, sample, representation):
    """Valida y explica corpus, representación y alcance de los ensayos guardados.

    No selecciona ni recalcula clusters. Separa comparaciones inductivas y las
    exploraciones transductivas, cuyos n no representan el mismo ajuste.
    """
    # numpy y collections: validan dimensiones y cuentan familias del informe real.
    import numpy as np
    from collections import Counter
    if not isinstance(sample,pd.DataFrame) or not {'video_id','cluster'}.issubset(sample.columns) or len(sample)<3:
        raise ValueError('Clustering: muestra ausente o insuficiente para interpretar segmentación')
    if sample.video_id.isna().any() or sample.video_id.duplicated().any() or not sample.video_id.map(lambda value:isinstance(value,str) and bool(value.strip())).all():
        raise ValueError('Clustering: la muestra debe contener una fila por video_id utilizable')
    labels=pd.to_numeric(sample.cluster,errors='coerce')
    if not np.isfinite(labels).all() or (labels<0).any() or (labels!=np.floor(labels)).any():
        raise ValueError('Clustering: etiquetas finales ausentes, negativas o no enteras')
    selected=report.get('cluster_selected',{});trials=report.get('clustering')
    if not isinstance(trials,list) or not trials:
        raise ValueError('Clustering: faltan ensayos para explicar la comparación')
    nfeatures=representation.get('transformed_features');dimension=representation.get('svd_dimensions')
    for value in [nfeatures,dimension,representation.get('svd_input_features'),representation.get('fit_rows')]:
        if isinstance(value,bool) or not isinstance(value,int) or value<1:
            raise ValueError('Clustering: tamaños de representación no válidos')
    variance=representation.get('explained_variance_fraction')
    if not isinstance(variance,(float,int)) or not np.isfinite(variance) or not 0<=variance<=1+1e-8:
        raise ValueError('Clustering: varianza explicada SVD no válida')
    if representation['fit_rows']!=len(sample) or representation['svd_input_features']!=nfeatures or not dimension<min(nfeatures,len(sample)) or dimension!=selected.get('dimensions'):
        raise ValueError('Clustering: muestra, atributos, SVD e informe no concuerdan')
    if labels.nunique()!=selected.get('k'):
        raise ValueError('Clustering: grupos observados no coinciden con el resultado final')
    inductive=[row for row in trials if row.get('name') not in ['agglomerative','dbscan']]
    exploratory=[row for row in trials if row.get('name') in ['agglomerative','dbscan']]
    if not inductive:
        raise ValueError('Clustering: no hay comparaciones inductivas declaradas')
    for row in trials:
        if not isinstance(row.get('name'),str) or not row['name'] or not isinstance(row.get('dimensions'),int) or row['dimensions']<1:
            raise ValueError('Clustering: familia o dimensión de ensayo no válida')
        count=row.get('evaluation_n');fit_rows=row.get('n')
        if not isinstance(count,int) or not isinstance(fit_rows,int) or not 2<=count<=fit_rows:
            raise ValueError('Clustering: tamaños de ajuste/evaluación incoherentes en ensayo')
        if row in inductive and fit_rows!=len(sample):
            raise ValueError('Clustering: ensayos inductivos sobre corpus de tamaños distintos')
        if row in exploratory and (row.get('eligible') or row.get('selectable')):
            raise ValueError('Clustering: una exploración transductiva no puede declararse seleccionable')
    if len({row['evaluation_n'] for row in trials})!=1:
        raise ValueError('Clustering: tamaños de evaluación distintos; revisar la comparabilidad')
    families=Counter(row['name'] for row in inductive);other=Counter(row['name'] for row in exploratory)
    dimensions=sorted({row['dimensions'] for row in inductive})
    total_evaluation=inductive[0]['evaluation_n']
    lines=[f'**Entrada a la segmentación.** {format_number(len(sample),0)} vídeos distintos, una fila por vídeo del pool de desarrollo ampliado. Se conserva la primera aparición disponible al deduplicar; no se promedian todas sus apariciones ni se obtiene una muestra temporal prospectiva.',
           'El pool anterior se muestreó por estratos de categoría; después se deduplicaron vídeos y se tomó una muestra aleatoria con semilla. category no es atributo de entrada, objetivo de ajuste ni criterio para elegir el clustering, pero sí influyó en la composición por ese muestreo previo.',
           f'El pipeline guardado combina TF-IDF de título, tags y descripción, indicadores de región y atributos numéricos transformados. Columnas numéricas: {", ".join(representation["numeric_columns"])}. likes puede aparecer en perfiles descriptivos, pero no es entrada numérica de la segmentación actual.',
           f'Representación aprendida: {format_number(nfeatures,0)} atributos transformados → SVD de {format_number(dimension,0)} componentes en el modelo guardado; fracción de varianza explicada {format_number(variance*100,2)} % en esa representación de ajuste. No equivale a porcentaje de calidad, categorías acertadas o utilidad comercial.',
           f'Comparación registrada: {format_number(len(inductive),0)} ensayos inductivos; dimensiones SVD {dimensions}; familias y cantidades: '+', '.join(f'{name}={format_number(count,0)}' for name,count in sorted(families.items()))+'.',
           f'Cada ajuste inductivo usa {format_number(len(sample),0)} vídeos; la silueta se calcula sobre los mismos {format_number(total_evaluation,0)} índices de evaluación. Son observaciones del corpus ajustado, no una reserva de prueba externa de clustering. La igualdad de tamaños se valida aquí; la identidad de los índices proviene del código y la semilla compartidos.',
           f'Exploraciones separadas: {format_number(len(exploratory),0)} ensayos ('+', '.join(f'{name}={format_number(count,0)}' for name,count in sorted(other.items()))+'). Aglomerativo y DBSCAN se ajustan sobre los índices comunes; no aportan predict nativo para nuevas filas en esta interfaz y no compiten como ganadores inductivos.',
           f'Resultado guardado: {selected.get("name")}, {format_number(selected["k"],0)} grupos, SVD {format_number(dimension,0)}. Los números de grupo son identificadores arbitrarios, no etiquetas editoriales ni niveles de calidad. La regla de elección y las métricas se estudiarán después.',
           'SVD permite una representación densa de menor dimensión a partir de muchos atributos dispersos, sin centrar la matriz como PCA. Cambia la geometría: sus componentes mezclan señales y no son géneros predefinidos. Las proyecciones PCA/t-SNE de cuatro contadores del cuaderno son otro espacio y no validan por sí solas estos grupos.']
    return '\n\n'.join(lines)


# Punto de entrada: todos los auxiliares se definen antes de main.



def cluster_selection_summary(report, quality=None):
    """Contrasta criterios, banderas y ganador, y explica silueta y estabilidad.

    Reutiliza selection_details y sus constantes canónicas. No ajusta estimadores,
    relaja límites ni usa el solapamiento con categorías para elegir. Los tamaños
    de cada ensayo se contrastan como resumen, no como etiquetas reproducidas.
    """
    # numpy: valida números y reproduce medias de estabilidad.
    import numpy as np
    # clustering_experiments: mantiene una única regla de tamaño/estabilidad.
    from clustering_experiments import selection_details,MIN_CLUSTER_FRACTION,MIN_STABILITY_ARI
    trials=report.get('clustering') if isinstance(report,dict) else None
    selected=report.get('cluster_selected',{}) if isinstance(report,dict) else {}
    if not isinstance(trials,list) or not trials:
        raise ValueError('Clustering: faltan ensayos para verificar la selección')
    inductive=[row for row in trials if row.get('name') not in ['agglomerative','dbscan']]
    for row in inductive:
        sizes=row.get('cluster_sizes')
        if not isinstance(sizes,dict) or not sizes or any(isinstance(value,bool) or not isinstance(value,int) or value<=0 for value in sizes.values()) or sum(sizes.values())!=row.get('n'):
            raise ValueError('Clustering: tamaños de grupos ausentes o no compatibles con n')
        score=row.get('silhouette');groups=row.get('observed_evaluation_clusters');evaluation_n=row.get('evaluation_n')
        if score is not None and (isinstance(score,bool) or not isinstance(score,(float,int)) or not np.isfinite(score) or not -1<=score<=1):
            raise ValueError('Clustering: silueta no finita o fuera de [-1,1]')
        coverage=row.get('coverage')
        if not isinstance(coverage,(float,int)) or not np.isfinite(coverage) or not 0<=coverage<=1 or not isinstance(groups,int) or not isinstance(evaluation_n,int):
            raise ValueError('Clustering: cobertura o grupos evaluados incoherentes')
        technical=score is not None and coverage==1 and bool(row.get('converged')) and 1<groups<evaluation_n
        if not isinstance(row.get('eligible'),bool) or row['eligible']!=technical:
            raise ValueError('Clustering: eligible no concuerda con validez técnica declarada')
        stability=row.get('stability_ARI')
        if not isinstance(stability,list) or any(isinstance(value,bool) or not isinstance(value,(float,int)) or not np.isfinite(value) or not -1<=value<=1 for value in stability):
            raise ValueError('Clustering: réplicas ARI inválidas')
        mean=row.get('stability_ARI_mean')
        if stability:
            if not isinstance(mean,(float,int)) or not np.isfinite(mean) or not np.isclose(mean,np.mean(stability),rtol=1e-9,atol=1e-12):
                raise ValueError('Clustering: ARI medio no coincide con réplicas')
        elif mean is not None:
            raise ValueError('Clustering: ARI medio declarado sin réplicas')
        # Codificar grupos por posición conserva sus tamaños para aplicar la regla canónica.
        labels=np.repeat(np.arange(len(sizes)),list(sizes.values()))
        details=selection_details(labels,row['eligible'],stability)
        if not isinstance(row.get('selectable'),bool) or row['selectable']!=details['selectable'] or not np.isclose(row.get('min_cluster_fraction',float('nan')),details['min_cluster_fraction'],rtol=1e-9,atol=1e-12):
            raise ValueError('Clustering: selectable o fracción mínima no concuerda con la regla')
    valid=[row for row in inductive if row['eligible']]
    allowed=[row for row in inductive if row['selectable']]
    if not allowed:
        raise ValueError('Clustering: ninguna solución cumple; no se relajan límites ni se inventa ganador')
    expected=max(allowed,key=lambda row:row['silhouette'])
    if any(selected.get(key)!=expected.get(key) for key in ['name','k','dimensions']) or not np.isclose(selected.get('silhouette',float('nan')),expected['silhouette'],rtol=1e-9,atol=1e-12):
        raise ValueError('Clustering: el ganador no maximiza silueta entre soluciones que cumplen')
    full=(quality or {}).get('full_pipeline_stability',{})
    if full:
        replicas=full.get('replicate_ARI')
        if not isinstance(replicas,list) or not replicas or any(not isinstance(value,(float,int)) or not np.isfinite(value) or not -1<=value<=1 for value in replicas) or not np.isclose(full.get('mean_ARI',float('nan')),np.mean(replicas),rtol=1e-9,atol=1e-12):
            raise ValueError('Clustering: estabilidad del pipeline completo incoherente')
        if full.get('audit_training_video_overlap')!=0 or not full.get('all_preprocessing_refit') or not full.get('selected_recipe_fixed'):
            raise ValueError('Clustering: protocolo de auditoría del pipeline no coincide con lo declarado')
    best_geometry=max(valid,key=lambda row:row['silhouette'])
    best_stability_text=format_number(best_geometry['stability_ARI_mean'],4) if best_geometry['stability_ARI_mean'] is not None else 'no disponible (sin réplicas válidas)'
    lines=[f'**Selección de clustering.** {format_number(len(valid),0)} ensayos inductivos técnicamente elegibles; {format_number(len(allowed),0)} cumplen además tamaño y estabilidad. eligible exige convergencia declarada, cobertura completa y silueta calculable con más de un grupo y menos grupos que casos evaluados.',
           f'Regla adicional: cada grupo ocupa al menos {format_number(MIN_CLUSTER_FRACTION*100,0)} % del corpus y hay tres réplicas válidas con ARI medio ≥ {format_number(MIN_STABILITY_ARI,2)}. El límite de ARI se aplica a la media, no obliga a que cada réplica supere ese valor.',
           'La silueta de un caso compara a, distancia media dentro de su grupo, con b, menor distancia media a otro grupo: s=(b−a)/max(a,b). El informe usa distancia euclídea en SVD e índices comunes. Cerca de 1 indica separación relativa; cerca de 0, fronteras difusas; negativo, posible mejor proximidad a otro grupo. No es accuracy ni porcentaje de categorías acertadas.',
           f'Ganador entre quienes cumplen: {expected["name"]}, k={expected["k"]}, SVD={expected["dimensions"]}, silueta {format_number(expected["silhouette"],4)}. Tamaños de grupo: '+', '.join(format_number(value,0) for value in expected['cluster_sizes'].values())+f'; grupo mínimo {format_number(expected["min_cluster_fraction"]*100,2)} %.',
           f'Entre todos los técnicamente elegibles, la mayor silueta corresponde a {best_geometry["name"]}, k={best_geometry["k"]}, SVD={best_geometry["dimensions"]}: {format_number(best_geometry["silhouette"],4)}. Grupo mínimo {format_number(best_geometry["min_cluster_fraction"]*100,2)} %; ARI medio {best_stability_text}. '+('No cumple los límites adicionales.' if not best_geometry['selectable'] else 'También cumple los límites adicionales.'),
           f'Estabilidad del ganador con SVD fijo: réplicas [{"; ".join(format_number(value,4) for value in expected["stability_ARI"])}], media {format_number(expected["stability_ARI_mean"],4)}. Se reajusta el algoritmo en remuestreos del 80 % y se comparan asignaciones en los mismos índices; esos índices pueden participar en los remuestreos. No es una prueba sobre vídeos disjuntos.',
           'ARI compara dos particiones y ajusta por acuerdo esperado al azar; no cambia al permutar los números de grupo. Aquí compara referencia y réplica, no géneros editoriales. El campo ARI frente a category responde a otra pregunta y no interviene en la selección.']
    if full:
        lines.append(f'Diagnóstico distinto del pipeline completo: ARI medio {format_number(full["mean_ARI"],4)}, réplicas [{"; ".join(format_number(value,4) for value in full["replicate_ARI"])}]. Reaprende transformadores y SVD con receta fija; {format_number(full["training_videos"],0)} vídeos de ajuste y {format_number(full["audit_videos"],0)} de auditoría, sin IDs compartidos en esos ajustes. Es otra muestra interna de la fuente histórica; no mide una nueva selección de receta ni es validación externa. Este ARI no se usa en el filtro anterior.')
    policy=report.get('clustering_selection_policy')
    if not isinstance(policy,str) or not policy.strip():
        raise ValueError('Clustering: falta declarar el origen y alcance de la política de selección')
    lines.extend([localize_policy(policy),'Estabilidad alta no garantiza separación fuerte, géneros útiles ni segmentos comerciales. Los límites se añadieron después de observar la solución anterior: no se presentan como criterios a priori ni como requisitos del profesor. No se cambia la política al mirar ahora estas tablas.'])
    return '\n\n'.join(lines)


# Punto de entrada: todos los auxiliares se definen antes de main.

def cluster_profiles_summary(profiles, sample):
    """Describe tamaños y medianas y contrasta perfiles con los vídeos publicados.

    No asigna significado comercial ni causal. Las razones sólo se calculan con
    denominador positivo; las modas no se presentan como mayorías ni exclusivas.
    """
    # numpy: valida valores finitos y compara medianas con tolerancia numérica.
    import numpy as np
    required = ['cluster', 'n', 'views_median', 'likes_median', 'dislikes_median',
                'comments_median', 'title_length_median', 'top_category', 'top_region']
    if profiles.empty or not set(required).issubset(profiles.columns):
        raise ValueError('Los perfiles de clustering están vacíos o faltan columnas requeridas')
    if not {'video_id', 'cluster', 'category', 'views', 'likes', 'dislikes', 'comment_count'}.issubset(sample.columns):
        raise ValueError('La muestra de clustering no contiene las columnas necesarias para validar perfiles')
    if sample.video_id.isna().any() or sample.video_id.astype(str).str.strip().eq('').any() or sample.video_id.duplicated().any():
        raise ValueError('La muestra debe contener una identidad utilizable y única por vídeo')
    if profiles.cluster.duplicated().any() or profiles.cluster.isna().any() or sample.cluster.isna().any():
        raise ValueError('Los identificadores de grupo faltan o están duplicados en los perfiles')
    if set(profiles.cluster) != set(sample.cluster):
        raise ValueError('Los grupos de los perfiles no coinciden con la muestra')
    metrics = {'views_median': 'views', 'likes_median': 'likes',
               'dislikes_median': 'dislikes', 'comments_median': 'comment_count'}
    total = len(sample)
    lines = [f'### Qué caracteriza a los grupos ({format_number(total, 0)} vídeos únicos)',
             'Los identificadores son arbitrarios. n cuenta vídeos, no apariciones regionales. '
             'Las medianas resumen el valor central: no son medias, sumas ni tasas.']
    for row in profiles.itertuples():
        group = sample.loc[sample.cluster == row.cluster]
        if not np.isfinite(row.n) or row.n != len(group):
            raise ValueError(f'El tamaño del grupo {row.cluster} no coincide con la muestra')
        for column, source in metrics.items():
            values = pd.to_numeric(group[source], errors='coerce')
            if not np.isfinite(values).all() or (values < 0).any():
                raise ValueError(f'El grupo {row.cluster} contiene contadores inválidos en {source}')
            value = getattr(row, column)
            if not np.isfinite(value) or value < 0 or not np.isclose(value, values.median(), rtol=1e-10, atol=1e-8):
                raise ValueError(f'La mediana de {source} del grupo {row.cluster} no coincide con la muestra')
        if not np.isfinite(row.title_length_median) or row.title_length_median < 0:
            raise ValueError('La longitud mediana del título debe ser finita y no negativa')
        modes = group.category.dropna().mode()
        if modes.empty or row.top_category not in modes.values or pd.isna(row.top_region) or not str(row.top_region).strip():
            raise ValueError(f'La categoría modal o región del grupo {row.cluster} no es válida')
        category_n = int(group.category.eq(row.top_category).sum())
        lines.append(f'**Grupo {row.cluster}:** {format_number(row.n, 0)} vídeos '
                     f'({format_number(row.n / total * 100, 2)} %). Medianas: '
                     f'{format_number(row.views_median, 1)} visualizaciones; '
                     f'{format_number(row.likes_median, 1)} likes; '
                     f'{format_number(row.dislikes_median, 1)} dislikes; '
                     f'{format_number(row.comments_median, 1)} comentarios; '
                     f'{format_number(row.title_length_median, 1)} caracteres en el título. '
                     f'Categoría modal publicada: {row.top_category}, con {format_number(category_n, 0)} vídeos '
                     f'({format_number(category_n / row.n * 100, 2)} % del grupo); '
                     f'{"es una de varias categorías empatadas en frecuencia" if len(modes) > 1 else "es la categoría más frecuente"}. '
                     f'Región modal publicada: {row.top_region}. Una moda no implica mayoría ni exclusividad.')
    low, high = profiles.views_median.min(), profiles.views_median.max()
    if high == low:
        lines.append('Las medianas de visualizaciones son iguales: no se establece un orden de audiencia entre estos grupos.')
    elif low > 0:
        lines.append(f'La mayor mediana de visualizaciones es {format_number(high / low, 2)} veces la menor. '
                     'Es una razón de medianas, no el cociente para cada vídeo ni una tasa de interacción.')
    else:
        lines.append('La mediana menor de visualizaciones es cero: no se calcula una razón, porque dividir entre cero no está definido.')
    lines.append('Los perfiles describen esta muestra y la aparición retenida de cada vídeo; no representan su evolución temporal. '
                 'likes se describe, pero no es entrada numérica de la segmentación. '
                 'Título y región no se conservan en cluster_sample.csv: sus resúmenes publicados no se pueden reproducir sólo desde ese CSV. '
                 'Una diferencia descriptiva de audiencia no demuestra causalidad, una separación puramente temática ni utilidad comercial.')
    return '\n\n'.join(lines)




def cluster_category_summary(report, sample, counts, proportions):
    """Contrasta contingencias y métricas y explica asociación editorial sin elegir.

    Las proporciones se normalizan dentro de cada grupo. Se exige una categoría
    conocida por vídeo para que pandas no descarte filas silenciosamente.
    """
    # numpy: comprueba finitud y tolerancia de las evidencias numéricas guardadas.
    import numpy as np
    # sklearn.metrics: reproduce acuerdo ajustado e información mutua normalizada.
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    if sample.empty or not {'video_id','cluster','category'}.issubset(sample.columns):
        raise ValueError('La muestra editorial está vacía o faltan columnas requeridas')
    for column in ['video_id','cluster','category']:
        if sample[column].isna().any() or sample[column].astype(str).str.strip().eq('').any():
            raise ValueError(f'La muestra editorial contiene valores ausentes en {column}')
    if sample.video_id.duplicated().any():
        raise ValueError('La comparación editorial requiere una fila por vídeo')
    expected = pd.crosstab(sample.cluster, sample.category)
    fractions = expected.div(expected.sum(axis=1), axis=0)
    for label, frame, reference in [('recuentos',counts,expected),('proporciones',proportions,fractions)]:
        if frame.index.has_duplicates or frame.columns.has_duplicates or set(frame.index)!=set(reference.index) or set(frame.columns)!=set(reference.columns):
            raise ValueError(f'Los grupos o categorías de la tabla de {label} no coinciden con la muestra')
        values = frame.reindex(index=reference.index,columns=reference.columns).to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values<0).any() or not np.allclose(values,reference.to_numpy(),rtol=1e-10,atol=1e-10):
            raise ValueError(f'La tabla de {label} no coincide con las categorías de la muestra')
    ari = float(adjusted_rand_score(sample.category, sample.cluster))
    nmi = float(normalized_mutual_info_score(sample.category, sample.cluster))
    saved = report['cluster_interpretation']
    for key,value in [('ARI',ari),('NMI',nmi)]:
        if not np.isfinite(saved[key]) or not np.isclose(saved[key],value,rtol=1e-10,atol=1e-10):
            raise ValueError(f'La métrica editorial {key} no coincide con la muestra')
    lines = ['### Relación de grupos con categorías editoriales',
             f'Se comparan {format_number(len(sample),0)} vídeos, {len(expected)} grupos y {len(expected.columns)} categorías. '
             'Cada celda de la tabla de recuentos cuenta vídeos con ambas etiquetas. '
             'El CSV de proporciones contiene fracciones de 0 a 1; la tabla porcentual multiplica por 100. '
             'El denominador es el total del grupo (fila), no el total de la categoría (columna).']
    for cluster,row in expected.iterrows():
        top = row.sort_values(ascending=False,kind='stable').head(3)
        details = '; '.join(f'{category}: {format_number(value,0)} ({format_number(value/row.sum()*100,2)} %)' for category,value in top.items())
        lines.append(f'Grupo {cluster}, categorías más frecuentes: {details}. '
                     'En caso de empate, el orden de presentación no significa una diferencia de frecuencia.')
    lines.append(f'ARI frente a categorías: **{format_number(ari,4)}**. '
                 'Compara pares de vídeos y descuenta acuerdo esperado al azar; puede ser negativo. '
                 'No es el ARI de estabilidad entre réplicas ni un porcentaje de aciertos.')
    lines.append(f'NMI: **{format_number(nmi,4)}**. '
                 'Mide información compartida normalizada (media aritmética de las entropías). '
                 'No está ajustada por azar ni representa precisión o porcentaje de categorías recuperadas.')
    if len(expected)<2 or len(expected.columns)<2:
        lines.append('La comparación tiene una sola etiqueta en al menos una partición: es degenerada y no acredita recuperación temática, incluso si una convención de la métrica da un valor alto.')
    else:
        lines.append('Las dos particiones coinciden exactamente, salvo permutación de etiquetas.' if np.isclose(ari,1,rtol=0,atol=1e-12) else
                     'Las particiones no son equivalentes. Se informa la magnitud de ambas métricas sin convertir un umbral arbitrario en una prueba estadística de asociación.')
    lines.append('Tener distinto número de grupos y categorías limita la equivalencia uno a uno, pero no invalida las métricas. '
                 'Las categorías sirven como referencia imperfecta y no eligen el ganador; sí influyeron en la estratificación del pool anterior. '
                 'La comparación es descriptiva sobre la muestra de ajuste, no validación externa ni prueba de utilidad comercial.')
    return '\n\n'.join(lines)




def cluster_term_representation(root):
    """Recupera vocabulario y centros reconstruidos del pipeline final, sin fit.

    Devuelve nombres de atributos y coordenadas aproximadas en ese mismo orden.
    No interpreta las dimensiones SVD como palabras ni modifica el modelo.
    """
    # configparser: localiza el peso declarado en train.conf, sin fijar su nombre.
    import configparser
    # models: carga la interfaz pública de la segmentación guardado.
    from models import REGISTRY
    config = configparser.ConfigParser()
    config.read(Path(root)/'train.conf', encoding='utf-8')
    model = REGISTRY['cluster']().load(Path(root)/config['output']['cluster'])
    pipeline = model.pipeline_
    estimator = pipeline.named_steps['estimator']
    centers = estimator.means_ if hasattr(estimator, 'means_') else estimator.cluster_centers_
    return (pipeline.named_steps['features'].get_feature_names_out(),
            centers @ pipeline.named_steps['reduce'].components_)


def cluster_terms_summary(terms, names, centers):
    """Valida términos y pesos contra el modelo y redacta límites descriptivos.

    Conserva el orden exportado de los doce mayores pesos textuales por centro.
    Admite pesos negativos de una reconstrucción SVD: no son probabilidades.
    """
    # numpy: reproduce el orden por peso y comprueba dimensiones y finitud.
    import numpy as np
    names = np.asarray(names, dtype=str)
    centers = np.asarray(centers, dtype=float)
    if names.ndim != 1 or centers.ndim != 2 or centers.shape[1] != len(names) or not len(centers) or not np.isfinite(centers).all():
        raise ValueError('Los centros reconstruidos no tienen dimensiones o valores válidos')
    if len(set(names)) != len(names):
        raise ValueError('El vocabulario reconstruido contiene atributos duplicados')
    indices = np.flatnonzero(np.char.startswith(names,'text__'))
    if not len(indices):
        raise ValueError('No hay atributos textuales para describir los grupos')
    if terms.empty or not {'cluster','term','reconstructed_weight'}.issubset(terms.columns):
        raise ValueError('La tabla de términos está vacía o faltan columnas requeridas')
    if terms[['cluster','term']].isna().any().any() or terms.term.astype(str).str.strip().eq('').any() or terms.duplicated(['cluster','term']).any():
        raise ValueError('Hay términos vacíos o repetidos dentro de un grupo')
    if set(terms.cluster) != set(range(len(centers))):
        raise ValueError('Los grupos de la tabla de términos no coinciden con el modelo')
    lines = ['### Palabras y expresiones de los centros reconstruidos',
             f'Se contrastan {len(centers)} grupos y {len(indices)} atributos textuales. '
             'Los términos proceden del vocabulario TF-IDF de título, tags y descripción combinados: '
             'también pueden ser expresiones de varias palabras. No identifican por sí solos el campo de origen.']
    term_sets = []
    for cluster,center in enumerate(centers):
        selected = indices[np.argsort(center[indices])[-12:][::-1]]
        block = terms.loc[terms.cluster == cluster]
        expected = [name.removeprefix('text__') for name in names[selected]]
        weights = pd.to_numeric(block.reconstructed_weight,errors='coerce').to_numpy()
        if list(block.term) != expected or not np.isfinite(weights).all() or not np.allclose(weights,center[selected],rtol=1e-9,atol=1e-10):
            raise ValueError(f'Los términos, pesos u orden del grupo {cluster} no coinciden con el modelo guardado')
        term_sets.append(set(expected))
        lines.append(f'**Grupo {cluster}, {len(block)} términos en orden descendente:** '+
                     '; '.join(f'{term} ({format_number(weight,5)})' for term,weight in zip(expected,weights))+'.')
    shared = sorted(set.intersection(*term_sets))
    lines.append(f'Términos presentes entre los destacados de todos los grupos: {len(shared)}'+
                 (': '+', '.join(shared)+'.' if shared else '. No hay coincidencias en estas listas truncadas.'))
    if any('none' in group for group in term_sets):
        lines.append('Aparece el término literal none: no es el objeto None de Python ni demuestra por sí solo que haya datos ausentes; puede ser un marcador del contenido original. No se le asigna significado temático sin revisar sus apariciones.')
    markers = sorted(set.union(*term_sets) & {'com','https','http','www','https www','youtube com'})
    if markers:
        lines.append('Se observan tokens compatibles con enlaces o plantillas: '+', '.join(markers)+
                     '. Es un indicio descriptivo, no un recuento de documentos con enlaces ni una demostración de su causa.')
    # distinctive: ejemplos calculados dentro de las listas truncadas exportadas.
    for group, vocabulary in enumerate(term_sets):
        others = set.union(*(terms for index, terms in enumerate(term_sets) if index != group)) if len(term_sets) > 1 else set()
        distinctive = sorted(vocabulary - others)
        if distinctive:
            lines.append(f'**Lectura del grupo {group}:** términos destacados que no aparecen en '
                         'las listas truncadas de los demás grupos: ' + ', '.join(distinctive) + '. '
                         'Son pistas del vocabulario y de plantillas del perfil, no palabras exclusivas '
                         'de todos sus documentos ni etiquetas temáticas verificadas.')
    lines.append('**Conclusión.** Los términos reconstruidos y sus diferencias de vocabulario '
                 'aconsejan describir perfiles de contenido y formato, junto con audiencia y región. '
                 'Las listas de términos no bastan para nombrar géneros: la correspondencia con las categorías '
                 'editoriales se cuantifica por separado con ARI y NMI.')
    lines.append('El cálculo multiplica centros en SVD por sus componentes para volver aproximadamente al espacio de atributos. '
                 'No reconstruye documentos originales ni recupera información descartada. Los pesos son coordenadas aproximadas: '
                 'no son frecuencias, probabilidades, porcentajes ni coeficientes causales; no tienen que sumar uno y pueden ser negativos.')
    lines.append('Se muestran los mayores pesos dentro de cada grupo, no las palabras más exclusivas ni las diferencias respecto a otros centros. '
                 'Las palabras compartidas y las plantillas limitan las etiquetas temáticas. No se afirma que sean categorías aprendidas. '
                 'Eliminar enlaces, marcadores o palabras frecuentes sería otro preprocesamiento que habría que comparar con un protocolo declarado; no se cambia ahora mirando el resultado.')
    return '\n\n'.join(lines)




def numeric_projection_summary(stats):
    """Explica resultados PCA/t-SNE calculados sin confundirlos con el espacio del modelo.

    Las cargas se interpretan por magnitud; el signo de un eje PCA es arbitrario.
    No atribuye separación de clusters, causalidad ni calidad a la varianza retenida.
    """
    # numpy: contrasta dimensiones y finitud antes de narrar resultados.
    import numpy as np
    ratios = np.asarray(stats['explained_variance_ratio'],dtype=float)
    loadings = np.asarray(stats['loadings'],dtype=float)
    columns = stats['columns']
    if ratios.shape!=(2,) or loadings.shape!=(len(columns),2) or not np.isfinite(ratios).all() or not np.isfinite(loadings).all() or (ratios<0).any() or ratios.sum()>1+1e-10:
        raise ValueError('Las varianzas o cargas PCA no son válidas para la explicación')
    if stats['n']<3 or not 0<stats['perplexity']<stats['n']:
        raise ValueError('El tamaño de muestra o la perplexity de t-SNE no son válidos')
    maximum=np.abs(loadings[:,0]).max()
    dominant=[column for column,value in zip(columns,np.abs(loadings[:,0])) if np.isclose(value,maximum,rtol=1e-9,atol=1e-12)]
    lines=['### Cómo leer PCA y t-SNE numéricos',
           f'Se representan {format_number(stats["n"],0)} vídeos y cuatro contadores: '+', '.join(columns)+'. '
           f'Se rellenaron {format_number(stats["missing_filled_with_zero"],0)} valores ausentes con cero para esta visualización; '
           'no es la imputación aprendida del modelo. Se aplica log1p y después estandarización en esta muestra.',
           f'PCA conserva {format_number(ratios[0]*100,2)} % en PC1 y {format_number(ratios[1]*100,2)} % en PC2; '
           f'en conjunto, {format_number(ratios.sum()*100,2)} % de la varianza de estos cuatro contadores transformados. '
           'No es porcentaje de aciertos ni varianza del texto o del espacio TF-IDF + SVD.',
           'Mayor carga absoluta de PC1: '+', '.join(dominant)+'. '
           'Las cargas orientan el eje, sin demostrar causalidad. Su signo puede invertirse sin cambiar la proyección geométrica.',
           f't-SNE usa perplexity {stats["perplexity"]}, semilla {stats["seed"]} y un máximo de {stats["max_iter"]} iteraciones. '
           'La perplexity regula la escala de vecindad, no el número de grupos. Los ejes no son variables ni porcentajes de varianza; '
           'las distancias entre islas, sus tamaños y sus huecos no validan separación global. Una semilla fija facilita repetir este cálculo, no garantiza robustez frente a otras elecciones.',
           'Los colores son las etiquetas de la segmentación ya calculada: PCA y t-SNE no crean esos grupos aquí. '
           'Ambos gráficos usan sólo los cuatro contadores; likes figura en ellos aunque no sea entrada numérica de la segmentación. '
           'No muestran las distancias del modelo TF-IDF + metadatos + SVD ni sustituyen su silueta o estabilidad.']
    if stats['constant_columns']:
        lines.append('Contadores constantes en esta muestra: '+', '.join(stats['constant_columns'])+'. No aportan variación a PCA.')
    return '\n\n'.join(lines)




def model_space_summary(stats):
    """Explica las dos reducciones y el coloreado compartido sin validar grupos por apariencia."""
    # numpy: comprueba varianza y evita porcentajes imposibles en el texto.
    import numpy as np
    ratios = np.asarray(stats['explained_variance_ratio'],dtype=float)
    if ratios.shape!=(2,) or not np.isfinite(ratios).all() or (ratios<0).any() or ratios.sum()>1+1e-10:
        raise ValueError('La varianza PCA del espacio del modelo no es válida')
    if not stats['sample_and_labels_verified'] or stats['pipeline_refit']:
        raise ValueError('La explicación requiere muestra contrastada y pipeline final sin reajuste')
    return ('### Proyección del espacio real del modelo\n\n'
            f'Se representan {format_number(stats["n"],0)} vídeos, {stats["groups"]} grupos y {stats["categories"]} categorías. '
            f'El preprocesador guardado produce {format_number(stats["transformed_features"],0)} atributos de TF-IDF, numéricos y región; '
            f'el SVD aprendido los transforma en {stats["svd_dimensions"]} coordenadas. Se han contrastado muestra, orden y grupos con el informe, sin reajustar esa cadena de procesamiento.\n\n'
            f'Un PCA descriptivo posterior reduce esas coordenadas a dos ejes: PC1 conserva {format_number(ratios[0]*100,2)} % '
            f'y PC2 {format_number(ratios[1]*100,2)} %; suman {format_number(ratios.sum()*100,2)} % de la varianza del espacio SVD en esta muestra. '
            'No es el porcentaje de varianza de los cuatro contadores del gráfico anterior ni de aciertos de categorías.\n\n'
            f'El SVD guardado registra {format_number(stats["svd_explained_variance"]*100,2)} % de varianza explicada en su representación de ajuste. '
            'Es otra cantidad: no se multiplican ambos porcentajes para presentar una conservación exacta de información original.\n\n'
            'Cada vídeo ocupa exactamente las mismas coordenadas en ambos paneles, con ejes compartidos. Sólo cambia el color: '
            'a la izquierda, grupo predicho; a la derecha, categoría editorial. Los colores de ambos paneles no establecen correspondencias entre sus etiquetas. '
            'Las categorías no intervienen en el PCA de esta figura.\n\n'
            'La proyección 2D descarta variación: solapamiento no demuestra ausencia de separación en todas las dimensiones, '
            'y separación visual no acredita estabilidad, utilidad ni recuperación de géneros. La silueta se calculó en el SVD completo, no en este dibujo. '
            'La comparación editorial debe leerse con contingencia, ARI y NMI; esta figura describe datos de ajuste y no constituye validación externa.')




def cluster_selection_plot_summary(report):
    """Explica el gráfico de ensayos sin unir configuraciones ni elegir de nuevo.

    Reutiliza la validación canónica de criterios y ganador. Distingue número de
    grupos observado de parámetros como bandwidth o tipo de covarianza.
    """
    cluster_selection_summary(report)
    rows = [row for row in report['clustering'] if row['name'] not in ['agglomerative','dbscan']]
    visible = [row for row in rows if row['silhouette'] is not None]
    selected = report['cluster_selected']
    lines = ['### Comparar ensayos sin confundir silueta con admisibilidad',
             f'El gráfico contiene {len(visible)} ensayos inductivos con silueta disponible, de {len(rows)} registrados; '
             f'{sum(row["selectable"] for row in visible)} cumplen los límites de selección. '
             f'Quedan fuera {len(report["clustering"])-len(rows)} exploraciones transductivas. '
             'Cada punto es un ensayo. Eje horizontal: número de grupos; vertical: silueta. El color identifica dimensiones SVD. '
             'Círculo: cumple la regla; cruz: no cumple; estrella: solución seleccionada. No se conectan puntos con líneas.',
             'Varios puntos pueden compartir k y dimensión: representan configuraciones distintas, por ejemplo covarianzas GMM. '
             'En Mean Shift el número de grupos se obtiene con bandwidth; no se fijó k como en KMeans. '
             'Los puntos superpuestos pueden ocultarse entre sí: la tabla completa conserva todos los ensayos.']
    for family in sorted({row['name'] for row in rows}):
        trials = [row for row in rows if row['name']==family]
        plotted = [row for row in trials if row['silhouette'] is not None]
        lines.append(f'{family}: {len(trials)} ensayos, {len(plotted)} dibujados, '
                     f'{sum(row["selectable"] for row in plotted)} admisibles; SVD ensayados '+
                     ', '.join(str(value) for value in sorted({row['dimensions'] for row in trials}))+'.')
    lines.append(f'La estrella corresponde a {selected["name"]}, k={selected["k"]}, '
                 f'SVD={selected["dimensions"]}, silueta {format_number(selected["silhouette"],4)}. '
                 'Estar más arriba no basta: la selección también exige tamaño y estabilidad. '
                 'La regla posterior al primer estudio se declara como tal; no es un requisito atribuido al profesor.')
    lines.append('La figura resume búsquedas ya realizadas. No es una curva de aprendizaje, una interpolación entre configuraciones '
                 'ni una demostración de que aumentar k o dimensiones mejore el resultado. La silueta usa índices comunes, '
                 'pero cambiar SVD cambia la geometría. No se reajustan modelos ni límites mirando la figura.')
    return '\n\n'.join(lines)




def recommendation_selection_summary(result, protocol):
    """Explica elección en validación y separación de catálogos sin equiparar proxy y relevancia.

    Contrasta métricas acotadas, nombres únicos, consultas comunes y máximo en
    validación; los empates conservan el primer ensayo, como el entrenamiento.
    """
    # numpy: comprueba métricas finitas antes de presentar porcentajes.
    import numpy as np
    trials = result.get('trials',[])
    metric = 'category_agreement_at_5_PROXY'
    fields = [metric,'random_category_expectation','catalog_coverage','unique_channel_fraction_at_5','empty_query_fraction']
    if not trials or any(not isinstance(row.get('model'),str) or not row['model'].strip() for row in trials) or len({row['model'] for row in trials})!=len(trials):
        raise ValueError('Faltan variantes de recomendación o sus nombres están vacíos o repetidos')
    for row in trials:
        if any(not isinstance(row.get(field),(float,int)) or not np.isfinite(row[field]) or not 0<=row[field]<=1 for field in fields):
            raise ValueError('Las métricas de validación del recomendador deben ser finitas y estar entre cero y uno')
        if row.get('query_videos')!=protocol['validation_queries']:
            raise ValueError('Los ensayos de recomendación no comparten el número de consultas de validación reconstruido')
    for name,key in [('catalog_videos','selection_catalog'),('validation_queries','validation_queries'),('test_queries','test_queries'),('final_catalog_videos','final_catalog')]:
        if result.get(name)!=protocol[key] or not isinstance(protocol[key],int) or protocol[key]<=0:
            raise ValueError('Los tamaños del informe de recomendación no coinciden con el protocolo reconstruido')
    if len(protocol['pairwise_overlap'])!=3 or any(protocol['pairwise_overlap']) or protocol['model_refit'] or not protocol['test_query_order_verified'] or not protocol['final_catalog_order_verified']:
        raise ValueError('El protocolo de recomendación no acredita particiones disjuntas y catálogo final contrastado')
    winner = max(trials,key=lambda row:row[metric])
    if result['selected']!=winner['model'] or protocol['selected']!=winner['model']:
        raise ValueError('La variante seleccionada no maximiza el proxy de validación o no coincide con los pesos')
    lines=['### Recomendación por contenido: elegir antes de comprobar',
           'El objetivo es recuperar vídeos con texto similar a una consulta, no predecir una categoría ni usar valoraciones colaborativas. '
           'Se comparan representaciones de título, tags y descripción mediante vecinos de coseno. La categoría sólo sirve como proxy imperfecto de evaluación; no es entrada textual del recomendador.',
           f'Se parte de {format_number(protocol["unique_development_videos"],0)} vídeos únicos del pool de desarrollo y se toman '
           f'{format_number(protocol["final_catalog"],0)}. Para comparar variantes se indexan {format_number(protocol["selection_catalog"],0)} vídeos. '
           f'Los pools de validación y prueba tienen {format_number(protocol["validation_pool"],0)} y {format_number(protocol["test_pool"],0)}; '
           f'se consultan {format_number(protocol["validation_queries"],0)} y {format_number(protocol["test_queries"],0)}, respectivamente. '
           'Los tres conjuntos usados para comparar/evaluar no comparten IDs. Son particiones internas por vídeo; no garantizan canales disjuntos ni validación temporal o externa.',
           f'Se comparan {len(trials)} variantes sobre las mismas consultas de validación. Se maximiza coincidencia de categoría entre hasta cinco vecinos, '
           'con denominador fijo cinco por consulta; las consultas sin resultados no desaparecen del promedio. '
           'No equivale a la Precision@5 habitual, que mide qué proporción de las cinco primeras recomendaciones '
           'consideraría relevantes una persona: aquí el acierto es compartir categoría, una aproximación automática.']
    for row in trials:
        lines.append(f'{row["model"]}: proxy de validación {format_number(row[metric]*100,2)} % en '
                     f'{format_number(row["query_videos"],0)} consultas.')
    lines.append(f'Ganadora: **{winner["model"]}**, con {format_number(winner[metric]*100,2)} % en validación. '
                 'Sólo esa representación se comprobó después en prueba; el test no elige la variante. Los empates conservan el primer ensayo de la lista, sin desempatar mirando prueba.')
    lines.append(f'Después de evaluar, se reajustó la representación elegida sobre los {format_number(protocol["final_catalog"],0)} vídeos, '
                 f'incluidos los que antes eran consultas y {format_number(protocol["unused_in_comparison"],0)} que no entraron en las consultas reducidas. '
                 'El catálogo final guardado y su orden se han contrastado, sin fit en esta revisión. '
                 'Las métricas publicadas corresponden al catálogo de evaluación anterior, no a una evaluación independiente del modelo ampliado. '
                 'El nombre independent_proxy_test significa consultas internas disjuntas, no datos externos ni relevancia humana acreditada.')
    return '\n\n'.join(lines)




def recommendation_metrics_summary(result, diagnostics):
    """Valida denominadores y narra métricas proxy sin inventar juicios humanos.

    Recalcula proxy, consultas vacías y referencia aleatoria desde sus filas.
    Cobertura y diversidad se contrastan en dominio; su reproducción requiere
    las listas de vecinos, que se verifican en la auditoría diagnóstica separada.
    """
    # numpy: valida dominios, recuentos enteros y tolerancias de las métricas.
    import numpy as np
    required={'query_id','category_agreement_at_5_PROXY','random_category_expectation','returned','empty'}
    if diagnostics.empty or not required.issubset(diagnostics.columns):
        raise ValueError('Faltan filas o columnas del diagnóstico de consultas de recomendación')
    if diagnostics.query_id.isna().any() or diagnostics.query_id.astype(str).str.strip().eq('').any() or diagnostics.query_id.duplicated().any():
        raise ValueError('El diagnóstico requiere una identidad utilizable y única por consulta')
    metrics=result['independent_proxy_test'];n=len(diagnostics);catalog_n=result['catalog_videos']
    if not isinstance(catalog_n,int) or catalog_n<5 or metrics.get('query_videos')!=n or result['test_queries']!=n:
        raise ValueError('El tamaño del catálogo o de las consultas de prueba no coincide con el diagnóstico')
    for field in ['category_agreement_at_5_PROXY','random_category_expectation','catalog_coverage','unique_channel_fraction_at_5','empty_query_fraction']:
        value=metrics.get(field)
        if not isinstance(value,(float,int)) or not np.isfinite(value) or not 0<=value<=1:
            raise ValueError(f'La métrica de recuperación {field} debe ser finita y estar entre cero y uno')
    values=diagnostics[['category_agreement_at_5_PROXY','random_category_expectation','returned']].apply(pd.to_numeric,errors='coerce').to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values<0).any() or (values[:,:2]>1).any() or (values[:,2]>5).any() or not np.equal(values[:,2],np.floor(values[:,2])).all():
        raise ValueError('El diagnóstico contiene proporciones o números de vecinos inválidos')
    matches=5*values[:,0];returned=values[:,2]
    if not np.allclose(matches,np.round(matches),rtol=0,atol=1e-9) or (matches>returned+1e-9).any():
        raise ValueError('La coincidencia editorial no corresponde a recuentos enteros entre cero y los vecinos devueltos')
    if not all(isinstance(value,(bool,np.bool_)) for value in diagnostics['empty']):
        raise ValueError('El indicador empty debe ser booleano, no texto o números ambiguos')
    if not np.array_equal(diagnostics['empty'].to_numpy(),returned==0):
        raise ValueError('El indicador de consulta vacía contradice el número de vecinos')
    for field,observed in [('category_agreement_at_5_PROXY',values[:,0].mean()),('random_category_expectation',values[:,1].mean()),('empty_query_fraction',(returned==0).mean())]:
        if not np.isclose(metrics[field],observed,rtol=1e-9,atol=1e-10):
            raise ValueError(f'La métrica {field} no coincide con las consultas publicadas')
    coverage_count=metrics['catalog_coverage']*catalog_n
    if not np.isclose(coverage_count,round(coverage_count),rtol=0,atol=1e-8) or coverage_count>returned.sum()+1e-8:
        raise ValueError('La cobertura no corresponde a un número posible de vídeos distintos recuperados')
    empty_count=int((returned==0).sum());slots=5*n;match_count=int(round(matches.sum()))
    lines=['### Métricas de prueba: qué cuenta cada denominador',
           f'La representación {result["selected"]} se evalúa con {format_number(n,0)} consultas disjuntas del catálogo de '
           f'{format_number(catalog_n,0)} vídeos. Las métricas pertenecen a ese ajuste previo, no al catálogo final ampliado. '
           'Esta prueba interna no aporta juicios humanos de relevancia.',
           f'Coincidencia editorial top-5 proxy: {format_number(match_count,0)} coincidencias de categoría entre '
           f'{format_number(slots,0)} posiciones previstas: **{format_number(metrics["category_agreement_at_5_PROXY"]*100,2)} %**. '
           'Para cada consulta se divide entre cinco incluso si devuelve menos vecinos; después se promedian las consultas. '
           'Tampoco es la Precision@5 de relevancia juzgada por personas ni un porcentaje de usuarios satisfechos.',
           f'Se devolvieron {format_number(returned.sum(),0)} candidatos en total. Consultas sin resultados: '
           f'{format_number(empty_count,0)} de {format_number(n,0)}, **{format_number(metrics["empty_query_fraction"]*100,2)} %**. '
           'No se excluyen del promedio ni se inventan cinco vecinos para completarlas. empty significa lista vacía; '
           'por sí solo no identifica si faltaba vocabulario o no había similitud positiva.',
           f'Cobertura del catálogo: {format_number(round(coverage_count),0)} vídeos distintos recuperados entre '
           f'{format_number(catalog_n,0)} disponibles: **{format_number(metrics["catalog_coverage"]*100,2)} %**. '
           'Un mismo candidato repetido en varias consultas cuenta una sola vez. No es porcentaje de consultas atendidas ni tasa de aciertos.',
           f'Diversidad de canales: **{format_number(metrics["unique_channel_fraction_at_5"]*100,2)} %**. '
           'En cada lista se divide el número de nombres de canal distintos entre los vecinos realmente devueltos; '
           'se promedian las listas y una lista vacía aporta cero. No se divide siempre entre cinco y no es la fracción global de canales únicos. '
           'Se utilizan nombres de canal, no identidades verificadas de creadores.',
           'Las métricas responden a preguntas distintas: coincidencia editorial, alcance del catálogo, variedad de nombres de canal y falta de resultados. '
           'Ninguna demuestra por sí sola calidad, relevancia o satisfacción; una cobertura mayor tampoco es automáticamente mejor. '
           'Aquí se validan proxy, vacíos y referencia aleatoria con filas por consulta; cobertura y diversidad completas requieren listas de vecinos.']
    return '\n\n'.join(lines)




def recommendation_control_summary(result, audit):
    """Interpreta esperanza e incertidumbre condicional sin afirmar relevancia o significación."""
    if audit.get('status')!='passed' or not audit.get('catalog_and_model_fixed') or audit.get('model_fits')!=0:
        raise ValueError('La explicación requiere controles reproducidos sin ajuste de modelos')
    lower,upper=audit['ci95']
    lines=['### Referencia aleatoria e incertidumbre: alcance de los controles',
           f'Para cada consulta se calcula la fracción del catálogo que tiene su categoría. '
           f'Promediando las {format_number(audit["queries"],0)} consultas, la coincidencia esperada bajo selección uniforme de vecinos es '
           f'**{format_number(audit["random_expectation"]*100,2)} %**. '
           'Es una esperanza analítica con el catálogo fijo, no el resultado de ejecutar un recomendador aleatorio ni una probabilidad uniforme entre categorías. '
           'Se supone que se pueden devolver cinco candidatos incluso cuando el modelo textual no produce resultados.',
           f'El proxy observado es {format_number(audit["observed_proxy"]*100,2)} %. La diferencia modelo menos referencia es '
           f'**{format_number(audit["gain"]*100,2)} puntos porcentuales**, no ese porcentaje de mejora relativa ni de satisfacción.',
           f'Bootstrap: {format_number(audit["replicas"],0)} réplicas, cada una con {format_number(audit["queries"],0)} consultas '
           f'muestreadas con reemplazo, semilla {audit["seed"]}. Se remuestrea el resultado completo de cada consulta, no sus cinco vecinos como si fueran independientes. '
           'Se promedia cada réplica y se toman los percentiles 2,5 y 97,5.',
           f'Intervalo percentil nominal 95 % de la media proxy: **{format_number(lower*100,2)} %–{format_number(upper*100,2)} %**. '
           'Describe variación entre remuestreos de estas consultas con catálogo y modelo de evaluación fijos. '
           'No representa el rango de resultados de cada consulta ni la probabilidad de relevancia de una recomendación.',
           'Su cobertura nominal depende de supuestos: consultas representativas y dependencia suficientemente controlada. '
           'Separar IDs de vídeo no elimina la dependencia por canales o plantillas. No incorpora selección de representación, '
           f'otro entrenamiento, otro catálogo, el reajuste final a {format_number(result["final_catalog_videos"],0)} vídeos, cambios temporales ni incertidumbre de juicios humanos. '
           'No se interpreta como una probabilidad del 95 % de que un parámetro fijo esté dentro de este intervalo concreto.',
           'La diferencia observada no lleva aquí su propio intervalo ni un contraste formal frente a la referencia. '
           'Que el valor puntual aleatorio quede por debajo del intervalo proxy no se presenta como p-valor o prueba formal de superioridad. '
           'Las 500 réplicas producen extremos aproximados con variación Monte Carlo; una semilla fija da repetibilidad, no exactitud ni validez externa.']
    if audit['constant_query_scores']:
        lines.append('Todas las consultas tienen el mismo proxy: el remuestreo no genera dispersión y el intervalo puede colapsar. Eso no demuestra ausencia de incertidumbre fuera de esta muestra.')
    return '\n\n'.join(lines)




def cluster_region_summary(counts, profiles):
    """Describe asociación entre grupo y región, sin atribuirle causalidad."""
    # numpy: contrasta recuentos y tamaños para evitar tablas de otro experimento.
    import numpy as np
    expected=profiles.set_index('cluster')['n']
    if counts.empty or set(counts.index)!=set(expected.index) or not counts.columns.is_unique:
        raise ValueError('El cruce grupo-región no coincide con los perfiles')
    values=counts.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values<0).any() or not np.equal(values,np.floor(values)).all() or not np.array_equal(counts.sum(axis=1).reindex(expected.index),expected):
        raise ValueError('Los recuentos regionales no suman los tamaños de grupo')
    lines=['**Distribución regional de los grupos.** Porcentajes dentro de cada grupo:']
    for group,row in counts.iterrows():
        top=row.sort_values(ascending=False,kind='stable').head(2)
        lines.append(f'Grupo {group}: '+ '; '.join(f'{region}, {format_number(n,0)} vídeos ({format_number(n/row.sum()*100,2)} %)' for region,n in top.items())+'.')
    totals=counts.sum(axis=0)
    fractions=counts.loc[:,totals.gt(0)].div(totals.loc[totals.gt(0)],axis=1)
    reference=fractions.iloc[0].sort_values(ascending=False,kind='stable')
    if len(reference):
        selected=list(reference.head(2).index)
        if reference.index[-1] not in selected:selected.append(reference.index[-1])
        lines.append(f'Dentro de cada región, el grupo {fractions.index[0]} reúne '+ '; '.join(f'{format_number(fractions.loc[fractions.index[0],region]*100,2)} % de {region}' for region in selected)+'. El denominador ahora es el total de vídeos de esa región, no el tamaño del grupo.')
    lines.append('Audiencia mediana por grupo: '+ '; '.join(f'{row.cluster}: {format_number(row.views_median,1)} visualizaciones' for row in profiles.itertuples())+'.')
    audience='Las medianas de audiencia difieren entre grupos.' if profiles.views_median.nunique()>1 else 'Las medianas de audiencia coinciden entre grupos.'
    region_diff=fractions.shape[1]>1 and not np.allclose(fractions.to_numpy(),fractions.iloc[:,0].to_numpy()[:,None],rtol=1e-9,atol=1e-12)
    regional='La distribución de grupos varía entre regiones.' if region_diff else 'En estas proporciones no cambia la distribución de grupos entre regiones.'
    lines.append(audience+' '+regional+' **Conclusión:** los grupos ofrecen perfiles distintos de audiencia y composición regional; no se interpretan como categorías temáticas. Los términos reconstruidos permiten contrastar también plantillas e indicios lingüísticos. Estos cruces describen asociación y no miden qué atributo causa o domina la separación; para eso haría falta una ablación específica.')
    lines.append('Región es un atributo del modelo y las apariciones retenidas dependen del muestreo. Este cruce describe asociación en la muestra, no causa de la agrupación ni segmentos comerciales validados.')
    return '\n\n'.join(lines)


def recommendation_coverage_summary(result, diagnostics):
    """Contextualiza cobertura mediante el máximo de candidatos distintos posible."""
    recommendation_metrics_summary(result,diagnostics)
    n=len(diagnostics);catalog=result['catalog_videos'];returned=int(diagnostics.returned.sum())
    planned=min(5*n,catalog);actual=min(returned,catalog)
    covered=round(result['independent_proxy_test']['catalog_coverage']*catalog)
    ratio=format_number(covered/actual*100,2)+' %' if actual else 'no estimable: no se devolvieron candidatos'
    return (f'**Alcance de la cobertura.** Con {format_number(n,0)} consultas y hasta cinco vecinos, el techo teórico es '
            f'{format_number(planned,0)} candidatos distintos: {format_number(planned/catalog*100,2)} % del catálogo. '
            f'Con las listas realmente devueltas, el techo es {format_number(actual,0)}: {format_number(actual/catalog*100,2)} %. '
            f'Se recuperaron {format_number(covered,0)} distintos ({ratio} de ese último techo). '
            'Son límites combinatorios; no son objetivos de calidad ni demuestran que recuperar más candidatos aporte relevancia.')


def global_conclusions(root):
    """Redacta las conclusiones globales visibles del cuaderno a partir de los resultados guardados.

    Interpreta cada problema (qué significa la cifra, qué decisión justifica y dónde
    está su límite) y después extrae conclusiones transversales. Todas las cifras se
    leen de los informes de esta ejecución; no hay valores escritos a mano.
    """
    # numpy: recall por clase y confusión principal a partir de la matriz de prueba.
    import numpy as np
    root = Path(root)
    report = json.loads((root/'reports/evaluation.json').read_text(encoding='utf-8'))
    quality = json.loads((root/'reports/quality_checks.json').read_text(encoding='utf-8'))
    tasks = report['tasks']
    pct = lambda value: format_number(value*100, 2) + ' %'

    # Categoría: mejor y peor clase con soporte suficiente y su confusión principal.
    cat = tasks['category']; holdout = cat['holdout']
    labels = holdout['labels']; matrix = np.asarray(holdout['confusion_matrix'], dtype=float)
    support = matrix.sum(axis=1)
    recall = np.divide(np.diag(matrix), support, out=np.zeros_like(support), where=support > 0)
    # eligible: clases frecuentes (al menos 100 apariciones) para no extraer conclusiones de soportes mínimos.
    eligible = [i for i in range(len(labels)) if support[i] >= 100]
    best = max(eligible, key=lambda i: recall[i]); worst = min(eligible, key=lambda i: recall[i])
    off = matrix[worst].copy(); off[worst] = 0; target = int(off.argmax())
    # weakest: clase con soporte y menor recuperación, aunque sea minoritaria.
    weakest = min((i for i in range(len(labels)) if support[i] > 0), key=lambda i: recall[i])
    cat_f1 = holdout['macro_f1']; cat_base = cat['baseline_holdout']['macro_f1']
    cat_channel = quality['tasks']['category']['test']['macro_f1']
    ci = cat['confidence_interval']

    # Moderación: aciertos, alertas y prevalencia de la clase positiva.
    def moderation(task):
        result = tasks[task]; m = result['holdout']['confusion_matrix']
        positive = result['holdout']['labels'].index('1')
        tp = m[positive][positive]; actual = sum(m[positive]); predicted = sum(row[positive] for row in m)
        total = sum(map(sum, m))
        return {'f1': result['holdout']['macro_f1'], 'base': result['baseline_holdout']['macro_f1'],
                'tp': tp, 'actual': actual, 'predicted': predicted, 'total': total,
                'recall': tp/actual if actual else 0.0, 'precision': tp/predicted if predicted else 0.0,
                'prevalence': actual/total if total else 0.0,
                'channel': quality['tasks'][task]['test']['macro_f1'],
                'channel_base': quality['tasks'][task]['baseline_test']['macro_f1']}
    comments, ratings = moderation('comments_disabled'), moderation('ratings_disabled')

    # Regresión: reducción del error frente a la referencia y concentración de los mayores errores.
    likes, ratio = tasks['likes'], tasks['ratio']
    reduction = lambda r: 1 - r['holdout']['mae']/r['baseline_holdout']['mae']
    likes_errors = pd.read_csv(root/'reports/likes_largest_errors.csv')
    ratio_errors = pd.read_csv(root/'reports/ratio_largest_errors.csv')
    ratio_under = int((ratio_errors.actual > ratio_errors.predicted).sum())
    ablation = ratio.get('ratio_ablation') or {}

    # Segmentación y recomendación.
    cluster = report['cluster_selected']; interp = report['cluster_interpretation']
    profiles = pd.read_csv(root/'reports/cluster_profiles.csv').sort_values('views_median', ascending=False)
    high, low = profiles.iloc[0], profiles.iloc[-1]
    rec = report['recommendation']; test = rec['independent_proxy_test']
    proxy, random_proxy = test['category_agreement_at_5_PROXY'], rec['controls']['random_category_expectation']
    proxy_ci = rec['controls'].get('proxy_ci95')

    ratio_word = 'apenas' if ratio['holdout']['r2'] < 0.3 else 'sólo en parte'
    lines = ['### Respuesta a la pregunta guía', '',
             'La pregunta era qué se puede estimar sobre vídeos que llegan a tendencias, cómo segmentarlos y qué contenido '
             'recuperar como similar. La respuesta que dan los datos es desigual según el problema, y esa desigualdad es '
             'en sí misma una conclusión:', '',
             f'- **Se estima bien** el número de «me gusta» cuando se conocen los demás contadores del vídeo (R² '
             f'{format_number(likes["holdout"]["r2"],3)}), y **se clasifica la categoría** muy por encima de la referencia '
             f'(macro-F1 {format_number(cat_f1,3)} frente a {format_number(cat_base,3)}).',
             f'- **Se detecta sólo en parte** si un creador desactivará comentarios o valoraciones: el modelo encuentra '
             f'{pct(comments["recall"])} y {pct(ratings["recall"])} de los casos reales, con muchas falsas alarmas.',
             f'- La proporción «me gusta/no me gusta» **{ratio_word} es predecible** con la información disponible '
             f'(R² {format_number(ratio["holdout"]["r2"],3)}).',
             f'- Los vídeos se separan en **{cluster["k"]} segmentos de audiencia**, no en temas.',
             f'- El recomendador devuelve vídeos de la misma categoría **{format_number(proxy/random_proxy,1)} veces** más a menudo que el azar.', '',
             '### Lectura de cada problema: qué significa, qué decisión justifica y dónde está el límite', '',
             '| Problema | Resultado en prueba | Interpretación | Uso razonable |', '|---|---|---|---|',
             f'| Categoría (pedido) | macro-F1 {format_number(cat_f1,3)}; referencia {format_number(cat_base,3)} | '
             f'El texto del vídeo contiene información real sobre su categoría, pero las categorías genéricas se confunden entre sí. | '
             'Sugerir la categoría y revisar las clases minoritarias. |',
             f'| Comentarios desactivados (opcional) | {format_number(comments["tp"],0)} de {format_number(comments["actual"],0)} detectados; '
             f'{pct(comments["precision"])} de las alertas son correctas | Señal útil, pero rara ({pct(comments["prevalence"])} de los vídeos). | '
             'Priorizar revisiones, no decidir automáticamente. |',
             f'| Valoraciones desactivadas (opcional) | {format_number(ratings["tp"],0)} de {format_number(ratings["actual"],0)} detectados; '
             f'{pct(ratings["precision"])} de las alertas son correctas | Más difícil que comentarios: menos precisión con una prevalencia parecida. | '
             'Sólo como filtro previo con revisión humana. |',
             f'| Número de «me gusta» (pedido) | MAE {format_number(likes["holdout"]["mae"],0)}; referencia {format_number(likes["baseline_holdout"]["mae"],0)} | '
             f'Error un {pct(reduction(likes))} menor; la relación entre contadores es muy estable. | '
             'Estimar «me gusta» ausentes o anómalos, no prever popularidad antes de publicar. |',
             f'| Proporción «me gusta/no me gusta» (opcional) | MAE {format_number(ratio["holdout"]["mae"],2)}; referencia {format_number(ratio["baseline_holdout"]["mae"],2)} | '
             f'Mejora del {pct(reduction(ratio))}; la cola de proporciones altas queda sin explicar. | '
             'Orden de magnitud orientativo; no para vídeos concretos. |',
             f'| Segmentación (pedido) | {cluster["name"]}, k={cluster["k"]}; silueta {format_number(cluster["silhouette"],3)}; ARI {format_number(interp["ARI"],3)} | '
             'Grupos separados por audiencia y región, casi independientes de la categoría. | Describir la muestra; no sustituye a las categorías. |',
             f'| Recomendación (pedido) | {pct(proxy)} de coincidencia de categoría; azar {pct(random_proxy)} | '
             'Recupera contenido textual parecido, a menudo del mismo tema. | Sugerir vídeos similares; su utilidad real requiere evaluación humana. |', '',
             '### Interpretación razonada por problema', '',
             f'**Categoría.** Macro-F1 {format_number(cat_f1,3)} (intervalo por remuestreo {format_number(ci["low"],3)}–{format_number(ci["high"],3)}) '
             f'frente a {format_number(cat_base,3)} de la referencia: el texto de título, etiquetas y descripción permite reconocer la categoría. '
             f'El acierto no es uniforme. Entre las categorías con al menos 100 apariciones, «{labels[best]}» recupera {pct(recall[best])} de sus vídeos, '
             f'mientras que «{labels[worst]}» sólo {pct(recall[worst])}; su confusión principal es «{labels[target]}» ({pct(off[target]/support[worst])} de la clase). '
             f'Las clases pequeñas son las menos fiables: «{labels[weakest]}» ({format_number(support[weakest],0)} apariciones) sólo recupera {pct(recall[weakest])}. '
             'Las categorías con vocabulario propio se reconocen mejor que las genéricas, que comparten palabras con muchas otras. '
             'Es coherente con que el propio etiquetado de YouTube sea ambiguo en esas categorías, aunque no se ha medido por separado. '
             f'En el diagnóstico por canales disjuntos, macro-F1 baja a {format_number(cat_channel,3)}; sugiere que parte de lo aprendido '
             'depende de creadores concretos, aunque ese diagnóstico se reajusta con otra muestra.', '',
             f'**Comentarios y valoraciones desactivados.** Son casos raros: {pct(comments["prevalence"])} y {pct(ratings["prevalence"])} de las '
             f'apariciones de prueba. En comentarios, un modelo que dijera siempre «habilitado» acertaría {pct(1-comments["prevalence"])} de las veces y no '
             'detectaría ningún caso; por eso se evalúa con macro-F1 y se fija el umbral con F2, que da más peso a encontrar positivos. '
             f'Con ese criterio, en comentarios {pct(comments["precision"])} de las alertas son correctas, unas '
             f'{format_number(comments["precision"]/comments["prevalence"],0)} veces más que marcando al azar; en valoraciones, '
             f'{pct(ratings["precision"])} ({format_number(ratings["precision"]/ratings["prevalence"],0)} veces el azar). '
             'Se acepta conscientemente un número alto de falsas alarmas a cambio de no perder la mayoría de los casos. '
             f'Por canales disjuntos, comentarios queda en {format_number(comments["channel"],3)} frente a {format_number(comments["channel_base"],3)} de '
             f'referencia y valoraciones en {format_number(ratings["channel"],3)} frente a {format_number(ratings["channel_base"],3)}: '
             'la detección de comentarios desactivados no se sostiene en canales nuevos. Es compatible con que esa decisión dependa más '
             'de cada creador que del contenido, aunque esta explicación es una hipótesis.', '',
             f'**Número de «me gusta».** El error medio baja de {format_number(likes["baseline_holdout"]["mae"],0)} a '
             f'{format_number(likes["holdout"]["mae"],0)} y R² llega a {format_number(likes["holdout"]["r2"],3)}. El resultado es alto porque '
             'el modelo usa contadores contemporáneos (visualizaciones, comentarios, «no me gusta»), muy relacionados con los «me gusta». '
             'Por tanto, mide bien la coherencia entre contadores, pero no demuestra que se pueda anticipar el éxito de un vídeo antes de publicarlo. '
             f'Las 100 apariciones con mayor error corresponden a sólo {format_number(likes_errors.video_id.nunique(),0)} vídeos muy populares: '
             'el error medio depende mucho de unos pocos casos extremos.', '',
             f'**Proporción «me gusta/no me gusta».** La mejora es pequeña (MAE {format_number(ratio["holdout"]["mae"],2)} frente a '
             f'{format_number(ratio["baseline_holdout"]["mae"],2)}; R² {format_number(ratio["holdout"]["r2"],3)}). '
             f'En las 100 apariciones con mayor error, {format_number(ratio_under,0)} son subestimaciones: el modelo no reconoce los vídeos con '
             'proporciones excepcionalmente altas. ']
    if {'without_views_comment_count_mae', 'with_counters_mae'} <= set(ablation):
        lines[-1] += (f'En la ablación de validación cruzada, retirar visualizaciones y comentarios sólo mueve el MAE de '
                      f'{format_number(ablation["with_counters_mae"],2)} a {format_number(ablation["without_views_comment_count_mae"],2)}: '
                      'esos contadores aportan poco a esta tarea. ')
    lines[-1] += ('Es un resultado desfavorable que se declara: la aceptación relativa de un vídeo depende de factores que '
                  'los datos disponibles no recogen.')
    lines += ['',
              f'**Segmentación.** De {format_number(len(report["clustering"]),0)} configuraciones exploradas se elige {cluster["name"]} con '
              f'k={cluster["k"]} sobre {cluster["dimensions"]} dimensiones SVD (silueta {format_number(cluster["silhouette"],3)}, una separación moderada). '
              f'ARI {format_number(interp["ARI"],3)} y NMI {format_number(interp["NMI"],3)} indican que los grupos casi no coinciden con las categorías. '
              f'Sus perfiles sí difieren en audiencia: mediana de {format_number(high.views_median,0)} visualizaciones en el grupo {int(high.cluster)} '
              f'frente a {format_number(low.views_median,0)} en el grupo {int(low.cluster)}, con predominio de {high.top_region} y {low.top_region}, respectivamente. '
              'La estructura que se encuentra se asocia más con alcance y región que con temática. Es una descripción: no se ha medido qué atributo domina la separación.', '',
              f'**Recomendación.** Se comparan {len(rec["trials"])} representaciones en validación y se comprueba la elegida, {rec["selected"]}, en '
              f'{format_number(test["query_videos"],0)} consultas de prueba: {pct(proxy)} de los vecinos comparten categoría '
              + (f'(intervalo {pct(proxy_ci[0])}–{pct(proxy_ci[1])}) ' if proxy_ci else '') +
              f'frente a {pct(random_proxy)} esperable al azar. Las listas son variadas ({pct(test["unique_channel_fraction_at_5"])} de canales '
              f'distintos por lista) y sólo {pct(test["empty_query_fraction"])} de las consultas se quedan sin resultados. '
              'La coincidencia de categoría es una aproximación: indica que se recupera contenido afín, no que la recomendación sea útil para una persona.', '',
              '### Conclusiones transversales', '',
              '1. **Lo más predecible es lo contemporáneo.** Las tareas que usan contadores del propio vídeo (número de «me gusta») obtienen '
              'resultados muy altos; las que sólo usan contenido (categoría, moderación) quedan en niveles moderados. La popularidad se describe '
              'mejor de lo que se anticipa.',
              '2. **El canal pesa más de lo que muestra la partición por vídeo.** La caída al separar canales en categoría y comentarios indica que '
              'parte del aprendizaje es propio de cada creador. Para aplicar los modelos a canales nuevos habría que validar por canal desde el principio.',
              '3. **En clases desbalanceadas, la exactitud engaña.** Por eso se usan macro-F1, matrices de confusión, curvas precisión–recuperación '
              'y umbrales calibrados fuera de la prueba. Así se estudia y se mitiga el sesgo hacia la clase mayoritaria, como pide el reto opcional.',
              '4. **La estructura natural de los datos no es la editorial.** Ni los grupos ni el espacio del modelo reproducen las categorías: '
              'la audiencia y la región se asocian más con los grupos que el tema.',
              '5. **Los resultados desfavorables se mantienen.** La proporción «me gusta/no me gusta», la precisión de valoraciones y la caída por canal '
              'se presentan con sus cifras; no se retocan umbrales ni se seleccionan modelos mirando la prueba para mejorarlos.', '',
              '### Qué no permiten afirmar estos resultados', '',
              '- Que alguna variable **cause** una categoría, una desactivación o una popularidad mayor: coeficientes y cruces describen asociaciones.',
              '- Que los modelos **anticipen** el éxito antes de publicar: likes y proporción usan señales contemporáneas.',
              '- Que funcionen igual en **otras épocas o canales**: no hay validación temporal ni externa, y la prueba interna se consultó durante el desarrollo.',
              '- Que las recomendaciones sean **útiles para usuarios**: no hay juicios humanos ni datos de uso.', '',
              '### Próximos pasos que se desprenden de lo observado', '',
              '- Validar agrupando por canal, porque es donde más cae el rendimiento.',
              '- Estimar «me gusta» sólo con información disponible al publicar, para responder a una pregunta de negocio distinta.',
              '- Evaluar el recomendador con valoraciones humanas o datos de uso, como se propone fuera del enunciado.',
              '- Repetir la evaluación con datos de un periodo posterior para medir la estabilidad temporal.', '',
              '### Conclusión final', '',
              'La práctica resuelve los cuatro problemas pedidos y los dos opcionales con modelos elegidos por validación y comprobados en datos reservados. '
              'Los resultados muestran que el contenido textual basta para orientar la categoría y recuperar vídeos afines, que los contadores contemporáneos '
              'explican casi todo el número de «me gusta» y que la moderación y la proporción de valoraciones dependen en gran parte de factores que el dataset '
              'no recoge. El valor del trabajo no está sólo en las cifras, sino en saber qué decisiones permite tomar cada modelo y cuáles no.']
    return '\n'.join(lines) + '\n'


def executive_summary(root, report):
    """Publica una lectura breve de decisiones y límites; conserva el informe detallado aparte."""
    root=Path(root)
    lines=['# Resultados y decisiones: lectura principal','',ALCANCE_RETOS,'',
           '| Objetivo | Enunciado | Modelo elegido | Métrica de prueba | Referencia |','|---|---|---|---:|---:|']
    for task,row in report['tasks'].items():
        metric='macro_f1' if 'macro_f1' in row['holdout'] else 'mae'
        name='macro-F1' if metric=='macro_f1' else 'MAE'
        scope = 'Opcional para subir nota' if task in RETOS_OPCIONALES else 'Pedido'
        lines.append(f'| {task} | {scope} | {row["selected"]} | {name}: {format_number(row["holdout"][metric],3)} | {format_number(row["baseline_holdout"][metric],3)} |')
    cluster=report['cluster_selected'];rec=report['recommendation']
    lines += [f'| cluster | Pedido | {cluster["name"]}, k={cluster["k"]}, SVD={cluster["dimensions"]} | Silueta: {format_number(cluster["silhouette"],4)} | No es clasificación editorial |',
              f'| recommend | Pedido | {rec["selected"]} | Coincidencia editorial proxy: {format_number(rec["independent_proxy_test"]["category_agreement_at_5_PROXY"]*100,2)} % | Azar esperado: {format_number(rec["controls"]["random_category_expectation"]*100,2)} % |','',
              'Macro-F1: mayor es mejor. MAE: menor es mejor, en unidades del objetivo. Cada familia se eligió por validación cruzada y después se refinó; los clasificadores binarios ajustan el umbral para F2 con predicciones OOF. El test no selecciona recetas.','',
              '## Decisiones que condicionan la interpretación','',
              'Categoría y moderación utilizan contenido previo a las interacciones. Likes y ratio son estimaciones contemporáneas con señales declaradas; no predicciones garantizadas antes de publicar.', '',
              'Clustering maximiza silueta entre soluciones que cumplen tamaño y estabilidad. La regla se incorporó tras el primer estudio y se declara post hoc. Los grupos no equivalen a géneros ni a segmentos comerciales validados.', '',
              f'La recomendación se eligió en validación y se comprobó con un catálogo de {format_number(rec["catalog_videos"],0)} vídeos; el peso final se reajustó después con {format_number(rec["final_catalog_videos"],0)}. Las métricas no evalúan independientemente ese catálogo ampliado.','']
    quality=json.loads((root/'reports/quality_checks.json').read_text(encoding='utf-8'))
    comments=quality['tasks']['comments_disabled'];ratings=report['tasks']['ratings_disabled']
    lines += ['## Límites y resultados desfavorables','',
              '**Comentarios desactivados.** ' + task_evaluation_takeaway('comments_disabled',report['tasks']['comments_disabled']), '',
              '**Valoraciones desactivadas.** ' + task_evaluation_takeaway('ratings_disabled',report['tasks']['ratings_disabled']), '',
              '**Proporción likes/dislikes.** ' + task_evaluation_takeaway('ratio',report['tasks']['ratio']), '',
              f'Comentarios por canales disjuntos: macro-F1 {format_number(comments["test"]["macro_f1"],3)} frente a {format_number(comments["baseline_test"]["macro_f1"],3)} de referencia. Es un diagnóstico reajustado, distinto de evaluar los pesos finales.', '',
              f'Valoraciones: F2 con umbral OOF {format_number(ratings["positive_f2"],3)}; con umbral predeterminado {format_number(f2_from_confusion(ratings["default_threshold_holdout"]["confusion_matrix"]),3)}. Se informa la comparación y se conserva la política elegida en entrenamiento.', '',
              'La prueba interna se consultó durante el desarrollo; no hay validación externa prospectiva. Los intervalos son condicionales al ajuste y no incluyen incertidumbre de selección. El alcance de cada diagnóstico se explica en su apartado. La coincidencia editorial mide similitud de categoría. La evaluación humana de la utilidad de las recomendaciones se propone como ampliación y queda fuera de lo solicitado en el ejercicio.','',
              ]
    if (root/'reports/cluster_by_region.csv').exists():
        lines += ['',cluster_region_summary(pd.read_csv(root/'reports/cluster_by_region.csv',index_col=0),pd.read_csv(root/'reports/cluster_profiles.csv'))]
    # La concentración del error de likes queda visible también en la lectura principal.
    errors = pd.read_csv(root/'reports/likes_largest_errors.csv')
    lines += ['', '### Dónde se concentra el error del número de «me gusta»', '',
              *conclusion_paragraphs(largest_errors_summary('likes', report['tasks']['likes'], errors))]
    lines += ['', generalization_summary(root, report, quality), '',f'Correspondencia editorial de la segmentación: ARI {format_number(report["cluster_interpretation"]["ARI"],4)} y NMI {format_number(report["cluster_interpretation"]["NMI"],4)}. Comparan grupos y categorías, no estabilidad entre réplicas.']
    lines += ['',recommendation_coverage_summary(rec,pd.read_csv(root/'reports/recommendation_query_diagnostics.csv'))]
    lines += ['', '## Consulta del detalle', '',
              'El cuaderno conserva gráficos, interpretación y tablas desplegables junto a cada objetivo. [Informe detallado](results_details.md): refinamiento, métricas, coeficientes, regiones y limitaciones. [Datos](data_findings.md), [calidad](quality_report.md), [temario](temario_coverage.md).']
    return '\n'.join(lines)+'\n'




def task_evaluation_takeaway(task, result):
    """Resume el resultado por tarea con denominadores explícitos y sin cambiar decisiones."""
    if 'macro_f1' in result['holdout']:
        classification_summary(task,result['holdout'])
        text=f'Macro-F1 de prueba {format_number(result["holdout"]["macro_f1"],3)}, referencia {format_number(result["baseline_holdout"]["macro_f1"],3)}.'
        if task in ['comments_disabled','ratings_disabled']:
            labels=result['holdout']['labels'];positive=labels.index('1')
            matrix=result['holdout']['confusion_matrix'];tp=matrix[positive][positive]
            actual=sum(matrix[positive]);predicted=sum(row[positive] for row in matrix)
            recall=format_number(tp/actual*100,2)+' %' if actual else 'no estimable (sin casos positivos)'
            precision=format_number(tp/predicted*100,2)+' %' if predicted else 'no estimable (sin predicciones positivas)'
            text+=f' Detecta {format_number(tp,0)} de {format_number(actual,0)} positivos reales: recall {recall}; precisión positiva {precision}. No son porcentajes de exactitud global.'
        else:
            missing=sum(sum(row)==0 for row in result['holdout']['confusion_matrix'])
            text+=f' Categorías sin soporte en prueba: {missing}; no se estima su fiabilidad con esa reserva.'
        return text
    regression_summary(task,result)
    r2=result['holdout'].get('r2')
    r2text=format_number(r2,3) if r2 is not None else 'no estimable'
    return f'MAE de prueba {format_number(result["holdout"]["mae"],3)} frente a {format_number(result["baseline_holdout"]["mae"],3)} de referencia; R² {r2text}. Es una estimación en la escala declarada, no garantía de precisión para cada vídeo.'


def editorial_detail(text):
    """Concentra advertencias generales idénticas conservando resultados y contexto específicos."""
    prefixes=('No se comparan magnitudes entre escalas','Un peso negativo reduce','text__ identifica','No se reajusta el modelo mirando estas asociaciones.')
    seen=set();paragraphs=[]
    for paragraph in text.split('\n\n'):
        if paragraph.startswith(prefixes):
            if paragraph in seen:continue
            seen.add(paragraph)
        paragraphs.append(paragraph)
    return '\n\n'.join(paragraphs)






def generalization_summary(root, report, quality):
    """Relaciona la evaluación por vídeo y por canal sin confundir sus protocolos."""
    lines = ['### Qué cambia al separar canales']
    drops = []
    for task in ['category', 'comments_disabled', 'ratings_disabled']:
        check = quality['tasks'][task]
        if check.get('status') != 'passed':
            continue
        # original y separated: métricas de evaluaciones internas distintas.
        original = report['tasks'][task]['holdout']['macro_f1']
        separated = check['test']['macro_f1']
        baseline = check['baseline_test']['macro_f1']
        drops.append(separated < original)
        lines.append(f'**{task_description(task).capitalize()}:** macro-F1 por vídeo **{format_number(original,3)}**; '
                     f'por canal **{format_number(separated,3)}**; referencia por canal '
                     f'**{format_number(baseline,3)}**. '
                     + ('El resultado por canal queda por debajo de su referencia.' if separated < baseline
                        else 'El resultado por canal supera su referencia.'))
    lines.append('**Conclusión.** El rendimiento de la partición por vídeo no basta para acreditar '
                 'generalización a otros canales. '
                 + ('La caída observada es compatible con que parte de las asociaciones textuales dependa de creadores, nombres propios o plantillas ' if any(drops) else
                    'Aunque no se observe una caída en estas tareas, las asociaciones pueden depender de creadores, nombres propios o plantillas ') +
                 
                 'recurrentes. Se relaciona así con los coeficientes exportados, sin demostrar '
                 'memorización ni atribuirles toda la diferencia: cambian también muestras y calibración. '
                 'Para una aplicación a canales nuevos habría que usar esa separación en la '
                 'validación de desarrollo y comprobarla después con una prueba independiente.')
    return '\n\n'.join(lines)


def regional_composition_summary(data, category_fractions, *, region_names=None):
    """Interpreta composición y repetición con denominadores y países legibles.

    region_names recibe el catálogo de nombres ya utilizado por el cuaderno.
    Los códigos desconocidos se conservan; no se deduce nacionalidad del creador.
    """
    # region_labels: presenta código y país sin duplicar el catálogo regional.
    region_labels = {code: f'{code} ({name})'
                     for code, name in (region_names or {}).items() if name}
    lines = ['**Conclusión del cruce categoría × región.** Categoría dominante en cada lista:']
    for region in category_fractions.columns:
        block = category_fractions[region]
        if block.sum() <= 0:
            continue
        label = block.idxmax()
        lines.append(f'{region_labels.get(region, region)}: {label}, **{format_number(block.loc[label]*100,2)} %** de sus apariciones.')
    differs = category_fractions.shape[1] > 1 and bool(category_fractions.sub(category_fractions.iloc[:,0], axis=0).abs().gt(1e-10).any().any())
    lines.append(('La mezcla editorial varía entre listas regionales. ' if differs else 'En este cruce la mezcla editorial no varía entre listas regionales. ') + 'Incluir region permite al '
                 'modelo representar esas diferencias observadas; su aportación incremental sólo '
                 'se demostraría comparando con una ablación sin ese atributo.')
    # repetition: compara listas por apariciones por vídeo, sin asignar nacionalidad.
    repetition = data.groupby('region').agg(appearances=('video_id','size'), videos=('video_id','nunique'))
    repetition['appearances_per_video'] = repetition.appearances / repetition.videos
    high = repetition.appearances_per_video.idxmax()
    low = repetition.appearances_per_video.idxmin()
    lines.append(f'**Repetición regional.** {region_labels.get(high, high)} registra **{format_number(repetition.loc[high,"appearances_per_video"],2)}** '
                 f'apariciones por vídeo, frente a **{format_number(repetition.loc[low,"appearances_per_video"],2)}** '
                 f'en {region_labels.get(low, low)}. Una lista con más repetición da más peso a las mismas identidades en '
                 'las métricas por aparición. Esto ayuda a interpretar los errores regionales, '
                 'pero no demuestra por sí solo por qué cambia el MAE (error absoluto medio). Las regiones identifican '
                 'listas de tendencias, no la nacionalidad de los creadores.')
    return '\n\n'.join(lines)


def quality_report_text(report):
    """Presenta diagnóstico por canal, costes y estabilidad con formato español.

    Sólo redacta evidencias recibidas. El umbral no se selecciona de nuevo tras
    observar costes de prueba; los escenarios no representan costes acordados.
    """
    lines = ['# Validaciones adicionales de calidad', '',
             'Son comprobaciones **internas** de la misma fuente histórica. Se congelan las recetas '
             'y se separan componentes vídeo–nombre de canal en ajuste, calibración y prueba. '
             'Los nombres no son identificadores canónicos de creador.', '',
             'Filas de ajuste, calibración y prueba: **' + human_value(report['sample_rows']) + '**. '
             'Solapamientos entre particiones: ' + human_value(report['partitions']['overlap_counts_train_cal_train_test_cal_test']) + '.', '',
             '| Tarea | Métrica | Prueba por canal | Referencia | Entrenamiento |',
             '|---|---|---:|---:|---:|']
    for task, result in report['tasks'].items():
        if result['status'] != 'passed':
            lines.append(f'| {task} | {result["status"]} | — | — | — |')
            continue
        metric = 'macro_f1' if 'macro_f1' in result['test'] else 'mae'
        lines.append(f'| {task} | {metric} | {format_number(result["test"][metric],3)} | '
                     f'{format_number(result["baseline_test"][metric],3)} | {format_number(result["train"][metric],3)} |')
    for task in ['comments_disabled', 'ratings_disabled']:
        result = report['tasks'][task]
        if result['status'] != 'passed':
            continue
        lines += ['', f'## {task_description(task).capitalize()}',
                  f'F2 por canal: **{format_number(result["positive_f2"],3)}** con calibración separada; '
                  f'**{format_number(result["default_positive_f2"],3)}** con umbral predeterminado. '
                  f'Umbral elegido en calibración: **{format_number(result["threshold"],6)}**.', '',
                  '| Coste FN | Coste/1.000 predeterminado | Coste/1.000 calibrado |', '|---:|---:|---:|']
        scenarios = result['cost_scenarios']
        for row in scenarios:
            lines.append(f'| {format_number(row["false_negative_unit_cost"],0)} | '
                         f'{format_number(row["default_cost_per_1000"],1)} | {format_number(row["calibrated_cost_per_1000"],1)} |')
        worse = sum(row['calibrated_cost_per_1000'] > row['default_cost_per_1000'] for row in scenarios)
        lines.append(f'**Conclusión sobre costes.** El umbral calibrado cuesta más en **{format_number(worse,0)} de '
                     f'{format_number(len(scenarios),0)}** escenarios considerados. Optimizar F2 '
                     'no equivale a minimizar el coste: aumentar la detección puede generar más falsas alarmas. '
                     'No hay una política operativa avalada por estos escenarios; se conserva el umbral '
                     'seleccionado en calibración y se informa el balance observado. '
                     'Una política futura requeriría acordar costes y optimizarlos en desarrollo, no en esta prueba.')
    stability = report['full_pipeline_stability']
    lines += ['', '## Estabilidad de toda la segmentación',
              f'ARI medio **{format_number(stability["mean_ARI"],3)}**; réplicas '
              + human_value(stability['replicate_ARI']) + '. En cada remuestreo se reajustan '
              f'TF-IDF, imputación, escala, SVD y centroides. Los **{format_number(stability["audit_videos"],0)}** '
              'vídeos de auditoría quedan fuera de todos esos ajustes.',
              'La receta está fijada. La estabilidad observada indica reproducibilidad de la partición '
              'bajo estos remuestreos internos; no valida su utilidad temática ni una nueva selección de receta.', '',
              '## Límites y posibles ampliaciones',
              'Una fuente externa y los identificadores canónicos de canal permitirían comprobar la '
              'generalización. El diagnóstico Unicode describe sistemas de escritura, no identifica idiomas.', '',
              'La evaluación humana de las recomendaciones queda fuera de lo solicitado en el ejercicio. '
              'Se propone como ampliación: requeriría un protocolo y valoraciones independientes de personas.']
    return '\n'.join(lines) + '\n'


def localize_policy(text):
    """Localiza decimales en el texto descriptivo de política, conservando el registro técnico."""
    # re: distingue decimales de versiones o nombres de archivo con varios puntos.
    import re
    return re.sub(r'(?<![\w.])(\d+)\.(\d+)(?![\w.])', lambda match: match[1]+','+match[2], text)


def conclusion_paragraphs(text):
    """Selecciona conclusiones redactadas para mostrarlas antes del detalle técnico."""
    return [paragraph for paragraph in text.split('\n\n') if paragraph.startswith('**Conclusión')]




def date_views_summary(stats):
    """Explica una sola distribución por calendario, sus horas de publicación y su cola."""
    cutoff=stats['cutoff'];panel=stats['panels'][1]
    lines=[f'**Lectura del gráfico principal.** Cada barra superior cuenta apariciones con '
           f'**0 a {cutoff} días de calendario** entre publicación y tendencias. '
           'No se conoce la hora real de entrada en tendencias. Los valores fuera de la ventana '
           'se cuentan abajo y no se llevan al último día.',
           f'**Cobertura.** Arriba hay **{format_number(panel["visible"],0)} de '
           f'{format_number(panel["valid"],0)} apariciones comparables**, '
           f'**{format_number(panel["visible"]/panel["valid"]*100,2)} %**. '
           f'Por encima de {cutoff} días: **{format_number(sum(panel["tail"]),0)}** '
           f'(**{format_number(sum(panel["tail"])/panel["valid"]*100,2)} %**). '
           f'Por debajo de 0 días: **{format_number(panel["underflow"],0)}**.']
    if stats.get('hour_labels'):
        lines.append('**Tonos de azul.** Representan la hora de publicación UTC en cuatro franjas: '
                     +', '.join(stats['hour_labels'])+'. Se incluye el inicio y se excluye el final. '
                     'La altura sigue contando apariciones; cada porcentaje se divide por el total '
                     'de su barra diaria. Las etiquetas se redondean a enteros y se muestran donde caben. '
                     'El desglose completo conserva los porcentajes con dos decimales.')
        rows=panel.get('hour_rows',[])
        if rows:
            largest=max(row['Total de la barra'] for row in rows)
            day=next(row['Intervalo de días'] for row in rows if row['Total de la barra']==largest)
            strongest=max((row for row in rows if row['Intervalo de días']==day),key=lambda row:row['Apariciones'])
            lines.append(f'**Lectura concreta.** En una de las barras de mayor volumen, el día **{day}**, '
                         f'la franja **{strongest["Franja UTC"]}** reúne '
                         f'**{format_number(strongest["Apariciones"],0)} de {format_number(largest,0)} apariciones**: '
                         f'**{format_number(strongest["Porcentaje de la barra"],2)} % de esa barra**. '
                         'Describe el intervalo observado y no demuestra un horario óptimo de publicación.')
    lines.append('**Cola inferior.** Forma parte del mismo conjunto de apariciones que la vista principal. '
                 'Cada porcentaje se divide por todos los pares válidos, no sólo por los casos de la cola. '
                 f'El primer intervalo significa más de {cutoff} días y hasta '
                 f'{min(value for value in (30,60,365) if value>cutoff)}, incluido su extremo superior. '
                 'El último reúne los valores superiores a 365 días; los originales no se recortan.')
    lines.append('**Conclusión.** La vista principal permite estudiar la concentración inicial y '
                 'la cola cuantifica las apariciones tardías sin ocultarlas. Son apariciones regionales '
                 'y diarias, con vídeos repetidos, no tiempos hasta la primera entrada de vídeos únicos. '
                 'Las franjas describen publicaciones UTC, no horas de entrada ni horas locales de los creadores.')
    return '\n\n'.join(lines)


def date_diagnostic_summary(stats):
    """Explica la precisión temporal como diagnóstico opcional, sin otro histograma."""
    hourly,calendar=stats['panels']
    return ('**Qué comprueba este diagnóstico.** La lista de tendencias sólo registra el día. '
            'Representarlo a las 00:00 y restar la hora de publicación introduce fracciones '
            'de día y posibles negativos. La comparación comprueba esa precisión; no es otra '
            'distribución independiente del comportamiento de los vídeos.\n\n'
            '| Comprobación | Diferencias negativas |\n|---|---:|\n'
            f'| Resta con hora de publicación (A) | **{format_number(hourly["negative"],0)}** |\n'
            f'| Diferencia por calendario, usada en el gráfico principal | **{format_number(calendar["negative"],0)}** |\n\n'
            '**Ejemplos conceptuales, no filas adicionales del dataset:**\n\n'
            '| Publicación y lista de tendencias | Resta con hora | Calendario |\n|---|---:|---:|\n'
            '| Publicación el día 14 a las 18:00 UTC; lista del día 15 | 6 horas = 0,25 días | 1 día |\n'
            '| Publicación el día 15 a las 16:00 UTC; lista del mismo día | -16 horas = -0,67 días | 0 días |\n\n'
            'Al agrupar en intervalos de un día, las mismas apariciones cambian principalmente '
            'de posición entre las dos medidas; por eso no se presentan dos histogramas casi iguales. '
            'Los negativos de A no prueban que el vídeo entrara antes de publicarse. '
            'Para detectar una fecha de tendencias anterior al día de publicación se utiliza el calendario.')


# Todas las funciones de presentación están definidas antes de ejecutar el informe.
if __name__ == '__main__':
    print(main())
