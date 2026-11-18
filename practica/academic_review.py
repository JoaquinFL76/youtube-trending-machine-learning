"""Matriz del temario y balance verificable de convergencia de la ejecución actual."""
# metric_utils: comparte el cálculo de F2 y validaciones de matrices de confusión.
# Se importa: f2_from_confusion.
from metric_utils import f2_from_confusion
# json: lee y escribe datos estructurados, configuraciones y evidencias en JSON.
import json
# math: proporciona operaciones y comprobaciones matemáticas escalares.
import math
# pathlib: maneja rutas de archivos y carpetas mediante objetos, como Path.
# Se importa: Path.
from pathlib import Path
# pandas: organiza, limpia y analiza datos tabulares mediante DataFrame y Series.
import pandas as pd
# results_report: redacta conclusiones y límites a partir de evidencias calculadas.
# Se importa: format_number.
from results_report import format_number


def write_coverage(report, directory):
    """Genera cobertura desde ensayos válidos y artefactos disponibles, sin entrenar.

    Distingue técnicas probadas, ampliaciones y descartes metodológicos. No presenta
    Naive Bayes como cobertura completa de redes bayesianas.
    Escribe CSV y Markdown y devuelve el texto para incorporarlo al cuaderno.
    """
    rows = [
        ('Regresión lineal y logística', 'Probado', 'Ridge, LogisticRegression y variantes log1p; regularización por CV'),
        ('Perceptrón multicapa', 'Probado', 'MLPClassifier/Regressor con SVD y escalado latente dentro de CV; iteraciones y avisos en convergence_review.json'),
        ('Árboles y Random Forest', 'Probado', 'Clasificación/regresión; mismos folds y referencias'),
        ('KNN y distancias', 'Probado', 'Vecinos con coseno y recuperación de contenido'),
        ('SVM lineal y kernel RBF', 'Probado', 'LinearSVC/LinearSVR frente a SVC/SVR RBF; SVD y escalado latente dentro de cada fold'),
        ('PCA y t-SNE', 'Probado', 'Visualización descriptiva; no validan por sí mismas los clusters'),
        ('TruncatedSVD', 'Probado', 'Texto disperso; dimensiones comparadas y preprocesamiento persistido'),
        ('LDA supervisado', 'Probado', 'Clasificación con SVD de 30/60 dimensiones dentro de cada fold y covarianza con/sin regularización'),
        ('KMeans, MiniBatch, Mean Shift y GMM', 'Probado', 'Mismos datos e índices; bandwidth/componentes/covarianza; selección sin categorías'),
        ('DBSCAN y aglomerativo', 'Probado como exploración', 'Cobertura y ARI/NMI; ajuste transductivo separado de los candidatos inductivos'),
        ('Redes bayesianas', 'No incorporado como modelo general', 'MultinomialNB es una referencia restringida; un DAG requeriría justificar estructura y discretización dentro de CV'),
        ('HMM', 'No incorporado', 'Exigiría formular estados/secuencias y validación temporal; objetivos actuales contemporáneos'),
        ('XGBoost', 'Probado', 'Árboles por gradiente en CPU para clasificación y regresión con TF-IDF y metadatos. Regresión directa y log1p'),
        ('IA generativa, RAG y agentes', 'Contenido optativo no incorporado', 'No necesario para los objetivos del enunciado; sin servicios externos'),
    ]
    requirements = {
        'Regresión lineal y logística': {'ridge', 'logistic'},
        'Perceptrón multicapa': {'mlp_classifier', 'mlp_regressor'},
        'Árboles y Random Forest': {'tree_classifier', 'tree_regressor', 'forest_classifier', 'forest_regressor'},
        'KNN y distancias': {'knn_classifier', 'knn_regressor'},
        'SVM lineal y kernel RBF': {'svm', 'svr', 'svm_rbf', 'svr_rbf'},
        'LDA supervisado': {'lda_classifier'},
        'XGBoost': {'xgb_classifier', 'xgb_regressor', 'xgb_regressor_log_target'},
    }
    trials = [trial for result in report.get('tasks', {}).values() for trial in result.get('trials', [])]
    attempted = {trial['model'] for trial in trials}
    successful = {trial['model'] for trial in trials
                  if 'error' not in trial and isinstance(trial.get('cv_score'), (int, float))
                  and math.isfinite(trial['cv_score'])}
    cluster_trials = report.get('clustering', [])
    cluster_attempted = {trial['name'] for trial in cluster_trials}
    cluster_successful = {trial['name'] for trial in cluster_trials if 'error' not in trial
                         and trial.get('silhouette') is not None}
    checked = []
    for block, state, evidence in rows:
        required = requirements.get(block)
        present, done = attempted, successful
        if block == 'KMeans, MiniBatch, Mean Shift y GMM':
            required = {'kmeans', 'minibatch', 'meanshift', 'gmm'}
            present, done = cluster_attempted, cluster_successful
        elif block == 'DBSCAN y aglomerativo':
            required = {'dbscan', 'agglomerative'}
            present, done = cluster_attempted, cluster_successful
        if required:
            missing = required - done
            if missing:
                state = 'Parcial' if required & done else ('Sin resultado válido' if required & present else 'No ejecutado')
            evidence += '; evaluation.json: ' + ', '.join(sorted(required & done))
            if missing:
                evidence += '; sin resultado válido: ' + ', '.join(sorted(missing))
        elif block == 'PCA y t-SNE':
            projections = ['reduction_pca_tsne.png']
            existing = [name for name in projections if (directory / name).is_file()]
            state = 'Probado' if len(existing) == len(projections) else ('Parcial' if existing else 'Sin evidencia disponible')
            evidence += '; cuaderno, sección 8: PCA y t-SNE; gráfico regenerable' if existing else '; sin gráfico disponible'
        elif block == 'TruncatedSVD':
            valid = [trial for trial in cluster_trials if trial.get('dimensions', 0) > 0 and trial.get('silhouette') is not None]
            state = 'Probado' if valid else 'Sin resultado válido'
            evidence += '; evaluation.json/clustering: ' + str(len(valid)) + ' resultados con dimensiones y silhouette'
        checked.append((block, state, evidence))
    rows = checked
    frame = pd.DataFrame(rows, columns=['bloque', 'estado', 'evidencia o justificación'])
    frame.to_csv(directory / 'temario_coverage.csv', index=False)
    lines = ['# Cobertura del módulo 6', '',
             'El estado distingue experimentos ejecutados de ampliaciones no incorporadas. Las fuentes son las presentaciones y notebooks del módulo y la explicación de la práctica; no se afirma haber probado todas las técnicas existentes.', '',
             '| Bloque | Estado | Evidencia o justificación |', '|---|---|---|']
    lines.extend('| ' + ' | '.join(row) + ' |' for row in rows)
    lines += ['', 'El [glosario de la práctica](../GUIA_DE_CONSULTA.md#glosario) explica cada técnica, incluso las que quedan fuera de la comparación.']
    lines += ['', 'Los retos opcionales se presentan como ampliaciones del problema. La matriz documenta el alcance metodológico y enlaza la cobertura con los experimentos y sus limitaciones. Fuentes: presentaciones 1–8 y notebooks 1–10 del módulo; explicación de la práctica, 00:19:18–00:22:43.']
    lines += ['', 'Los avisos de cada búsqueda pueden proceder de configuraciones descartadas; no identifican necesariamente al ganador. final_fit en convergence_review.json registra por separado los avisos del ajuste final de cada modelo. La ausencia de aviso no demuestra un óptimo global.', '', 'AIC/BIC se informan sólo para GMM y son comparables entre configuraciones de la misma representación y muestra, no entre familias o dimensiones SVD distintas.']
    text = '\n'.join(lines) + '\n'
    (directory / 'temario_coverage.md').write_text(text, encoding='utf-8')
    return text


def write_convergence(report, directory):
    """Exporta avisos de búsquedas y ajustes finales sin atribuirlos indebidamente."""
    convergence = []
    for task, result in report['tasks'].items():
        for trial in result['trials']:
            review = trial.get('convergence_review', {})
            convergence.append({'task': task, 'stage': 'familias', 'model': trial['model'],
                                'error': trial.get('error'), **review})
        convergence.append({'task': task, 'stage': 'refinamiento', 'model': result['selected'],
                            **result.get('refinement', {}).get('convergence_review', {})})
    final_path = directory / 'final_training.json'
    final = json.loads(final_path.read_text(encoding='utf-8')) if final_path.exists() else {}
    record = {'scope': 'Avisos agregados de todas las configuraciones de cada búsqueda; pueden corresponder a candidatos descartados. final_fit separa los avisos del ganador al ajustarse con la receta congelada.',
              'searches': convergence, 'final_fit': final.get('models', {})}
    (directory / 'convergence_review.json').write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding='utf-8')
    pd.DataFrame([{'task': row['task'], 'stage': row['stage'], 'model': row['model'],
                   'attempts': len(row.get('attempts', [])), 'status': row.get('status', 'error'),
                   'remaining_warnings': ' | '.join(row.get('remaining_convergence_warnings', []))}
                  for row in convergence]).to_csv(directory / 'convergence_review.csv', index=False)


def update_readme(report, directory, root):
    """Actualiza las secciones de resultados con evidencias, sin modificar modelos."""
    # README se recalcula con esta ejecución para no conservar cifras anteriores.
    readme_path = root / 'README.md'
    readme = readme_path.read_text(encoding='utf-8')
    start = readme.index('## Resultados principales de la prueba interna')
    end = readme.index('## Lectura de resultados', start)
    summary = ['## Resultados principales de la prueba interna', '',
               '| Objetivo | Resultado actual |', '|---|---:|']
    for task, label in [('category', 'Categoría'), ('likes', 'Likes'), ('ratio', 'Ratio')]:
        metrics = report['tasks'][task]['holdout']
        value = (f"Macro-F1 {format_number(metrics['macro_f1'])}" if task == 'category'
                 else f"MAE {format_number(metrics['mae'])}; R² {format_number(metrics['r2'])}")
        summary.append(f'| {label} | {value} |')
    profiles = pd.read_csv(directory / 'cluster_profiles.csv')
    total = int(profiles.n.sum())
    summary += ['', f"**Alcance de la segmentación:** el grupo mayor concentra {format_number(100 * profiles.n.max()/total, 1)} % de {format_number(total, 0)} vídeos y el menor contiene {int(profiles.n.min())}. La selección por silhouette es geométrica; no acredita por sí sola grupos temáticos ni utilidad comercial. Consultar los perfiles y las alternativas.", '']
    quality = json.loads((directory / 'quality_checks.json').read_text(encoding='utf-8'))
    comments = quality['tasks']['comments_disabled']
    ratings = report['tasks']['ratings_disabled']
    matrix = ratings['default_threshold_holdout']['confusion_matrix']
    default_f2 = f2_from_confusion(matrix)
    summary += ['', f"**Límites medidos:** en el diagnóstico por canales disjuntos, comentarios obtiene macro-F1 {format_number(comments['test']['macro_f1'])} frente a {format_number(comments['baseline_test']['macro_f1'])} de referencia. Este diagnóstico reajusta las recetas con otra partición y calibración; no evalúa directamente los pesos entregados. En valoraciones, F2 pasa de {format_number(default_f2)} con el umbral predeterminado a {format_number(ratings['positive_f2'])} con el umbral OOF. Las conclusiones detallan si mejora o empeora y mantienen la política elegida en entrenamiento.", '',
                'Las cifras corresponden al holdout interno histórico por vídeo; no acreditan validación externa prospectiva ni relevancia humana del recomendador.', '',
                '## Lectura de resultados']
    readme = readme[:start] + '\n'.join(summary) + readme[end + len('## Lectura de resultados'):]
    # re: busca y transforma texto mediante expresiones regulares.
    import re
    evidence_path = directory / 'notebook_execution.json'
    if evidence_path.exists():
        execution = json.loads(evidence_path.read_text(encoding='utf-8'))
        training = execution if execution.get('training_mode') == 'full_from_scratch' else execution.get('full_training_reference', {})
        if execution.get('status') == 'passed' and 'seconds' in training:
            minutes = training['seconds'] / 60
            readme = re.sub(r'la (?:última ejecución completa|ejecución) desde cero registrada tardó .*?(?:según el equipo y la carga|el tiempo depende del equipo y de la carga)\.',
                            f"la última ejecución completa desde cero registrada tardó {format_number(minutes, 1)} minutos; el tiempo depende del equipo y de la carga.", readme)
    readme_path.write_text(readme, encoding='utf-8')


def main():
    """Orquesta cobertura, convergencia y resumen desde los informes actuales."""
    root = Path(__file__).resolve().parent
    directory = root / 'reports'
    report = json.loads((directory / 'evaluation.json').read_text(encoding='utf-8'))
    text = write_coverage(report, directory)
    write_convergence(report, directory)
    update_readme(report, directory, root)
    return text


# Punto de entrada: al ejecutar este archivo directamente, __name__ vale "__main__".
# El bloque siguiente inicia su ejecución; al importarlo desde otro archivo, no se ejecuta.
if __name__ == '__main__':
    print(main())
