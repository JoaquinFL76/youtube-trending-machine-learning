"""Comparación agrupada, umbral OOF, ajuste ampliado y una evaluación final reservada."""
# metric_utils: comparte el cálculo de F2 y validaciones de matrices de confusión.
# Se importa: f2_from_precision_recall.
from metric_utils import f2_from_precision_recall
# argparse: interpreta y valida los argumentos recibidos por la línea de comandos.
import argparse
# configparser: lee y escribe las secciones de configuración de train.conf.
import configparser
# hashlib: calcula huellas digitales para comprobar integridad y procedencia.
import hashlib
# json: lee y escribe datos estructurados, configuraciones y evidencias en JSON.
import json
# platform: consulta versiones e información del entorno de ejecución.
import platform
# importlib.metadata: consulta la versión instalada de la dependencia experimental.
# Se importa: version.
from importlib.metadata import version
# tempfile: crea archivos y directorios temporales para operaciones aisladas.
import tempfile
# time: mide duraciones y consulta referencias de tiempo del proceso.
import time
# warnings: emite, captura o filtra avisos sin confundirlos con errores.
import warnings
# gc: permite liberar objetos que ya no se necesitan mediante el recolector de memoria.
import gc
# os: accede a variables de entorno y servicios del sistema operativo.
import os
# refinement: refina hiperparámetros de la familia seleccionada con validación cruzada.
# Se importa: refine.
from refinement import refine
# train_models: ajusta y guarda los modelos ganadores a partir de recetas congeladas.
# Se importa: fit_selected.
from train_models import fit_selected
# pathlib: maneja rutas de archivos y carpetas mediante objetos, como Path.
# Se importa: Path.
from pathlib import Path
# numpy: realiza cálculos numéricos y operaciones con vectores y matrices.
import numpy as np
# pandas: organiza, limpia y analiza datos tabulares mediante DataFrame y Series.
import pandas as pd
# sklearn: aporta algoritmos de aprendizaje automático y su versión de biblioteca.
import sklearn
# joblib: gestiona cachés y persistencia de objetos usados en los experimentos.
# Se importa: Memory.
from joblib import Memory
# sklearn.base: clona estimadores sin reutilizar el estado aprendido de otro ajuste.
# Se importa: clone.
from sklearn.base import clone
# sklearn.model_selection: organiza particiones, validación cruzada y búsquedas de parámetros.
# Se importa: GroupShuffleSplit, StratifiedGroupKFold, GroupKFold, GridSearchCV.
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold, GroupKFold, GridSearchCV
# sklearn.metrics: calcula métricas para evaluar predicciones, grupos y similitudes.
# Se importa: f1_score, fbeta_score, balanced_accuracy_score, classification_report, confusion_matrix, mean_absolute_error, root_mean_squared_error, r2_score, average_precision_score, precision_recall_curve, precision_score, recall_score.
from sklearn.metrics import (f1_score, fbeta_score, balanced_accuracy_score, classification_report,
    confusion_matrix, mean_absolute_error, root_mean_squared_error, r2_score, average_precision_score,
    precision_recall_curve, precision_score, recall_score)
# models: expone las siete clases finales y el registro que relaciona tareas con clases.
# Se importa: REGISTRY.
from models import REGISTRY
# models.core: comparte validación, transformaciones, factorías y persistencia de modelos.
# Se importa: features, transformer, Pipeline, estimator_from_name, positive_scores.
from models.core import features, transformer, Pipeline, estimator_from_name, positive_scores
# unsupervised_analysis: coordina los experimentos de clustering y recomendación.
# Se importa: run_unsupervised.
from unsupervised_analysis import run_unsupervised

# TASKS: objetivos supervisados que se evalúan por separado.
TASKS = ['category', 'comments_disabled', 'ratings_disabled', 'likes', 'ratio']
# CLASSIFICATION: subconjunto de objetivos con etiquetas discretas.
CLASSIFICATION = TASKS[:3]

def dump(path, value):
    """Serializa value como JSON UTF-8 en path, convirtiendo tipos auxiliares a texto.

    No crea el directorio padre; propaga errores de escritura. Devuelve None."""
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding='utf-8')

def candidates(classification):
    """Construye familias y rejillas de clasificación o regresión sin ajustarlas.

    Devuelve tuplas (nombre, estimador nuevo, parámetros de Pipeline). En regresión
    compara también log1p del objetivo; la referencia mantiene su escala original."""
    # suffix: sufijo que distingue familias de clasificación y regresión.
    suffix = '_classifier' if classification else '_regressor'
    # specifications: familias y rejillas antes de expandir objetivos logarítmicos.
    specifications = ([('baseline'+suffix, {}), ('knn'+suffix, {'n_neighbors':[3,7]}),
        ('tree'+suffix, {'max_depth':[10,20]}), ('logistic', {'C':[0.3,2.0]}),
        ('svm', {'C':[0.3,2.0]}), ('svm_rbf', {'C':[1.,3.], 'gamma':['scale']}), ('naive_bayes', {'alpha':[0.5,2.0]}),
        ('sgd', {'alpha':[0.0001,0.001]}), ('forest'+suffix, {'max_depth':[10,20]}),
        ('mlp'+suffix, {'alpha':[0.0001,0.01]}),
        ('lda_classifier', {'shrinkage': [None, 'auto']}),
        ('xgb_classifier', {'max_depth': [3, 6]})] if classification else
        [('baseline'+suffix, {}), ('knn'+suffix, {'n_neighbors':[3,7]}), ('tree'+suffix, {'max_depth':[10,20]}),
         ('ridge', {'alpha':[1.,10.]}), ('svr', {'C':[0.1,1.]}),
         ('svr_rbf', {'C':[1.,3.], 'gamma':['scale']}), ('forest'+suffix, {'max_depth':[10,20]}),
         ('extra_trees', {'max_depth':[10,20]}), ('mlp'+suffix, {'alpha':[0.0001,0.01]}),
         ('xgb_regressor', {'max_depth': [3, 6]})])
    # result: lista de familias, estimadores nuevos y rejillas que se compararán por CV.
    result = []
    for name, grid in specifications:
        result.append((name, estimator_from_name(name), {'estimator__'+k:v for k,v in grid.items()}))
        if not classification and not name.startswith('baseline'):
            result.append((name+'_log_target', estimator_from_name(name+'_log_target'), {'estimator__regressor__'+k:v for k,v in grid.items()}))
    return result

# training_data: comparte la partición por vídeo y el muestreo de entrenamiento y prueba.
# Se importa: group_sample, split_video_pools.
from training_data import group_sample, split_video_pools, build_group_folds

def evaluate(ytrain, ytest, prediction, classification):
    """Calcula métricas en escala original, sin ajustar ni seleccionar modelos.

    En clasificación usa la unión de clases de ytrain/ytest y cero para divisiones
    indefinidas; devuelve métricas, soportes y matriz. En regresión devuelve
    MAE, RMSE y R². prediction debe estar alineada con ytest."""
    if classification:
        # labels: orden de etiquetas que fija matrices y medias macro.
        labels = sorted(set(ytrain) | set(ytest))
        return {'macro_f1':float(f1_score(ytest,prediction,labels=labels,average='macro',zero_division=0)),
            'balanced_accuracy':float(balanced_accuracy_score(ytest,prediction)),
            'classification_report':classification_report(ytest,prediction,labels=labels,zero_division=0,output_dict=True),
            'confusion_matrix':confusion_matrix(ytest,prediction,labels=labels).tolist(), 'labels':[str(x) for x in labels]}
    return {'mae':float(mean_absolute_error(ytest,prediction)), 'rmse':float(root_mean_squared_error(ytest,prediction)), 'r2':float(r2_score(ytest,prediction))}

def ci_by_video(y, pred, groups, classification, seed, labels):
    """Estima intervalos percentiles del 95 % con 150 remuestreos por vídeo.

    Cada réplica pondera juntas todas las apariciones del mismo grupo. Devuelve
    un diccionario de límites y, en binario, precisión/recall positivos.
    El intervalo es condicional al modelo: no incorpora selección ni cambio temporal."""
    # codes: código de grupo de cada fila para expandir pesos por vídeo; ids: identificadores únicos usados en
    # muestreo o recuperación.
    codes, ids = pd.factorize(groups)
    # rng: generador aleatorio local, sin modificar el estado global.
    rng = np.random.default_rng(seed)
    # values: métrica de cada réplica bootstrap para obtener percentiles 2,5 y 97,5; positive_precision:
    # precisiones positivas por réplica o en la evaluación actual; positive_recall: sensibilidades positivas
    # por réplica o en la evaluación actual.
    values, positive_precision, positive_recall = [], [], []
    for _ in range(150):
        # weights: multiplicidades de remuestreo asignadas juntas a todas las filas de cada vídeo.
        weights = rng.multinomial(len(ids), np.full(len(ids), 1/len(ids)))[codes]
        score = (f1_score(y,pred,labels=labels,average='macro',zero_division=0,sample_weight=weights)
                 if classification else mean_absolute_error(y,pred,sample_weight=weights))
        values.append(score)
        if classification and set(labels)=={0,1}:
            positive_precision.append(float(precision_score(y,pred,sample_weight=weights,zero_division=0)))
            positive_recall.append(float(recall_score(y,pred,sample_weight=weights,zero_division=0)))
    # result: resultado local devuelto o añadido al informe.
    result = {'metric':'macro_f1' if classification else 'mae', 'method':'percentiles bootstrap por video_id; 150 réplicas; condicional al modelo ajustado',
              'low':float(np.quantile(values,.025)), 'high':float(np.quantile(values,.975))}
    if positive_recall:
        result['positive_recall_interval'] = np.quantile(positive_recall,[.025,.975]).tolist()
        result['positive_precision_interval'] = np.quantile(positive_precision,[.025,.975]).tolist()
        result['positive_rows'] = int(np.sum(np.asarray(y)==1))
        result['positive_video_groups'] = int(pd.Series(np.asarray(groups)[np.asarray(y)==1]).nunique())
    return result

def representativeness(full, samples):
    """Compara proporciones de cada muestra con el corpus completo.

    Devuelve una tabla por campo, valor y muestra; difference_pp está en puntos
    porcentuales. Describe composición, sin usar la prueba para ajustar modelos."""
    tables = []
    for field in ['category','region','comments_disabled','ratings_disabled']:
        # reference: distribución de proporciones del corpus de referencia.
        reference = full[field].value_counts(normalize=True, dropna=False)
        for name, sample in samples.items():
            # observed: proporciones de la muestra que se comparan con el corpus.
            observed = sample[field].value_counts(normalize=True, dropna=False)
            for value in reference.index.union(observed.index):
                tables.append({'field':field,'value':str(value),'sample':name,'full_fraction':reference.get(value,0),
                    'sample_fraction':observed.get(value,0), 'difference_pp':100*(observed.get(value,0)-reference.get(value,0))})
    return pd.DataFrame(tables)

# sklearn.decomposition: reduce dimensiones mediante técnicas como PCA y SVD.
# Se importa: TruncatedSVD.
from sklearn.decomposition import TruncatedSVD
# sklearn.preprocessing: codifica categorías, escala atributos y aplica transformaciones.
# Se importa: StandardScaler.
from sklearn.preprocessing import StandardScaler
# convergence_review: revisa avisos y realiza reintentos acotados de convergencia.
# Se importa: fit_search_review.
from convergence_review import fit_search_review


def fit_search(task, x, y, splits, cache, report_dir):
    """Selecciona la mejor familia por CV sobre particiones agrupadas prefijadas.

    Ajusta transformadores dentro de cada fold y conserva tiempos, avisos y errores
    en report_dir. Devuelve (ensayos, mejor tupla); maximiza macro-F1 o −MAE.
    Lanza RuntimeError si ninguna familia es válida. No consulta el holdout."""
    # trials: ensayos acumulados con puntuaciones, tiempos y avisos; best: tupla con puntuación, nombre,
    # estimador y diagnóstico de la mejor familia.
    trials, best = [], None
    for name, estimator, params in candidates(task in CLASSIFICATION):
        dump(report_dir/'progress.json', {'task':task,'candidate':name,'status':'running'})
        print('Comparando',task,name,flush=True)
        if name in ['logistic','ridge','ridge_log_target']:
            # params: hiperparámetros del estimador o rutas de Pipeline del bloque.
            params = dict(params, **{'features__text__ngram_range':[(1,1),(1,2)]})
        # pipe: Pipeline de preprocesamiento y estimación dentro de CV.
        steps = [('features', transformer(task))]
        if 'rbf' in name or name.startswith('mlp') or name == 'lda_classifier':
            dimensions = [30, 60] if name == 'lda_classifier' else [60] if name.startswith('mlp') else [30]
            steps.append(('reduce', TruncatedSVD(n_components=dimensions[0], random_state=42)))
            steps.append(('latent_scale', StandardScaler()))
            params = dict(params, **{'reduce__n_components': dimensions})
        steps.append(('estimator', estimator))
        pipe = Pipeline(steps, memory=cache)
        # search: muestra pequeña empleada para seleccionar familias e hiperparámetros.
        search = GridSearchCV(pipe, params, cv=splits, scoring='f1_macro' if task in CLASSIFICATION else 'neg_mean_absolute_error',
                              return_train_score=True, n_jobs=1, error_score='raise')
        # start: reloj monotónico del ajuste del candidato.
        start = time.monotonic()
        try:
            review = fit_search_review(search, features(x), y)
            i = search.best_index_
            # trial: diagnóstico del candidato evaluado en esta iteración.
            trial = {'model':name, 'cv_score':float(search.best_score_),
                'cv_std':float(search.cv_results_['std_test_score'][i]),
                'cv_train_score':float(search.cv_results_['mean_train_score'][i]),
                'fold_scores':[float(search.cv_results_[f'split{k}_test_score'][i]) for k in range(len(splits))],
                'params':dict(search.best_params_, **review['iteration_parameters']), 'seconds':time.monotonic()-start,
                'warnings':review['warnings'], 'convergence_review':review}
            trials.append(trial)
            if best is None or search.best_score_ > best[0]:
                # best: tupla con puntuación, nombre, estimador y diagnóstico de la mejor familia.
                best = (search.best_score_, name, search.best_estimator_, trial)
        except (ValueError, TypeError) as exc:
            trials.append({'model':name,'error':str(exc),'seconds':time.monotonic()-start})
        dump(report_dir/f'{task}_trials.json', trials)
    if best is None:
        raise RuntimeError(f'Ningún candidato válido para {task}')
    return trials, best

def learning_curve(task, winner, search, y, splits, seed):
    """Reajusta clones del ganador con tres tamaños del entrenamiento de cada fold.

    Devuelve puntuaciones de entrenamiento/validación y sus tamaños efectivos.
    La validación permanece fija y ningún vídeo cruza el límite del fold.
    No extrapola la saturación al pool ampliado ni al corpus completo."""
    # points: resultados acumulados para los tres tamaños de entrenamiento.
    points = []
    for fraction in [.25,.5,1.0]:
        # scores: puntuaciones alineadas con filas o folds del bloque; train_scores: puntuaciones en ajuste
        # para contrastar con validación; sizes: filas efectivas por fold al conservar grupos completos.
        scores, train_scores, sizes = [], [], []
        for train, valid in splits:
            # small: subconjunto de vídeos completos para un punto de la curva.
            small = group_sample(search.iloc[train], max(100,int(len(train)*fraction)), seed)
            idx = small.index
            # estimator: estimador concreto de la familia, distinto del preprocesador.
            estimator = clone(winner)
            estimator.fit(features(small),y.loc[idx])
            # pred: predicciones alineadas con las etiquetas de evaluación; fitted: predicciones del clon
            # sobre su propio subconjunto de ajuste.
            pred, fitted = estimator.predict(features(search.iloc[valid])), estimator.predict(features(small))
            if task in CLASSIFICATION:
                score = f1_score(y.iloc[valid],pred,average='macro',zero_division=0)
                # train_score: métrica sobre las filas usadas para ajustar el clon.
                train_score = f1_score(y.loc[idx],fitted,average='macro',zero_division=0)
            else:
                # score: valor de la métrica del candidato o réplica actual; train_score: métrica sobre las
                # filas usadas para ajustar el clon.
                score, train_score = mean_absolute_error(y.iloc[valid],pred), mean_absolute_error(y.loc[idx],fitted)
            sizes.append(len(small))
            scores.append(float(score))
            train_scores.append(float(train_score))
        points.append({'fraction':fraction,'mean_train_rows':float(np.mean(sizes)), 'validation_mean':float(np.mean(scores)),
                       'validation_std':float(np.std(scores)), 'train_mean':float(np.mean(train_scores)), 'fold_scores':scores})
    return points

def tune_threshold(winner, search, y, splits):
    """Elige un umbral que maximiza F2 usando puntuaciones fuera de fold.

    Cada fila recibe una puntuación de un clon que no vio su vídeo. Devuelve
    (umbral, diagnóstico OOF, vector de puntuaciones). La selección previa de
    hiperparámetros impide interpretar este OOF como validación anidada independiente."""
    # scores: puntuaciones alineadas con filas o folds del bloque.
    scores = np.empty(len(search))
    for train, valid in splits:
        fitted = clone(winner).fit(features(search.iloc[train]), y.iloc[train])
        scores[valid] = positive_scores(fitted, features(search.iloc[valid]))
    # precision: precisión de la clase positiva a lo largo de los umbrales; recall: sensibilidad de la clase
    # positiva a lo largo de los umbrales; thresholds: umbrales asociados a los pares de precisión y recall.
    precision, recall, thresholds = precision_recall_curve(y, scores)
    # f2: F2: media armónica que pondera recall cuatro veces más que precisión.
    f2 = f2_from_precision_recall(precision[:-1], recall[:-1])
    i = int(np.argmax(f2))
    return float(thresholds[i]), {'objective':'F2 OOF: recall pesa más; elección académica, pendiente validar costes reales',
        'oof_average_precision':float(average_precision_score(y,scores)), 'oof_prevalence':float(np.mean(y)),
        'oof_precision':float(precision[i]),'oof_recall':float(recall[i]),'oof_f2':float(f2[i]),
        'warning':'OOF reutiliza la selección de hiperparámetros; no es una estimación anidada independiente'}, scores

def _fit_supervised_task(task, search, final, test, conf, base, seed, report_dir, training_manifest, cache):
    """Compara, ajusta y evalúa una sola tarea con particiones ya reservadas.

    search/final/test son DataFrames disjuntos por vídeo donde corresponde.
    conf/base/report_dir fijan rutas y folds; seed conserva reproducibilidad.
    training_manifest identifica la ejecución y cache acelera transformaciones CV.
    Guarda pesos y diagnósticos, y devuelve (métricas, receta) sin elegir con test.
    """
    print('TAREA',task,flush=True)
    # x: entradas de la tarea o fixture local, alineadas con sus etiquetas.
    x = search.dropna(subset=[task]).reset_index(drop=True)
    # train: filas o índices de ajuste de la partición local.
    train = final.dropna(subset=[task]).reset_index(drop=True)
    # holdout: filas de prueba con objetivo válido; no participan en el ajuste.
    holdout = test.dropna(subset=[task]).reset_index(drop=True)
    # y: objetivo alineado por fila con las entradas de ajuste; yfinal: objetivo válido de la muestra
    # de ajuste ampliado; ytest: objetivo real de las filas reservadas para evaluación.
    y, yfinal, ytest = x[task],train[task],holdout[task]
    # classification: indicador que determina métrica y familia de estimadores.
    classification = task in CLASSIFICATION
    if task != 'category':
        # y: objetivo alineado por fila con las entradas de ajuste; yfinal: objetivo válido de la
        # muestra de ajuste ampliado; ytest: objetivo real de las filas reservadas para evaluación.
        y,yfinal,ytest = [pd.to_numeric(v).astype(int if classification else float) for v in [y,yfinal,ytest]]
    # cv: particionador de validación cruzada agrupada.
    splits = build_group_folds(x, y, conf.getint('training','folds'), classification, seed)
    # trials: ensayos acumulados con puntuaciones, tiempos y avisos; best: tupla con puntuación,
    # nombre, estimador y diagnóstico de la mejor familia.
    trials, best = fit_search(task,x,y,splits,cache,report_dir)
    # winner: mejor candidato según validación, sin seleccionar por la prueba.
    winner = clone(best[2])
    winner.memory = None
    # selected_params: parámetros elegidos por CV, mantenidos al ampliar el ajuste.
    selected_params = best[3]['params']
    if conf.getint('training', 'refinement_trials', fallback=0):
        winner, selected_params, refined_score, refinement = refine(
            task, winner, best[3], x, y, splits, cache, report_dir, seed,
            conf.getint('training', 'refinement_trials'))
        winner.memory = None
        best = (refined_score, best[1], winner, dict(best[3], params=selected_params))
    else:
        refinement = None
    # threshold: punto operativo binario seleccionado con OOF, sin usar el test; threshold_info:
    # diagnóstico y alcance de la elección del umbral OOF; oof_scores: puntuación de cada fila
    # obtenida fuera de su fold de ajuste.
    threshold, threshold_info, oof_scores = None,None,None
    if task in CLASSIFICATION[1:]:
        # threshold: punto operativo binario seleccionado con OOF, sin usar el test; threshold_info:
        # diagnóstico y alcance de la elección del umbral OOF; oof_scores: puntuación de cada fila
        # obtenida fuera de su fold de ajuste.
        threshold,threshold_info,oof_scores = tune_threshold(winner,x,y,splits)
    # curves: tres puntos de aprendizaje con tamaños efectivos y dispersión entre folds.
    curves = learning_curve(task,winner,x,y,splits,seed)
    # ablation: comparación condicional de atributos manteniendo folds e hiperparámetros.
    ablation = None
    if task == 'ratio':
        # without: clon para la ablación que excluye views/comment_count del objetivo ratio.
        without = clone(winner).set_params(features=transformer('ratio',ratio_counters=False))
        without.set_params(**{k:v for k,v in selected_params.items() if k.startswith('features__')})
        # Mismos folds y receta elegida; experimento diagnóstico, no nueva consulta al test.
        # scores: puntuaciones alineadas con filas o folds del bloque.
        scores = []
        for a,b in splits:
            # fitted: clon ajustado en el fold de la ablación, sin contadores de ratio.
            fitted = clone(without).fit(features(x.iloc[a]),y.iloc[a])
            scores.append(float(mean_absolute_error(y.iloc[b],fitted.predict(features(x.iloc[b])))))
        # ablation: comparación condicional de atributos manteniendo folds e hiperparámetros.
        ablation = {'without_views_comment_count_mae':float(np.mean(scores)), 'without_fold_mae':scores,
                    'with_counters_mae':-best[0], 'limitation':'Hiperparámetros seleccionados con contadores; ablación condicional, no búsqueda anidada'}
    # model: instancia de la tarea que se ajusta, carga o comprueba.
    model = fit_selected(task, train, yfinal, {'model':best[1], 'params':selected_params, 'threshold':None})
    # raw_prediction: predicciones antes de aplicar el umbral seleccionado en OOF.
    raw_prediction = model.predict(holdout)
    model.decision_threshold_ = threshold
    # pred: predicciones alineadas con las etiquetas de evaluación; fitted_pred: predicciones de
    # entrenamiento para diagnosticar sobreajuste.
    pred, fitted_pred = model.predict(holdout),model.predict(train)
    # baseline: referencia simple ajustada sobre el mismo entrenamiento.
    baseline = estimator_from_name('baseline_classifier' if classification else 'baseline_regressor')
    baseline.fit(np.zeros((len(train),1)),yfinal)
    # baseline_pred: predicciones de referencia sobre las mismas filas de prueba.
    baseline_pred = baseline.predict(np.zeros((len(holdout),1)))
    # result: métricas del ganador, referencia, incertidumbre y diagnósticos de esta tarea.
    result = {'selected':best[1],'trials':trials,'refinement':refinement,'train_rows':len(train),'test_rows':len(holdout),'search_rows':len(x),
        'train':evaluate(yfinal,yfinal,fitted_pred,classification), 'holdout':evaluate(yfinal,ytest,pred,classification),
        'baseline_holdout':evaluate(yfinal,ytest,baseline_pred,classification),
        'confidence_interval':ci_by_video(ytest,pred,holdout.video_id,classification,seed,sorted(set(yfinal)|set(ytest))),
        'learning_curve':curves,'unseen_test_labels':[str(v) for v in set(ytest)-set(yfinal)] if classification else [],
        'threshold':threshold,'threshold_selection':threshold_info,'ratio_ablation':ablation}
    # errors: filas evaluadas con valor real y predicción para inspeccionar fallos.
    errors = holdout[['video_id','region','title','category']].copy()
    # errors['actual']: objetivo observado, conservado para revisar errores; errors['predicted']:
    # predicción alineada con la fila evaluada.
    errors['actual'],errors['predicted'] = np.asarray(ytest),pred
    if not classification:
        # errors['absolute_error']: diferencia absoluta entre objetivo real y predicción.
        errors['absolute_error'] = np.abs(np.asarray(ytest)-pred)
        errors.sort_values('absolute_error',ascending=False).head(100).to_csv(report_dir/f'{task}_largest_errors.csv',index=False)
    else:
        errors.loc[errors.actual != errors.predicted].head(100).to_csv(report_dir/f'{task}_error_examples.csv',index=False)
    # region_results: métricas por región con sus soportes.
    region_results = []
    for region, indices in holdout.groupby('region').groups.items():
        local = evaluate(yfinal,ytest.loc[indices],np.asarray(pred)[indices],classification)
        region_results.append({'region':region,'rows':len(indices), **{k:v for k,v in local.items() if k in ['macro_f1','balanced_accuracy','mae','rmse','r2']}})
    pd.DataFrame(region_results).to_csv(report_dir/f'{task}_by_region.csv',index=False)
    if threshold_info:
        test_scores = positive_scores(model.pipeline_,features(holdout))
        # r: informe o valores de recall según el contexto de lectura.
        p,r,t = precision_recall_curve(ytest,test_scores)
        pd.DataFrame({'precision':p,'recall':r,'threshold':np.r_[t,np.nan]}).to_csv(report_dir/f'{task}_precision_recall.csv',index=False)
        result['default_threshold_holdout'] = evaluate(yfinal,ytest,raw_prediction,True)
        result['average_precision'] = float(average_precision_score(ytest,test_scores))
        result['positive_prevalence'] = float(np.mean(ytest))
        result['positive_precision'] = float(precision_score(ytest,pred,zero_division=0))
        result['positive_recall'] = float(recall_score(ytest,pred,zero_division=0))
        result['positive_f2'] = float(fbeta_score(ytest,pred,beta=2,zero_division=0))
    model.training_manifest_ = dict(training_manifest,task=task,selected=best[1],params=selected_params)
    model.save(base/conf['output'][task])
    # Coeficientes asociados a nombres: asociaciones condicionadas, no efectos causales.
    # estimator: estimador concreto de la familia, distinto del preprocesador.
    estimator = model.pipeline_.named_steps['estimator']
    # estimator: estimador concreto de la familia, distinto del preprocesador.
    estimator = getattr(estimator,'regressor_',estimator)
    coefficients_path = report_dir / f'{task}_coefficients.csv'
    if hasattr(estimator,'coef_'):
        # coefs: coeficientes del estimador expresados en dos dimensiones.
        coefs = np.atleast_2d(estimator.coef_)
        # names: nombres de atributos transformados o recursos que se empaquetan.
        if 'reduce' in model.pipeline_.named_steps:
            names = [f'svd__component_{index + 1}' for index in range(coefs.shape[1])]
        else:
            names = model.pipeline_.named_steps['features'].get_feature_names_out()
        # labels: orden de etiquetas que fija matrices y medias macro.
        labels = list(getattr(estimator,'classes_',[]))
        # rows: registros tabulares acumulados para exportar el diagnóstico.
        rows = []
        for k, values in enumerate(coefs):
            for idx in np.argsort(np.abs(values))[-15:][::-1]:
                rows.append({'class':str(labels[k] if len(labels)==len(coefs) else labels[-1] if labels else task),'feature':names[idx],'coefficient':float(values[idx])})
        pd.DataFrame(rows).to_csv(coefficients_path, index=False)
    else:
        coefficients_path.unlink(missing_ok=True)
    return result, {'model': best[1], 'params': selected_params, 'threshold': threshold}


def fingerprint(path, conf):
    """Vincula datos, configuración experimental, código activo y versiones.

    Las secciones model:* son salidas de selección, no entradas de búsqueda.
    El entrenador lee esas secciones; la caché contrasta además sus recetas.
    """
    base = path.parent
    settings = {section: dict(conf[section]) for section in conf.sections() if not section.startswith('model:')}
    digest = hashlib.sha256(json.dumps(settings, sort_keys=True).encode())
    with (base/conf['data']['input']).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    for source in ['model_selection.py','train_models.py','refinement.py','cluster_diagnostics.py','unsupervised_analysis.py','recommendation_analysis.py','convergence_review.py','clustering_experiments.py','training_data.py','metric_utils.py']:
        digest.update((base/source).read_bytes())
    for source in sorted((base/'models').glob('*.py')):
        digest.update(source.read_bytes())
    digest.update(json.dumps({'python':platform.python_version(),'sklearn':sklearn.__version__,'numpy':np.__version__,'pandas':pd.__version__,'xgboost':version('xgboost')},sort_keys=True).encode())
    return digest.hexdigest()


def run(config_path, resume=False, task=None, finish=True):
    """Ejecuta búsqueda agrupada, ajuste ampliado y evaluación de las siete tareas.

    config_path fija datos, salidas y presupuesto; resume sólo acepta una ejecución
    completa con huellas idénticas en informe y siete modelos. Devuelve el informe
    y escribe pesos, recetas y diagnósticos. La prueba no selecciona hiperparámetros.
    Una caché incompatible produce ValueError; una configuración ausente,
    FileNotFoundError. Sin resume repite los ajustes y reemplaza resultados."""
    # path: ruta del recurso que se lee, guarda o verifica en este bloque.
    path = Path(config_path).resolve()
    # conf: configuración leída de train.conf, con rutas relativas a su ubicación.
    conf = configparser.ConfigParser()
    if not conf.read(path,encoding='utf-8'):
        raise FileNotFoundError(path)
    # base: raíz resuelta a partir de la configuración o evidencia histórica; seed: semilla que fija muestreo
    # y particiones reproducibles.
    base, seed = path.parent, conf.getint('training','seed',fallback=42)
    # report_dir: directorio de evidencias del experimento.
    report_dir = base/conf['output']['reports']
    report_dir.mkdir(parents=True,exist_ok=True)
    # data_path: CSV procesado cuya huella identifica los datos del experimento.
    data_path = base/conf['data']['input']
    signature = fingerprint(path, conf)
    if resume and (report_dir/'evaluation.json').exists():
        # existing: informe previo que sólo se reutiliza si coinciden todas las huellas.
        existing = json.loads((report_dir/'evaluation.json').read_text(encoding='utf-8'))
        if all(f'model:{name}' in conf for name in REGISTRY):
            # train_models: ajusta y guarda los modelos ganadores a partir de recetas congeladas.
            # Se importa: read_specifications.
            from train_models import read_specifications
            configured = read_specifications(conf)
            stored = json.loads((base/'models/recipes.json').read_text(encoding='utf-8'))
            if not (all(configured[name]['model'] == stored[name]['model'] and configured[name]['params'] == stored[name]['params'] and configured[name].get('threshold') == stored[name].get('threshold') for name in REGISTRY)):
                raise ValueError('La configuración final no coincide con las recetas')
        if existing.get('fingerprint') == signature and existing.get('complete') and all((base/conf['output'][t]).is_file() for t in [*TASKS,'cluster','recommend']):
            for task, cls in REGISTRY.items():
                obj = cls().load(base/conf['output'][task])
                if obj.manifest_['training'].get('fingerprint') != signature:
                    raise ValueError('Los artefactos no corresponden al informe; ejecutar sin --resume')
            print('Evaluación y siete artefactos coinciden con datos, configuración y código.',flush=True)
            return existing
        raise ValueError('La caché no coincide o está incompleta; ejecutar sin --resume')
    # full: corpus completo antes de limitar las muestras de modelado.
    full = pd.read_csv(data_path)
    # full['ratio']: objetivo likes/dislikes; denominadores cero y valores no finitos quedan ausentes.
    full['ratio'] = full.likes/full.dislikes.replace(0,np.nan)
    # full['ratio']: objetivo likes/dislikes; denominadores cero y valores no finitos quedan ausentes.
    full['ratio'] = full.ratio.replace([np.inf,-np.inf],np.nan)
    train_pool, test_pool = split_video_pools(full, seed)
    # search: muestra pequeña empleada para seleccionar familias e hiperparámetros.
    search = group_sample(train_pool,conf.getint('training','max_rows'),seed).reset_index(drop=True)
    # final: muestra ampliada de entrenamiento que contiene la muestra de búsqueda.
    final = group_sample(train_pool,conf.getint('training','final_rows'),seed).reset_index(drop=True)
    # test: muestra de evaluación reservada y ajena a la selección del bloque.
    test = group_sample(test_pool,conf.getint('training','test_rows'),seed+1).reset_index(drop=True)
    if not (not set(final.video_id) & set(test.video_id)):
        raise ValueError('Comprobación incumplida en model_selection.py: not set(final.video_id) & set(test.video_id)')
    if not (set(search.video_id) <= set(final.video_id)):
        raise ValueError('Comprobación incumplida en model_selection.py: set(search.video_id) <= set(final.video_id)')
    representativeness(full,{'search':search,'final_train':final,'holdout':test}).to_csv(report_dir/'representativeness.csv',index=False)
    # report: informe del experimento con métricas, protocolo y procedencia.
    report = {'fingerprint':signature,'seed':seed,'sampling_rows':len(search),'full_rows':len(full), 'final_train_rows':len(final),'holdout_rows':len(test),
        'protocol':'Holdout por video_id antes de muestrear; muestreo por categoría y grupos; CV sólo en búsqueda; ajuste final ampliado sin holdout',
        'scope':'Estimación para otros vídeos de tendencias de estas regiones/época; no validación temporal o por canal',
        'prior_review':'Diseño revisado tras una evaluación exploratoria anterior. Este holdout interno no sustituye una validación externa prospectiva.',
        'environment':{'python':platform.python_version(),'sklearn':sklearn.__version__,'numpy':np.__version__,'pandas':pd.__version__,'xgboost':version('xgboost')},
        'ratio_definition':'likes/dislikes; denominador cero excluido','tasks':{}}
    # recipes: recetas persistidas para que el constructor sin argumentos reproduzca la selección.
    recipes = {}
    previous_path = report_dir/'evaluation.json'
    recipe_path = report_dir/'selection_recipes.json'
    if task is not None and task != 'category' and previous_path.exists():
        previous = json.loads(previous_path.read_text(encoding='utf-8'))
        if previous.get('fingerprint') != signature:
            raise ValueError('Completar la fase anterior con las mismas fuentes y datos')
        report['tasks'] = previous['tasks']
        recipes = json.loads(recipe_path.read_text(encoding='utf-8'))['recipes']
    # training_manifest: procedencia común de datos, código, semilla y tamaños del ajuste.
    training_manifest = {'fingerprint':signature,'seed':seed,'search_rows':len(search),'final_rows':len(final),'test_rows':len(test)}
    # Sólo las muestras son necesarias a partir de aquí: liberar el corpus/pools completos.
    del full, train_pool, test_pool
    gc.collect()
    with tempfile.TemporaryDirectory(prefix='youtube_cv_',dir=base) as tmp:
        # cache: caché temporal de transformaciones de los folds.
        cache = Memory(tmp,verbose=0)
        for task in (TASKS if task is None else [] if task == '__finish__' else [task]):
            # Cada tarea produce una receta y sus diagnósticos; run sólo coordina y consolida.
            result, selected_recipe = _fit_supervised_task(
                task, search, final, test, conf, base, seed, report_dir, training_manifest, cache)
            report['tasks'][task] = result
            recipes[task] = selected_recipe
            dump(report_dir/'evaluation.json', report)
            dump(recipe_path, {'fingerprint':signature, 'recipes':recipes})
    if not finish:
        return report
    if not (set(report['tasks']) == set(TASKS)):
        raise ValueError('Completar las cinco tareas antes de finalizar')
    unsupervised, unsupervised_recipes = run_unsupervised(final,base,conf,seed,training_manifest)
    report.update(unsupervised)
    recipes.update(unsupervised_recipes)
    report['complete'] = True
    dump(base/'models/recipes.json',recipes)
    dump(report_dir/'evaluation.json',report)
    dump(report_dir/'progress.json',{'status':'complete','fingerprint':signature})
    return report

# Punto de entrada: al ejecutar este archivo directamente, __name__ vale "__main__".
# El bloque siguiente inicia su ejecución; al importarlo desde otro archivo, no se ejecuta.
if __name__ == '__main__':
    # parser: contrato de argumentos de la línea de comandos.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    parser.add_argument('--resume',action='store_true',help='Sólo reutiliza una ejecución íntegra con huella idéntica')
    # args: argumentos ya validados por argparse.
    args = parser.parse_args()
    run(args.config,args.resume)

