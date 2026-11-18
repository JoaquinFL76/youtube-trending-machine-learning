"""Calibración del punto operativo OOF en el pool final, separada de la búsqueda pequeña."""
# pathlib: maneja rutas de archivos y carpetas mediante objetos, como Path.
# Se importa: Path.
from pathlib import Path
# datetime: representa fechas, horas y zonas horarias para registrar la ejecución.
# Se importa: datetime, timezone.
from datetime import datetime, timezone
# configparser: lee y escribe las secciones de configuración de train.conf.
import configparser
# gc: permite liberar objetos que ya no se necesitan mediante el recolector de memoria.
import gc
# hashlib: calcula huellas digitales para comprobar integridad y procedencia.
import hashlib
# json: lee y escribe datos estructurados, configuraciones y evidencias en JSON.
import json
# time: mide duraciones y consulta referencias de tiempo del proceso.
import time
# tempfile: crea archivos y directorios temporales para operaciones aisladas.
import tempfile
# warnings: emite, captura o filtra avisos sin confundirlos con errores.
import warnings
# numpy: realiza cálculos numéricos y operaciones con vectores y matrices.
import numpy as np
# pandas: organiza, limpia y analiza datos tabulares mediante DataFrame y Series.
import pandas as pd
# sklearn.model_selection: organiza particiones, validación cruzada y búsquedas de parámetros.
# Se importa: StratifiedGroupKFold.
from sklearn.model_selection import StratifiedGroupKFold
# sklearn.metrics: calcula métricas para evaluar predicciones, grupos y similitudes.
# Se importa: average_precision_score, precision_recall_curve, precision_score, recall_score, fbeta_score.
from sklearn.metrics import average_precision_score, precision_recall_curve, precision_score, recall_score, fbeta_score
# models: expone las siete clases finales y el registro que relaciona tareas con clases.
# Se importa: REGISTRY.
from models import REGISTRY
# models.core: comparte validación, transformaciones, factorías y persistencia de modelos.
# Se importa: features, positive_scores.
from models.core import features, positive_scores
# model_selection: compara modelos y conserva métricas, particiones y huellas del estudio.
# Se importa: tune_threshold, evaluate, ci_by_video, dump.
from model_selection import tune_threshold, evaluate, ci_by_video, dump
# training_data: comparte la partición por vídeo y el muestreo de entrenamiento y prueba.
# Se importa: group_sample, split_video_pools.
from training_data import group_sample, split_video_pools, build_group_folds

def save_model(model,path):
    """Guarda primero en un directorio temporal y reemplaza pickle y manifiesto.

    model debe estar ajustado y path debe tener un directorio padre existente.
    Devuelve None. Cada reemplazo es atómico; el par de archivos no constituye
    una transacción conjunta y se valida su coherencia al reutilizar la entrega."""
    with tempfile.TemporaryDirectory(dir=path.parent,prefix='calibration_save_') as tmp:
        # staged: archivo temporal completo antes de reemplazar el destino.
        staged=Path(tmp)/path.name
        model.save(staged)
        staged.replace(path)
        staged.with_suffix('.manifest.json').replace(path.with_suffix('.manifest.json'))

def run(config_path, report, resume=False):
    """Calibra los dos umbrales binarios usando OOF del pool ampliado de entrenamiento.

    config_path reproduce particiones; report debe corresponder a una fase base
    completa. resume reutiliza únicamente huellas coherentes con los pesos.
    Conserva el diagnóstico inicial, actualiza umbrales e informes y devuelve report.
    Rechaza fase incompleta o manifiestos incoherentes con ValueError. La revisión
    posterior a evaluación interna exige validación externa para generalizar."""
    if not report.get('complete'):
        raise ValueError('Completar primero la búsqueda, ajuste y modelos de la fase base')
    # path: ruta del recurso que se lee, guarda o verifica en este bloque.
    path=Path(config_path).resolve()
    # root: directorio de la práctica, independiente del directorio de invocación.
    root=path.parent
    # conf: configuración leída de train.conf, con rutas relativas a su ubicación.
    conf=configparser.ConfigParser()
    conf.read(path,encoding='utf-8')
    # out: destino de informes o resultado intermedio según la operación local.
    out=root/conf['output']['reports']
    # signature: SHA-256 que vincula exactamente datos, configuración, código y versiones.
    signature=hashlib.sha256(Path(__file__).read_bytes()+(root/'experiment.py').read_bytes()+report['fingerprint'].encode()).hexdigest()
    # previous: metadatos de la fase previa que se contrastan antes de reutilizarla.
    previous=report.get('calibration',{})
    # tasks: tareas binarias a las que se aplica la calibración ampliada.
    tasks=['comments_disabled','ratings_disabled']
    if resume and previous.get('fingerprint')==signature and previous.get('complete'):
        for task in tasks:
            # model: instancia de la tarea que se ajusta, carga o comprueba.
            model=REGISTRY[task]().load(root/conf['output'][task])
            if model.training_manifest_.get('calibration_fingerprint')!=signature:
                raise ValueError('La calibración del informe y los pesos no coincide')
        print('Calibración OOF ampliada verificada por huella.',flush=True)
        return report
    # started: reloj monotónico de inicio para medir duración sin cambios de hora del sistema.
    started=time.monotonic()
    # seed: semilla que fija muestreo y particiones reproducibles.
    seed=report['seed']
    # columns: esquema explícito que conserva el orden de arrays y columnas.
    columns=['video_id','category','title','tags','description','publish_time','region',*tasks]
    # full: corpus completo antes de limitar las muestras de modelado.
    full=pd.read_csv(root/conf['data']['input'],usecols=columns)
    train_pool, test_pool = split_video_pools(full, seed)
    # train: filas o índices de ajuste de la partición local.
    train=group_sample(train_pool,conf.getint('training','final_rows'),seed).reset_index(drop=True)
    # test: muestra de evaluación reservada y ajena a la selección del bloque.
    test=group_sample(test_pool,conf.getint('training','test_rows'),seed+1).reset_index(drop=True)
    if not (not set(train.video_id)&set(test.video_id)):
        raise ValueError('Comprobación incumplida en calibrate_thresholds.py: not set(train.video_id)&set(test.video_id)')
    del full, train_pool, test_pool
    gc.collect()
    # recipes: recetas persistidas para que el constructor sin argumentos reproduzca la selección.
    recipes=json.loads((root/'models/recipes.json').read_text(encoding='utf-8'))
    # phase: evidencia de la calibración OOF ampliada, separada del entrenamiento base.
    phase={'fingerprint':signature,'base_fingerprint':report['fingerprint'],'complete':False,
        'protocol':'Hiperparámetros seleccionados en búsqueda pequeña; umbral F2 OOF con tres folds agrupados en pool final; test sólo evaluación',
        'limitation':'OOF reutiliza selección previa de hiperparámetros; no es CV anidada. El rediseño es posterior a evaluación interna y necesita validación externa.',
        'started_utc':datetime.now(timezone.utc).isoformat(),'fresh_oof_training':True,'tasks':{}}
    for task in tasks:
        print('Calibración ampliada',task,flush=True)
        dump(out/'progress.json',{'task':task,'phase':'expanded_oof_threshold','status':'running'})
        # x: entradas de la tarea o fixture local, alineadas con sus etiquetas.
        x=train.dropna(subset=[task]).reset_index(drop=True)
        # holdout: filas de prueba con objetivo válido; no participan en el ajuste.
        holdout=test.dropna(subset=[task]).reset_index(drop=True)
        # y: objetivo alineado por fila con las entradas de ajuste.
        y=pd.to_numeric(x[task]).astype(int)
        # ytest: objetivo real de las filas reservadas para evaluación.
        ytest=pd.to_numeric(holdout[task]).astype(int)
        # result: resultado local devuelto o añadido al informe.
        result=report['tasks'][task]
        if not (len(x)==result['train_rows'] and len(holdout)==result['test_rows']):
            raise ValueError("Comprobación incumplida en calibrate_thresholds.py: len(x)==result['train_rows'] and len(holdout)==result['test_rows']")
        # model: instancia de la tarea que se ajusta, carga o comprueba.
        model=REGISTRY[task]().load(root/conf['output'][task])
        # cv: particionador de validación cruzada agrupada.
        splits=build_group_folds(x,y,conf.getint('training','folds'),True,seed)
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter('always')
            # threshold: punto operativo binario seleccionado con OOF, sin usar el test; info: diagnóstico de
            # la calibración OOF y sus límites de interpretación.
            threshold,info,_=tune_threshold(model.pipeline_,x,y,splits)
        info['warnings']=sorted({str(item.message) for item in recorded})
        if 'search_threshold_selection' not in result:
            result['search_threshold_selection']=result.get('threshold_selection')
            result['search_threshold']=result.get('threshold')
            result['search_threshold_holdout']=result['holdout']
            result['search_threshold_positive_f2']=result.get('positive_f2')
        info.update({'pool':'ajuste final ampliado','oof_rows':len(x),'folds':len(splits),
                     'fold_train_rows':[len(a) for a,b in splits],'fold_validation_rows':[len(b) for a,b in splits]})
        model.decision_threshold_=threshold
        # pred: predicciones alineadas con las etiquetas de evaluación.
        pred=model.predict(holdout)
        # train_pred: predicciones sobre las mismas filas usadas para ajustar.
        train_pred=model.predict(x)
        # scores: puntuaciones alineadas con filas o folds del bloque.
        scores=positive_scores(model.pipeline_,features(holdout))
        result.update({'threshold':threshold,'threshold_selection':info,
            'train':evaluate(y,y,train_pred,True),'holdout':evaluate(y,ytest,pred,True),
            'confidence_interval':ci_by_video(ytest,pred,holdout.video_id,True,seed,[0,1]),
            'positive_precision':float(precision_score(ytest,pred,zero_division=0)),
            'positive_recall':float(recall_score(ytest,pred,zero_division=0)),
            'positive_f2':float(fbeta_score(ytest,pred,beta=2,zero_division=0)),
            'average_precision':float(average_precision_score(ytest,scores)),'positive_prevalence':float(ytest.mean())})
        # p: precisión positiva de cada umbral de la curva PR de prueba; r: recall positivo de cada umbral de
        # la curva PR de prueba; t: umbrales de la curva PR; se añade NaN al punto terminal sin umbral.
        p,r,t=precision_recall_curve(ytest,scores)
        pd.DataFrame({'precision':p,'recall':r,'threshold':np.r_[t,np.nan]}).to_csv(out/f'{task}_precision_recall.csv',index=False)
        # errors: filas evaluadas con valor real y predicción para inspeccionar fallos.
        errors=holdout[['video_id','region','title','category']].copy()
        # errors['actual']: objetivo observado, conservado para revisar errores; errors['predicted']:
        # predicción alineada con la fila evaluada.
        errors['actual'],errors['predicted']=np.asarray(ytest),pred
        errors.loc[errors.actual!=errors.predicted].drop_duplicates('video_id').head(100).to_csv(out/f'{task}_error_examples.csv',index=False)
        # regional: diagnósticos regionales acumulados.
        regional=[]
        for region,indices in holdout.groupby('region').groups.items():
            m=evaluate(y,ytest.loc[indices],np.asarray(pred)[indices],True)
            regional.append({'region':region,'rows':len(indices),'macro_f1':m['macro_f1'],'balanced_accuracy':m['balanced_accuracy']})
        pd.DataFrame(regional).to_csv(out/f'{task}_by_region.csv',index=False)
        model.training_manifest_['calibration_fingerprint']=signature
        model.training_manifest_['threshold_oof_rows']=len(x)
        save_model(model,root/conf['output'][task])
        recipes[task]['threshold']=threshold
        phase['tasks'][task]={'threshold':threshold,'oof_rows':len(x),'oof_average_precision':info['oof_average_precision'],
            'oof_f2':info['oof_f2'],'fold_train_rows':info['fold_train_rows'],'fold_validation_rows':info['fold_validation_rows'],'warnings':info['warnings']}
    phase.update(complete=True,seconds=time.monotonic()-started,finished_utc=datetime.now(timezone.utc).isoformat())
    report['calibration']=phase
    report['protocol']='Holdout por video_id antes de muestrear; búsqueda y CV en muestra pequeña; calibración F2 OOF sólo en pool final de entrenamiento; prueba para evaluación interna'
    dump(root/'models/recipes.json',recipes)
    dump(out/'evaluation.json',report)
    dump(out/'calibration_execution.json',phase)
    dump(out/'progress.json',{'status':'complete','fingerprint':report['fingerprint'],'calibration_fingerprint':signature})
    return report

# Punto de entrada: al ejecutar este archivo directamente, __name__ vale "__main__".
# El bloque siguiente inicia su ejecución; al importarlo desde otro archivo, no se ejecuta.
if __name__=='__main__':
    # argparse: interpreta y valida los argumentos recibidos por la línea de comandos.
    import argparse
    # parser: contrato de argumentos de la línea de comandos.
    # args: argumentos ya validados por argparse.
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    args=parser.parse_args()
    # conf: configuración leída de train.conf, con rutas relativas a su ubicación.
    conf=configparser.ConfigParser()
    conf.read(args.config,encoding='utf-8')
    # path: ruta del recurso que se lee, guarda o verifica en este bloque.
    path=Path(args.config).resolve()
    # report: informe del experimento con métricas, protocolo y procedencia.
    report=json.loads((path.parent/conf['output']['reports']/'evaluation.json').read_text(encoding='utf-8'))
    run(path,report)
