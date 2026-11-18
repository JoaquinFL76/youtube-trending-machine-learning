"""Ajuste y serialización de los siete ganadores configurados; no busca modelos."""
# pathlib: maneja rutas de archivos y carpetas mediante objetos, como Path.
# Se importa: Path.
from pathlib import Path
# argparse: interpreta y valida los argumentos recibidos por la línea de comandos.
import argparse
# configparser: lee y escribe las secciones de configuración de train.conf.
import configparser
# json: lee y escribe datos estructurados, configuraciones y evidencias en JSON.
import json
# numpy: realiza cálculos numéricos y operaciones con vectores y matrices.
import numpy as np
# pandas: organiza, limpia y analiza datos tabulares mediante DataFrame y Series.
import pandas as pd
# models: expone las siete clases finales y el registro que relaciona tareas con clases.
# Se importa: REGISTRY.
from models import REGISTRY
# models.core: comparte validación, transformaciones, factorías y persistencia de modelos.
# Se importa: estimator_from_name, cluster_estimator.
from models.core import estimator_from_name, cluster_estimator
# warnings: emite, captura o filtra avisos sin confundirlos con errores.
import warnings


def fit_selected(task, data, target, specification):
    """Ajusta la clase pública con una receta explícita y devuelve el modelo.

    specification contiene familia, parámetros y umbral previamente seleccionados.
    El umbral se aplica sin consultar etiquetas de prueba ni recalibrarlo aquí.
    """
    if task == 'cluster':
        estimator = cluster_estimator(specification['model'], specification['k'], specification['params'])
    elif task == 'recommend':
        estimator = None
    else:
        estimator = estimator_from_name(specification['model'])
    model = REGISTRY[task](estimator=estimator, params=specification['params'])
    # El ajuste final no cambia la receta ni oculta una falta de convergencia.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        model.fit(data, target)
    model.fit_warnings_ = sorted({str(item.message) for item in caught})
    model.decision_threshold_ = specification.get('threshold')
    return model


def read_specifications(conf):
    """Lee las siete recetas de las secciones model:*; rechaza configuraciones incompletas."""
    return {task: {'model': conf[f'model:{task}']['family'],
                   'params': json.loads(conf[f'model:{task}']['params']),
                   'threshold': json.loads(conf[f'model:{task}'].get('threshold', 'null')),
                   **({'k': conf.getint(f'model:{task}', 'k')} if task == 'cluster' else {})}
            for task in REGISTRY}


def run(config_path):
    """Lee datos/rutas/recetas de config_path y entrena únicamente los ganadores.

    Conserva la división por vídeo y los presupuestos del experimento. No ejecuta
    búsquedas ni elige umbrales. Guarda pesos y un registro del ajuste final.
    """
    # model_selection: compara modelos y conserva métricas, particiones y huellas del estudio.
    # Se importa: fingerprint.
    from model_selection import fingerprint
    # training_data: comparte la partición por vídeo y el muestreo de entrenamiento y prueba.
    # Se importa: group_sample, split_video_pools.
    from training_data import group_sample, split_video_pools
    path = Path(config_path).resolve()
    root = path.parent
    conf = configparser.ConfigParser()
    if not conf.read(path, encoding='utf-8'):
        raise FileNotFoundError(f'No existe el archivo de configuración: {path}')
    specifications = read_specifications(conf)
    report_dir = root / conf['output']['reports']
    report_dir.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(root / conf['data']['input'])
    data['ratio'] = (data.likes / data.dislikes.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
    seed = conf.getint('training', 'seed')
    train, _ = split_video_pools(data, seed)
    final = group_sample(train, conf.getint('training', 'final_rows'), seed).reset_index(drop=True)
    signature = fingerprint(path, conf)
    evidence = {'status': 'passed', 'mode': 'fixed_winners_only', 'fingerprint': signature, 'models': {}}
    for task, spec in specifications.items():
        if task in ['cluster', 'recommend']:
            unique = final.drop_duplicates('video_id')
            count = conf.getint('training', 'cluster_rows' if task == 'cluster' else 'recommend_rows')
            block = unique.sample(n=min(count, len(unique)), random_state=seed).reset_index(drop=True)
            target = None
        else:
            block = final.dropna(subset=[task]).reset_index(drop=True)
            target = block[task] if task == 'category' else pd.to_numeric(block[task]).astype(int if task.endswith('_disabled') else float)
        print('Ajuste final', task, len(block), 'filas', flush=True)
        model = fit_selected(task, block, target, spec)
        model.training_manifest_ = {'fingerprint': signature, 'selected': spec['model'], 'fit_rows': len(block), 'mode': 'fixed_winners_only'}
        if task.endswith('_disabled'):
            # El umbral está congelado desde la calibración del experimento.
            report_path = report_dir / 'evaluation.json'
            if report_path.exists():
                report = json.loads(report_path.read_text(encoding='utf-8'))
                if report.get('fingerprint') == signature and report.get('calibration',{}).get('complete'):
                    model.training_manifest_['calibration_fingerprint'] = report['calibration']['fingerprint']
        model.save(root / conf['output'][task])
        evidence['models'][task] = {'rows': len(block), 'family': spec['model'], 'threshold': spec.get('threshold'), 'warnings': model.fit_warnings_}
    (report_dir / 'final_training.json').write_text(json.dumps(evidence, indent=2, ensure_ascii=False), encoding='utf-8')
    return evidence


# Punto de entrada: al ejecutar este archivo directamente, __name__ vale "__main__".
# El bloque siguiente inicia su ejecución; al importarlo desde otro archivo, no se ejecuta.
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    run(parser.parse_args().config)
