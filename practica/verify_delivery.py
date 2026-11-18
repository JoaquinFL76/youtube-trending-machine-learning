"""Comprueba el cuaderno y la inferencia de los siete modelos entregados."""
# pathlib y configparser: localizan la entrega y leen sus rutas.
from pathlib import Path
import configparser
# datetime y json: fechan y guardan el resultado de la comprobación.
from datetime import datetime, timezone
import json
# hashlib: comprueba la conservación de pesos y manifiestos.
import hashlib
# os, subprocess y sys: invocan la misma interfaz que utiliza el menú.
import os
import subprocess
import sys
# numpy y pandas: preparan consultas y contrastan sus respuestas.
import numpy as np
import pandas as pd
# nbformat: valida el formato y las salidas del cuaderno.
import nbformat
# models: relaciona tareas con las siete clases finales.
from models import REGISTRY


def main():
    """Valida ejecución, procedencia e inferencia sin guardar nuevos pesos."""
    # root y conf: carpeta del proyecto y rutas relativas de cada modelo.
    root=Path(__file__).resolve().parent
    conf=configparser.ConfigParser()
    if not conf.read(root/'train.conf',encoding='utf-8'):
        raise ValueError('Falta train.conf junto al cuaderno.')
    notebook=nbformat.read(root/'exploratory_analysis.ipynb',as_version=4)
    nbformat.validate(notebook)
    code=[cell for cell in notebook.cells if cell.cell_type=='code']
    if [cell.execution_count for cell in code]!=list(range(1,len(code)+1)):
        raise ValueError('Ejecuta el cuaderno completo en orden antes de comprobar la entrega.')
    if any(output.output_type=='error' for cell in code for output in cell.outputs):
        raise ValueError('El cuaderno contiene un error. Revisa la celda correspondiente.')
    # sample: las mismas entradas de ejemplo que se consultan desde el BAT.
    example=root/'data/example.csv'
    sample=pd.read_csv(example)
    if sample.empty:
        raise ValueError(f'El CSV de ejemplo no contiene observaciones: {example}')
    report=json.loads((root/'reports/evaluation.json').read_text(encoding='utf-8'))
    if not report.get('complete') or not report.get('calibration',{}).get('complete'):
        raise ValueError('El experimento o su calibración están incompletos.')
    # artifacts: huellas anteriores a las consultas.
    artifacts={}
    for path in (root/'artifacts').iterdir():
        if path.suffix=='.pkl' or path.name.endswith('.manifest.json'):
            with path.open('rb') as stream:
                artifacts[path]=hashlib.file_digest(stream,'sha256').hexdigest()
    summary={'status':'passed','executed_utc':datetime.now(timezone.utc).isoformat(),
             'notebook_code_cells':len(code),'notebook_errors':0,'sequential_counts':True,'models':{}}
    env=dict(os.environ,PYTHONIOENCODING='utf-8',PYTHONDONTWRITEBYTECODE='1')
    for task,cls in REGISTRY.items():
        # model: estado persistido con su esquema y procedencia comprobados por load.
        path=root/conf['output'][task]
        model=cls().load(path)
        if model.manifest_['training']['fingerprint']!=report['fingerprint']:
            raise ValueError(f'La huella de entrenamiento de {task} difiere del informe.')
        if task in ('comments_disabled','ratings_disabled'):
            if model.training_manifest_['calibration_fingerprint']!=report['calibration']['fingerprint']:
                raise ValueError(f'La huella de calibración de {task} difiere del informe.')
            if model.decision_threshold_!=report['tasks'][task]['threshold']:
                raise ValueError(f'El umbral de {task} difiere del elegido en entrenamiento.')
        batch=model.predict(sample)
        single=model.predict(sample.iloc[0])
        array=model.predict(sample.reindex(columns=model.columns_).to_numpy())
        if len(batch)!=len(sample) or len(single)!=1 or len(array)!=len(sample):
            raise ValueError(f'{task}: la respuesta tiene un número de filas incorrecto.')
        if task=='recommend':
            if batch[0]!=single[0] or batch!=array:
                raise ValueError('Las recomendaciones difieren entre fila, lote y array.')
        elif np.asarray(batch).dtype.kind in 'iuf':
            np.testing.assert_allclose(batch[:1],single)
            np.testing.assert_allclose(batch,array)
        else:
            np.testing.assert_array_equal(batch[:1],single)
            np.testing.assert_array_equal(batch,array)
        # La invocación CLI debe devolver las mismas respuestas que la clase.
        result=subprocess.run([sys.executable,str(root/'inference_model.py'),task,str(path),str(example)],
                              cwd=root,capture_output=True,text=True,encoding='utf-8',env=env)
        if result.returncode:
            raise ValueError(f'La consulta de {task} falló: {result.stderr.strip()}')
        cli=json.loads(result.stdout)
        expected=batch if task=='recommend' else np.asarray(batch).tolist()
        if cli!=expected:
            raise ValueError(f'{task}: la respuesta CLI difiere de la clase.')
        summary['models'][task]={'pickle_bytes':path.stat().st_size,
                                'single_batch_numpy_cli':'passed','manifest_task':model.manifest_['task']}
    try:
        REGISTRY['category']().load(root/conf['output']['likes'])
    except ValueError:
        summary['wrong_task_rejected']=True
    else:
        raise ValueError('La carga debe rechazar pesos de una tarea distinta.')
    for path,previous in artifacts.items():
        with path.open('rb') as stream:
            if hashlib.file_digest(stream,'sha256').hexdigest()!=previous:
                raise ValueError(f'Cambiaron pesos o manifiestos durante la consulta: {path}')
    summary['weights_unchanged']=True
    summary['expanded_calibration']=True
    summary['base_training_evidence']={'fingerprint':report['fingerprint'],
                                      'calibration_fingerprint':report['calibration']['fingerprint'],'complete':True}
    (root/'reports/verification.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
