"""Experimentos separados, calibración y congelación de los ganadores."""
# pathlib: maneja rutas de archivos y carpetas mediante objetos, como Path.
# Se importa: Path.
from pathlib import Path
# configparser: lee y escribe las secciones de configuración de train.conf.
import configparser
# json: lee y escribe datos estructurados, configuraciones y evidencias en JSON.
import json
# os: accede a variables de entorno y servicios del sistema operativo.
import os
# model_selection: compara modelos y conserva métricas, particiones y huellas del estudio.
# Se importa: run como select, TASKS, fingerprint.
from model_selection import run as select, TASKS, fingerprint
# calibrate_thresholds: calibra umbrales binarios con predicciones OOF de entrenamiento.
# Se importa: run como calibrate.
from calibrate_thresholds import run as calibrate
# train_models: ajusta y guarda los modelos ganadores a partir de recetas congeladas.
# Se importa: run como train_final.
from train_models import run as train_final


def freeze_config(path):
    """Escribe las recetas elegidas en model:* sin alterar entradas de búsqueda."""
    conf=configparser.ConfigParser()
    conf.read(path,encoding='utf-8')
    recipes=json.loads((path.parent/'models/recipes.json').read_text(encoding='utf-8'))
    for task,spec in recipes.items():
        conf[f'model:{task}']={'family':spec['model'],'params':json.dumps(spec['params']),'threshold':json.dumps(spec.get('threshold'))}
        if 'k' in spec: conf[f'model:{task}']['k']=str(spec['k'])
    with path.open('w',encoding='utf-8') as stream: conf.write(stream)


class Experiment:
    """Coordina tareas explícitas y conserva resultados antes de continuar."""

    def __init__(self,config_path,resume=False):
        """Fija configuración y modo; resume exige un estudio completo compatible."""
        self.path=Path(config_path).resolve()
        self.resume=resume
        self.report=select(self.path,resume=True) if resume else None

    def compare(self,task):
        """Compara familias y refina task sin elegir con prueba.

        ML_RESUME_INTERRUPTED=1 recupera tareas terminadas tras comprobar sus
        huellas y cargar los pesos; el modo normal vuelve a ejecutar la tarea.
        """
        if task not in TASKS: raise ValueError(task)
        if self.resume: return self.report['tasks'][task]
        if os.environ.get('ML_RESUME_INTERRUPTED')=='1':
            # models: expone las siete clases finales y el registro que relaciona tareas con clases.
            # Se importa: REGISTRY.
            from models import REGISTRY
            conf=configparser.ConfigParser()
            conf.read(self.path,encoding='utf-8')
            previous_path=self.path.parent/'reports/evaluation.json'
            if previous_path.exists():
                previous=json.loads(previous_path.read_text(encoding='utf-8'))
                if previous.get('fingerprint')==fingerprint(self.path,conf) and task in previous.get('tasks',{}):
                    model=REGISTRY[task]().load(self.path.parent/conf['output'][task])
                    if not (model.manifest_['training']['fingerprint']==previous['fingerprint']):
                        raise ValueError("Comprobación incumplida en experiment.py: model.manifest_['training']['fingerprint']==previous['fingerprint']")
                    self.report=previous
                    print('Tarea recuperada con huella verificada:',task,flush=True)
                    return previous['tasks'][task]
        self.report=select(self.path,task=task,finish=False)
        return self.report['tasks'][task]

    def finish(self):
        """Ejecuta no supervisados, calibra y ajusta los ganadores congelados."""
        if self.resume: return calibrate(self.path,self.report,resume=True)
        self.report=select(self.path,task='__finish__')
        self.report=calibrate(self.path,self.report,resume=False)
        freeze_config(self.path)
        self.report['final_training']=train_final(self.path)
        (self.path.parent/'reports/evaluation.json').write_text(json.dumps(self.report,indent=2,ensure_ascii=False),encoding='utf-8')
        return self.report


def run(config_path,resume=False):
    """Ejecuta el mismo flujo por pasos que muestra el cuaderno."""
    study=Experiment(config_path,resume)
    for task in TASKS: study.compare(task)
    return study.finish()
