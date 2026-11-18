"""Ejecuta todas las celdas con un kernel nuevo del mismo Python que este proceso."""
# pathlib: maneja rutas de archivos y carpetas mediante objetos, como Path.
# Se importa: Path.
from pathlib import Path
# datetime: representa fechas, horas y zonas horarias para registrar la ejecución.
# Se importa: datetime, timezone.
from datetime import datetime, timezone, timedelta
# hashlib: calcula huellas digitales para comprobar integridad y procedencia.
import hashlib
# math: comprueba que umbrales y duraciones sean números finitos.
import math
# re: exige el formato hexadecimal de las huellas SHA-256.
import re
# json: lee y escribe datos estructurados, configuraciones y evidencias en JSON.
import json
# os: accede a variables de entorno y servicios del sistema operativo.
import os
# sys: consulta el intérprete activo y sus flujos de entrada y salida.
import sys
# time: mide duraciones y consulta referencias de tiempo del proceso.
import time
# tempfile: prepara archivos completos en el mismo volumen antes de publicarlos.
import tempfile
# nbformat: lee, escribe y valida la estructura de cuadernos Jupyter.
import nbformat
# nbclient: ejecuta las celdas del cuaderno mediante un kernel de Jupyter.
# Se importa: NotebookClient.
from nbclient import NotebookClient


def save_full_training_evidence(root, notebook_path, evidence):
    """Valida antes de archivar y publica instantáneas en una carpeta exclusiva.

    Lee cada entrada una vez y comprueba completitud, huellas y modo. Prepara
    copias y manifiesto local antes de publicar la carpeta; actualiza el índice
    fijo atómicamente. Un fallo del índice conserva la carpeta recuperable.
    No constituye una transacción frente a cortes eléctricos o concurrencia.
    """
    # shutil: retira únicamente la carpeta temporal de una preparación fallida.
    import shutil
    # evaluation: ruta de la evaluación que se archivará junto al cuaderno.
    evaluation = root/'reports/evaluation.json'
    try:
        if evidence.get('status') != 'passed' or evidence.get('training_mode') != 'full_from_scratch':
            raise ValueError('Sólo se archiva una ejecución completa nueva y superada')
        # notebook_bytes y evaluation_bytes: entradas exactas, leídas una sola vez.
        notebook_bytes = notebook_path.read_bytes()
        evaluation_bytes = evaluation.read_bytes()
        base = json.loads(evaluation_bytes.decode('utf-8'))
        if not isinstance(base, dict) or base.get('complete') is not True:
            raise ValueError('La evaluación no declara complete=true')
        calibration = base.get('calibration')
        if not isinstance(calibration, dict) or calibration.get('complete') is not True:
            raise ValueError('La calibración no declara complete=true')
        for value in (base.get('fingerprint'), calibration.get('fingerprint')):
            if not isinstance(value, str) or re.fullmatch(r'[0-9a-f]{64}', value) is None:
                raise ValueError('La evaluación debe incluir huellas SHA-256 válidas')
        if calibration['fingerprint'] != evidence.get('calibration_fingerprint'):
            raise ValueError('La calibración no coincide con la evidencia de ejecución')
        notebook_hash = hashlib.sha256(notebook_bytes).hexdigest()
        if notebook_hash != evidence.get('notebook_sha256'):
            raise ValueError('El cuaderno cambió después de registrar su huella')
    except (OSError, ValueError) as exc:
        raise ValueError(f'No se puede archivar evidencia completa de {evaluation}: {exc}') from exc
    # directory: destino histórico; no se crea antes de validar las entradas.
    directory = root/'historial_tecnico/entrenamientos'
    directory.mkdir(parents=True, exist_ok=True)
    # stage: carpeta exclusiva; fecha con microsegundos y sufijo aleatorio sin colisiones.
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')
    stage = Path(tempfile.mkdtemp(prefix='.pending_'+stamp+'_', dir=directory))
    final = stage.with_name(stage.name.removeprefix('.pending_'))
    proof = dict(evidence, notebook_snapshot=(final/'notebook.ipynb').relative_to(root).as_posix(),
                 base_fingerprint=base['fingerprint'],
                 base_evaluation_snapshot=(final/'evaluation.json').relative_to(root).as_posix(),
                 base_evaluation_sha256=hashlib.sha256(evaluation_bytes).hexdigest(),
                 scope='Notebook completo, búsqueda, ajuste, calibración OOF y diagnósticos nuevos')
    try:
        # Las copias aún no son un archivo histórico publicado.
        (stage/'notebook.ipynb').write_bytes(notebook_bytes)
        (stage/'evaluation.json').write_bytes(evaluation_bytes)
        (stage/'proof.json').write_text(json.dumps(proof, indent=2), encoding='utf-8')
        stage.rename(final)
        _atomic_write_text(root/'historial_tecnico/full_training_execution.json', json.dumps(proof, indent=2))
    except Exception as exc:
        if stage.exists():
            shutil.rmtree(stage)
        raise OSError(f'No se completó el archivado histórico. Si ya se publicó, se conserva en {final}: {exc}') from exc


def archive_previous_coefficients(root):
    """Copia informes previos y después los retira, restaurándolos ante fallo ordinario.

    No toca pesos. Un fallo de restauración conserva las copias y muestra su ruta.
    No es una transacción frente a corte eléctrico ni escritores concurrentes.
    """
    # shutil: conserva copias completas antes de retirar informes originales.
    import shutil
    # previous: informes regulares, ordenados para un recorrido reproducible.
    previous = sorted(path for path in (root/'reports').glob('*_coefficients.csv') if path.is_file())
    if not previous:
        return
    # destination: carpeta exclusiva que guarda copias recuperables del lote.
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')
    destination = root/'historial_tecnico'/('coeficientes_previos_'+stamp)
    destination.mkdir(parents=True)
    # removed: originales retirados que habría que restaurar si falla el lote.
    removed = []
    try:
        for path in previous:
            shutil.copy2(path, destination/path.name)
        for path in previous:
            path.unlink()
            removed.append(path)
    except OSError as exc:
        try:
            for path in removed:
                shutil.copy2(destination/path.name, path)
        except OSError as recovery_error:
            raise OSError(f'Falló el archivado y su restauración: {recovery_error}. '
                          f'Copias de recuperación: {destination}') from exc
        raise OSError(f'No se pudo archivar el lote de coeficientes; originales conservados o '
                      f'restaurados. Copias disponibles: {destination}: {exc}') from exc


def _prepare_notebook(root):
    """Valida cuaderno e informe antes de archivar coeficientes o iniciar un kernel.

    Devuelve documento y evidencia anterior. Rechaza entradas ausentes, vacías,
    mal formadas o de esquema inválido con ruta y causa. Registra fallos iniciales
    por separado, sin sobrescribir el informe de la última ejecución del cuaderno.
    """
    # path y previous_path: entradas de arranque, aún sin modificar.
    path = root/'exploratory_analysis.ipynb'
    previous_path = root/'reports/notebook_execution.json'
    try:
        try:
            notebook = nbformat.read(path, as_version=4)
            nbformat.validate(notebook)
        except (OSError, ValueError, nbformat.ValidationError) as exc:
            raise ValueError(f'No se puede cargar un cuaderno válido desde {path}: {exc}') from exc
        if not any(cell.cell_type == 'code' for cell in notebook.cells):
            raise ValueError(f'El cuaderno no contiene celdas de código: {path}')
        try:
            previous = json.loads(previous_path.read_text(encoding='utf-8')) if previous_path.exists() else {}
        except (OSError, ValueError) as exc:
            raise ValueError(f'No se puede leer el informe anterior {previous_path}: {exc}') from exc
        if not isinstance(previous, dict):
            raise ValueError(f'El informe anterior debe ser un objeto JSON: {previous_path}')
        if previous.get('full_training_reference') is not None and not isinstance(previous['full_training_reference'], dict):
            raise ValueError(f'full_training_reference debe ser un objeto JSON: {previous_path}')
        if (os.environ.get('ML_REUSE_TRAINING') != '1' and
                os.environ.get('ML_RESUME_INTERRUPTED') != '1'):
            archive_previous_coefficients(root)
    except (OSError, ValueError) as exc:
        # record: fallo previo al kernel; no pretende ser evidencia de celdas ejecutadas.
        record = {'status':'failed', 'phase':'startup', 'kernel_started':False,
                  'error_type':type(exc).__name__, 'error':str(exc),
                  'executed_utc':datetime.now(timezone.utc).isoformat()}
        try:
            directory = root/'historial_tecnico/errores_arranque'
            directory.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')
            (directory/(stamp+'.json')).write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
        except OSError as log_error:
            print(f'No se pudo guardar el diagnóstico de arranque: {log_error}', file=sys.stderr)
        raise
    return notebook, previous


def clean_progress_outputs(notebook, root):
    """Archiva avances conocidos tras unir fragmentos stdout adyacentes.

    Respeta el orden de tablas, stderr, errores y datos científicos. No cambia
    fuentes ni contadores y no filtra mensajes por su contenido desfavorable.
    Los registros detallados quedan fuera del ZIP académico.
    """
    prefixes = ('Comparando ', 'Refinamiento ', 'TAREA ', 'Clustering ',
                'Ajuste final ', 'Validación por canal:')
    removed = []
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type != 'code':
            continue
        outputs, pending = [], []

        def flush_stdout():
            """Filtra un bloque stdout completo sin cruzar otras clases de salida."""
            if not pending:
                return
            text = ''.join(item.get('text', '') for item in pending)
            retained = []
            for line in text.splitlines(keepends=True):
                if line.startswith(prefixes):
                    removed.append({'cell': index, 'text': line.rstrip()})
                else:
                    retained.append(line)
            if retained:
                output = pending[0].copy()
                output['text'] = ''.join(retained)
                outputs.append(output)
            pending.clear()

        for output in cell.get('outputs', []):
            if output.get('output_type') == 'stream' and output.get('name') == 'stdout':
                pending.append(output)
            else:
                flush_stdout()
                outputs.append(output)
        flush_stdout()
        cell['outputs'] = outputs
    record = {'removed_progress_lines': len(removed), 'scientific_displays_preserved': True,
              'technical_log_in_delivery': False,
              'scope': 'Se archiva únicamente progreso conocido; el registro externo no es necesario para ejecutar la entrega'}
    if removed:
        directory = root / 'historial_tecnico' / 'salidas_progreso'
        directory.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')
        log = directory / (stamp + '.json')
        log.write_text(json.dumps(removed, indent=2, ensure_ascii=False), encoding='utf-8')
        record['technical_log'] = log.relative_to(root).as_posix()
    return record



def _atomic_write_text(path, content):
    """Publica texto UTF-8 mediante reemplazo; conserva el archivo anterior si falla.

    El temporal reside junto al destino. No garantiza durabilidad frente a cortes
    eléctricos ni coordina escritores concurrentes. Propaga fallos con su causa.
    """
    # temporary: ruta del archivo preparado que se cerrará antes del reemplazo.
    temporary = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                         prefix='.'+path.name+'_', suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(content)
        os.replace(temporary, path)
    except OSError as exc:
        raise OSError(f'No se pudo publicar el archivo {path}: {exc}') from exc
    finally:
        if temporary is not None and temporary.exists():
            try:
                temporary.unlink()
            except OSError as cleanup_error:
                print(f'No se pudo retirar el temporal {temporary}: {cleanup_error}', file=sys.stderr)


def _calibration_evidence(root, started_utc):
    """Valida evidencia OOF y devuelve modo y huella, sin entrenar ni recalibrar.

    Un informe ausente sólo se admite si evaluation.json no declara calibración
    completa. Exige fechas con zona, finalización coherente, SHA-256 y tareas
    binarias completas. Contrasta huellas con evaluación cuando está disponible.
    La comparación temporal describe procedencia; no valida los pesos del modelo.
    Admite hasta cinco minutos de desfase futuro del reloj, de forma explícita.
    """
    # path y evaluation_path: evidencias que se leen sin sobrescribirlas.
    path = root/'reports/calibration_execution.json'
    evaluation_path = root/'reports/evaluation.json'
    try:
        # evaluation: referencia opcional para comprobar coherencia entre informes.
        evaluation = {}
        if evaluation_path.exists():
            evaluation = json.loads(evaluation_path.read_text(encoding='utf-8'))
            if not isinstance(evaluation, dict):
                raise ValueError(f'La evaluación debe ser un objeto: {evaluation_path}')
        declared = evaluation.get('calibration', {})
        if not isinstance(declared, dict):
            raise ValueError(f'calibration debe ser un objeto en {evaluation_path}')
        if not path.exists():
            if declared.get('complete') is True:
                raise ValueError('Falta el informe de una calibración declarada completa')
            return {}
        calibration = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(calibration, dict) or calibration.get('complete') is not True:
            raise ValueError('Se requiere un objeto con complete=true')
        for key in ('fingerprint', 'base_fingerprint'):
            value = calibration.get(key)
            if not isinstance(value, str) or re.fullmatch(r'[0-9a-f]{64}', value) is None:
                raise ValueError(f'{key} debe ser una huella SHA-256 hexadecimal de 64 caracteres')
        # dates: instantes con zona, normalizados a UTC antes de compararlos.
        dates = {}
        for key in ('started_utc', 'finished_utc'):
            value = calibration.get(key)
            if not isinstance(value, str):
                raise ValueError(f'Falta una fecha ISO textual en {key}')
            instant = datetime.fromisoformat(value)
            if instant.tzinfo is None or instant.utcoffset() is None:
                raise ValueError(f'{key} debe incluir zona horaria')
            dates[key] = instant.astimezone(timezone.utc)
        if dates['finished_utc'] < dates['started_utc']:
            raise ValueError('finished_utc es anterior a started_utc')
        if dates['finished_utc'] > datetime.now(timezone.utc)+timedelta(minutes=5):
            raise ValueError('La calibración tiene fechas futuras fuera de la tolerancia de cinco minutos')
        seconds = calibration.get('seconds')
        if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds < 0:
            raise ValueError('seconds debe ser una duración finita no negativa')
        tasks = calibration.get('tasks')
        if not isinstance(tasks, dict):
            raise ValueError('tasks debe ser un objeto')
        for name in ('comments_disabled', 'ratings_disabled'):
            task = tasks.get(name)
            if not isinstance(task, dict):
                raise ValueError(f'Falta la tarea de calibración {name}')
            threshold, rows = task.get('threshold'), task.get('oof_rows')
            if type(threshold) not in (int, float) or not math.isfinite(threshold):
                raise ValueError(f'Umbral no finito o ausente en {name}')
            if type(rows) is not int or rows <= 0:
                raise ValueError(f'oof_rows debe ser un entero positivo en {name}')
        if evaluation:
            if evaluation.get('fingerprint') != calibration['base_fingerprint']:
                raise ValueError(f'base_fingerprint no coincide con {evaluation_path}')
            if declared.get('complete') is not True or declared.get('fingerprint') != calibration['fingerprint']:
                raise ValueError(f'La calibración completa y su huella no coinciden con {evaluation_path}')
        if started_utc.tzinfo is None or started_utc.utcoffset() is None:
            raise ValueError('El inicio de ejecución debe incluir zona horaria')
        return {'calibration_mode': 'fresh_oof' if dates['started_utc'] >= started_utc else 'verified_cache',
                'calibration_fingerprint': calibration['fingerprint'],
                'calibration_report_validated': True}
    except (OSError, ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f'Informe de calibración inválido en {path}: {exc}') from exc


def main():
    """Ejecuta todas las celdas en un kernel nuevo y registra orden, modo y huellas.

    Conserva evidencia incluso ante fallo; propaga excepciones de ejecución."""
    # root: directorio de la práctica, independiente del directorio de invocación.
    root=Path(__file__).resolve().parent
    # path: ruta del recurso que se lee, guarda o verifica en este bloque.
    path=root/'exploratory_analysis.ipynb'
    # notebook: documento nbformat cuyas celdas se construyen, ejecutan o verifican.
    notebook, previous = _prepare_notebook(root)
    full_reference = previous.get('full_training_reference')
    if previous.get('training_mode') == 'full_from_scratch' and previous.get('status') == 'passed':
        full_reference = {key: previous[key] for key in ['executed_utc', 'seconds', 'notebook_sha256',
                          'calibration_fingerprint', 'resumed_interrupted'] if key in previous}
        full_reference['scope'] = 'Ejecución histórica desde cero anterior a esta revisión de presentación; no es una nueva ejecución del cuaderno actual'
    for cell in notebook.cells:
        if cell.cell_type=='code':
            cell.outputs=[]
            cell.execution_count=None
    # started: reloj monotónico de inicio para medir duración sin cambios de hora del sistema.
    started=time.monotonic()
    # started_utc: instante UTC utilizado para distinguir calibración nueva de reutilizada.
    started_utc=datetime.now(timezone.utc)
    # kernel_started: se confirma al recibir un evento de celda tras el arranque.
    kernel_started = False
    def progress(**kwargs):
        """Notifica el índice de la celda que NotebookClient va a ejecutar."""
        nonlocal kernel_started
        kernel_started = True
        # Guarda también al comenzar la siguiente celda, tras finalizar la anterior.
        _atomic_write_text(path, nbformat.writes(notebook))
        print('Ejecutando celda',kwargs['cell_index'],flush=True)
    def persist(**kwargs):
        """Guarda el notebook tras cada celda ejecutada para conservar evidencia parcial."""
        _atomic_write_text(path, nbformat.writes(notebook))
    # status y phase: éxito sólo tras terminar celdas y comprobar su orden.
    status, phase = 'failed', 'client_preparation'
    # failure: excepción principal; no debe quedar oculta por un fallo del informe.
    failure = None
    try:
        # client: configura un kernel nuevo con el intérprete de esta práctica.
        client = NotebookClient(notebook, timeout=None, resources={'metadata': {'path': str(root)}},
                                on_cell_start=progress, on_cell_executed=persist)
        client.km = client.create_kernel_manager()
        client.km.kernel_spec.argv = [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}']
        phase = 'execution'
        client.execute()
        phase = 'validation'
        # counts: secuencia de ejecución de todas las celdas de código.
        counts = [c.execution_count for c in notebook.cells if c.cell_type == 'code']
        if counts != list(range(1, len(counts)+1)):
            raise ValueError(f'Orden de ejecución de celdas no válido: {counts}')
        status = 'passed'
    except BaseException as exc:
        # También conserva diagnóstico al interrumpir, y siempre vuelve a propagar.
        failure = exc
        raise
    finally:
        # evidence: el fallo no debe afirmar que todas las celdas se completaron.
        evidence = {'status': status, 'phase': phase, 'kernel_started': kernel_started,
                    'fresh_kernel': kernel_started, 'all_cells': status == 'passed',
                    'python': sys.executable,
                    'training_mode': 'verified_cache' if os.environ.get('ML_REUSE_TRAINING') == '1' else 'full_from_scratch',
                    'executed_utc': datetime.now(timezone.utc).isoformat(),
                    'seconds': time.monotonic()-started,
                    'execution_counts': [c.execution_count for c in notebook.cells if c.cell_type == 'code']}
        if evidence['training_mode'] == 'full_from_scratch' and os.environ.get('ML_RESUME_INTERRUPTED') == '1':
            evidence['resumed_interrupted'] = True
        if failure is not None:
            evidence['error_type'] = type(failure).__name__
            evidence['error'] = str(failure)
        if evidence['training_mode'] == 'verified_cache' and full_reference:
            evidence['full_training_reference'] = full_reference
        # finalization_error: errores secundarios se registran sin ocultar el principal.
        finalization_error = None
        try:
            cleanup = clean_progress_outputs(notebook, root) if status == 'passed' else {}
            evidence['progress_cleanup'] = cleanup
            _atomic_write_text(path, nbformat.writes(notebook))
            evidence['notebook_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
            evidence.update(_calibration_evidence(root, started_utc))
        except Exception as exc:
            finalization_error = exc
            evidence['status'] = 'failed'
            evidence['finalization_error'] = f'{type(exc).__name__}: {exc}'
        try:
            _atomic_write_text(root/'reports/notebook_execution.json', json.dumps(evidence, indent=2))
        except Exception as exc:
            if failure is not None or finalization_error is not None:
                print(f'No se pudo guardar la evidencia: {exc}', file=sys.stderr)
            else:
                raise
        if finalization_error is not None:
            if failure is None:
                raise finalization_error
            print(f'Fallo adicional al conservar evidencia: {finalization_error}', file=sys.stderr)
        if evidence['status'] == 'passed' and evidence['training_mode'] == 'full_from_scratch':
            try:
                save_full_training_evidence(root, path, evidence)
            except Exception as exc:
                evidence['status'] = 'failed'
                evidence['phase'] = 'archiving'
                evidence['archive_error'] = f'{type(exc).__name__}: {exc}'
                try:
                    _atomic_write_text(root/'reports/notebook_execution.json', json.dumps(evidence, indent=2))
                except Exception as report_error:
                    print(f'No se pudo registrar el fallo de archivado: {report_error}', file=sys.stderr)
                raise
    print('Notebook completado desde kernel limpio',flush=True)

# Punto de entrada: al ejecutar este archivo directamente, __name__ vale "__main__".
# El bloque siguiente inicia su ejecución; al importarlo desde otro archivo, no se ejecuta.
if __name__=='__main__':
    main()
