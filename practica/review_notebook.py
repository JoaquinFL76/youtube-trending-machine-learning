"""Abre un repaso visual en JupyterLab existente con el kernel Python de la práctica."""
# argparse: interpreta opciones de preparación y apertura del repaso.
import argparse
# configparser: obtiene las rutas de datos y pesos sin cambiar train.conf.
import configparser
# datetime: identifica cuándo se creó cada copia de estudio.
from datetime import datetime, timezone
# json: guarda el kernel y la información de procedencia de la copia.
import json
# importlib.metadata y platform: consultan las versiones sin cargar ni actualizar bibliotecas.
from importlib.metadata import version
import platform
# os: prepara variables limitadas al visor y sus procesos hijos.
import os
# pathlib: maneja rutas portables relativas a la carpeta de la práctica.
from pathlib import Path
# shutil: copia archivos independientes y localiza programas existentes.
import shutil
# subprocess: comprueba intérpretes y abre Jupyter sin interpretar órdenes de shell.
import subprocess
# sys: identifica el Python 3.12 que ejecutará las celdas.
import sys
# tempfile: crea carpetas de sesión exclusivas dentro de la práctica.
import tempfile
# nbformat: lee el cuaderno y prepara sus celdas sin ejecutar código científico.
import nbformat


def find_jupyter_python():
    """Localiza un Python con JupyterLab ya instalado; nunca instala paquetes.

    ML_JUPYTER_PYTHON permite elegir una instalación concreta. Después se prueban
    el entorno activo, Anaconda/Miniconda habituales y los programas del PATH.
    El intérprete encontrado sirve el visor, no ejecuta los modelos de la práctica.
    """
    explicit = os.environ.get('ML_JUPYTER_PYTHON')
    candidates = [Path(explicit)] if explicit else [Path(sys.executable)]
    if not explicit:
        for key in ['CONDA_PREFIX', 'ANACONDA_HOME']:
            if os.environ.get(key):
                candidates.append(Path(os.environ[key])/'python.exe')
        home = Path.home()
        for parent in [home, Path(os.environ.get('ProgramData', r'C:\ProgramData')),
                       Path(os.environ.get('LOCALAPPDATA', str(home/'AppData/Local')))]:
            candidates.extend(parent/name/'python.exe' for name in ['anaconda3','miniconda3'])
        for program in ['jupyter-lab', 'conda']:
            located = shutil.which(program)
            if located:
                candidates.append(Path(located).parent.parent/'python.exe')
    seen = set()
    for candidate in candidates:
        if str(candidate).casefold() in seen or not candidate.is_file():
            continue
        seen.add(str(candidate).casefold())
        try:
            check = subprocess.run([str(candidate), '-B', '-c',
                                    "import importlib.util,sys;sys.exit(0 if importlib.util.find_spec('jupyterlab') else 1)"],
                                   capture_output=True, timeout=20)
        except (OSError, subprocess.TimeoutExpired):
            continue
        if check.returncode == 0:
            return candidate.resolve()
    if explicit:
        raise ValueError('ML_JUPYTER_PYTHON no apunta a un Python con JupyterLab disponible. Revisa esa ruta.')
    raise ValueError('No se encuentra JupyterLab. Si lo tienes en Anaconda, indica su python.exe mediante '
                     'ML_JUPYTER_PYTHON. Si aún no tienes visor, consulta «Repaso visual» en README.md para '
                     'prepararlo. También puedes abrir el cuaderno ya ejecutado y revisar sus salidas.')


def _relative_path(value):
    """Exige rutas internas para que una sesión de estudio no escriba fuera de su copia."""
    path = Path(value)
    if path.is_absolute() or '..' in path.parts:
        raise ValueError(f'El repaso requiere rutas relativas dentro de la práctica en train.conf: {value}')
    return path


def visual_diagnostic():
    """Muestra visor, kernel y versiones sin crear una sesión ni ejecutar celdas."""
    jupyter = find_jupyter_python()
    check = subprocess.run([str(jupyter),'-B','-c',
        "import json,sys;from importlib.metadata import version;print(json.dumps({'python':sys.executable,'jupyterlab':version('jupyterlab'),'jupyter_server':version('jupyter_server'),'jupyter_client':version('jupyter_client')}))"],
        capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=30)
    if check.returncode:
        raise RuntimeError('El Python configurado no puede consultar JupyterLab. Revisa ML_JUPYTER_PYTHON y la instalación existente de Anaconda.')
    server = json.loads(check.stdout)
    print('DIAGNÓSTICO DEL REPASO VISUAL')
    print('Visor Jupyter: '+str(jupyter))
    print('Versiones del visor: JupyterLab '+server['jupyterlab']+'; servidor '+server['jupyter_server']+'; cliente '+server['jupyter_client'])
    print('Kernel de cálculo: '+sys.executable)
    print('Python del kernel: '+platform.python_version())
    for name in ['numpy','pandas','scikit-learn','ipykernel','jupyter_client']:
        print(name+': '+version(name))
    print('Selecciona Python 3.12 · Práctica ML en el cuaderno. No se ha instalado ni ejecutado ningún modelo.')
    return {'server':server,'kernel_python':sys.executable,'kernel_python_version':platform.python_version()}


def prepare_review(root):
    """Crea una copia independiente y deja vacías las salidas para ejecutar paso a paso.

    Conserva exactamente el código de las celdas, los datos y los pesos. Si falta
    el CSV procesado, lo genera únicamente en la copia desde dataset.zip o data/raw.
    Cada llamada crea una sesión nueva; no sobrescribe sesiones ni la entrega.
    """
    root = Path(root).resolve()
    source = root/'exploratory_analysis.ipynb'
    if not source.is_file():
        raise FileNotFoundError('Falta exploratory_analysis.ipynb junto al BAT. Extrae la práctica completa.')
    conf = configparser.ConfigParser()
    if not conf.read(root/'train.conf', encoding='utf-8'):
        raise FileNotFoundError('Falta train.conf junto al BAT.')
    if not conf.has_option('data','input') or not conf.has_option('output','reports'):
        raise ValueError('train.conf debe contener data.input y output.reports. Restaura la configuración de la entrega.')
    data_path = _relative_path(conf['data']['input'])
    report_path = _relative_path(conf['output']['reports'])
    for key, value in conf['output'].items():
        relative = _relative_path(value)
        if key != 'reports' and not (root/relative).is_file():
            raise FileNotFoundError(f'Falta el modelo guardado para {key}: {relative}. Restaura los pesos de la entrega para revisar sus resultados.')
    notebook = nbformat.read(source, as_version=4)
    nbformat.validate(notebook)
    sessions = root/'historial_tecnico/repaso_visual'
    sessions.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_')
    study = Path(tempfile.mkdtemp(prefix=stamp, dir=sessions))
    print('Preparando copia de estudio: ' + str(study), flush=True)
    # Las copias son archivos independientes; no se crean enlaces que compartan sus escrituras.
    for item in root.glob('*.py'):
        shutil.copy2(item, study/item.name)
    for name in ['train.conf','README.md','GUIA_DE_CONSULTA.md','ENUNCIADO.txt','requirements.txt','requirements-lock.txt']:
        if (root/name).is_file():
            shutil.copy2(root/name, study/name)
    for name in ['models', 'artifacts', report_path.as_posix()]:
        if (root/name).is_dir():
            shutil.copytree(root/name, study/name, ignore=shutil.ignore_patterns('__pycache__'))
    for relative in [data_path, data_path.with_suffix('.audit.json'), Path('data/example.csv')]:
        if (root/relative).is_file():
            (study/relative).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(root/relative, study/relative)
    if not (study/data_path).is_file():
        # preprocess_data: reutiliza la preparación canónica, sin volver a entrenar.
        from preprocess_data import preprocess
        raw = root/'dataset.zip' if (root/'dataset.zip').is_file() else root/'data/raw'
        if not raw.exists():
            raise FileNotFoundError('Faltan el CSV procesado y dataset.zip. Coloca el dataset junto al BAT para preparar la copia de estudio.')
        print('Preparando los datos dentro de la copia de estudio...', flush=True)
        preprocess(raw, study/data_path)
    if not (study/data_path.with_suffix('.audit.json')).is_file():
        raise FileNotFoundError('Falta la auditoría asociada al CSV procesado. Verifica primero los datos desde la opción 1 del BAT.')
    # Sólo se cambian presentación y salidas de la copia: el código sigue siendo el original.
    count = sum(cell.cell_type == 'code' for cell in notebook.cells)
    labeled_cells = []
    position = 0
    for cell in notebook.cells:
        if cell.cell_type == 'code':
            position += 1
            label = f'**Celda de código {position} de {count}**'
            if labeled_cells and labeled_cells[-1].cell_type == 'markdown':
                labeled_cells[-1].source += '\n\n' + label
            else:
                labeled_cells.append(nbformat.v4.new_markdown_cell(label))
            cell.outputs = []
            cell.execution_count = None
            cell.metadata.pop('execution', None)
        labeled_cells.append(cell)
    notebook.cells = labeled_cells
    notebook.cells.insert(0, nbformat.v4.new_markdown_cell(
        '# Repaso visual paso a paso\n\n'
        f'Estás ejecutando una copia de `{source.name}`, con **{count} celdas de código numeradas**. '
        'Las salidas de cada celda se muestran debajo de su código al ejecutarla.\n\n'
        '1. Lee el texto y selecciona la primera celda de código, la que comienza con las importaciones.\n'
        '2. Pulsa **Ctrl + Enter** para ejecutarla y quedarte en ella. **Shift + Enter** ejecuta y selecciona la siguiente.\n'
        '3. Espera mientras aparece `[*]`. Después lee los mensajes de `print`, las tablas, los gráficos y sus comentarios.\n'
        '4. Selecciona la siguiente celda cuando hayas terminado de revisar la salida.\n\n'
        'Avanza de arriba abajo porque las celdas comparten variables. Algunas producen varios gráficos y explicaciones.\n\n'
        'Esta copia calcula las salidas del análisis y reutiliza modelos, comparaciones y umbrales tras comprobar sus huellas. '
        'Debido a que estás en una copia, puedes guardarla con las tablas, gráficos y explicaciones obtenidos, y añadir tus propios apuntes en celdas de texto (Markdown) para repasarlos después.'))
    notebook.metadata['kernelspec'] = {'name':'python3', 'display_name':'Python 3.12 · Práctica ML', 'language':'python'}
    nbformat.write(notebook, study/source.name)
    (study/'historial_tecnico').mkdir(exist_ok=True)
    (study/'historial_tecnico/sesion_visual.json').write_text(json.dumps({
        'source':str(root), 'study':str(study), 'code_cells':count,
        'python_kernel':sys.executable, 'training_mode':'verified_reuse',
        'created_utc':datetime.now(timezone.utc).isoformat()},indent=2,ensure_ascii=False),encoding='utf-8')
    return study


def review_environment(study, jupyter_python):
    """Separa el servidor Anaconda del kernel .venv y mantiene temporales y ajustes en la copia."""
    state = Path(study)/'historial_tecnico/jupyter'
    locations = {key:state/name for key,name in [('JUPYTER_DATA_DIR','data'),
                 ('JUPYTER_RUNTIME_DIR','runtime'),('JUPYTER_CONFIG_DIR','config'),
                 ('IPYTHONDIR','ipython'),('TEMP','tmp'),('TMP','tmp')]}
    for location in locations.values():
        location.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, **{key:str(value) for key,value in locations.items()},
               ML_REUSE_TRAINING='1', ML_RESUME_INTERRUPTED='', PYTHONDONTWRITEBYTECODE='1',
               PYTHONIOENCODING='utf-8', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
    env['JUPYTER_PATH'] = str(locations['JUPYTER_DATA_DIR'])
    # El kernel utiliza las bibliotecas de la práctica, aunque el visor proceda de Anaconda.
    kernel = locations['JUPYTER_DATA_DIR']/'kernels/python3'
    kernel.mkdir(parents=True, exist_ok=True)
    (kernel/'kernel.json').write_text(json.dumps({
        'argv':[sys.executable,'-B','-m','ipykernel_launcher','-f','{connection_file}'],
        'display_name':'Python 3.12 · Práctica ML', 'language':'python',
        'env':{'PATH':os.environ.get('PATH',''), 'CONDA_PREFIX':'', 'ML_REUSE_TRAINING':'1'}
    },indent=2,ensure_ascii=False),encoding='utf-8')
    prefix = Path(jupyter_python).parent
    if (prefix/'conda-meta').is_dir():
        # Equivale a las rutas necesarias del acceso directo Anaconda, sin cambiar la carpeta activa.
        paths = [prefix,prefix/'Library/mingw-w64/bin',prefix/'Library/usr/bin',prefix/'Library/bin',prefix/'Scripts']
        env['PATH'] = os.pathsep.join(map(str,paths))+os.pathsep+env.get('PATH','')
        env['CONDA_PREFIX'] = str(prefix)
    return env


def main():
    """Abre el visor existente; preparar una copia no ejecuta las celdas ni instala programas."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-only', action='store_true', help='Prepara una sesión sin abrir el navegador.')
    parser.add_argument('--no-browser', action='store_true', help='Inicia el visor sin abrir automáticamente el navegador.')
    parser.add_argument('--diagnostico', action='store_true', help='Muestra visor, kernel y versiones sin preparar una sesión.')
    args = parser.parse_args()
    try:
        if args.diagnostico:
            visual_diagnostic()
            return 0
        jupyter = find_jupyter_python()
        study = prepare_review(Path(__file__).resolve().parent)
        env = review_environment(study, jupyter)
        if args.prepare_only:
            print(json.dumps({'study':str(study),'jupyter_python':str(jupyter),'kernel_python':sys.executable},ensure_ascii=False))
            return 0
        print('Visor Jupyter: '+str(jupyter), flush=True)
        print('Python de las celdas: '+sys.executable, flush=True)
        print('Cuaderno de estudio: '+str(study/'exploratory_analysis.ipynb'), flush=True)
        print('En el navegador, Ctrl + Enter ejecuta la celda seleccionada y te deja en ella.', flush=True)
        print('Mantén esta ventana abierta. Ctrl+C cierra el visor; la copia de estudio se conserva.', flush=True)
        command = [str(jupyter),'-B','-m','jupyterlab','exploratory_analysis.ipynb',
                   '--ServerApp.root_dir='+str(study),
                   '--LabApp.default_url=/lab/tree/exploratory_analysis.ipynb',
                   '--ServerApp.ip=127.0.0.1','-y']
        if args.no_browser:
            command.append('--no-browser')
        with subprocess.Popen(command, cwd=study, env=env) as server:
            try:
                code = server.wait()
                if code:
                    print('El visor no ha terminado correctamente. Ejecuta ejecutar_practica.bat --diagnostico-visual y consulta las incidencias del README.', file=sys.stderr)
                return code
            except KeyboardInterrupt:
                try:
                    server.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    server.terminate()
                    server.wait(timeout=10)
                print('Repaso visual cerrado. La copia de estudio se conserva en '+str(study))
                return 0
    except PermissionError as exc:
        print('No se puede abrir el repaso visual: comprueba los permisos de la carpeta y del Python configurado. Detalle: '+str(exc), file=sys.stderr)
        return 2
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError) as exc:
        print('No se puede abrir el repaso visual: '+str(exc), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
