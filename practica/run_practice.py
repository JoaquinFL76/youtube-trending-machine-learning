"""Preprocesa datos reales y ejecuta el notebook completo; puede verificar y empaquetar."""
# pathlib: maneja rutas de archivos y carpetas mediante objetos, como Path.
# Se importa: Path.
from pathlib import Path
# argparse: interpreta y valida los argumentos recibidos por la línea de comandos.
import argparse
# subprocess: ejecuta otros programas o scripts y permite comprobar su resultado.
import subprocess
# sys: consulta el intérprete activo y sus flujos de entrada y salida.
import sys
# json: lee y escribe datos estructurados, configuraciones y evidencias en JSON.
import json
# os: accede a variables de entorno y servicios del sistema operativo.
import os
# tempfile: fija el directorio temporal sólo dentro del proceso actual.
import tempfile

# errno: identifica el error de disco lleno sin depender del idioma del sistema.
import errno
# importlib: comprueba que los paquetes pueden cargarse y consulta sus versiones instaladas.
from importlib import import_module, metadata
# platform y struct: identifican la implementación, versión y arquitectura de Python.
import platform
import struct
# zipfile: distingue un ZIP corrupto de otros errores de entrada.
from zipfile import BadZipFile

PYTHON_REUSE = '3.12.14'


def check_environment(root, mode):
    """Comprueba el intérprete, el entorno fijado y la carga de los paquetes principales."""
    version = platform.python_version()
    bits = struct.calcsize('P') * 8
    print(f'Python: {version} ({bits} bits) · {sys.executable}', flush=True)
    if sys.version_info[:2] != (3, 12) or bits != 64 or platform.python_implementation() != 'CPython':
        print('ERROR: se necesita CPython 3.12 de 64 bits. Consulta el paso 1 de README.md.', file=sys.stderr)
        return 2
    # expected_python: parche asociado a la evaluación presente en esta copia.
    # Si se reentrena en otro parche 3.12, se puede reutilizar el nuevo experimento con ese parche.
    expected_python = PYTHON_REUSE
    evaluation = root/'reports/evaluation.json'
    if evaluation.is_file() and mode != 'train':
        expected_python = json.loads(evaluation.read_text(encoding='utf-8')).get('environment', {}).get('python', PYTHON_REUSE)
    if version != expected_python:
        message = (f'Para verificar o repasar con los modelos de esta copia se necesita Python {expected_python}. '
                   'Selecciona ese intérprete o usa la opción 2 en una copia para reproducir con tu versión 3.12.')
        if mode in ('verify', 'visual'):
            print('ERROR: ' + message, file=sys.stderr)
            return 2
        print('Aviso: ' + message, flush=True)

    lock = root/'requirements-lock.txt'
    if not lock.is_file() or not (root/'requirements.txt').is_file():
        print('ERROR: faltan requirements.txt o requirements-lock.txt. Extrae el ZIP académico completo.', file=sys.stderr)
        return 2
    entries = [line.partition('#')[0].strip() for line in lock.read_text(encoding='utf-8-sig').splitlines()]
    if not any(entries):
        print(f'ERROR: {lock.name} está vacío. Restaura el archivo del ZIP.', file=sys.stderr)
        return 2
    missing, different = [], []
    for entry in filter(None, entries):
        package, separator, expected = entry.partition('==')
        if not separator or not package or not expected:
            print(f'ERROR: entrada inválida en {lock.name}: {entry}. Restaura el archivo del ZIP.', file=sys.stderr)
            return 2
        try:
            actual = metadata.version(package)
        except metadata.PackageNotFoundError:
            missing.append(entry)
        else:
            if actual != expected:
                different.append(f'{package} {actual} (se requiere {expected})')
    if missing or different:
        if missing:
            print('ERROR: faltan paquetes: ' + ', '.join(missing), file=sys.stderr)
        if different:
            print('ERROR: versiones distintas de las fijadas: ' + ', '.join(different), file=sys.stderr)
        print(f'En PowerShell, desde la carpeta de la práctica:\n'
              f'  & "{sys.executable}" -m pip install -r requirements.txt\n'
              f'  & "{sys.executable}" -m pip check', file=sys.stderr)
        return 2

    for module in ('numpy', 'pandas', 'scipy.linalg', 'sklearn.pipeline', 'joblib',
                   'matplotlib._path', 'nbformat', 'nbclient', 'jupyter_client', 'ipykernel.kernelapp', 'xgboost'):
        try:
            import_module(module)
        except Exception as exc:
            print(f'ERROR: {module} está instalado pero no se puede importar ({type(exc).__name__}: {exc}).\n'
                  'Recrea el entorno de la práctica con el Python indicado, reinstala requirements.txt y ejecuta pip check. '
                  'Si aparece DLL load failed, consulta el apartado de incidencias de README.md.', file=sys.stderr)
            return 2
    print('Versiones fijadas e importaciones principales comprobadas.', flush=True)
    return 0


def explain_io_error(exc):
    """Indica cómo recuperar fallos comunes de escritura en la carpeta de la práctica."""
    if getattr(exc, 'errno', None) == errno.ENOSPC or getattr(exc, 'winerror', None) == 112:
        return 'Libera espacio en la unidad donde extrajiste la práctica y vuelve a intentarlo.'
    if isinstance(exc, PermissionError):
        return 'Usa una carpeta con permisos de escritura y cierra los archivos de la práctica que estén abiertos.'
    return 'Revisa la ruta, los permisos y el espacio disponible. Consulta README.md si persiste.'

def configure_text_streams():
    """Usa UTF-8 en las salidas de run_practice.py, también si se redirigen en Windows.

    No cambia la configuración del sistema ni el entorno del proceso llamante.
    Admite flujos de prueba sin reconfigure; conserva errores de escritura reales.
    """
    for stream in [sys.stdout, sys.stderr]:
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8')


def run_python(arguments, root, capture_output=False):
    """Ejecuta Python con salidas UTF-8 coherentes con la captura de run_practice.py.

    arguments fija el script o módulo; root fija el directorio de trabajo. Copia
    el entorno sin mutarlo y devuelve CompletedProcess. El consumidor comprueba
    el código de retorno para poder guardar el registro antes de propagar un fallo.
    """
    environment = dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONDONTWRITEBYTECODE='1')
    completed = subprocess.run([sys.executable, '-B', *arguments], cwd=root, env=environment,
                               capture_output=capture_output, text=True, encoding='utf-8')
    if capture_output and not all(isinstance(value, str) for value in [completed.stdout, completed.stderr]):
        raise RuntimeError('No se pudieron capturar ambas salidas del subproceso como texto UTF-8')
    return completed


def main():
    """Orquesta limpieza, notebook y comprobaciones opcionales con el mismo Python.

    Los subprocesos fallidos detienen la ejecución antes de anunciar éxito."""
    configure_text_streams()
    # parser: contrato de argumentos de la línea de comandos.
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',type=Path,help='Dataset original ZIP; si se omite, usa data/raw')
    parser.add_argument('--verify',action='store_true')
    parser.add_argument('--package',action='store_true',help='Empaquetado de desarrollo con package_submission.py')
    parser.add_argument('--check-environment', choices=('environment', 'consult', 'visual'),
                        help='Comprueba Python y las dependencias sin ejecutar el cuaderno')
    # args: argumentos ya validados por argparse.
    args=parser.parse_args()
    # root: directorio de la práctica, independiente del directorio de invocación.
    root=Path(__file__).resolve().parent
    try:
        if args.check_environment:
            return check_environment(root, args.check_environment)
        mode = 'verify' if os.environ.get('ML_REUSE_TRAINING') == '1' else 'train'
        if check_environment(root, mode):
            return 2
        execute_practice(root, args)
    except BadZipFile as exc:
        print(f'ERROR al leer el dataset: {exc}. Vuelve a descargar el ZIP original del enunciado y guárdalo como dataset.zip.', file=sys.stderr)
        return 2
    except OSError as exc:
        print(f'ERROR de acceso: {exc}. {explain_io_error(exc)}', file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f'ERROR en los datos o la configuración: {exc}. Comprueba que has extraído toda la práctica '
              'y que usas el dataset original del enunciado.', file=sys.stderr)
        return 2
    except subprocess.CalledProcessError as exc:
        command = Path(exc.cmd[2]).name
        help_by_step = {
            'execute_notebook.py': ('Consulta reports/notebook_execution.json. Si el error menciona huellas o caché '
                                    'incompatible, restaura el ZIP completo, el dataset y el entorno documentado. '
                                    'Para reproducir con otro Python 3.12 usa la opción 2 en una copia.'),
            '-m': 'Consulta reports/unit_tests.log para localizar la prueba que falló.',
        }
        print(f'ERROR en {command} (código {exc.returncode}). '
              + help_by_step.get(command, 'Lee el mensaje anterior y consulta las incidencias de README.md.'), file=sys.stderr)
        return 2
    return 0


def execute_practice(root, args):
    """Prepara los datos y ejecuta el cuaderno y los controles que se solicitaron."""
    required = ['preprocess_data.py', 'execute_notebook.py', 'exploratory_analysis.ipynb', 'train.conf']
    if args.verify:
        required.append('verify_delivery.py')
    if args.package:
        required.append('package_submission.py')
    missing = [name for name in required if not (root/name).is_file()]
    if missing:
        raise ValueError('Faltan archivos de la entrega: ' + ', '.join(missing))

    # preprocess_data: integra las regiones después de comprobar el entorno.
    from preprocess_data import preprocess

    # temporary_root: temporales y conexiones locales, fuera de la entrega académica.
    temporary_root = root/'historial_tecnico/temporales'
    temporary_root.mkdir(parents=True, exist_ok=True)
    os.environ['TEMP'] = os.environ['TMP'] = str(temporary_root)
    os.environ['JUPYTER_RUNTIME_DIR'] = str(temporary_root/'jupyter')
    tempfile.tempdir = str(temporary_root)
    print('Preparando el dataset y la auditoría de datos…', flush=True)
    preprocess(args.archive if args.archive else root/'data/raw',root/'data/processed/videos.csv')
    print('Datos preparados. Iniciando exploratory_analysis.ipynb…', flush=True)
    run_python([str(root/'execute_notebook.py')], root).check_returncode()
    if args.verify:
        # La comprobación académica contrasta las siete interfaces.
        run_python(['verify_delivery.py'], root).check_returncode()
    if args.package:
        run_python(['package_submission.py'], root).check_returncode()

# Punto de entrada: al ejecutar este archivo directamente, __name__ vale "__main__".
# El bloque siguiente inicia su ejecución; al importarlo desde otro archivo, no se ejecuta.
if __name__=='__main__':
    raise SystemExit(main())
