"""Enriquecimiento regional y limpieza común, sin ajustar transformadores."""
# argparse: interpreta y valida los argumentos recibidos por la línea de comandos.
import argparse
# json: lee y escribe datos estructurados, configuraciones y evidencias en JSON.
import json
# hashlib: calcula huellas digitales para comprobar integridad y procedencia.
import hashlib
# io: permite analizar exactamente los bytes cuya huella se ha auditado.
from io import BytesIO
# os: reemplaza archivos preparados y restaura salidas ante fallos de publicación.
import os
# shutil: conserva las salidas anteriores durante la publicación.
import shutil
# tempfile: crea archivos y directorios temporales para operaciones aisladas.
import tempfile
# zipfile: lee, crea y comprueba archivos comprimidos ZIP.
import zipfile
# pathlib: maneja rutas de archivos y carpetas mediante objetos, como Path.
# Se importa: Path.
from pathlib import Path
# pandas: organiza, limpia y analiza datos tabulares mediante DataFrame y Series.
import pandas as pd

def _load_category_mapping(path):
    """Lee un JSON regional UTF-8 y devuelve su diccionario ID textual a nombre.

    path debe identificar un archivo con objeto raíz e items de tipo lista. Cada
    elemento debe tener id textual no vacío o entero, y snippet.title textual no
    vacío. Se admiten listas vacías para mantener el respaldo por consenso.
    Rechaza IDs duplicados dentro del mismo archivo, incluso con igual nombre o
    representados como 10 y "10". Los mismos IDs en regiones distintas son válidos.
    No modifica nombres ni inventa categorías. Los errores incluyen ruta y, cuando
    procede, posición del elemento. Devuelve un diccionario independiente; errores
    de estructura, codificación o sintaxis producen ValueError y los de acceso,
    FileNotFoundError u OSError con contexto. No escribe ni transforma los CSV.
    """
    # path: ruta del catálogo regional; la ubicación absoluta identifica los errores.
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'No se encuentra el JSON de categorías: {path.resolve()}')
    if not path.is_file():
        raise ValueError(f'La ruta del JSON de categorías no es un archivo: {path.resolve()}')
    try:
        # content: texto original; la decodificación es estricta para no alterar etiquetas.
        content = path.read_text(encoding='utf-8')
    except UnicodeDecodeError as exc:
        raise ValueError(f'El JSON de categorías no contiene UTF-8 válido: {path.resolve()}') from exc
    except OSError as exc:
        raise OSError(f'No se puede leer el JSON de categorías {path.resolve()}: {exc}') from exc
    try:
        # document: estructura Python interpretada, todavía pendiente de validar.
        document = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f'JSON de categorías mal formado en {path.resolve()}: '
            f'línea {exc.lineno}, columna {exc.colno}. {exc.msg}') from exc
    if not isinstance(document, dict):
        raise ValueError(f'El JSON de categorías debe tener un objeto raíz: {path.resolve()}')
    if 'items' not in document or not isinstance(document['items'], list):
        raise ValueError(f'El JSON de categorías debe contener una lista "items": {path.resolve()}')
    # mapping: correspondencias validadas; no se sobrescriben identificadores repetidos.
    mapping = {}
    for position, item in enumerate(document['items'], start=1):
        # context: archivo y posición humana, empezando en 1, para localizar un defecto.
        context = f'{path.resolve()}, elemento {position} de "items"'
        if not isinstance(item, dict):
            raise ValueError(f'Cada categoría debe ser un objeto: {context}')
        if 'id' not in item or type(item['id']) not in (str, int) or not str(item['id']).strip():
            raise ValueError(f'Falta un "id" válido: se requiere texto no vacío o entero: {context}')
        # identifier: el mismo tipo textual que utiliza el cruce original con category_id.
        identifier = str(item['id'])
        if identifier in mapping:
            raise ValueError(f'ID de categoría duplicado "{identifier}" en {context}')
        # snippet: objeto de metadatos; se valida antes de acceder a title.
        snippet = item.get('snippet')
        if not isinstance(snippet, dict):
            raise ValueError(f'Falta un objeto "snippet" válido: {context}')
        # title: nombre literal; se conserva exactamente si es una cadena no vacía.
        title = snippet.get('title')
        if not isinstance(title, str) or not title.strip():
            raise ValueError(f'Falta un "snippet.title" de texto no vacío: {context}')
        mapping[identifier] = title
    return mapping


def _stream_sha256(stream):
    """Calcula SHA-256 desde la posición actual de un flujo binario, por bloques."""
    # digest y block: acumulador y fragmento de un MiB, sin cargar otro CSV entero.
    digest = hashlib.sha256()
    while block := stream.read(1024 * 1024):
        digest.update(block)
    return digest.hexdigest()


def load_processed_data(csv_path):
    """Lee CSV y auditoría verificados y devuelve (DataFrame, diccionario).

    Exige esquema mínimo de tabla, recuentos regionales y generales coherentes,
    tamaño y SHA-256 del CSV de salida. Conserva video_id como texto. Un informe
    antiguo sin huella debe regenerarse mediante preprocess, nunca completarse
    por suposición. No corrige entradas ni modifica archivos. Identifica las
    rutas y conserva causas de errores. No admite escritores concurrentes.
    """
    # csv_path y audit_path: par producido por preprocess en una misma carpeta.
    csv_path = Path(csv_path).resolve()
    audit_path = csv_path.with_suffix('.audit.json')
    try:
        audit = json.loads(audit_path.read_text(encoding='utf-8'))
        if not isinstance(audit, dict):
            raise ValueError('La auditoría debe ser un objeto JSON')
        regions = audit.get('regions')
        if not isinstance(regions, dict) or not regions:
            raise ValueError('regions debe contener al menos un objeto regional')
        # counts: campos usados por la auditoría visible del cuaderno.
        counts = ['output_rows', 'exact_duplicates_removed', 'missing_keys_removed',
                  'invalid_identity_rows_removed', 'nonstandard_ids_retained',
                  'videos_with_multiple_categories', 'negative_exact_timestamp_delays',
                  'negative_calendar_day_delays', 'removed_total_rows']
        for key in counts:
            if type(audit.get(key)) is not int or audit[key] < 0:
                raise ValueError(f'{key} debe ser un entero no negativo')
        if audit['output_rows'] == 0:
            raise ValueError('La auditoría declara cero filas de salida')
        for region, item in regions.items():
            if not isinstance(region, str) or not region or not isinstance(item, dict):
                raise ValueError('Cada región debe tener nombre y objeto de auditoría')
            for key in ['input_rows', 'replacement_characters_in_source', 'replacement_characters_introduced']:
                if type(item.get(key)) is not int or item[key] < 0:
                    raise ValueError(f'{region}.{key} debe ser un entero no negativo')
            if item.get('encoding') != 'utf-8':
                raise ValueError(f'Codificación no admitida en {region}')
        if sum(item['input_rows'] for item in regions.values()) != audit['output_rows'] + audit['removed_total_rows']:
            raise ValueError('No cuadran filas regionales, eliminadas y finales')
        if audit['removed_total_rows'] != sum(audit[key] for key in ['exact_duplicates_removed', 'missing_keys_removed', 'invalid_identity_rows_removed']):
            raise ValueError('No cuadran los motivos de eliminación')
        fingerprint = audit.get('output_sha256')
        if not isinstance(fingerprint, str) or len(fingerprint) != 64 or any(c not in '0123456789abcdef' for c in fingerprint):
            raise ValueError('Falta output_sha256 válido; vuelva a ejecutar el preprocesamiento')
        if type(audit.get('output_bytes')) is not int or audit['output_bytes'] <= 0:
            raise ValueError('Falta output_bytes válido; vuelva a ejecutar el preprocesamiento')
        # stream: se valida y analiza el mismo archivo abierto; sin segunda apertura.
        with csv_path.open('rb') as stream:
            size = stream.seek(0, 2)
            stream.seek(0)
            if size != audit['output_bytes'] or _stream_sha256(stream) != fingerprint:
                raise ValueError('CSV y auditoría no coinciden por tamaño o SHA-256')
            stream.seek(0)
            frame = pd.read_csv(stream, encoding='utf-8', dtype={'video_id': 'string'})
        required = {'video_id', 'region', 'category', 'publish_time', 'trending_date',
                    'views', 'likes', 'dislikes', 'comment_count'}
        if required - set(frame.columns):
            raise ValueError(f'Columnas ausentes en el CSV procesado: {sorted(required-set(frame.columns))}')
        if len(frame) != audit['output_rows']:
            raise ValueError('Las filas del CSV no coinciden con output_rows')
        if frame[['video_id', 'region', 'category']].isna().any().any():
            raise ValueError('El CSV procesado contiene identidad, región o categoría ausente')
        if not set(frame.region).issubset(regions):
            raise ValueError('El CSV contiene regiones no registradas en auditoría')
        return frame, audit
    except (OSError, ValueError, TypeError) as exc:
        raise ValueError(f'No se puede cargar el par procesado {csv_path} y {audit_path}: {exc}') from exc


def _write_outputs(result, audit, destination):
    """Prepara CSV y JSON antes de publicarlos; restaura salidas si falla un reemplazo.

    Los temporales se crean en el mismo volumen que el destino. Cada reemplazo es
    atómico a nivel de archivo, no constituye una transacción frente a cortes de
    corriente ni escritores concurrentes. Si la restauración falla, se conservan
    las copias de recuperación y se informa de su ruta. No modifica entradas.
    """
    # audit_path: informe asociado; destination contiene el CSV elegido.
    audit_path = destination.with_suffix('.audit.json')
    destination.parent.mkdir(parents=True, exist_ok=True)
    # stage: carpeta de preparación y recuperación, independiente de las entradas.
    stage = Path(tempfile.mkdtemp(prefix='.preprocess_', dir=destination.parent))
    # published y previous: destinos reemplazados y copias de su estado anterior.
    published, previous = [], {}
    try:
        result.to_csv(stage/'new.csv', index=False, encoding='utf-8')
        # La huella corresponde al CSV de salida exacto, no a los CSV regionales.
        with (stage/'new.csv').open('rb') as stream:
            audit['output_sha256'] = _stream_sha256(stream)
        audit['output_bytes'] = (stage/'new.csv').stat().st_size
        (stage/'new.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
        for index, target in enumerate((destination, audit_path)):
            previous[target] = target.exists()
            if previous[target]:
                shutil.copy2(target, stage/f'previous_{index}')
        for prepared, target in ((stage/'new.csv', destination), (stage/'new.json', audit_path)):
            os.replace(prepared, target)
            published.append(target)
    except Exception as exc:
        # Sólo se captura para restaurar; el error nunca se convierte en éxito.
        try:
            for target in reversed(published):
                index = (destination, audit_path).index(target)
                if previous[target]:
                    os.replace(stage/f'previous_{index}', target)
                else:
                    target.unlink(missing_ok=True)
        except OSError as recovery_error:
            raise OSError(f'Fallo al guardar y restaurar salidas. Copias de recuperación: {stage}. '
                          f'Restauración: {recovery_error}') from exc
        shutil.rmtree(stage)
        raise OSError(f'No se pudieron guardar CSV y auditoría en {destination.parent}: {exc}') from exc
    else:
        shutil.rmtree(stage)


def preprocess(source, destination):
    """Une CSV regionales y resuelve sus categorías con los JSON de origen.

    source acepta directorio o ZIP; destination recibe CSV y auditoría JSON.
    Prioriza el JSON regional y admite únicamente consenso inequívoco como respaldo.
    Conserva UTF-8 válido, audita sustituciones, fechas e identidades, y elimina
    duplicados exactos y claves inválidas. No aprende imputaciones ni escalas.
    Devuelve el DataFrame limpio; rechaza rutas inseguras, esquemas incompletos
    y categorías ambiguas mediante ValueError. Una entrada inexistente produce
    FileNotFoundError con la ruta consultada; propaga errores de lectura/escritura."""
    # destination: ruta elegida por quien llama; se valida frente a las entradas.
    destination = Path(destination).resolve()
    # source: entrada original o fuente que se procesa en este bloque.
    source = Path(source)
    # Comprobar existencia antes de distinguir archivo y carpeta evita errores engañosos.
    if not source.exists():
        raise FileNotFoundError(
            f'No se encuentra la entrada de datos: {source.resolve()}. '
            'Compruebe la ruta indicada; debe ser un ZIP existente o una carpeta con CSV/JSON regionales.')
    if source.is_file():
        if source.resolve() in (destination, destination.with_suffix('.audit.json')):
            raise ValueError(f'El destino sobrescribiría el ZIP de entrada: {source.resolve()}')
        if not zipfile.is_zipfile(source):
            raise ValueError(f'El archivo no es un ZIP válido: {source.resolve()}. Indique el ZIP original o una carpeta con CSV/JSON regionales.')
        # temporary_root: extracción local al proyecto, también al llamar preprocess directamente.
        temporary_root = Path(__file__).resolve().parent/'historial_tecnico/temporales'
        temporary_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='youtube_raw_', dir=temporary_root) as tmp:
            # raw: contenido original o directorio temporal antes de transformar.
            raw = Path(tmp).resolve()
            with zipfile.ZipFile(source) as archive:
                if any(not (raw/item.filename).resolve().is_relative_to(raw) for item in archive.infolist()):
                    raise ValueError('Ruta insegura dentro del ZIP')
                archive.extractall(raw)
            return preprocess(raw,destination)
    # Tras descartar archivos, la entrada debe ser una carpeta utilizable.
    if not source.is_dir():
        raise ValueError(f'La entrada no es un archivo ZIP ni una carpeta: {source.resolve()}')
    # frames: tablas regionales pendientes de concatenar.
    frames = []
    # files: archivos seleccionados para lectura o entrega.
    files = sorted(path for path in source.rglob('*videos.csv') if path.is_file())
    if not files:
        raise ValueError(f'No se encuentran CSV regionales con el patrón *videos.csv en {source.resolve()}. Compruebe el contenido del dataset.')
    # inputs: rutas originales que ninguna de las dos salidas puede sobrescribir.
    inputs = files + list(source.rglob('*_category_id.json'))
    if any(path.resolve() in (destination, destination.with_suffix('.audit.json')) for path in inputs):
        raise ValueError(f'El destino CSV o JSON sobrescribiría un archivo de entrada: {destination}')
    # category_mappings: cada archivo se lee y valida una sola vez para todo el flujo.
    category_mappings = {path: _load_category_mapping(path)
                         for path in sorted(source.rglob('*_category_id.json'))}
    # catalog: reúne los nombres regionales validados sin sobrescribir alternativas.
    catalog = {}
    for mapping in category_mappings.values():
        for identifier, title in mapping.items():
            catalog.setdefault(identifier, set()).add(title)
    # consensus: categorías con un único nombre entre JSON regionales.
    consensus = {key: next(iter(values)) for key, values in catalog.items() if len(values) == 1}
    # audit: registro de limpieza, incidencias y decisiones sin aprender transformaciones.
    audit = {'regions': {}, 'fallback_policy': 'JSON regional primero; consenso entre JSON restantes si falta; error si ambiguo'}
    for csv in files:
        # region: región de aparición en tendencias; no nacionalidad del creador.
        region = csv.stem.removesuffix('videos')
        if region in audit['regions']:
            raise ValueError(f'CSV regional duplicado para {region}: {csv.resolve()}')
        matches = list(source.rglob(region + '_category_id.json'))
        if len(matches) != 1:
            raise ValueError(f'JSON único requerido para {region}: {matches}')
        # mapping: correspondencia ID→nombre procedente del JSON regional.
        mapping = category_mappings[matches[0]]
        # Un byte inválido no justifica reinterpretar todo el documento como Latin-1.
        # raw: contenido original o directorio temporal antes de transformar.
        try:
            raw = csv.read_bytes()
        except OSError as exc:
            raise OSError(f'No se puede leer el CSV regional {region}: {csv.resolve()}: {exc}') from exc
        # existing_replacements: caracteres U+FFFD que ya estaban presentes en los bytes originales.
        existing_replacements = raw.count(b'\xef\xbf\xbd')
        # introduced_replacements: sustituciones introducidas al decodificar secuencias UTF-8 inválidas.
        introduced_replacements = raw.decode('utf-8', errors='replace').count('\ufffd') - existing_replacements
        if introduced_replacements / max(len(raw), 1) > .001:
            raise ValueError(f'Demasiados errores UTF-8 en {csv.name}; revisar codificación explícitamente')
        # encoding_audit: procedencia UTF-8 y sustituciones del archivo regional.
        encoding_audit = {'encoding':'utf-8', 'errors_policy':'replace sólo secuencias inválidas, sin reinterpretar texto válido',
                          'replacement_characters_in_source':existing_replacements,
                          'replacement_characters_introduced':introduced_replacements,
                          'source_sha256':hashlib.sha256(raw).hexdigest()}
        try:
            # df: tabla de los mismos bytes auditados, sin reabrir el archivo.
            df = pd.read_csv(BytesIO(raw), encoding='utf-8', encoding_errors='replace',
                             dtype={'category_id': 'string', 'video_id': 'string'})
        except (pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
            raise ValueError(f'CSV vacío o mal formado en región {region}: {csv.resolve()}: {exc}') from exc
        del raw
        if df.empty:
            raise ValueError(f'CSV sin observaciones en región {region}: {csv.resolve()}')
        # required: columnas mínimas exigidas al archivo regional.
        required = {'video_id','category_id','views','likes','dislikes','comment_count','publish_time','trending_date'}
        if required - set(df.columns):
            raise ValueError(f'Columnas requeridas ausentes en {region}: {sorted(required-set(df.columns))}')
        # category_ids: claves textuales; se recortan sólo espacios exteriores, sin convertir números.
        category_ids = df['category_id'].str.strip()
        # invalid_ids: filas sin identificador, incluyendo campos vacíos o sólo espacios.
        invalid_ids = category_ids.isna() | category_ids.eq('').fillna(False)
        if invalid_ids.any():
            raise ValueError(
                f'category_id ausente o vacío en región {region}, archivo {csv.resolve()}: '
                f'{int(invalid_ids.sum())} filas. Revise los identificadores del CSV.')
        df['category_id'] = category_ids
        # df['category']: nombre obtenido del JSON regional o del consenso inequívoco.
        df['category'] = df['category_id'].astype(str).map(mapping).astype('string')
        # missing: máscara de categorías sin correspondencia regional.
        missing = df['category'].isna()
        audit['regions'][region] = {'input_rows':len(df), 'fallback_ids':df.loc[missing,'category_id'].value_counts().to_dict(), **encoding_audit}
        df.loc[missing,'category'] = df.loc[missing,'category_id'].astype(str).map(consensus)
        if df['category'].isna().any():
            # unresolved: frecuencias de IDs sin nombre incluso tras el respaldo.
            unresolved = df.loc[df['category'].isna(), 'category_id'].value_counts()
            # details: muestra acotada para evitar mensajes enormes con entradas defectuosas.
            details = ', '.join(f'{identifier!r}: {int(count)} filas'
                                for identifier, count in unresolved.head(20).items())
            raise ValueError(
                f'Categorías sin correspondencia en región {region}, archivo {csv.resolve()}: '
                f'{int(unresolved.sum())} filas, {len(unresolved)} IDs. '
                f'Detalle (máximo 20): {details}. '
                'Revise el JSON regional y los IDs del CSV; no se convierten decimales ni ceros iniciales.')
        # df: tabla de observaciones del bloque; cada fila mantiene su unidad original.
        df = df.drop(columns='category_id')
        # df['region']: región de aparición conservada como atributo nominal.
        df['region'] = region
        # quality: recuentos separados por columna; no se redondean ni imputan datos.
        quality = {'numeric': {}, 'boolean': {}, 'dates': {}}
        for col in ['views', 'likes', 'dislikes', 'comment_count']:
            original = df[col]
            # float64 fija el formato del CSV, cuyos bytes forman parte de la huella de reutilización.
            numeric = pd.to_numeric(original, errors='coerce').astype('float64')
            nonfinite = numeric.isin([float('inf'), float('-inf')])
            negative = numeric.lt(0) & ~nonfinite
            numeric = numeric.mask(nonfinite | negative)
            quality['numeric'][col] = {
                'missing_in_source': int(original.isna().sum()),
                'not_numeric': int((original.notna() & pd.to_numeric(original, errors='coerce').isna()).sum()),
                'nonfinite': int(nonfinite.sum()), 'negative': int(negative.sum()),
                'fractional_retained': int((numeric.notna() & numeric.mod(1).ne(0)).sum())}
            df[col] = numeric
        for col in ['comments_disabled', 'ratings_disabled', 'video_error_or_removed']:
            if col in df:
                original = df[col]
                tokens = original.astype('string').str.strip().str.lower()
                mapped = tokens.map({'true': 1, 'false': 0, '1': 1, '0': 0, '1.0': 1, '0.0': 0})
                quality['boolean'][col] = {
                    'missing_in_source': int(original.isna().sum()),
                    'unrecognized': int((original.notna() & mapped.isna()).sum())}
                df[col] = mapped
            else:
                quality['boolean'][col] = {'column_absent': True}
        for col, date_format in [('publish_time', 'ISO8601'), ('trending_date', '%y.%d.%m')]:
            original = df[col]
            parsed = pd.to_datetime(original, format=date_format, errors='coerce', utc=True)
            quality['dates'][col] = {
                'missing_in_source': int(original.isna().sum()),
                'invalid': int((original.notna() & parsed.isna()).sum())}
            df[col] = parsed
        audit['regions'][region]['quality'] = quality
        # Normalizar antes de deduplicar permite reconocer copias con espacios exteriores.
        df['video_id'] = df['video_id'].str.strip()
        frames.append(df)
    # combined: concatenación regional previa a eliminar duplicados.
    combined = pd.concat(frames, ignore_index=True)
    # result: resultado local devuelto o añadido al informe.
    result = combined.drop_duplicates()
    audit['exact_duplicates_removed'] = len(combined)-len(result)
    # missing_keys: máscara de filas sin identidad o categoría utilizable.
    missing_keys = result.video_id.isna() | result.category.isna()
    audit['missing_keys_removed'] = int(missing_keys.sum())
    # result: resultado local devuelto o añadido al informe.
    result = result.loc[~missing_keys].copy()
    # invalid: entrada o máscara rechazada por el contrato del bloque.
    invalid = result.video_id.astype(str).str.strip().str.casefold().isin(['', '#name?', 'nan', 'none'])
    audit['invalid_identity_rows_removed'] = int(invalid.sum())
    # result: resultado local devuelto o añadido al informe.
    result = result.loc[~invalid].copy()
    # result['video_id']: identidad normalizada para agrupar todas las apariciones del vídeo.
    result['video_id'] = result.video_id.astype(str).str.strip()
    if result.empty:
        raise ValueError('No quedan filas con identidad utilizable después de la limpieza; no se escriben salidas.')
    audit['nonstandard_ids_retained'] = int((~result.video_id.str.fullmatch(r'[A-Za-z0-9_-]{11}')).sum())
    audit['identity_policy'] = 'Excluir claves ausentes y marcadores de error; conservar otros IDs no estándar y auditarlos, sin inventar identidad compartida'
    audit['videos_with_multiple_categories'] = int(result.groupby('video_id').category.nunique().gt(1).sum())
    # delay: diferencia temporal en días, conservando la hora de publicación.
    delay = (result.trending_date-result.publish_time).dt.total_seconds()/86400
    # calendar_delay: diferencia de días normalizados; separa la resolución diaria de la horaria.
    calendar_delay = (result.trending_date.dt.normalize()-result.publish_time.dt.normalize()).dt.days
    audit['negative_exact_timestamp_delays'] = int(delay.lt(0).sum())
    audit['negative_calendar_day_delays'] = int(calendar_delay.lt(0).sum())
    audit['date_policy'] = 'trending_date sólo tiene día; un retraso entre -1 y 0 días no prueba una anomalía. No se usa trending_date como predictor. Negativos de calendario se conservan y auditan: no hay zona horaria regional suficiente para corregirlos.'
    audit['output_rows'] = len(result)
    audit['removed_exact_duplicates_or_missing_keys'] = audit['exact_duplicates_removed'] + audit['missing_keys_removed']
    audit['removed_total_rows'] = sum(x['input_rows'] for x in audit['regions'].values()) - len(result)
    _write_outputs(result, audit, destination)
    return result

# Punto de entrada: al ejecutar este archivo directamente, __name__ vale "__main__".
# El bloque siguiente inicia su ejecución; al importarlo desde otro archivo, no se ejecuta.
if __name__ == '__main__':
    # parser: contrato de argumentos de la línea de comandos.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('raw_path')
    parser.add_argument('output_path')
    # args: argumentos ya validados por argparse.
    args = parser.parse_args()
    print(preprocess(args.raw_path, args.output_path).shape)

