"""Inferencia por CLI: tarea, pesos y archivo CSV de entrada."""
# argparse: interpreta y valida los argumentos recibidos por la línea de comandos.
import argparse
# json: lee y escribe datos estructurados, configuraciones y evidencias en JSON.
import json
# sys: separa avisos y errores de las predicciones JSON de stdout.
import sys
# pickle: identifica errores esperables al leer pesos serializados.
import pickle
# pathlib: distingue archivos y carpetas y comprueba la extensión del CSV.
from pathlib import Path
# numpy: realiza cálculos numéricos y operaciones con vectores y matrices.
import numpy as np
# pandas: organiza, limpia y analiza datos tabulares mediante DataFrame y Series.
import pandas as pd
# models: expone las siete clases finales y el registro que relaciona tareas con clases.
# Se importa: REGISTRY.
from models import REGISTRY
# warnings: captura avisos del modelo para explicarlos sin mostrar líneas de código.
import warnings
# urllib.parse: construye enlaces a partir de identificadores sin consultar Internet.
from urllib.parse import quote
# results_report: reutiliza el formato español de cifras de los informes.
from results_report import format_number
# models.core: identifica el aviso de columnas adicionales, sin modificar los modelos.
from models.core import IgnoredColumnsWarning

# CATEGORY_NAMES: traducciones de presentación; se conserva también la etiqueta original.
CATEGORY_NAMES = {
    'Film & Animation': 'Cine y animación', 'Autos & Vehicles': 'Vehículos',
    'Music': 'Música', 'Pets & Animals': 'Mascotas y animales', 'Sports': 'Deportes',
    'Travel & Events': 'Viajes y eventos', 'Gaming': 'Videojuegos',
    'People & Blogs': 'Personas y blogs', 'Comedy': 'Comedia',
    'Entertainment': 'Entretenimiento', 'News & Politics': 'Noticias y política',
    'Howto & Style': 'Tutoriales y estilo', 'Education': 'Educación',
    'Science & Technology': 'Ciencia y tecnología',
    'Nonprofits & Activism': 'Organizaciones y activismo',
    'Shows': 'Programas', 'Movies': 'Películas', 'Trailers': 'Tráilers',
}


def _display_text(value, fallback=''):
    """Normaliza un texto escalar para mostrarlo en una línea sin controles de terminal."""
    if value is None or pd.isna(value):
        return fallback
    # text: conserva caracteres imprimibles y normaliza espacios y saltos de línea.
    text = ' '.join(str(value).split())
    return ''.join(character for character in text if character.isprintable()) or fallback


# RECOMMENDATION_ROW_ERRORS: sólo estas incidencias de entrada admiten recuperación por fila.
RECOMMENDATION_ROW_ERRORS = {
    'Consulta vacía:': 'Falta contenido. Aporta título, etiquetas o descripción, o un ID conocido sin campos de texto.',
    'ID desconocido:': 'El ID no está en el catálogo. Aporta título, etiquetas o descripción del vídeo.',
    'Consulta sin vocabulario conocido;': 'El texto no contiene términos del vocabulario aprendido. Añade etiquetas o descripción, o consulta por un ID conocido.',
}


def _predict_for_cli(model, data, row_issues=None):
    """Conserva filas válidas de un lote del recomendador sin cambiar su API ni sus pesos.

    Los lotes válidos siguen usando una sola llamada. Si falla una entrada prevista,
    divide el lote hasta aislarla; conserva posiciones y devuelve [] en esa fila.
    Registra su causa en row_issues. Una entrada única o un lote totalmente inválido
    mantiene un error global; fallos estructurales o del modelo siempre se propagan.
    """
    if getattr(model, 'task', None) != 'recommend' or len(data) < 2:
        return model.predict(data)
    issues = row_issues if row_issues is not None else {}

    def predict_block(first, last):
        """Resuelve un bloque contiguo y subdivide sólo errores de entrada conocidos."""
        try:
            result = model.predict(data.iloc[first:last])
        except ValueError as exc:
            note = next((note for prefix, note in RECOMMENDATION_ROW_ERRORS.items()
                         if str(exc).startswith(prefix)), None)
            if note is None:
                raise
            if last-first == 1:
                issues[first] = note
                return [[]]
            middle = (first+last)//2
            return predict_block(first, middle) + predict_block(middle, last)
        if len(result) != last-first:
            raise ValueError('El recomendador no ha devuelto una respuesta por fila; no se pueden asociar los resultados.')
        return list(result)

    result = predict_block(0, len(data))
    if len(issues) == len(data):
        raise ValueError(f'No se pudo procesar ninguna de las {format_number(len(data), 0)} consultas. '
                         f'Primera incidencia, fila 1: {issues[0]}')
    return result


def _predict_with_notes(model, data, row_issues=None):
    """Predice sin cambiar el modelo y convierte los avisos en notas breves en stderr.

    No elimina avisos: resume los de columnas extra y conserva el mensaje de los
    demás. Su alcance termina con esta llamada y no cambia filtros permanentes.
    """
    with warnings.catch_warnings(record=True) as notices:
        warnings.simplefilter('always', IgnoredColumnsWarning)
        try:
            result = _predict_for_cli(model, data, row_issues)
        finally:
            # seen: evita repetir el mismo aviso cuando varias observaciones lo provocan.
            seen = set()
            for notice in notices:
                if issubclass(notice.category, IgnoredColumnsWarning):
                    extra = len(set(data.columns) - set(model.columns_))
                    message = (f'Nota: columnas adicionales no utilizadas en esta tarea: {extra}. '
                               'Esto no impide la consulta.')
                else:
                    message = 'Aviso del modelo: ' + _display_text(str(notice.message))
                if message not in seen:
                    print(message, file=sys.stderr)
                    seen.add(message)
    return result


def _format_text_results(task, data, result, source, model, row_issues=None):
    """Relaciona cada fila con su resultado y explica las siete tareas en español.

    Usa posiciones de fila, no el índice del DataFrame, para evitar atribuciones
    incorrectas. Presenta cifras sin cambiar predicciones; no inventa probabilidades
    ni significados para los grupos. En recomendación, un ID conocido puede tomar
    su título del catálogo guardado únicamente para identificar la consulta.
    """
    if len(result) != len(data):
        raise ValueError('El número de resultados no coincide con los vídeos del CSV; no se pueden asociar con seguridad.')
    # descriptions: propósito y límite de cada tarea, sin métricas prefijadas.
    descriptions = {
        'category': ('Categoría del vídeo', 'La categoría es una estimación del modelo a partir de los datos del CSV.'),
        'comments_disabled': ('Estado de los comentarios', 'Se predice si los comentarios están habilitados; no se comprueba su estado actual en YouTube.'),
        'ratings_disabled': ('Estado de las valoraciones', 'Se predice si las valoraciones están habilitadas; no se comprueba su estado actual en YouTube.'),
        'likes': ('Estimación de likes', 'Utiliza datos disponibles del vídeo, incluidos otros contadores; no es una previsión anterior a publicarlo.'),
        'ratio': ('Proporción de likes/dislikes', 'Estima cuántos likes corresponden a cada dislike con la información disponible.'),
        'cluster': ('Asignación a un grupo', 'Agrupa por texto y otras características. El número identifica un grupo; no es una categoría ni una nota de calidad.'),
        'recommend': ('Vídeos recomendados', 'Busca textos parecidos dentro del catálogo guardado. La similitud no es una probabilidad de que el vídeo te guste.'),
    }
    heading, note = descriptions[task]
    lines = [f'RESULTADOS: {heading}', f'Archivo utilizado: {source}',
             f'Vídeos consultados: {format_number(len(data), 0)}', note]
    # titles: sólo identifica consultas por ID; no interviene en la similitud calculada.
    titles = {}
    if task == 'recommend' and hasattr(model, 'catalog_') and 'title' in model.catalog_:
        titles = dict(zip(model.catalog_.video_id.astype(str), model.catalog_.title))
    for position, ((_, row), value) in enumerate(zip(data.iterrows(), result), 1):
        video_id = _display_text(row.get('video_id'))
        title = _display_text(row.get('title')) or _display_text(titles.get(video_id), 'Sin título disponible')
        lines += ['', f'Vídeo {format_number(position, 0)}: {title}']
        if video_id:
            lines.append('ID: ' + video_id)
        if row_issues and position-1 in row_issues:
            lines.append('Consulta no procesada: ' + row_issues[position-1])
            continue
        if task == 'category':
            original = _display_text(value, 'Categoría no disponible')
            translated = CATEGORY_NAMES.get(original)
            label = f'{translated} ({original})' if translated else original
            lines.append('Categoría estimada: ' + label)
        elif task in ['comments_disabled', 'ratings_disabled']:
            if value not in [0, 1]:
                raise ValueError('El clasificador no ha devuelto un estado válido; se esperaba 0 o 1.')
            subject = 'Comentarios' if task == 'comments_disabled' else 'Valoraciones'
            state = ('deshabilitados' if value == 1 else 'habilitados') if task == 'comments_disabled' else ('deshabilitadas' if value == 1 else 'habilitadas')
            lines.append(f'Predicción: {subject.lower()} {state}.')
        elif task in ['likes', 'ratio']:
            if not np.isfinite(float(value)):
                raise ValueError(f'El modelo ha devuelto una estimación no finita para el vídeo {position}.')
            if float(value) < 0:
                lines.append(f'El modelo devuelve {format_number(float(value), 2)}; este valor negativo no tiene interpretación como {task}. No se corrige automáticamente.')
            elif task == 'likes':
                lines.append(f'Likes estimados: aproximadamente {format_number(float(value), 0)}.')
            else:
                lines.append(f'Proporción estimada: {format_number(float(value), 2)} likes por cada dislike.')
        elif task == 'cluster':
            lines.append(f'Grupo asignado: {_display_text(value)}.')
        else:
            if not value:
                lines.append('No se han encontrado recomendaciones con similitud positiva para este vídeo.')
            for rank, item in enumerate(value, 1):
                candidate_id = _display_text(item.get('video_id'))
                candidate_title = _display_text(item.get('title'), 'Sin título disponible')
                lines.append(f'  {rank}. {candidate_title}')
                lines.append(f'     Similitud textual: {format_number(float(item["similarity"]), 3)}.')
                if candidate_id:
                    lines.append(f'     ID: {candidate_id}')
                    lines.append('     Enlace: https://www.youtube.com/watch?v=' + quote(candidate_id, safe=''))
    return '\n'.join(lines)


def _csv_selection_prompt(count):
    """Explica las opciones disponibles sin mostrar un rango absurdo cuando sólo hay un CSV."""
    if count == 0:
        return 'Escribe otra ruta o S para cancelar la consulta: '
    numbers = '1' if count == 1 else f'1-{count}'
    return f'Elige un número ({numbers}), escribe otra ruta o S para cancelar la consulta: '


def _select_input_csv(value):
    """Si la ruta es una carpeta, permite elegir uno de sus CSV sin recorrer subcarpetas.

    Lista archivos .csv sin distinguir mayúsculas, ordenados por nombre. Admite
    número, una ruta directa, R para solicitar otra ruta y S para cancelar. Devuelve una ruta o None cuando
    se cancela o termina la entrada. Los mensajes van a stderr para reservar
    stdout a las predicciones JSON. No abre pesos ni modifica archivos.
    """
    # current: ruta que puede ser un archivo o la carpeta que se desea explorar.
    current = Path(value)
    while current.is_dir():
        try:
            # files: archivos CSV inmediatos; se excluyen carpetas con extensión .csv.
            files = sorted((path for path in current.iterdir()
                            if path.is_file() and path.suffix.lower() == '.csv'),
                           key=lambda path: path.name.casefold())
        except OSError as exc:
            raise ValueError(f'No se puede listar la carpeta: {current}. Comprueba sus permisos de lectura.') from exc
        print(f'\nArchivos CSV en: {current}', file=sys.stderr)
        for number, path in enumerate(files, 1):
            print(f'{number}. {path.name}', file=sys.stderr)
        if not files:
            print('No hay archivos CSV en esta carpeta; no se han buscado en subcarpetas.', file=sys.stderr)
        print('Puedes escribir otra ruta directamente. R también permite pedirla; S cancela.', file=sys.stderr)
        while True:
            # choice: texto escrito por el usuario; nunca se ejecuta como comando.
            print(_csv_selection_prompt(len(files)), end='', file=sys.stderr, flush=True)
            try:
                choice = input().strip()
            except (EOFError, KeyboardInterrupt):
                return None
            if choice.upper() == 'S':
                return None
            if choice.upper() == 'R':
                print('Nueva ruta de archivo o carpeta [Intro = data/example.csv]: ',
                      end='', file=sys.stderr, flush=True)
                try:
                    current = Path(input().strip().strip('"') or 'data/example.csv')
                except (EOFError, KeyboardInterrupt):
                    return None
                break
            try:
                # index: posición de la lista; sólo se acepta dentro del rango mostrado.
                index = int(choice) - 1
            except ValueError:
                # candidate: ruta literal, con o sin comillas; nunca se interpreta como código.
                path_text = choice.strip('"')
                candidate = Path(path_text)
                if path_text and (candidate.is_absolute() or '/' in path_text or '\\' in path_text
                                  or candidate.suffix.lower() == '.csv' or candidate.exists()):
                    current = candidate
                    break
                index = -1
            if 0 <= index < len(files):
                return str(files[index])
            print('Opción no válida. Escribe un número de la lista, otra ruta o S para cancelar.'
                  if files else 'Opción no válida. Escribe otra ruta o S para cancelar.', file=sys.stderr)
    return str(current)


def _read_input_csv(value):
    """Lee un CSV con errores breves que indican la causa y cómo corregirla.

    Valida archivo y extensión antes de cargar pesos. Conserva la lectura UTF-8
    y las reglas de pandas; no modifica datos ni transforma columnas. Distingue
    carpeta, ausencia, permisos, archivo vacío, codificación y formato ilegible.
    """
    # path: archivo solicitado; las rutas relativas se interpretan desde la carpeta activa.
    path = Path(value)
    if path.is_dir():
        raise ValueError(f'Has indicado una carpeta: {path}. Selecciona un archivo CSV concreto, por ejemplo data/example.csv.')
    if not path.exists():
        raise FileNotFoundError(f'No existe el archivo CSV: {path}. Revisa la ruta o usa data/example.csv.')
    if path.suffix.lower() != '.csv':
        raise ValueError(f'El archivo no tiene extensión .csv: {path}. Selecciona un archivo CSV de vídeos.')
    try:
        # data: una fila por vídeo consultado; no se carga ningún modelo en este paso.
        data = pd.read_csv(path)
    except IsADirectoryError as exc:
        raise ValueError(f'Has indicado una carpeta: {path}. Selecciona un archivo CSV concreto.') from exc
    except FileNotFoundError as exc:
        raise FileNotFoundError(f'No existe el archivo CSV: {path}. Revisa la ruta o usa data/example.csv.') from exc
    except PermissionError as exc:
        raise ValueError(f'No se puede leer el CSV: {path}. Comprueba los permisos y si otro programa lo tiene bloqueado.') from exc
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f'El CSV está vacío: {path}. Debe contener cabeceras y al menos un vídeo; puedes usar data/example.csv.') from exc
    except UnicodeDecodeError as exc:
        raise ValueError(f'El CSV no está codificado en UTF-8: {path}. Para probar usa data/example.csv. Si es un CSV regional original, prepáralo con preprocess_data.py; si es propio, guárdalo como CSV UTF-8.') from exc
    except pd.errors.ParserError as exc:
        raise ValueError(f'No se puede interpretar el CSV: {path}. Revisa las comillas y el número de columnas; utiliza comas como separador.') from exc
    except OSError as exc:
        raise ValueError(f'No se puede abrir el CSV: {path}. Comprueba la ruta y los permisos de lectura.') from exc
    if data.empty:
        raise ValueError(f'El CSV no contiene observaciones: {path}. Añade al menos una fila de vídeo después de la cabecera.')
    return data


def main():
    """Lee tarea, pickle y CSV; muestra JSON por defecto o resultados explicados.

    Valida la tarea y el CSV; opcionalmente permite elegirlo en una carpeta.
    Delega el esquema y los pesos en la clase; cancelar la selección devuelve código 3."""
    # p: parser de los tres argumentos posicionales de inferencia.
    p = argparse.ArgumentParser()
    p.add_argument('model_type', choices=REGISTRY)
    p.add_argument('weights')
    p.add_argument('input_file')
    p.add_argument('--formato', choices=['json', 'texto'], default='json',
                   help='JSON para programas; texto explicado para consultar desde el menú.')
    p.add_argument('--seleccionar-csv', action='store_true',
                   help='Si la entrada es una carpeta, permite elegir uno de sus CSV.')
    # a: tarea, archivo de pesos y CSV recibidos y validados por la CLI.
    a = p.parse_args()
    try:
        # input_path: archivo directo o seleccionado; cancelar no produce predicciones.
        input_path = _select_input_csv(a.input_file) if a.seleccionar_csv else a.input_file
        if input_path is None:
            p.exit(3, 'Consulta cancelada. No se han generado predicciones.\n')
        # data: entrada validada antes de cargar pesos; los errores de ruta no abren modelos.
        data = _read_input_csv(input_path)
        # model: estado aprendido recuperado del disco, sin reentrenarlo ni guardarlo.
        model = REGISTRY[a.model_type]().load(a.weights)
        # Una consulta por ID conocido se resuelve contra el catálogo guardado.
        id_only = a.model_type=='recommend' and 'video_id' in data and not set(data)&{'title','tags','description'}
        if not id_only:
            missing = sorted(set(model.columns_)-set(data.columns))
            if missing:
                print('Aviso: columnas ausentes: '+', '.join(missing)+'. Se aplicará el tratamiento de ausentes admitido por el modelo.',file=sys.stderr)
        for column in ['views','dislikes','comment_count']:
            if column in data and column in model.columns_:
                values = pd.to_numeric(data[column],errors='coerce').to_numpy(dtype=float)
                invalid = int((~np.isfinite(values) | (values<0)).sum())
                if invalid:
                    print(f'Aviso: {column} contiene {invalid} valores ausentes o inválidos; el modelo aplicará su imputación aprendida.',file=sys.stderr)
        # El menú explica los resultados; la CLI original conserva exactamente su JSON.
        # Ambos formatos resumen los avisos en stderr; stdout mantiene sólo los resultados.
        # row_issues: causas por posición; las predicciones mantienen el orden del CSV.
        row_issues = {}
        result = _predict_with_notes(model, data, row_issues)
        if a.formato == 'texto':
            print(_format_text_results(a.model_type, data, result, input_path, model, row_issues))
        else:
            for position, note in sorted(row_issues.items()):
                print(f'Aviso: fila {format_number(position+1, 0)} sin recomendaciones. {note} Las demás filas conservan sus resultados.', file=sys.stderr)
            print(json.dumps(result.tolist() if isinstance(result,np.ndarray) else result,ensure_ascii=True))
    except (OSError,ValueError,RuntimeError,EOFError,KeyError,pickle.UnpicklingError) as exc:
        p.exit(2,f'Error de inferencia: {exc}\n')

# Punto de entrada: al ejecutar este archivo directamente, __name__ vale "__main__".
# El bloque siguiente inicia su ejecución; al importarlo desde otro archivo, no se ejecuta.
if __name__ == '__main__':
    main()
