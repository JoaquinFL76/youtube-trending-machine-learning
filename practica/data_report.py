"""Informe descriptivo reproducible sobre el conjunto completo."""
# pathlib: maneja rutas de archivos y carpetas mediante objetos, como Path.
# Se importa: Path.
from pathlib import Path
# json: lee y escribe datos estructurados, configuraciones y evidencias en JSON.
import json
# pandas: organiza, limpia y analiza datos tabulares mediante DataFrame y Series.
import pandas as pd
# results_report: presenta cifras en formato español y negrita Markdown.
from results_report import highlight_number

def corpus_overview(rows, unique_videos, regions, categories):
    """Redacta el tamaño y la cobertura a partir de los recuentos ya calculados."""
    return (
        f'El CSV unificado contiene {highlight_number(rows)} observaciones y '
        f'{highlight_number(unique_videos)} identificadores de vídeo, en '
        f'{highlight_number(regions)} regiones y {highlight_number(categories)} categorías.'
    )


def main(*, echo=True):
    """Escribe el resumen JSON y devuelve el informe descriptivo en Markdown.

    echo permite imprimirlo al ejecutar el script directamente. El cuaderno
    recibe el mismo texto y lo presenta como Markdown.
    """
    # root: directorio de la práctica, independiente del directorio de invocación.
    root = Path(__file__).resolve().parent
    # df: tabla de observaciones del bloque; cada fila mantiene su unidad original.
    df = pd.read_csv(root/'data/processed/videos.csv')
    # out: destino de informes o resultado intermedio según la operación local.
    out = root/'reports'
    out.mkdir(exist_ok=True)
    # summary: resumen estructurado para dejar evidencia verificable.
    summary = {
        'rows':len(df), 'unique_videos':df.video_id.nunique(),
        'regions':df.region.value_counts().to_dict(),
        'categories':df.category.value_counts().to_dict(),
        'missing':df.isna().sum().to_dict(),
        'comments_disabled_fraction':float(df.comments_disabled.mean()),
        'ratings_disabled_fraction':float(df.ratings_disabled.mean()),
        'zero_dislikes_fraction':float(df.dislikes.eq(0).mean()),
        'counter_quantiles':df[['views','likes','dislikes','comment_count']].quantile([0,.5,.9,.99,1]).to_dict(),
    }
    # moderation: tasas regionales de desactivación, ponderadas por apariciones.
    moderation = df.groupby('region')[['comments_disabled','ratings_disabled']].mean()
    # peak_comments: región con mayor tasa observada de comentarios desactivados.
    peak_comments = moderation.comments_disabled.idxmax()
    # peak_ratings: región con mayor tasa observada de valoraciones desactivadas.
    peak_ratings = moderation.ratings_disabled.idxmax()
    # audit: registro de limpieza, incidencias y decisiones sin aprender transformaciones.
    audit = json.loads((root/'data/processed/videos.audit.json').read_text(encoding='utf-8'))
    # introduced: total de sustituciones UTF-8 introducidas, separado del daño previo.
    introduced = sum(item.get('replacement_characters_introduced',0) for item in audit['regions'].values())
    # dates: fechas convertidas con ausentes explícitos cuando falla el parseo.
    dates = pd.to_datetime(df.trending_date,errors='coerce',utc=True)
    # monthly: cobertura por mes, con días observados para contextualizar meses parciales.
    monthly = pd.DataFrame({'month':dates.dt.strftime('%Y-%m'),'day':dates.dt.normalize()}).groupby('month').agg(rows=('day','size'),observed_days=('day','nunique'))
    # monthly['rows_per_observed_day']: volumen dividido por días observados para contextualizar meses
    # parciales.
    monthly['rows_per_observed_day'] = monthly.rows/monthly.observed_days
    summary['monthly_coverage'] = monthly.to_dict(orient='index')
    (out/'data_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False,default=str),encoding='utf-8')
    # Los recuentos, las tasas y los cuantiles proceden del resumen ya calculado.
    category, appearances = next(iter(summary['categories'].items()))
    views = summary['counter_quantiles']['views']
    likes = summary['counter_quantiles']['likes']
    size = corpus_overview(summary['rows'], summary['unique_videos'],
                           len(summary['regions']), len(summary['categories']))
    paragraphs = [
        '# Resultados descriptivos del dataset completo',
        '## Tamaño y cobertura',
        size + ' Cada observación es una aparición en tendencias.',
        f'La cobertura va del **{dates.min().date()}** al **{dates.max().date()}**. '
        f'El primer mes incluye {highlight_number(monthly.observed_days.iloc[0])} días observados '
        f'y el último {highlight_number(monthly.observed_days.iloc[-1])}. '
        'Los registros por día observado permiten comparar esos meses parciales. '
        'Esta cobertura es insuficiente para estimar estacionalidad anual.',
        '## Categorías y moderación',
        f'La categoría más frecuente es **{category}**, con '
        f'{highlight_number(appearances / summary["rows"] * 100, 1, " %")} de las apariciones.',
        f'Los comentarios están desactivados en el '
        f'{highlight_number(summary["comments_disabled_fraction"] * 100, 2, " %")} '
        f'y las valoraciones en el {highlight_number(summary["ratings_disabled_fraction"] * 100, 2, " %")}. '
        'Este desbalance motiva el uso de **macro-F1**, exactitud equilibrada, '
        'recall por clase y matrices de confusión.',
        f'La mayor tasa regional de comentarios desactivados se observa en **{peak_comments}** '
        f'({highlight_number(moderation.loc[peak_comments, "comments_disabled"] * 100, 2, " %")}). '
        f'En valoraciones se observa en **{peak_ratings}** '
        f'({highlight_number(moderation.loc[peak_ratings, "ratings_disabled"] * 100, 2, " %")}). '
        'Estas diferencias orientan la revisión de errores por región. '
        'Atribuirlas a causas culturales o decisiones de la plataforma requeriría otras evidencias.',
        '## Calidad de los datos y fechas',
        f'La auditoría retira {highlight_number(audit.get("invalid_identity_rows_removed", 0))} filas '
        'con marcadores que impiden identificar el vídeo. '
        f'Conserva y audita {highlight_number(audit.get("nonstandard_ids_retained", 0))} registros '
        f'con IDs no estándar y {highlight_number(audit.get("videos_with_multiple_categories", 0))} '
        'vídeos con categorías contradictorias. Las apariciones del mismo vídeo se mantienen juntas.',
        f'La lectura conserva UTF-8 y sustituye {highlight_number(introduced)} secuencias inválidas. '
        'La auditoría registra los recuentos y las huellas de origen por región.',
        f'Hay {highlight_number(audit.get("negative_exact_timestamp_delays", 0))} diferencias horarias '
        f'negativas entre tendencia y publicación y {highlight_number(audit.get("negative_calendar_day_delays", 0))} '
        'diferencias negativas al comparar por día. Las listas de tendencias registran el día. '
        'Los desfases dentro de un mismo día deben interpretarse con esa resolución.',
        '## Contadores y alcance',
        f'El {highlight_number(summary["zero_dislikes_fraction"] * 100, 2, " %")} '
        'de las apariciones tiene cero dislikes. El objetivo likes/dislikes se calcula '
        'en apariciones con dislikes positivos.',
        f'Las medianas son {highlight_number(views[.5])} visualizaciones y '
        f'{highlight_number(likes[.5])} likes. En likes, el percentil 99 es '
        f'{highlight_number(likes[.99])} y el máximo {highlight_number(likes[1])}. '
        'La comparación entre la escala original y la logarítmica ayuda a estudiar la cola larga.',
        'La repetición de vídeos exige separar los conjuntos por **video_id**. '
        'El estudio describe los vídeos observados en tendencias.',
        'Las tablas completas están en `data_summary.json`, la auditoría regional en '
        '`data/processed/videos.audit.json` y las métricas predictivas en `evaluation.json`.',
    ]
    # results_report: comparte la terminología española de las salidas.
    from results_report import clarify_narrative
    text = clarify_narrative('\n\n'.join(paragraphs) + '\n')
    (out/'data_findings.md').write_text(text, encoding='utf-8')
    if echo:
        print(text)
    return text


# Punto de entrada: al ejecutar este archivo directamente, __name__ vale "__main__".
# El bloque siguiente inicia su ejecución; al importarlo desde otro archivo, no se ejecuta.
if __name__ == '__main__':
    main()
