# Resultados descriptivos del dataset completo

## Tamaño y cobertura

El CSV unificado contiene **361.234** observaciones y **184.286** identificadores de vídeo, en **10** regiones y **18** categorías. Cada observación es una aparición en tendencias.

La cobertura va del **2017-11-14** al **2018-06-14**. El primer mes incluye **17** días observados y el último **14**. Los registros por día observado permiten comparar esos meses parciales. Esta cobertura es insuficiente para estimar estacionalidad anual.

## Categorías y moderación

La categoría más frecuente es **Entertainment**, con **28,8 %** de las apariciones.

Los comentarios están desactivados en el **2,17 %** y las valoraciones en el **1,86 %**. Este desbalance motiva el uso de **macro-F1**, exactitud equilibrada, recuperación por clase y matrices de confusión.

La mayor tasa regional de comentarios desactivados se observa en **JP** (**6,69 %**). En valoraciones se observa en **JP** (**6,88 %**). Estas diferencias orientan la revisión de errores por región. Atribuirlas a causas culturales o decisiones de la plataforma requeriría otras evidencias.

## Calidad de los datos y fechas

La auditoría retira **2.138** filas con marcadores que impiden identificar el vídeo. Conserva y audita **7** registros con IDs no estándar y **137** vídeos con categorías contradictorias. Las apariciones del mismo vídeo se mantienen juntas.

La lectura conserva UTF-8 y sustituye **121** secuencias inválidas. La auditoría registra los recuentos y las huellas de origen por región.

Hay **18.106** diferencias horarias negativas entre tendencia y publicación y **0** diferencias negativas al comparar por día. Las listas de tendencias registran el día. Los desfases dentro de un mismo día deben interpretarse con esa resolución.

## Contadores y alcance

El **2,51 %** de las apariciones tiene cero dislikes. El objetivo likes/dislikes se calcula en apariciones con dislikes positivos.

Las medianas son **177.042** visualizaciones y **3.540** likes. En likes, el percentil 99 es **595.421** y el máximo **5.613.827**. La comparación entre la escala original y la logarítmica ayuda a estudiar la cola larga.

La repetición de vídeos exige separar los conjuntos por **video_id**. El estudio describe los vídeos observados en tendencias.

Las tablas completas están en `data_summary.json`, la auditoría regional en `data/processed/videos.audit.json` y las métricas predictivas en `evaluation.json`.
