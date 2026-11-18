# Interpretación de la segmentación

ARI frente a categoría: **0,0060**; NMI: **0,0109**. La coincidencia con categorías editoriales es débil: los grupos no recuperan géneros de forma fiable.

ARI=0 corresponde aproximadamente al acuerdo esperado al azar; NMI mide información compartida. Las categorías no eligen el modelo ni el número de grupos.

Las proyecciones comparten coordenadas del espacio TF-IDF + metadatos + SVD del modelo. La proyección 2D pierde información y no sustituye silueta.

cluster_category_proportions.csv guarda fracciones entre 0 y 1 por grupo; el cuaderno las muestra también como porcentajes. cluster_terms.csv contiene términos reconstruidos desde centros o medias de la familia; son aproximaciones descriptivas y pueden reflejar plantillas o sistemas de escritura.

## Criterio de selección

Regla añadida después de observar una segmentación previa del 99,5 %: cada grupo debe contener al menos el 5 % de los vídeos y tres réplicas válidas de estabilidad con ARI medio >= 0,80; se maximiza silueta entre quienes cumplen. No se usan categorías editoriales ni etiquetas de prueba. Son límites metodológicos de esta revisión, no requisitos del profesor ni criterios fijados antes del primer estudio.

## Estabilidad medida

Silueta de la solución seleccionada: **0,202**. Estabilidad con SVD fijo: ARI medio **0,992**; 3 réplicas [0,996; 0,984; 0,996], rango 0,984–0,996. La media debe leerse junto con las réplicas: no acredita reproducción uniforme ni utilidad de los grupos. Al reajustar todo el preprocesamiento con la receta congelada, ARI medio **0,984**; réplicas [0,988; 0,988; 0,976]. El diagnóstico utiliza 5000 vídeos de entrenamiento y 1000 de auditoría interna, con 0 IDs compartidos. No es validación externa. Pocas réplicas limitan la interpretación; el balance se describe con los tamaños reales de los perfiles. Estabilidad, silueta y utilidad son conceptos distintos.

## Tamaños y utilidad

Distribución de tamaños: el grupo mayor reúne 51,4 % de los 6000 vídeos; el menor contiene 2914. Los tamaños deben valorarse junto con perfiles, estabilidad y el objetivo de negocio; silueta no valida utilidad por sí solo.
