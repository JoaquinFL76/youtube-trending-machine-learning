# Alcance y posibles mejoras

Este repositorio conserva la versión entregada de la práctica. Los siguientes puntos distinguen lo que se ha demostrado de las mejoras que requerirían nuevos experimentos.

## Segmentación

La solución guardada es KMeans con dos grupos y silueta de 0,2022. Su correspondencia con las categorías es reducida: índice de Rand ajustado (ARI) de 0,0060 e información mutua normalizada (NMI) de 0,0109. Describe perfiles de audiencia y composición regional, sin demostrar segmentos temáticos útiles.

Una revisión futura debería limpiar enlaces y marcadores vacíos del texto, comparar una representación sólo textual con la combinación actual y estudiar números de grupos mayores mediante estabilidad, tamaños e interpretación. Un número concreto de grupos debe justificarse por resultados; no se propone como ganador sin medirlo.

## Cobertura del entrenamiento y generalización

La búsqueda supervisada usa 5.955 filas. El ajuste final utiliza 29.926 filas para categoría, comentarios, valoraciones y likes; ratio utiliza 29.232 tras excluir objetivos no definidos. El desarrollo completo contiene 288.833 apariciones. Esta reducción hace viable el experimento, pero deja margen para entrenar con más datos y volver a medir la brecha entre ajuste y validación.

Las apariciones de cada vídeo se mantienen juntas. Esto no garantiza independencia por canal: el diagnóstico con canales separados reduce el macro-F1 de categoría a 0,383 y el de comentarios a 0,380. La prueba interna se ha consultado durante el desarrollo; no se presenta como una evaluación externa intacta.

## Regresión de likes y ratio

Las señales de interacción se observan en el mismo momento que el objetivo. La estimación de likes alcanza un MAE de 14.202,119 frente a 57.088,142 de referencia, pero unos pocos vídeos concentran gran parte del error. La versión entregada incluye vídeos con valoraciones desactivadas; una comparación futura debería estudiar su exclusión y declarar de forma consistente la nueva población de entrenamiento y evaluación.

El ratio mantiene un resultado modesto: R² de 0,132. Las mejoras sobre su cola de valores extremos son hipótesis que exigirían validación, sin retirar los ejemplos difíciles de la prueba para mejorar las cifras.

## Recomendación

El 44,60 % de coincidencia editorial, frente al 14,32 % esperado al azar, se mide sobre un catálogo de 6.000 vídeos. No acredita utilidad para personas ni evalúa independientemente el catálogo final ampliado a 10.000. La evaluación humana sería una extensión fuera del alcance solicitado.

La demostración entregada identifica la consulta por ID y puede devolver títulos repetidos con distintos IDs. Mostrar también el título de entrada y aplicar una regla explícita de diversidad por título son mejoras posibles. Si se modifica la selección de vecinos, deben comprobarse de nuevo sus efectos.

## Legibilidad y evolución

La modularización permite reutilizar lógica y verificar el flujo, aunque obliga a recorrer módulos para seguir algunas comparaciones. Una versión didáctica posterior podría desarrollar dentro del cuaderno al menos una comparación y una búsqueda completas, resumir mejor cada decisión y reducir explicaciones repetidas.

Estas mejoras no están implementadas en esta publicación. El código, los modelos y las salidas de `practica/` son los de la versión académica conservada.
