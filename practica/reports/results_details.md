# Informe de resultados y decisiones

La pregunta guía del cuaderno se formuló a partir del enunciado. Los retos pedidos abarcan categoría, número de likes, segmentación y recomendación. Los retos opcionales para subir nota abarcan la deshabilitación de comentarios o valoraciones y la proporción de likes y dislikes. La práctica aborda comentarios y valoraciones por separado. El [glosario de la práctica](../GUIA_DE_CONSULTA.md#glosario) explica las familias, las métricas y los términos del protocolo.

Datos limpios: **361.234 filas**. Búsqueda: **5.955**; ajuste final ampliado: **29.926**; prueba: **7.951**. La muestra de búsqueda queda incluida en el ajuste final. Ningún video_id cruza entrenamiento y prueba.

La muestra se estratifica aproximadamente por categoría conservando grupos completos; representativeness.csv cuantifica diferencias por región, categoría y etiquetas minoritarias. La prueba representa otros vídeos de tendencias del mismo periodo, no vídeos futuros ni canales nuevos.

La prueba interna se ha examinado durante el desarrollo y no constituye una validación externa prospectiva.

| Objetivo | Enunciado | Modelo | Entrenamiento | Prueba | Referencia en la misma prueba | IC 95 % por vídeo |
|---|---|---|---:|---:|---:|---|
| category | Pedido | svm | 0,934 | 0,527 | 0,025 | [0,501, 0,554] |
| comments_disabled | Opcional para subir nota | svm | 0,978 | 0,670 | 0,495 | [0,630, 0,722] |
| ratings_disabled | Opcional para subir nota | logistic | 0,727 | 0,591 | 0,495 | [0,560, 0,620] |
| likes | Pedido | svr_log_target | 4.504,118 | 14.202,119 | 57.088,142 | [6.812,051, 21.461,701] |
| ratio | Opcional para subir nota | ridge_log_target | 15,671 | 26,984 | 32,116 | [24,311, 29,199] |

Clasificación: macro-F1, mayor es mejor. Regresión: MAE en unidades originales, menor es mejor. Los intervalos remuestrean vídeos y son condicionales al ajuste; no incluyen incertidumbre de selección ni cambio temporal.

El macro-F1 de train/test y su intervalo usan la unión de etiquetas de ajuste y prueba, con zero_division=0. Una clase sin soporte de prueba aporta cero por esa convención y no permite medir su fiabilidad. La CV usa el scorer sklearn f1_macro con la unión de clases presentes en y_true o y_pred de cada partición. Los candidatos comparten particiones, pero no necesariamente las clases promediadas: predecir una clase ausente añade un F1 cero. La escasez de clases también afecta a la variación. No se ha cambiado esta convención para conservar el experimento registrado.

## Resultados que condicionan la interpretación

En el diagnóstico interno por canales disjuntos, comments_disabled obtiene macro-F1 0,380, frente a la referencia 0,495. Se reajusta la receta con otra partición y calibración: no es una evaluación directa de los pesos finales. La evaluación por vídeo y el diagnóstico por canal evalúan condiciones distintas; ninguno constituye validación externa prospectiva.

En ratings_disabled, F2 pasa de 0,318 con el umbral predeterminado a 0,302 con el umbral OOF. Se conserva la política elegida en entrenamiento y se informa de su resultado; no se cambia el umbral mirando la prueba.

La estrategia de calibración se revisó durante el desarrollo tras examinar resultados internos; se conserva la trazabilidad de ambas etapas y no constituye validación externa prospectiva.

La selección de familias y el refinamiento usan macro-F1; el umbral posterior optimiza F2 en OOF. Son objetivos distintos. Comparar con AP dentro de CV es una mejora experimental pendiente.

## Decisiones por problema

### category
Comparación condicionada por la representación: svm_rbf: SVD 30; mlp_classifier: SVD 60; lda_classifier: SVD 60. Los candidatos sin SVD usan la representación TF-IDF y sus metadatos. Se comparan recetas completas, por lo que las diferencias combinan algoritmo y representación. El refinamiento explora configuraciones de la familia ganadora dentro del presupuesto descrito.
Segunda fase: 13 configuraciones, incluida la referencia; CV inicial 0,4221, refinada 0,4557. Las rejillas y los avisos están en evaluation.json y se muestran en el cuaderno. La selección no consulta la prueba.
CV seleccionada: 0,4557 ± 0,0176; particiones [0,479878; 0,438526; 0,448577]. Parámetros: `{'estimator__C': 1.0, 'estimator__class_weight': 'balanced', 'features__text__max_features': 8000, 'features__text__min_df': 2, 'features__text__ngram_range': [1, 1], 'features__text__sublinear_tf': True, 'estimator__max_iter': 30000}`. El signo de la MAE en CV es negativo por convención de sklearn.
Macro-F1 de prueba 0,527, frente a 0,025 del clasificador mayoritario. La diferencia con entrenamiento es +0,407: debe leerse junto con la curva de aprendizaje y el soporte por clase. No todas las categorías tienen la misma fiabilidad.
Clases sin soporte en esta prueba: ['Movies', 'Trailers']. Como lectura complementaria, macro-F1 sólo sobre las 16 clases observadas: 0,593, referencia 0,029. Esta alternativa no reemplaza la definición global ni su intervalo remuestreo con reemplazo.
Categorías con menor F1 y al menos 20 ejemplos: Shows: F1 0,169, recuperación 22,7 %, n=22; Travel & Events: F1 0,217, recuperación 14,7 %, n=34; Nonprofits & Activism: F1 0,348, recuperación 49,2 %, n=63. Priorizar estos errores para revisión asistida; un promedio global no acredita clasificación fiable de todas las categorías.
**Diagnóstico regional: categoría del vídeo.** 10 regiones y 7.951 apariciones; soportes entre 339 y 955. Son filas de prueba, no vídeos únicos ni personas de esas nacionalidades.

Comparación descriptiva por balanced_accuracy: mayor es favorable. Regiones con al menos 100 filas: 10. El filtro no garantiza soporte de cada clase, independencia ni significación estadística.

Valor menos favorable entre las elegibles: JP, balanced_accuracy 0,4523, n=339. 

Clases con casos reales por región: entre 12 y 16, de 18 etiquetas declaradas. Exactitud equilibrada promedia recuperación sólo de clases presentes; esos conjuntos pueden diferir entre regiones.

evaluate fija las etiquetas con la unión de las clases del ajuste y las respuestas de cada región; macro-F1 incluye los ceros convencionales para etiquetas de esa unión sin casos ni predicciones en la región. No se interpreta como una media sólo de clases presentes. La tabla de soportes no contiene errores ni predicciones por clase.

La composición de objetivos, idiomas, canales y vídeos puede variar entre regiones; no se atribuye la diferencia al país. Un mismo vídeo puede aparecer en varias regiones dentro de prueba. Se describe una prueba interna, sin cambiar recetas ni elegir otro umbral con estos resultados.
**Coeficientes: categoría del vídeo.** 270 pesos exportados para 18 clases/salidas. Se conserva hasta 15 por clase/salida, ordenados por valor absoluto; no es el vector completo y la ausencia de un atributo no prueba peso cero.

text__ identifica TF-IDF del contenido combinado, no necesariamente una palabra del título; num__ usa variables imputadas, transformadas con log1p y escaladas; cat__ identifica indicadores de región. Los pesos no están en la escala de las columnas originales.

SVM lineal: el peso modifica la puntuación de decisión de la salida correspondiente. Un peso positivo eleva esa puntuación al aumentar ese atributo transformado, manteniendo los demás fijos; no es una probabilidad.

Ejemplos de hasta tres categorías con mayor soporte real entre las exportadas; el extracto completo queda en el detalle. Cada salida tiene su propio vector, no una importancia global común.

Salida Entertainment: `text__entertainment` = 2,1525; `text__trailers` = -1,7315; `text__ncast` = 1,7090.

En el extracto de esta salida: pesos positivos 14, negativos 1, ceros 0. Estos recuentos no describen todos los pesos del modelo.

Salida People & Blogs: `text__jeremstar` = 2,2131; `text__shin` = 1,9791; `text__nмоя` = 1,9410.

En el extracto de esta salida: pesos positivos 13, negativos 2, ceros 0. Estos recuentos no describen todos los pesos del modelo.

Salida Music: `text__lnk` = 3,1597; `text__album` = 2,4104; `text__lyrics` = 2,4081.

En el extracto de esta salida: pesos positivos 15, negativos 0, ceros 0. Estos recuentos no describen todos los pesos del modelo.

Un peso negativo reduce esa puntuación o respuesta transformada, no implica efecto perjudicial. La contribución de un caso requiere peso × valor transformado, más intercepto y el resto de atributos. TF-IDF se normaliza: añadir una palabra puede cambiar otras coordenadas; los atributos están relacionados.

No se comparan magnitudes entre escalas o modelos como importancia universal, ni se concluye causalidad por términos, canales o regiones.

La clasificación multiclase compara puntuaciones de varias salidas; elevar una no garantiza cambiar la etiqueta final.

**Lectura conjunta.** Los términos también pueden recoger nombres propios y plantillas recurrentes de creadores. El diagnóstico por canales disjuntos (apartado de calidad) comprueba si el rendimiento se mantiene al cambiar ese contexto; un coeficiente aislado no demuestra memorización.

No se reajusta el modelo mirando estas asociaciones.
**Curva de aprendizaje: categoría del vídeo.** 3 tamaños y 3 particiones. Eje X: media de filas efectivas de ajuste por partición, de 955,0 a 3.970,0; no es el número de vídeos ni el tamaño del ajuste final.

Validación macro-F1: 0,2926 → 0,4557; aumenta. Mayor es mejor: mejora numérica entre extremos. Pasos consecutivos favorables: 2 de 2; no se presupone una evolución monótona.

En el último tamaño, entrenamiento 0,9854 y validación 0,4557. Brecha orientada para que un valor positivo indique ventaja del entrenamiento: 0,5298. Es un diagnóstico condicionado a los datos y al protocolo, no una prueba aislada de sobreajuste.

Barras: media de validación ± desviación entre particiones, calculada con ddof=0; última desviación 0,0176. Las barras de error de las curvas de aprendizaje muestran la desviación entre particiones. No son intervalos de confianza ni intervalos de predicción individual para un vídeo. Las particiones comparten parte del entrenamiento y su número es reducido.

Para cada tamaño se ajustan clones sin estado aprendido, usando vídeos completos de la parte de ajuste; la validación de la partición permanece fija. El preprocesamiento se aprende dentro de la cadena de procesamiento. La receta ya fue seleccionada con estos datos de desarrollo: esta curva no es validación anidada independiente ni utiliza la reserva de prueba.

**Conclusión para categoría del vídeo.** Con **955,0** → **3.970,0** filas medias de ajuste por partición, el macro-F1 de validación pasa de **0,2926** a **0,4557**. Se observa mejora dentro del rango medido; merece contrastar una muestra mayor mediante validación de desarrollo, sin extrapolar una ganancia fuera de ese rango.

Macro-F1 se calcula con las clases presentes en respuestas o predicciones de cada evaluación; entrenamiento y validación pueden promediar conjuntos de clases distintos. La brecha no es una comparación perfectamente homogénea de etiquetas.

Estos 3 tamaños en la muestra de búsqueda no demuestran saturación ni predicen qué ocurriría al ampliar el corpus. Una mejora observada puede motivar otro experimento; no garantiza que más datos resuelvan el error ni justifica reajustar decisiones con la prueba.
Convergencia pendiente en búsquedas por familias: ['mlp_classifier']; avisos de convergencia resueltos tras reintento: []; otros avisos registrados: ['lda_classifier']; candidatos fallidos: []. Los otros avisos no se interpretan como falta de convergencia. Cada aviso agrupa configuraciones de una búsqueda y no identifica necesariamente al ganador.

### comments_disabled
Comparación condicionada por la representación: svm_rbf: SVD 30; mlp_classifier: SVD 60; lda_classifier: SVD 30. Los candidatos sin SVD usan la representación TF-IDF y sus metadatos. Se comparan recetas completas, por lo que las diferencias combinan algoritmo y representación. El refinamiento explora configuraciones de la familia ganadora dentro del presupuesto descrito.
Segunda fase: 13 configuraciones, incluida la referencia; CV inicial 0,5767, refinada 0,5949. Las rejillas y los avisos están en evaluation.json y se muestran en el cuaderno. La selección no consulta la prueba.
CV seleccionada: 0,5949 ± 0,0630; particiones [0,679366; 0,528312; 0,576943]. Parámetros: `{'estimator__C': 3.0, 'estimator__class_weight': None, 'features__text__max_features': 16000, 'features__text__min_df': 2, 'features__text__ngram_range': [1, 1], 'features__text__sublinear_tf': True, 'estimator__max_iter': 30000}`. El signo de la MAE en CV es negativo por convención de sklearn.
Umbral elegido únicamente con OOF en ajuste final ampliado (29.926 filas) para maximizar F2 (prioridad académica a la recuperación). Recuperación de positivos en prueba: 46,5 %, frente a 32,4 % con el umbral predeterminado; precisión 28,8 %, soporte 142. AP 0,419, referencia aleatoria por prevalencia 0,018.
F2 de prueba: 0,414 con el umbral OOF y 0,367 con el predeterminado. Falsos positivos: 163 frente a 13; verdaderos positivos: 66 frente a 46. El cambio de recuperación debe leerse junto con la precisión y F2; por sí solo no acredita una mejora de la utilidad.
La calibración inicial sobre la muestra de búsqueda obtuvo F2 de prueba 0,308; se conserva como diagnóstico. La fase final utiliza OOF del conjunto ampliado, sin seleccionar el umbral con las etiquetas de prueba.
El balance entre falsas alarmas y omisiones depende de la tarea: no se propone automatizar decisiones sobre creadores. Una aplicación de cribado necesitaría revisión humana, costes explícitos y nueva validación. Las curvas PR y ambas matrices permiten valorar el intercambio sin seleccionar de nuevo en prueba.
**Diagnóstico regional: comentarios desactivados.** 10 regiones y 7.951 apariciones; soportes entre 339 y 955. Son filas de prueba, no vídeos únicos ni personas de esas nacionalidades.

Comparación descriptiva por balanced_accuracy: mayor es favorable. Regiones con al menos 100 filas: 10. El filtro no garantiza soporte de cada clase, independencia ni significación estadística.

Valor menos favorable entre las elegibles: US, balanced_accuracy 0,4981, n=792. 

Clases con casos reales por región: entre 1 y 2, de 2 etiquetas declaradas. Exactitud equilibrada promedia recuperación sólo de clases presentes; esos conjuntos pueden diferir entre regiones.

Soporte positivo real por región: CA=12; DE=22; FR=18; GB=0; IN=19; JP=17; KR=8; MX=10; RU=23; US=13.

Sin positivos reales: GB. Una exactitud equilibrada alta o perfecta en esas regiones no demuestra detección de la clase 1; la recuperación de positivos no se puede estimar.

evaluate fija las etiquetas con la unión de las clases del ajuste y las respuestas de cada región; macro-F1 incluye los ceros convencionales para etiquetas de esa unión sin casos ni predicciones en la región. No se interpreta como una media sólo de clases presentes. La tabla de soportes no contiene errores ni predicciones por clase.

La composición de objetivos, idiomas, canales y vídeos puede variar entre regiones; no se atribuye la diferencia al país. Un mismo vídeo puede aparecer en varias regiones dentro de prueba. Se describe una prueba interna, sin cambiar recetas ni elegir otro umbral con estos resultados.
**Coeficientes: comentarios desactivados.** 15 pesos exportados para 1 clases/salidas. Se conserva hasta 15 por clase/salida, ordenados por valor absoluto; no es el vector completo y la ausencia de un atributo no prueba peso cero.

SVM lineal: el peso modifica la puntuación de decisión de la salida correspondiente. Un peso positivo eleva esa puntuación al aumentar ese atributo transformado, manteniendo los demás fijos; no es una probabilidad.

Salida 1: `text__grosse` = 2,4929; `text__mhonividente` = 2,4618; `text__rama` = 2,3940.

En el extracto de esta salida: pesos positivos 14, negativos 1, ceros 0. Estos recuentos no describen todos los pesos del modelo.

En clasificación binaria, la etiqueta depende de la puntuación o probabilidad y del umbral de decisión; cambiar una contribución no garantiza cruzar ese umbral.

**Lectura conjunta.** Los términos también pueden recoger nombres propios y plantillas recurrentes de creadores. El diagnóstico por canales disjuntos (apartado de calidad) comprueba si el rendimiento se mantiene al cambiar ese contexto; un coeficiente aislado no demuestra memorización.

No se reajusta el modelo mirando estas asociaciones.
**Curva de aprendizaje: comentarios desactivados.** 3 tamaños y 3 particiones. Eje X: media de filas efectivas de ajuste por partición, de 953,7 a 3.970,0; no es el número de vídeos ni el tamaño del ajuste final.

Validación macro-F1: 0,5049 → 0,5949; aumenta. Mayor es mejor: mejora numérica entre extremos. Pasos consecutivos favorables: 2 de 2; no se presupone una evolución monótona.

En el último tamaño, entrenamiento 0,9935 y validación 0,5949. Brecha orientada para que un valor positivo indique ventaja del entrenamiento: 0,3986. Es un diagnóstico condicionado a los datos y al protocolo, no una prueba aislada de sobreajuste.

Barras: media de validación ± desviación entre particiones, calculada con ddof=0; última desviación 0,0630. Las barras de error de las curvas de aprendizaje muestran la desviación entre particiones. No son intervalos de confianza ni intervalos de predicción individual para un vídeo. Las particiones comparten parte del entrenamiento y su número es reducido.

Para cada tamaño se ajustan clones sin estado aprendido, usando vídeos completos de la parte de ajuste; la validación de la partición permanece fija. El preprocesamiento se aprende dentro de la cadena de procesamiento. La receta ya fue seleccionada con estos datos de desarrollo: esta curva no es validación anidada independiente ni utiliza la reserva de prueba.

**Conclusión para comentarios desactivados.** Con **953,7** → **3.970,0** filas medias de ajuste por partición, el macro-F1 de validación pasa de **0,5049** a **0,5949**. Se observa mejora dentro del rango medido; merece contrastar una muestra mayor mediante validación de desarrollo, sin extrapolar una ganancia fuera de ese rango.

Macro-F1 se calcula con las clases presentes en respuestas o predicciones de cada evaluación; entrenamiento y validación pueden promediar conjuntos de clases distintos. La brecha no es una comparación perfectamente homogénea de etiquetas.

Las predicciones de los clones usan el umbral predeterminado del estimador; no incorporan el umbral OOF final. No se equipara esta línea con el rendimiento del modelo final calibrado.

Estos 3 tamaños en la muestra de búsqueda no demuestran saturación ni predicen qué ocurriría al ampliar el corpus. Una mejora observada puede motivar otro experimento; no garantiza que más datos resuelvan el error ni justifica reajustar decisiones con la prueba.
Convergencia pendiente en búsquedas por familias: []; avisos de convergencia resueltos tras reintento: ['mlp_classifier']; otros avisos registrados: []; candidatos fallidos: []. Los otros avisos no se interpretan como falta de convergencia. Cada aviso agrupa configuraciones de una búsqueda y no identifica necesariamente al ganador.

### ratings_disabled
Comparación condicionada por la representación: svm_rbf: SVD 30; mlp_classifier: SVD 60; lda_classifier: SVD 60. Los candidatos sin SVD usan la representación TF-IDF y sus metadatos. Se comparan recetas completas, por lo que las diferencias combinan algoritmo y representación. El refinamiento explora configuraciones de la familia ganadora dentro del presupuesto descrito.
Segunda fase: 13 configuraciones, incluida la referencia; CV inicial 0,5521, refinada 0,5555. Las rejillas y los avisos están en evaluation.json y se muestran en el cuaderno. La selección no consulta la prueba.
CV seleccionada: 0,5555 ± 0,0237; particiones [0,588542; 0,53418; 0,54377]. Parámetros: `{'estimator__C': 1.0, 'estimator__class_weight': 'balanced', 'features__text__max_features': 8000, 'features__text__min_df': 2, 'features__text__ngram_range': [1, 1], 'features__text__sublinear_tf': True, 'estimator__max_iter': 3000}`. El signo de la MAE en CV es negativo por convención de sklearn.
Umbral elegido únicamente con OOF en ajuste final ampliado (29.926 filas) para maximizar F2 (prioridad académica a la recuperación). Recuperación de positivos en prueba: 41,7 %, frente a 49,4 % con el umbral predeterminado; precisión 14,4 %, soporte 156. AP 0,172, referencia aleatoria por prevalencia 0,020.
F2 de prueba: 0,302 con el umbral OOF y 0,318 con el predeterminado. Falsos positivos: 386 frente a 509; verdaderos positivos: 65 frente a 77. El cambio de recuperación debe leerse junto con la precisión y F2; por sí solo no acredita una mejora de la utilidad.
El objetivo F2 optimizado en OOF no mejora en esta reserva de prueba. La variabilidad de la minoría y el traslado entre modelos OOF y ajuste completo limitan la calibración. Un siguiente experimento debería separar una partición de calibración dentro del entrenamiento o usar validación anidada, acordar costes y validarse externamente. Se conserva la política predefinida y se informa el fallo; no se elige de nuevo mirando esta prueba.
La calibración inicial sobre la muestra de búsqueda obtuvo F2 de prueba 0,292; se conserva como diagnóstico. La fase final utiliza OOF del conjunto ampliado, sin seleccionar el umbral con las etiquetas de prueba.
El balance entre falsas alarmas y omisiones depende de la tarea: no se propone automatizar decisiones sobre creadores. Una aplicación de cribado necesitaría revisión humana, costes explícitos y nueva validación. Las curvas PR y ambas matrices permiten valorar el intercambio sin seleccionar de nuevo en prueba.
**Diagnóstico regional: valoraciones desactivadas.** 10 regiones y 7.951 apariciones; soportes entre 339 y 955. Son filas de prueba, no vídeos únicos ni personas de esas nacionalidades.

Comparación descriptiva por balanced_accuracy: mayor es favorable. Regiones con al menos 100 filas: 10. El filtro no garantiza soporte de cada clase, independencia ni significación estadística.

Valor menos favorable entre las elegibles: CA, balanced_accuracy 0,4994, n=834. 

Clases con casos reales por región: entre 1 y 2, de 2 etiquetas declaradas. Exactitud equilibrada promedia recuperación sólo de clases presentes; esos conjuntos pueden diferir entre regiones.

Soporte positivo real por región: CA=6; DE=19; FR=14; GB=0; IN=7; JP=20; KR=44; MX=14; RU=28; US=4.

Sin positivos reales: GB. Una exactitud equilibrada alta o perfecta en esas regiones no demuestra detección de la clase 1; la recuperación de positivos no se puede estimar.

evaluate fija las etiquetas con la unión de las clases del ajuste y las respuestas de cada región; macro-F1 incluye los ceros convencionales para etiquetas de esa unión sin casos ni predicciones en la región. No se interpreta como una media sólo de clases presentes. La tabla de soportes no contiene errores ni predicciones por clase.

La composición de objetivos, idiomas, canales y vídeos puede variar entre regiones; no se atribuye la diferencia al país. Un mismo vídeo puede aparecer en varias regiones dentro de prueba. Se describe una prueba interna, sin cambiar recetas ni elegir otro umbral con estos resultados.
**Coeficientes: valoraciones desactivadas.** 15 pesos exportados para 1 clases/salidas. Se conserva hasta 15 por clase/salida, ordenados por valor absoluto; no es el vector completo y la ausencia de un atributo no prueba peso cero.

Regresión logística: los pesos modifican los log-odds de la clase positiva; la función logística los convierte en probabilidad estimada. El coeficiente no es un cambio directo de probabilidad. El umbral OOF decide después y no altera estos pesos.

Salida 1: `text__hilton` = 5,2491; `text__girona` = 4,9893; `text__лайк` = 4,7131.

En el extracto de esta salida: pesos positivos 14, negativos 1, ceros 0. Estos recuentos no describen todos los pesos del modelo.

En clasificación binaria, la etiqueta depende de la puntuación o probabilidad y del umbral de decisión; cambiar una contribución no garantiza cruzar ese umbral.

**Lectura conjunta.** Los términos también pueden recoger nombres propios y plantillas recurrentes de creadores. El diagnóstico por canales disjuntos (apartado de calidad) comprueba si el rendimiento se mantiene al cambiar ese contexto; un coeficiente aislado no demuestra memorización.

No se reajusta el modelo mirando estas asociaciones.
**Curva de aprendizaje: valoraciones desactivadas.** 3 tamaños y 3 particiones. Eje X: media de filas efectivas de ajuste por partición, de 950,7 a 3.970,0; no es el número de vídeos ni el tamaño del ajuste final.

Validación macro-F1: 0,5103 → 0,5555; aumenta. Mayor es mejor: mejora numérica entre extremos. Pasos consecutivos favorables: 2 de 2; no se presupone una evolución monótona.

En el último tamaño, entrenamiento 0,7752 y validación 0,5555. Brecha orientada para que un valor positivo indique ventaja del entrenamiento: 0,2197. Es un diagnóstico condicionado a los datos y al protocolo, no una prueba aislada de sobreajuste.

Barras: media de validación ± desviación entre particiones, calculada con ddof=0; última desviación 0,0237. Las barras de error de las curvas de aprendizaje muestran la desviación entre particiones. No son intervalos de confianza ni intervalos de predicción individual para un vídeo. Las particiones comparten parte del entrenamiento y su número es reducido.

Para cada tamaño se ajustan clones sin estado aprendido, usando vídeos completos de la parte de ajuste; la validación de la partición permanece fija. El preprocesamiento se aprende dentro de la cadena de procesamiento. La receta ya fue seleccionada con estos datos de desarrollo: esta curva no es validación anidada independiente ni utiliza la reserva de prueba.

**Conclusión para valoraciones desactivadas.** Con **950,7** → **3.970,0** filas medias de ajuste por partición, el macro-F1 de validación pasa de **0,5103** a **0,5555**. Se observa mejora dentro del rango medido; merece contrastar una muestra mayor mediante validación de desarrollo, sin extrapolar una ganancia fuera de ese rango.

Macro-F1 se calcula con las clases presentes en respuestas o predicciones de cada evaluación; entrenamiento y validación pueden promediar conjuntos de clases distintos. La brecha no es una comparación perfectamente homogénea de etiquetas.

Las predicciones de los clones usan el umbral predeterminado del estimador; no incorporan el umbral OOF final. No se equipara esta línea con el rendimiento del modelo final calibrado.

Estos 3 tamaños en la muestra de búsqueda no demuestran saturación ni predicen qué ocurriría al ampliar el corpus. Una mejora observada puede motivar otro experimento; no garantiza que más datos resuelvan el error ni justifica reajustar decisiones con la prueba.
Convergencia pendiente en búsquedas por familias: []; avisos de convergencia resueltos tras reintento: ['mlp_classifier']; otros avisos registrados: []; candidatos fallidos: []. Los otros avisos no se interpretan como falta de convergencia. Cada aviso agrupa configuraciones de una búsqueda y no identifica necesariamente al ganador.

### likes
Comparación condicionada por la representación: svr_rbf: SVD 30; svr_rbf_log_target: SVD 30; mlp_regressor: SVD 60; mlp_regressor_log_target: SVD 60. Los candidatos sin SVD usan la representación TF-IDF y sus metadatos. Se comparan recetas completas, por lo que las diferencias combinan algoritmo y representación. El refinamiento explora configuraciones de la familia ganadora dentro del presupuesto descrito.
Segunda fase: 13 configuraciones, incluida la referencia; CV inicial -9.666,6390, refinada -9.178,7625. Las rejillas y los avisos están en evaluation.json y se muestran en el cuaderno. La selección no consulta la prueba.
CV seleccionada: -9.178,7625 ± 1.390,8962; particiones [-10.300,150307; -10.017,624043; -7.218,513241]. Parámetros: `{'estimator__regressor__C': 0.1, 'features__text__max_features': 16000, 'features__text__min_df': 4, 'features__text__ngram_range': [1, 2], 'features__text__sublinear_tf': True, 'estimator__regressor__max_iter': 60000}`. El signo de la MAE en CV es negativo por convención de sklearn.
**Lectura de regresión: número de «me gusta».** 7.951 apariciones de prueba. MAE y RMSE están en la escala original: likes; no son errores en logaritmos ni porcentajes.

MAE 14.202,12: media de las distancias absolutas entre valor real y predicción. No es la mediana del error ni garantiza que cada caso quede dentro de ese margen.

RMSE 67.497,26: raíz de la media de los errores al cuadrado. Da más peso a errores grandes; su distancia respecto al MAE no permite contar cuántos extremos hay sin mirar los casos.

Frente a la referencia de mediana del ajuste, evaluada sobre las mismas filas (MAE 57.088,14), el MAE disminuye. Reducción absoluta referencia − modelo: 42.886,02; reducción relativa: 75,12 %. Un valor negativo indica empeoramiento; no es un porcentaje de aciertos.

R² 0,9568: 1 − SSE/SST, respecto al predictor constante igual a la media de los valores reales de esta prueba. Es un denominador matemático de evaluación; no se usa esa media para entrenar el modelo ni la referencia sencilla de mediana.

R² no es el porcentaje de predicciones correctas ni una garantía de error pequeño. Con un objetivo constante SST=0 y con menos de dos casos la definición habitual requiere tratamiento especial; estos valores agregados no permiten diagnosticar por sí solos esas situaciones.

Estas magnitudes no se comparan directamente entre likes y ratio porque sus unidades y escalas son distintas. Describen modelos ya fijados sobre una prueba interna ya examinada; no justifican reoptimizar con sus etiquetas.
La diferencia RMSE–MAE refleja sensibilidad a errores extremos; los 100 mayores errores se entregan con título, región y objetivo real. Un R² bajo o negativo sigue siendo un límite aunque la MAE mejore respecto a la mediana.
**Diagnóstico regional: número de «me gusta».** 10 regiones y 7.951 apariciones; soportes entre 339 y 955. Son filas de prueba, no vídeos únicos ni personas de esas nacionalidades.

Comparación descriptiva por mae: menor es favorable. Regiones con al menos 100 filas: 10. El filtro no garantiza soporte de cada clase, independencia ni significación estadística.

Valor menos favorable entre las elegibles: GB, mae 44.545,6479, n=858. 

MAE global se reconstruye ponderando los MAE regionales por filas; RMSE global requiere raíz de la media ponderada de RMSE². R² global no es el promedio de los R² regionales: las medias y dispersiones locales difieren. Un MAE regional alto puede coexistir con R² alto.

La composición de objetivos, idiomas, canales y vídeos puede variar entre regiones; no se atribuye la diferencia al país. Un mismo vídeo puede aparecer en varias regiones dentro de prueba. Se describe una prueba interna, sin cambiar recetas ni elegir otro umbral con estos resultados.
**Coeficientes: número de «me gusta».** 15 pesos exportados para 1 clases/salidas. Se conserva hasta 15 por clase/salida, ordenados por valor absoluto; no es el vector completo y la ausencia de un atributo no prueba peso cero.

Regresión con objetivo log1p: los pesos lineales actúan sobre log1p(y); expm1 devuelve la predicción a la escala original. Un coeficiente de 1 no significa sumar un like ni una unidad de ratio, y no se aplica un multiplicador sin considerar el cambio real del atributo transformado.

Salida likes: `text__news` = -1,6133; `text__bts` = 1,4146; `text__instagram` = 1,1868.

En el extracto de esta salida: pesos positivos 5, negativos 10, ceros 0. Estos recuentos no describen todos los pesos del modelo.

En regresión se estima una respuesta numérica, no una etiqueta de clase; su interpretación debe respetar las transformaciones del objetivo y la escala de salida.

No se reajusta el modelo mirando estas asociaciones.
**Curva de aprendizaje: número de «me gusta».** 3 tamaños y 3 particiones. Eje X: media de filas efectivas de ajuste por partición, de 958,3 a 3.970,0; no es el número de vídeos ni el tamaño del ajuste final.

Validación MAE: 9.655,7313 → 9.178,7625; disminuye. Menor es mejor: mejora numérica entre extremos. Pasos consecutivos favorables: 2 de 2; no se presupone una evolución monótona.

En el último tamaño, entrenamiento 3.539,9235 y validación 9.178,7625. Brecha orientada para que un valor positivo indique ventaja del entrenamiento: 5.638,8390. Es un diagnóstico condicionado a los datos y al protocolo, no una prueba aislada de sobreajuste.

Barras: media de validación ± desviación entre particiones, calculada con ddof=0; última desviación 1.390,8962. Las barras de error de las curvas de aprendizaje muestran la desviación entre particiones. No son intervalos de confianza ni intervalos de predicción individual para un vídeo. Las particiones comparten parte del entrenamiento y su número es reducido.

Para cada tamaño se ajustan clones sin estado aprendido, usando vídeos completos de la parte de ajuste; la validación de la partición permanece fija. El preprocesamiento se aprende dentro de la cadena de procesamiento. La receta ya fue seleccionada con estos datos de desarrollo: esta curva no es validación anidada independiente ni utiliza la reserva de prueba.

**Conclusión para número de «me gusta».** Con **958,3** → **3.970,0** filas medias de ajuste por partición, el MAE de validación pasa de **9.655,73** a **9.178,76**. Se observa mejora dentro del rango medido; merece contrastar una muestra mayor mediante validación de desarrollo, sin extrapolar una ganancia fuera de ese rango.

Estos 3 tamaños en la muestra de búsqueda no demuestran saturación ni predicen qué ocurriría al ampliar el corpus. Una mejora observada puede motivar otro experimento; no garantiza que más datos resuelvan el error ni justifica reajustar decisiones con la prueba.
Convergencia pendiente en búsquedas por familias: ['svr_log_target']; avisos de convergencia resueltos tras reintento: ['mlp_regressor_log_target']; otros avisos registrados: []; candidatos fallidos: []. Los otros avisos no se interpretan como falta de convergencia. Cada aviso agrupa configuraciones de una búsqueda y no identifica necesariamente al ganador.
Avisos de convergencia restantes en el refinamiento: ['Liblinear failed to converge, increase the number of iterations.']. Su atribución a una configuración concreta no queda demostrada por el resumen agregado.

### ratio
Comparación condicionada por la representación: svr_rbf: SVD 30; svr_rbf_log_target: SVD 30; mlp_regressor: SVD 60; mlp_regressor_log_target: SVD 60. Los candidatos sin SVD usan la representación TF-IDF y sus metadatos. Se comparan recetas completas, por lo que las diferencias combinan algoritmo y representación. El refinamiento explora configuraciones de la familia ganadora dentro del presupuesto descrito.
Segunda fase: 13 configuraciones, incluida la referencia; CV inicial -26,4568, refinada -26,4568. Las rejillas y los avisos están en evaluation.json y se muestran en el cuaderno. La selección no consulta la prueba.
CV seleccionada: -26,4568 ± 2,7475; particiones [-25,977629; -30,035706; -23,357036]. Parámetros: `{'estimator__regressor__alpha': 1.0, 'features__text__max_features': 8000, 'features__text__min_df': 2, 'features__text__ngram_range': [1, 1], 'features__text__sublinear_tf': True, 'estimator__regressor__max_iter': None}`. El signo de la MAE en CV es negativo por convención de sklearn.
**Lectura de regresión: proporción «me gusta/no me gusta».** 7.733 apariciones de prueba. MAE y RMSE están en la escala original: unidades del cociente likes/dislikes (no puntos porcentuales); no son errores en logaritmos ni porcentajes.

MAE 26,98: media de las distancias absolutas entre valor real y predicción. No es la mediana del error ni garantiza que cada caso quede dentro de ese margen.

RMSE 61,94: raíz de la media de los errores al cuadrado. Da más peso a errores grandes; su distancia respecto al MAE no permite contar cuántos extremos hay sin mirar los casos.

Frente a la referencia de mediana del ajuste, evaluada sobre las mismas filas (MAE 32,12), el MAE disminuye. Reducción absoluta referencia − modelo: 5,13; reducción relativa: 15,98 %. Un valor negativo indica empeoramiento; no es un porcentaje de aciertos.

R² 0,1324: 1 − SSE/SST, respecto al predictor constante igual a la media de los valores reales de esta prueba. Es un denominador matemático de evaluación; no se usa esa media para entrenar el modelo ni la referencia sencilla de mediana.

R² no es el porcentaje de predicciones correctas ni una garantía de error pequeño. Con un objetivo constante SST=0 y con menos de dos casos la definición habitual requiere tratamiento especial; estos valores agregados no permiten diagnosticar por sí solos esas situaciones.

Estas magnitudes no se comparan directamente entre likes y ratio porque sus unidades y escalas son distintas. Describen modelos ya fijados sobre una prueba interna ya examinada; no justifican reoptimizar con sus etiquetas.
La diferencia RMSE–MAE refleja sensibilidad a errores extremos; los 100 mayores errores se entregan con título, región y objetivo real. Un R² bajo o negativo sigue siendo un límite aunque la MAE mejore respecto a la mediana.
**Ablación en la proporción «me gusta/no me gusta».** Se retiran conjuntamente views y comment_count, manteniendo la familia, la receta seleccionada y las mismas particiones por vídeo. No se retiran likes y dislikes: ya están excluidos como predictores para evitar revelar directamente el cociente.
MAE medio con contadores 26,4568; sin ellos 26,7153. Con contadores el MAE es menor. Diferencia sin − con: 0,2585 unidades de ratio; positivo favorece incluirlos.
Reducción relativa sobre la variante sin contadores: 0,97 %. Es una diferencia de error, no porcentaje de aciertos ni prueba de significación.
Comparación emparejada: menor MAE con contadores en 2 de 3 particiones; mayor en 1; empate en 0. No se supone un beneficio uniforme.

| Partición | MAE con contadores | MAE sin contadores | Sin − con |
|---|---:|---:|---:|
| 1 | 25,9776 | 26,7623 | 0,7847 |
| 2 | 30,0357 | 29,7172 | -0,3185 |
| 3 | 23,3570 | 23,6664 | 0,3093 |

Los hiperparámetros se seleccionaron con contadores; la variante sin ellos no recibió otra búsqueda. La ablación es condicional a esa receta y no demuestra que todo modelo sin contadores sea peor. Las dos variables se retiran juntas: no se identifica su aportación individual.
Las particiones comparten parte del entrenamiento y su número es reducido; una diferencia media pequeña no acredita significación ni causalidad. Ambos ajustes reaprenden sus transformadores dentro de la partición; no se supone que todos los pesos o escalas permanezcan idénticos al quitar variables.
views y comment_count son señales contemporáneas. Esta comparación no convierte el ejercicio en predicción antes de publicar, no garantiza disponibilidad futura y no autoriza cambiar atributos tras observar la reserva de prueba. Se conserva el resultado del estudio de desarrollo.
**Diagnóstico regional: proporción «me gusta/no me gusta».** 10 regiones y 7.733 apariciones; soportes entre 309 y 926. Son filas de prueba, no vídeos únicos ni personas de esas nacionalidades.

Comparación descriptiva por mae: menor es favorable. Regiones con al menos 100 filas: 10. El filtro no garantiza soporte de cada clase, independencia ni significación estadística.

Valor menos favorable entre las elegibles: GB, mae 36,4311, n=858. 

MAE global se reconstruye ponderando los MAE regionales por filas; RMSE global requiere raíz de la media ponderada de RMSE². R² global no es el promedio de los R² regionales: las medias y dispersiones locales difieren. Un MAE regional alto puede coexistir con R² alto.

La composición de objetivos, idiomas, canales y vídeos puede variar entre regiones; no se atribuye la diferencia al país. Un mismo vídeo puede aparecer en varias regiones dentro de prueba. Se describe una prueba interna, sin cambiar recetas ni elegir otro umbral con estos resultados.
**Coeficientes: proporción «me gusta/no me gusta».** 15 pesos exportados para 1 clases/salidas. Se conserva hasta 15 por clase/salida, ordenados por valor absoluto; no es el vector completo y la ausencia de un atributo no prueba peso cero.

Regresión con objetivo log1p: los pesos lineales actúan sobre log1p(y); expm1 devuelve la predicción a la escala original. Un coeficiente de 1 no significa sumar un like ni una unidad de ratio, y no se aplica un multiplicador sin considerar el cambio real del atributo transformado.

Salida ratio: `text__guardian` = -3,1126; `text__unge` = -2,8828; `text__seventeen` = 2,7629.

En el extracto de esta salida: pesos positivos 10, negativos 5, ceros 0. Estos recuentos no describen todos los pesos del modelo.

En regresión se estima una respuesta numérica, no una etiqueta de clase; su interpretación debe respetar las transformaciones del objetivo y la escala de salida.

No se reajusta el modelo mirando estas asociaciones.
**Curva de aprendizaje: proporción «me gusta/no me gusta».** 3 tamaños y 3 particiones. Eje X: media de filas efectivas de ajuste por partición, de 934,0 a 3.884,7; no es el número de vídeos ni el tamaño del ajuste final.

Validación MAE: 27,4509 → 26,4568; disminuye. Menor es mejor: mejora numérica entre extremos. Pasos consecutivos favorables: 2 de 2; no se presupone una evolución monótona.

En el último tamaño, entrenamiento 13,0248 y validación 26,4568. Brecha orientada para que un valor positivo indique ventaja del entrenamiento: 13,4320. Es un diagnóstico condicionado a los datos y al protocolo, no una prueba aislada de sobreajuste.

Barras: media de validación ± desviación entre particiones, calculada con ddof=0; última desviación 2,7475. Las barras de error de las curvas de aprendizaje muestran la desviación entre particiones. No son intervalos de confianza ni intervalos de predicción individual para un vídeo. Las particiones comparten parte del entrenamiento y su número es reducido.

Para cada tamaño se ajustan clones sin estado aprendido, usando vídeos completos de la parte de ajuste; la validación de la partición permanece fija. El preprocesamiento se aprende dentro de la cadena de procesamiento. La receta ya fue seleccionada con estos datos de desarrollo: esta curva no es validación anidada independiente ni utiliza la reserva de prueba.

**Conclusión para proporción «me gusta/no me gusta».** Con **934,0** → **3.884,7** filas medias de ajuste por partición, el MAE de validación pasa de **27,45** a **26,46**. Se observa mejora dentro del rango medido; merece contrastar una muestra mayor mediante validación de desarrollo, sin extrapolar una ganancia fuera de ese rango.

Estos 3 tamaños en la muestra de búsqueda no demuestran saturación ni predicen qué ocurriría al ampliar el corpus. Una mejora observada puede motivar otro experimento; no garantiza que más datos resuelvan el error ni justifica reajustar decisiones con la prueba.
Convergencia pendiente en búsquedas por familias: []; avisos de convergencia resueltos tras reintento: ['svr_log_target', 'mlp_regressor']; otros avisos registrados: []; candidatos fallidos: []. Los otros avisos no se interpretan como falta de convergencia. Cada aviso agrupa configuraciones de una búsqueda y no identifica necesariamente al ganador.

## Segmentación y recuperación

Segmentación: kmeans, k=2, SVD=15, silueta=0,202. Se probaron dimensiones 15/30/60; KMeans/MiniBatch con k=2/3/4/5/6/8/10/12; Mean Shift con dos estimaciones de ancho de banda y GMM con componentes 2/3/5/8 y covarianzas diag/tied. Aglomerativo y DBSCAN se conservan como exploración transductiva separada. La selección inductiva utiliza silueta en los mismos 1.000 índices; se informa estabilidad por remuestreo condicionada a una representación SVD fija, sin elegir con las categorías editoriales.

Regla añadida después de observar una segmentación previa del 99,5 %: cada grupo debe contener al menos el 5 % de los vídeos y tres réplicas válidas de estabilidad con ARI medio >= 0,80; se maximiza silueta entre quienes cumplen. No se usan categorías editoriales ni etiquetas de prueba. Son límites metodológicos de esta revisión, no requisitos del profesor ni criterios fijados antes del primer estudio.

Distribución de tamaños: el grupo mayor reúne 51,4 % de los 6000 vídeos; el menor contiene 2914. Los tamaños deben valorarse junto con perfiles, estabilidad y el objetivo de negocio; silueta no valida utilidad por sí solo.

Silueta de la solución seleccionada: **0,202**. Estabilidad con SVD fijo: ARI medio **0,992**; 3 réplicas [0,996; 0,984; 0,996], rango 0,984–0,996. La media debe leerse junto con las réplicas: no acredita reproducción uniforme ni utilidad de los grupos. Al reajustar todo el preprocesamiento con la receta congelada, ARI medio **0,984**; réplicas [0,988; 0,988; 0,976]. El diagnóstico utiliza 5000 vídeos de entrenamiento y 1000 de auditoría interna, con 0 IDs compartidos. No es validación externa. Pocas réplicas limitan la interpretación; el balance se describe con los tamaños reales de los perfiles. Estabilidad, silueta y utilidad son conceptos distintos.

### Qué caracteriza a los grupos (6.000 vídeos únicos)

Los identificadores son arbitrarios. n cuenta vídeos, no apariciones regionales. Las medianas resumen el valor central: no son medias, sumas ni tasas.

**Grupo 0:** 3.086 vídeos (51,43 %). Medianas: 161.045,5 visualizaciones; 4.061,5 likes; 196,0 dislikes; 607,5 comentarios; 52,0 caracteres en el título. Categoría modal publicada: Entertainment, con 984 vídeos (31,89 % del grupo); es la categoría más frecuente. Región modal publicada: CA. Una moda no implica mayoría ni exclusividad.

**Grupo 1:** 2.914 vídeos (48,57 %). Medianas: 18.095,5 visualizaciones; 273,5 likes; 16,0 dislikes; 40,0 comentarios; 53,0 caracteres en el título. Categoría modal publicada: Entertainment, con 774 vídeos (26,56 % del grupo); es la categoría más frecuente. Región modal publicada: MX. Una moda no implica mayoría ni exclusividad.

La mayor mediana de visualizaciones es 8,90 veces la menor. Es una razón de medianas, no el cociente para cada vídeo ni una tasa de interacción.

Los perfiles describen esta muestra y la aparición retenida de cada vídeo; no representan su evolución temporal. likes se describe, pero no es entrada numérica de la segmentación. Título y región no se conservan en cluster_sample.csv: sus resúmenes publicados no se pueden reproducir sólo desde ese CSV. Una diferencia descriptiva de audiencia no demuestra causalidad, una separación puramente temática ni utilidad comercial.

### Relación de grupos con categorías editoriales

Se comparan 6.000 vídeos, 2 grupos y 16 categorías. Cada celda de la tabla de recuentos cuenta vídeos con ambas etiquetas. El CSV de proporciones contiene fracciones de 0 a 1; la tabla porcentual multiplica por 100. El denominador es el total del grupo (fila), no el total de la categoría (columna).

Grupo 0, categorías más frecuentes: Entertainment: 984 (31,89 %); People & Blogs: 470 (15,23 %); News & Politics: 348 (11,28 %). En caso de empate, el orden de presentación no significa una diferencia de frecuencia.

Grupo 1, categorías más frecuentes: Entertainment: 774 (26,56 %); People & Blogs: 663 (22,75 %); News & Politics: 392 (13,45 %). En caso de empate, el orden de presentación no significa una diferencia de frecuencia.

ARI frente a categorías: **0,0060**. Compara pares de vídeos y descuenta acuerdo esperado al azar; puede ser negativo. No es el ARI de estabilidad entre réplicas ni un porcentaje de aciertos.

NMI: **0,0109**. Mide información compartida normalizada (media aritmética de las entropías). No está ajustada por azar ni representa precisión o porcentaje de categorías recuperadas.

Las particiones no son equivalentes. Se informa la magnitud de ambas métricas sin convertir un umbral arbitrario en una prueba estadística de asociación.

Tener distinto número de grupos y categorías limita la equivalencia uno a uno, pero no invalida las métricas. Las categorías sirven como referencia imperfecta y no eligen el ganador; sí influyeron en la estratificación del conjunto anterior. La comparación es descriptiva sobre la muestra de ajuste, no validación externa ni prueba de utilidad comercial.

### Palabras y expresiones de los centros reconstruidos

Se contrastan 2 grupos y 8000 atributos textuales. Los términos proceden del vocabulario TF-IDF de título, tags y descripción combinados: también pueden ser expresiones de varias palabras. No identifican por sí solos el campo de origen.

**Grupo 0, 12 términos en orden descendente:** com (0,04733); https (0,04319); www (0,03715); http (0,03268); https www (0,02832); youtube (0,02514); the (0,02422); instagram (0,02270); twitter (0,02260); youtube com (0,02220); facebook (0,02188); to (0,02150).

**Grupo 1, 12 términos en orden descendente:** none (0,05212); de (0,03454); com (0,03058); https (0,02736); www (0,02662); 2018 (0,02260); la (0,02148); http (0,01896); en (0,01859); https www (0,01823); youtube (0,01771); youtube com (0,01534).

Términos presentes entre los destacados de todos los grupos: 7: com, http, https, https www, www, youtube, youtube com.

Aparece el término literal none: no es el objeto None de Python ni demuestra por sí solo que haya datos ausentes; puede ser un marcador del contenido original. No se le asigna significado temático sin revisar sus apariciones.

Se observan tokens compatibles con enlaces o plantillas: com, http, https, https www, www, youtube com. Es un indicio descriptivo, no un recuento de documentos con enlaces ni una demostración de su causa.

**Lectura del grupo 0:** términos destacados que no aparecen en las listas truncadas de los demás grupos: facebook, instagram, the, to, twitter. Son pistas del vocabulario y de plantillas del perfil, no palabras exclusivas de todos sus documentos ni etiquetas temáticas verificadas.

**Lectura del grupo 1:** términos destacados que no aparecen en las listas truncadas de los demás grupos: 2018, de, en, la, none. Son pistas del vocabulario y de plantillas del perfil, no palabras exclusivas de todos sus documentos ni etiquetas temáticas verificadas.

**Conclusión.** Los términos reconstruidos y sus diferencias de vocabulario aconsejan describir perfiles de contenido y formato, junto con audiencia y región. Las listas de términos no bastan para nombrar géneros: la correspondencia con las categorías editoriales se cuantifica por separado con ARI y NMI.

El cálculo multiplica centros en SVD por sus componentes para volver aproximadamente al espacio de atributos. No reconstruye documentos originales ni recupera información descartada. Los pesos son coordenadas aproximadas: no son frecuencias, probabilidades, porcentajes ni coeficientes causales; no tienen que sumar uno y pueden ser negativos.

Se muestran los mayores pesos dentro de cada grupo, no las palabras más exclusivas ni las diferencias respecto a otros centros. Las palabras compartidas y las plantillas limitan las etiquetas temáticas. No se afirma que sean categorías aprendidas. Eliminar enlaces, marcadores o palabras frecuentes sería otro preprocesamiento que habría que comparar con un protocolo declarado; no se cambia ahora mirando el resultado.


Los rangos de audiencia son relativos a la muestra y no segmentos comerciales validados. Comparar contingencia, ARI y NMI evita confundir una separación por popularidad con géneros editoriales. En DBSCAN silueta excluye ruido: consultar siempre cobertura, sin premiar una solución que descarta casi todo.

Recomendador: char_tfidf. Coincidencia de categoría top-5 en consultas de prueba disjuntas: 44,6 %; cobertura del catálogo 18,8 %; fracción media de canales distintos en cada lista 82,9 %.

La categoría es un proxy imperfecto: dos vídeos de la misma categoría pueden ser irrelevantes y dos de distinta categoría resultar útiles. La evaluación humana de la utilidad queda fuera de lo solicitado en el ejercicio y se propone como ampliación con valoraciones independientes de personas.

## Alcance y siguientes decisiones

Likes y ratio son estimaciones contemporáneas; no predicciones antes de publicar. Moderación excluye contadores que revelarían la respuesta. Las asociaciones de coeficientes y correlaciones no son causales. La repetición de canales puede favorecer el rendimiento; falta una validación externa por tiempo/canal.

Para una aplicación operativa: acordar costes de errores, obtener relevancia humana, ampliar la validación temporal y monitorizar cambios de idioma y región. Estas acciones se derivan de las limitaciones observadas.

Las comprobaciones por canal y de estabilidad se explican en `reports/quality_report.md`. Los diagnósticos de consultas y de escritura se conservan en los CSV. La interpretación de los grupos está en `reports/cluster_interpretation.md`.
