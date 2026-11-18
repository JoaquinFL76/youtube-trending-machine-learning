# Resultados y decisiones: lectura principal

La pregunta guía del cuaderno se formuló a partir del enunciado. Los retos pedidos abarcan categoría, número de likes, segmentación y recomendación. Los retos opcionales para subir nota abarcan la deshabilitación de comentarios o valoraciones y la proporción de likes y dislikes. La práctica aborda comentarios y valoraciones por separado. El [glosario de la práctica](../GUIA_DE_CONSULTA.md#glosario) explica las familias, las métricas y los términos del protocolo.

| Objetivo | Enunciado | Modelo elegido | Métrica de prueba | Referencia |
|---|---|---|---:|---:|
| category | Pedido | svm | macro-F1: 0,527 | 0,025 |
| comments_disabled | Opcional para subir nota | svm | macro-F1: 0,670 | 0,495 |
| ratings_disabled | Opcional para subir nota | logistic | macro-F1: 0,591 | 0,495 |
| likes | Pedido | svr_log_target | MAE: 14.202,119 | 57.088,142 |
| ratio | Opcional para subir nota | ridge_log_target | MAE: 26,984 | 32,116 |
| grupo | Pedido | kmeans, k=2, SVD=15 | Silueta: 0,2022 | No es clasificación editorial |
| recommend | Pedido | char_tfidf | Coincidencia editorial proxy: 44,60 % | Azar esperado: 14,32 % |

Macro-F1: mayor es mejor. MAE: menor es mejor, en unidades del objetivo. Cada familia se eligió por validación cruzada y después se refinó; los clasificadores binarios ajustan el umbral para F2 con predicciones OOF. La prueba no selecciona recetas.

## Decisiones que condicionan la interpretación

Categoría y moderación utilizan contenido previo a las interacciones. Likes y ratio son estimaciones contemporáneas con señales declaradas; no predicciones garantizadas antes de publicar.

Segmentación maximiza silueta entre soluciones que cumplen tamaño y estabilidad. La regla se incorporó tras el primer estudio y se declara post hoc. Los grupos no equivalen a géneros ni a segmentos comerciales validados.

La recomendación se eligió en validación y se comprobó con un catálogo de 6.000 vídeos; el peso final se reajustó después con 10.000. Las métricas no evalúan independientemente ese catálogo ampliado.

## Límites y resultados desfavorables

**Comentarios desactivados.** Macro-F1 de prueba 0,670, referencia 0,495. Detecta 66 de 142 positivos reales: recuperación 46,48 %; precisión positiva 28,82 %. No son porcentajes de exactitud global.

**Valoraciones desactivadas.** Macro-F1 de prueba 0,591, referencia 0,495. Detecta 65 de 156 positivos reales: recuperación 41,67 %; precisión positiva 14,41 %. No son porcentajes de exactitud global.

**Proporción likes/dislikes.** MAE de prueba 26,984 frente a 32,116 de referencia; R² 0,132. Es una estimación en la escala declarada, no garantía de precisión para cada vídeo.

Comentarios por canales disjuntos: macro-F1 0,380 frente a 0,495 de referencia. Es un diagnóstico reajustado, distinto de evaluar los pesos finales.

Valoraciones: F2 con umbral OOF 0,302; con umbral predeterminado 0,318. Se informa la comparación y se conserva la política elegida en entrenamiento.

La prueba interna se consultó durante el desarrollo; no hay validación externa prospectiva. Los intervalos son condicionales al ajuste y no incluyen incertidumbre de selección. El alcance de cada diagnóstico se explica en su apartado. La coincidencia editorial mide similitud de categoría. La evaluación humana de la utilidad de las recomendaciones se propone como ampliación y queda fuera de lo solicitado en el ejercicio.


**Distribución regional de los grupos.** Porcentajes dentro de cada grupo:

Grupo 0: CA, 666 vídeos (21,58 %); RU, 499 vídeos (16,17 %).

Grupo 1: MX, 610 vídeos (20,93 %); FR, 498 vídeos (17,09 %).

Dentro de cada región, el grupo 0 reúne 84,73 % de CA; 73,91 % de GB; 31,49 % de JP. El denominador ahora es el total de vídeos de esa región, no el tamaño del grupo.

Audiencia mediana por grupo: 0: 161.045,5 visualizaciones; 1: 18.095,5 visualizaciones.

Las medianas de audiencia difieren entre grupos. La distribución de grupos varía entre regiones. **Conclusión:** los grupos ofrecen perfiles distintos de audiencia y composición regional; no se interpretan como categorías temáticas. Los términos reconstruidos permiten contrastar también plantillas e indicios lingüísticos. Estos cruces describen asociación y no miden qué atributo causa o domina la separación; para eso haría falta una ablación específica.

Región es un atributo del modelo y las apariciones retenidas dependen del muestreo. Este cruce describe asociación en la muestra, no causa de la agrupación ni segmentos comerciales validados.

### Dónde se concentra el error del número de «me gusta»

**Conclusión.** Las **100** apariciones de **3** vídeos concentran **48,10 %** del error absoluto y **89,53 %** del cuadrático. Un conjunto reducido de identidades pesa mucho en las métricas por aparición; el error cuadrático es especialmente sensible a esos casos. Conviene comprobar también métricas por vídeo en una futura evaluación, sin retirar ahora los errores difíciles de la prueba.

### Qué cambia al separar canales

**Categoría del vídeo:** macro-F1 por vídeo **0,527**; por canal **0,383**; referencia por canal **0,025**. El resultado por canal supera su referencia.

**Comentarios desactivados:** macro-F1 por vídeo **0,670**; por canal **0,380**; referencia por canal **0,495**. El resultado por canal queda por debajo de su referencia.

**Valoraciones desactivadas:** macro-F1 por vídeo **0,591**; por canal **0,523**; referencia por canal **0,495**. El resultado por canal supera su referencia.

**Conclusión.** El rendimiento de la partición por vídeo no basta para acreditar generalización a otros canales. La caída observada es compatible con que parte de las asociaciones textuales dependa de creadores, nombres propios o plantillas recurrentes. Se relaciona así con los coeficientes exportados, sin demostrar memorización ni atribuirles toda la diferencia: cambian también muestras y calibración. Para una aplicación a canales nuevos habría que usar esa separación en la validación de desarrollo y comprobarla después con una prueba independiente.

Correspondencia editorial de la segmentación: ARI 0,0060 y NMI 0,0109. Comparan grupos y categorías, no estabilidad entre réplicas.

**Alcance de la cobertura.** Con 300 consultas y hasta cinco vecinos, el techo teórico es 1.500 candidatos distintos: 25,00 % del catálogo. Con las listas realmente devueltas, el techo es 1.495: 24,92 %. Se recuperaron 1.130 distintos (75,59 % de ese último techo). Son límites combinatorios; no son objetivos de calidad ni demuestran que recuperar más candidatos aporte relevancia.

## Consulta del detalle

El cuaderno conserva gráficos, interpretación y tablas desplegables junto a cada objetivo. [Informe detallado](results_details.md): refinamiento, métricas, coeficientes, regiones y limitaciones. [Datos](data_findings.md), [calidad](quality_report.md), [temario](temario_coverage.md).
