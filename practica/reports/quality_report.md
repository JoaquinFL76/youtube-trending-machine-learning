# Validaciones adicionales de calidad

Son comprobaciones **internas** de la misma fuente histórica. Se congelan las recetas y se separan componentes vídeo–nombre de canal en ajuste, calibración y prueba. Los nombres no son identificadores canónicos de creador.

Filas de ajuste, calibración y prueba: **[11.980; 3.953; 5.981]**. Solapamientos entre particiones: channel_component: [0; 0; 0]; video_id: [0; 0; 0]; channel_title: [0; 0; 0].

| Tarea | Métrica | Prueba por canal | Referencia | Entrenamiento |
|---|---|---:|---:|---:|
| category | macro_f1 | 0,383 | 0,025 | 0,961 |
| comments_disabled | macro_f1 | 0,380 | 0,495 | 0,406 |
| ratings_disabled | macro_f1 | 0,523 | 0,495 | 0,654 |
| likes | mae | 13.025,446 | 43.201,925 | 3.791,839 |
| ratio | mae | 29,438 | 31,262 | 13,349 |

## comments_disabled
F2 por canal: **0,153** con calibración separada; **0,010** con umbral predeterminado. Umbral elegido en calibración: **-1,152121**.

| Coste FN | Coste/1.000 predeterminado | Coste/1.000 calibrado |
|---:|---:|---:|
| 1 | 23,6 | 464,3 |
| 2 | 44,5 | 468,1 |
| 5 | 107,2 | 479,7 |
| 10 | 211,7 | 498,9 |
**Conclusión sobre costes.** El umbral calibrado cuesta más en **4 de 4** escenarios considerados. Optimizar F2 no equivale a minimizar el coste: aumentar la detección puede generar más falsas alarmas. No hay una política operativa avalada por estos escenarios; se conserva el umbral seleccionado en calibración y se informa el balance observado. Una política futura requeriría acordar costes y optimizarlos en desarrollo, no en esta prueba.

## ratings_disabled
F2 por canal: **0,143** con calibración separada; **0,108** con umbral predeterminado. Umbral elegido en calibración: **0,327748**.

| Coste FN | Coste/1.000 predeterminado | Coste/1.000 calibrado |
|---:|---:|---:|
| 1 | 55,3 | 90,8 |
| 2 | 73,9 | 107,3 |
| 5 | 129,6 | 157,0 |
| 10 | 222,4 | 239,8 |
**Conclusión sobre costes.** El umbral calibrado cuesta más en **4 de 4** escenarios considerados. Optimizar F2 no equivale a minimizar el coste: aumentar la detección puede generar más falsas alarmas. No hay una política operativa avalada por estos escenarios; se conserva el umbral seleccionado en calibración y se informa el balance observado. Una política futura requeriría acordar costes y optimizarlos en desarrollo, no en esta prueba.

## Estabilidad de toda la segmentación
ARI medio **0,984**; réplicas [0,988024; 0,988024; 0,97612]. En cada remuestreo se reajustan TF-IDF, imputación, escala, SVD y centroides. Los **1.000** vídeos de auditoría quedan fuera de todos esos ajustes.
La receta está fijada. La estabilidad observada indica reproducibilidad de la partición bajo estos remuestreos internos; no valida su utilidad temática ni una nueva selección de receta.

## Límites y posibles ampliaciones
Una fuente externa y los identificadores canónicos de canal permitirían comprobar la generalización. El diagnóstico Unicode describe sistemas de escritura, no identifica idiomas.

La evaluación humana de las recomendaciones queda fuera de lo solicitado en el ejercicio. Se propone como ampliación: requeriría un protocolo y valoraciones independientes de personas.
