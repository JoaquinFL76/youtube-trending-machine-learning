# Cobertura del módulo 6

El estado distingue experimentos ejecutados de ampliaciones no incorporadas. Las fuentes son las presentaciones y notebooks del módulo y la explicación de la práctica; no se afirma haber probado todas las técnicas existentes.

| Bloque | Estado | Evidencia o justificación |
|---|---|---|
| Regresión lineal y logística | Probado | Ridge, LogisticRegression y variantes log1p; regularización por CV; evaluation.json: logistic, ridge |
| Perceptrón multicapa | Probado | MLPClassifier/Regressor con SVD y escalado latente dentro de CV; iteraciones y avisos en convergence_review.json; evaluation.json: mlp_classifier, mlp_regressor |
| Árboles y Random Forest | Probado | Clasificación/regresión; mismos folds y referencias; evaluation.json: forest_classifier, forest_regressor, tree_classifier, tree_regressor |
| KNN y distancias | Probado | Vecinos con coseno y recuperación de contenido; evaluation.json: knn_classifier, knn_regressor |
| SVM lineal y kernel RBF | Probado | LinearSVC/LinearSVR frente a SVC/SVR RBF; SVD y escalado latente dentro de cada fold; evaluation.json: svm, svm_rbf, svr, svr_rbf |
| PCA y t-SNE | Probado | Visualización descriptiva; no validan por sí mismas los clusters; cuaderno, sección 8: PCA y t-SNE; gráfico regenerable |
| TruncatedSVD | Probado | Texto disperso; dimensiones comparadas y preprocesamiento persistido; evaluation.json/clustering: 86 resultados con dimensiones y silhouette |
| LDA supervisado | Probado | Clasificación con SVD de 30/60 dimensiones dentro de cada fold y covarianza con/sin regularización; evaluation.json: lda_classifier |
| KMeans, MiniBatch, Mean Shift y GMM | Probado | Mismos datos e índices; bandwidth/componentes/covarianza; selección sin categorías; evaluation.json: gmm, kmeans, meanshift, minibatch |
| DBSCAN y aglomerativo | Probado como exploración | Cobertura y ARI/NMI; ajuste transductivo separado de los candidatos inductivos; evaluation.json: agglomerative, dbscan |
| Redes bayesianas | No incorporado como modelo general | MultinomialNB es una referencia restringida; un DAG requeriría justificar estructura y discretización dentro de CV |
| HMM | No incorporado | Exigiría formular estados/secuencias y validación temporal; objetivos actuales contemporáneos |
| XGBoost | Probado | Árboles por gradiente en CPU para clasificación y regresión con TF-IDF y metadatos. Regresión directa y log1p; evaluation.json: xgb_classifier, xgb_regressor, xgb_regressor_log_target |
| IA generativa, RAG y agentes | Contenido optativo no incorporado | No necesario para los objetivos del enunciado; sin servicios externos |

El [glosario de la práctica](../GUIA_DE_CONSULTA.md#glosario) explica cada técnica, incluso las que quedan fuera de la comparación.

Los retos opcionales se presentan como ampliaciones del problema. La matriz documenta el alcance metodológico y enlaza la cobertura con los experimentos y sus limitaciones. Fuentes: presentaciones 1–8 y notebooks 1–10 del módulo; explicación de la práctica, 00:19:18–00:22:43.

Los avisos de cada búsqueda pueden proceder de configuraciones descartadas; no identifican necesariamente al ganador. final_fit en convergence_review.json registra por separado los avisos del ajuste final de cada modelo. La ausencia de aviso no demuestra un óptimo global.

AIC/BIC se informan sólo para GMM y son comparables entre configuraciones de la misma representación y muestra, no entre familias o dimensiones SVD distintas.
