# Guía de consulta de la práctica

[Volver a la ruta breve del README](README.md#ruta-breve-para-corregir).

## Incidencias durante la corrección

Ante un fallo, leer primero el mensaje de la consola del BAT. Estas dos comprobaciones no entrenan, no cambian pesos ni instalan programas:

```powershell
.\ejecutar_practica.bat --comprobar-entorno
.\ejecutar_practica.bat --diagnostico-visual
```

La primera comprueba el Python y las dependencias principales; la segunda identifica por separado el **visor Jupyter** y el **kernel de cálculo**, con sus versiones. Es normal que el visor de Anaconda utilice otro Python: las celdas deben ejecutarse con **Python 3.12 · Práctica ML**, asociado a `.venv` o al intérprete de la práctica seleccionado por el BAT.

| Síntoma o mensaje | Causa probable | Qué hacer |
|---|---|---|
| Falta Python, no arranca o es de 32 bits. | El intérprete elegido no es adecuado o la `.venv` se ha movido. | Consulta el [paso 1 de la preparación](README.md#paso-1-disponer-de-cpython-312-de-64-bits), prepara CPython 3.12 de 64 bits y crea el entorno en esta carpeta. La primera línea del BAT indica qué intérprete eligió. |
| Faltan paquetes, el BAT indica versiones distintas o aparece `DLL load failed`. | El entorno está incompleto o no coincide con `requirements-lock.txt`. | Ejecuta la orden de instalación que muestra el BAT en el intérprete indicado y luego `pip check`. Si una biblioteca sigue sin importar, recrea el entorno de 64 bits. Para una DLL de ejecución de Visual C++ ausente, consulta el [redistribuible oficial de Microsoft](https://learn.microsoft.com/cpp/windows/latest-supported-vc-redist). |
| `No matching distribution found`, error de proxy o `No module named pip` al preparar el entorno. | Versión o arquitectura incompatible, red restringida o entorno sin pip. | Comprueba que Python sea 3.12 de 64 bits y que el equipo pueda acceder al índice de paquetes. Para una `.venv` de CPython sin pip, usa `python -m ensurepip --upgrade` desde su intérprete. En Conda, incluye `pip` al crear el entorno. |
| El dataset se informa como ZIP inválido o incompleto. | Se renombró un archivo distinto, se descargó incompleto o se extrajo el contenido. | Vuelve a descargar `youtube-new.zip` del enunciado y guárdalo como `dataset.zip` junto al BAT, todavía comprimido. |
| No se encuentra JupyterLab o falla `ML_JUPYTER_PYTHON`. | Anaconda está en una ruta distinta, o se ha indicado un Python sin JupyterLab. | Indicar el `python.exe` de la instalación existente mediante `ML_JUPYTER_PYTHON`, como en el [ejemplo de repaso visual](README.md#repaso-visual-ejecutar-y-comprender-una-celda-cada-vez), y repetir el diagnóstico. El BAT no reinstala Jupyter. |
| Se abre Anaconda, pero aparece otro kernel o faltan `sklearn`, `pandas`, etc. | Se está calculando con el Python de Anaconda en vez del entorno preparado de la práctica. | Elegir **Kernel → Change Kernel → Python 3.12 · Práctica ML**. Si no aparece, cerrar esa sesión y abrir la opción 4 del BAT, que registra el kernel dentro de su copia. |
| `Kernel died`, `kernel_info`, reinicios o desconexión. | Puede haber un fallo del proceso, un cliente de Jupyter incompatible, un kernel equivocado o falta de memoria. El mensaje por sí solo no demuestra que los pesos estén mal. | Guardar el mensaje completo de la consola y ejecutar el diagnóstico. Abrir de nuevo desde la opción 4 con el visor localizado por el BAT. Comprobar el kernel y cerrar ejecuciones paralelas. No actualizar en bloque las bibliotecas ni forzar las huellas. |
| El navegador no se abre, o una pestaña anterior ya no conecta. | El navegador predeterminado no se ha iniciado, el servidor se cerró o se está usando una URL de otra sesión. | Mantener abierta la consola y abrir la URL local completa que muestra ese arranque. Cada sesión puede usar un puerto distinto. Si la consola terminó, abrir de nuevo la opción 4. |
| Puerto ocupado o `Address already in use`. | Hay otro servidor utilizando ese puerto. | Jupyter prueba otros puertos; utilizar la URL que finalmente indique la consola. Si termina con error, cerrar únicamente la sesión de Jupyter que se reconoce como propia y volver a intentarlo. |
| La celda muestra `[*]`. | Sigue calculando o está esperando una respuesta; no equivale automáticamente a un error. | Esperar y observar la consola. No lanzar otra copia de la práctica a la vez. Si se quiere detener, usar el botón **■ Interrupt the kernel** y revisar el mensaje antes de continuar. |
| `NameError` al ejecutar una celda intermedia. | No se han ejecutado sus celdas anteriores o se ha reiniciado el kernel. | Reiniciar el kernel y ejecutar de arriba abajo, una celda cada vez. Las variables se comparten dentro de esa sesión. |
| No hay gráficos al abrir la copia de estudio. | Sus salidas empiezan vacías expresamente. | Ejecutar la celda correspondiente y buscar su salida debajo del código. Los gráficos aparecen en el cuaderno del navegador, no dentro de la consola negra del BAT. |
| Falta el CSV procesado, su auditoría o `dataset.zip`. | El ZIP académico no incluye el dataset completo. | Colocar el dataset del enunciado como `dataset.zip` junto al BAT. El repaso prepara los datos dentro de su copia si aún no existe el CSV; si existe sin auditoría, verificar los datos desde la opción 1. |
| Huellas incompatibles o versiones de pesos incompatibles. | Han cambiado datos, fuentes, configuración o versiones respecto a los modelos guardados. | Restaurar el conjunto coherente de la entrega y el entorno previsto. No editar huellas ni aceptar la reutilización por fuerza. Un experimento nuevo con la opción 2 es un proceso diferente y debe hacerse en una copia de trabajo. |
| Durante el entrenamiento aparece `Unable to cache to disk` y el aviso contiene una ruta muy larga. | Windows puede limitar la longitud de los archivos temporales de validación cruzada. | Extraer la práctica en una carpeta con una ruta más corta, como `D:/ML`, y repetir la opción 2. Puede utilizarse cualquier unidad con espacio y permiso de escritura. |
| Acceso denegado o falta de espacio al crear la copia. | La carpeta no permite escritura o la unidad no tiene espacio suficiente. | Extraer la práctica completa en una carpeta con escritura y espacio disponible. Cada sesión visual conserva sus propias copias de datos y modelos. No ejecutar desde dentro del ZIP comprimido. |
| `train.conf` no contiene las secciones o rutas esperadas. | La configuración se ha modificado o está incompleta. | Restaurar `train.conf` de la entrega. El modo de estudio exige rutas relativas internas para que sus escrituras se queden en la copia. |
| Anaconda Assistant pide una cuenta. | Es un panel adicional de Anaconda, independiente del cuaderno. | Se puede cerrar ese panel y seguir trabajando en el cuaderno. No es necesario iniciar sesión en ese asistente para ejecutar la práctica local. |
| Aparecen avisos de transporte TCP o del bucle de eventos. | Son avisos de la infraestructura Jupyter en Windows; su presencia no decide si una celda terminó bien. | Comprobar que la celda finaliza, muestra sus salidas y no genera una excepción. El visor de este modo escucha en `127.0.0.1`. Si la celda no responde, seguir el diagnóstico de kernel anterior; no ocultar el fallo ni dar por válida la ejecución. |

Si aparece otra incidencia, conservar **el mensaje completo**, la salida de `--diagnostico-visual`, la opción elegida y la celda donde ocurrió. Esta guía recoge escenarios previstos y comprobados cuando corresponde; no presupone que cualquier fallo posible sea recuperable automáticamente. La documentación de uso de las celdas está en [JupyterLab](https://jupyterlab.readthedocs.io/en/stable/user/notebook.html).

## Glosario

### Datos y atributos

| Término | Qué significa y para qué se usa |
|---|---|
| EDA | Análisis exploratorio de datos. Describe la composición, las ausencias, las distribuciones y las asociaciones antes de ajustar modelos. |
| Observación, vídeo único y soporte | Una observación es una aparición regional de un vídeo en tendencias. Un vídeo único se cuenta por identificador. El soporte es el número de ejemplos reales de una clase en el conjunto evaluado. |
| CSV y JSON | CSV guarda las apariciones en una tabla. JSON aporta el catálogo regional que traduce los identificadores de categoría a nombres. |
| Atributo, etiqueta y objetivo | Un atributo es una entrada del modelo. La etiqueta es la categoría o estado real de una fila. El objetivo es el resultado que se quiere predecir, también cuando es un número. |
| Valor ausente e imputación | Un dato ausente se identifica como NaN o vacío según la columna. Cuando corresponde, el transformador sustituye los valores ausentes con una regla aprendida en entrenamiento. |
| Mediana, percentil y cola larga | La mediana divide los datos ordenados por la mitad. Un percentil marca el valor por debajo del cual queda aproximadamente una fracción dada. Los percentiles altos muestran si unas pocas apariciones concentran magnitudes extremas. |
| Spearman y diagrama de caja | Spearman compara rangos y describe asociaciones crecientes o decrecientes. El diagrama de caja resume mediana, cuartiles y valores alejados para ver distribuciones con distinta dispersión. |
| TF-IDF | Frecuencia de término ponderada por la frecuencia inversa en documentos. Representa el contenido y da menos peso a términos muy comunes en el corpus. |
| Token, n-grama y vocabulario | Un token es una unidad de texto. Un n-grama reúne unidades consecutivas. El vocabulario enumera las entradas de texto que el transformador aprendió de sus datos de ajuste. |
| Escalado y codificación categórica | El escalado hace comparables las magnitudes numéricas en la representación. La codificación transforma una categoría como región en indicadores que el modelo puede usar. |
| SVD truncada | Descomposición en valores singulares que comprime una representación, conservando algunos componentes. Se aplica al texto disperso antes de ciertas familias y al formar el espacio de la segmentación. |
| Funciones de NumPy: log1p y expm1 | log1p transforma cantidades, incluidos los ceros, mediante logaritmo de uno más el valor. Comprime colas largas. expm1 invierte la transformación y devuelve las predicciones a las unidades originales. |

### Selección y evaluación

| Término | Qué significa y para qué se usa |
|---|---|
| Aprendizaje supervisado | Ajuste con respuestas conocidas. La clasificación predice etiquetas y la regresión estima cantidades. |
| Aprendizaje no supervisado | Busca estructura en los atributos. La segmentación forma grupos sin entrenar con las categorías editoriales como objetivo. |
| Partición, reserva de prueba y partición | Una partición divide datos por función. La reserva de prueba es la parte reservada para prueba. Una partición es una división de la validación cruzada que alterna el papel de ajuste y validación. |
| Validación cruzada agrupada y estratificación | Se ajusta y valida en varios particiones y se promedian sus puntuaciones. El grupo impide dividir un mismo vídeo entre ajuste y validación. La estratificación aproxima la distribución de clases cuando es posible. |
| Fuga de información | Ocurre cuando el ajuste dispone de información que debería quedar reservada. Se previene agrupando por vídeo y aprendiendo las transformaciones dentro de cada partición. |
| Cadena de procesamiento | Secuencia de transformadores y estimador. Conserva los pasos de preparación y predicción y permite ajustarlos juntos dentro de cada partición. |
| Hiperparámetro, rejilla y refinamiento | Un hiperparámetro se fija antes del ajuste. Una rejilla enumera configuraciones para compararlas. El refinamiento explora otras configuraciones de la familia que ganó en validación. |
| Referencia sencilla | Predicción sencilla con la que se compara el modelo en las mismas filas. Se usa, por ejemplo, la clase mayoritaria o la mediana del ajuste. |
| Desbalance y ponderación de clases | Hay desbalance cuando algunas clases son mucho menos frecuentes. La ponderación aumenta o reduce cuánto cuentan sus ejemplos durante el ajuste. |
| OOF, umbral y calibración del punto de decisión | OOF significa puntuaciones obtenidas fuera de la parte de ajuste de cada partición. Cada puntuación procede de un modelo que no vio el vídeo en su ajuste. El umbral convierte esa puntuación en una etiqueta y se escoge con las puntuaciones OOF. |
| Sobreajuste y curva de aprendizaje | Hay sobreajuste si el modelo se adapta al ajuste y pierde rendimiento en ejemplos reservados. La curva compara ambas puntuaciones a distintos tamaños de entrenamiento. |
| Ablación | Se retiran atributos concretos y se repite la comparación con las mismas particiones. En el ratio se usa para cuantificar el aporte de visualizaciones y comentarios. |
| Convergencia | Indica si el procedimiento de ajuste alcanzó su criterio de parada. Se registran avisos y reintentos para interpretar una búsqueda con ajustes incompletos. |
| IC del 95 % y remuestreo con reemplazo | El intervalo resume incertidumbre condicional a un modelo ajustado. El remuestreo con reemplazo selecciona vídeos permitiendo repeticiones con reemplazo para estimar cómo varía la métrica. |

### Familias supervisadas

| Término | Qué significa y para qué se usa |
|---|---|
| KNN | K vecinos más próximos. Predice a partir de los ejemplos cercanos. Se compara como clasificador y regresor, y la búsqueda de vecinos se usa también en recomendación. |
| Regresión logística y Ridge | La logística estima probabilidades de clase con una regla lineal. Ridge estima cantidades y penaliza coeficientes grandes para estabilizar el ajuste. |
| SVM lineal, SVC, SVR y RBF | Una máquina de vectores soporte usa márgenes. SVC clasifica y SVR estima valores. RBF es un kernel que permite relaciones no lineales basadas en cercanía. |
| Árbol, Random Forest y Extra Trees | Un árbol divide ejemplos mediante reglas sucesivas. Los bosques combinan árboles para reducir variabilidad. Extra Trees usa más aleatoriedad en los cortes. |
| Naive Bayes | Clasificador probabilístico que aproxima los atributos como independientes dentro de cada clase. Sirve de comparación rápida con el texto representado por frecuencias. |
| Covarianza y regularización | La covarianza resume cómo varían juntos dos atributos. Regularizar modera estimaciones inestables. LDA compara distintas formas de estimar esa relación. |
| SGD | Descenso de gradiente estocástico. Actualiza un modelo con pequeñas porciones de los datos y sirve como comparación lineal eficiente. |
| MLP | Perceptrón multicapa. Es una red neuronal con capas intermedias que permite relaciones no lineales. Aquí se prueba sobre una representación comprimida. |
| LDA | Análisis discriminante lineal supervisado. Compara las clases usando su centro y una covarianza común. Aquí se ensaya para clasificación con SVD y con o sin regularización de esa covarianza. |
| XGBoost | Biblioteca de árboles potenciados por gradiente. Añade árboles para corregir los errores de los anteriores. Compite en clasificación y regresión sobre las mismas particiones que las otras familias. |
| Redes bayesianas y HMM | Una red bayesiana expresa dependencias probabilísticas en un grafo. Un modelo oculto de Márkov, o HMM, describe secuencias con estados latentes. La comparación principal registra ambos como enfoques fuera del alcance de estos objetivos. |
| RAG | Generación aumentada mediante recuperación. Un sistema generativo usa documentos recuperados como contexto. El temario la recoge como contenido optativo y este ejercicio recupera vídeos por similitud de contenido. |

### Métricas de clasificación y regresión

| Término | Qué significa y para qué se usa |
|---|---|
| Matriz de confusión | Cruza etiquetas reales con etiquetas predichas. Permite ver qué clases se confunden y, en moderación, cuántos positivos se omiten o se anuncian de más. |
| Exactitud y exactitud equilibrada | La exactitud cuenta la fracción total de aciertos. La equilibrada promedia la recuperación de las clases para reflejar también las minoritarias. |
| Precisión y recuperación | La precisión mide qué fracción de los positivos predichos era realmente positiva. La recuperación o sensibilidad mide qué fracción de los positivos reales se detectó. |
| F1, macro-F1 y F2 | F1 combina precisión y recuperación. Macro-F1 promedia el F1 de las clases y sirve para elegir clasificadores. F2 da más peso a la recuperación y se usa al escoger el umbral de moderación. |
| Curva PR y AP | La curva de precisión y recuperación muestra su intercambio al variar el umbral. AP es la precisión promedio que resume el recorrido. Se informa junto a la frecuencia de la clase positiva. |
| MAE y RMSE | MAE promedia el error absoluto y sirve para comparar regresores en las unidades originales. RMSE da más peso a errores grandes y ayuda a identificar el impacto de los extremos. |
| R² | Coeficiente de determinación. Compara la suma de errores cuadrados con la de una referencia constante igual a la media. Indica cuánto mejora esa referencia en esta medida. |

### Segmentación, reducción y recomendación

| Término | Qué significa y para qué se usa |
|---|---|
| Grupo y segmentación | Un grupo reúne vídeos con atributos parecidos según la representación utilizada. Segmentación es la formación de grupos a partir de similitudes en atributos de vídeos. |
| KMeans y MiniBatch KMeans | KMeans asigna ejemplos al centro más cercano y actualiza los centros. MiniBatch estima los centros usando lotes pequeños. |
| Mean Shift y ancho de banda | Mean Shift busca zonas densas desplazando puntos hacia máximos locales. El ancho de banda fija la escala del entorno que se examina. |
| GMM y AIC/BIC | GMM modela los datos como mezcla de distribuciones gaussianas. El criterio de información de Akaike (AIC) y el bayesiano (BIC) ayudan a comparar configuraciones GMM sobre los mismos datos y representación, con penalización por complejidad. |
| DBSCAN y agrupación aglomerativa | DBSCAN agrupa zonas densas y marca puntos dispersos como ruido. El método aglomerativo fusiona grupos progresivamente. Ambos sirven para la exploración en la muestra. |
| Silueta y estabilidad | La silueta compara la cohesión dentro del grupo con la separación respecto a otros. La estabilidad comprueba cuánto se mantiene una partición al repetir el ajuste con muestras distintas. |
| ARI y NMI | El índice de Rand ajustado (ARI) mide acuerdo entre dos particiones ajustado por azar. La información mutua normalizada (NMI) mide información compartida. Se usan para cuantificar el solapamiento con categorías. El ARI también compara particiones de réplicas para estudiar estabilidad. |
| PCA y t-SNE | El análisis de componentes principales (PCA) proyecta sobre direcciones lineales de mayor varianza. t-SNE representa principalmente vecindades locales. Se muestran para explorar la estructura en dos dimensiones. |
| Similitud coseno | Compara la dirección de dos vectores de atributos. Se usa para ordenar candidatos de contenido parecido a una consulta. |
| Catálogo, consulta y top-k | El catálogo contiene los vídeos candidatos. La consulta es el vídeo o texto para el que se buscan vecinos. Top-k limita la lista a los k candidatos más próximos. |
| Cobertura, proxy y evaluación humana | La cobertura indica qué fracción del catálogo aparece en alguna recomendación. La coincidencia editorial sirve de proxy o indicador indirecto. Una revisión humana evalúa si las sugerencias son pertinentes para la consulta. |
| Precisión@k y NDCG | Precisión@k cuenta la fracción relevante entre las primeras k recomendaciones. La ganancia acumulada descontada normalizada (NDCG) da más peso a aciertos en posiciones altas. Ambas requieren juicios de relevancia para interpretarse como tales. |

### Ejecución y entrega

| Término | Qué significa y para qué se usa |
|---|---|
| Inferencia y receta | Inferencia es aplicar un modelo entrenado a datos nuevos. La receta identifica familia, transformaciones, hiperparámetros y umbral cuando procede. |
| Pickle, pesos y manifiesto | Pickle guarda el estado aprendido para recuperarlo con load. Los pesos y transformadores forman ese estado. El manifiesto registra el modelo, el esquema y su procedencia. |
| Huella SHA-256 y CRC | SHA-256 produce un identificador para comparar archivos y comprobar procedencia. El CRC del ZIP ayuda a detectar daños en sus entradas. |
| CLI, entorno virtual y kernel | CLI es la interfaz que recibe argumentos desde la terminal. El entorno virtual fija dependencias de la práctica. El kernel es el proceso que ejecuta las celdas del cuaderno. |
| DataFrame y array de NumPy | DataFrame es una tabla con nombres de columnas. Un array es una matriz sin esos nombres y necesita un esquema de entrada. Ambos sirven para ajustar y consultar las clases. |
| API y contrato | Una API es el conjunto de operaciones que un módulo ofrece. El contrato fija cómo se llaman y qué reciben y devuelven. Las siete clases comparten las operaciones exigidas en el enunciado. |

### Parámetros que aparecen en las tablas

| Término | Qué significa y para qué se usa |
|---|---|
| C, alpha y gamma | C controla cuánto se penalizan los errores en SVM y logística. Alpha fija la regularización en modelos como Ridge. Gamma controla el alcance de la cercanía que mide el kernel RBF. |
| n_estimators, max_depth y learning_rate | Cantidad de árboles, límite de profundidad y peso que aporta cada árbol nuevo. Se ajustan para controlar la complejidad de los bosques y de XGBoost cuando corresponde. |
| max_features, min_df, ngram_range y sublinear_tf | En TF-IDF fijan el tamaño del vocabulario, la frecuencia documental mínima, los tamaños de n-gramas y la forma de moderar repeticiones mediante una escala logarítmica. |
| shrinkage | Regularización de la estimación de la covarianza. En LDA se compara usarla o dejar la estimación sin ese ajuste. |
| n_components y k | Número de componentes que se conservan en una reducción o mezcla, según el estimador. En KMeans, k fija los grupos. En recomendación, k limita los vecinos devueltos. |
| eps y min_samples | En DBSCAN, eps fija el radio de vecindad y min_samples indica la densidad mínima para considerar un punto como núcleo. |
| random_state y n_jobs | La semilla fija elecciones aleatorias para poder repetir el experimento. n_jobs controla el paralelismo del ajuste. |


### Incidencias por fila al recomendar desde la CLI

En un CSV mixto, la CLI conserva las consultas válidas y señala las filas vacías, sin vocabulario conocido o con un ID desconocido. En JSON mantiene una posición por entrada: [] representa una consulta sin recomendaciones y stderr aclara si hubo un error de entrada; en texto se muestra la causa junto al vídeo. Si ninguna entrada es válida, termina con código 2. Los errores de integridad o del modelo siguen deteniendo la consulta. Esta recuperación pertenece a la interfaz; predict de la clase conserva su validación estricta del lote completo.
