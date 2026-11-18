# Machine Learning · YouTube Trending

## Pregunta guía y retos del enunciado

**Pregunta guía formulada a partir del enunciado.** ¿Qué podemos estimar sobre otros vídeos que alcanzan tendencias, cómo segmentarlos y qué contenido recuperar como similar?

| Área | Retos que pide el enunciado | Retos opcionales para subir nota |
|---|---|---|
| Clasificación | Predecir la categoría del vídeo (reto normal). | Estimar si se deshabilitan los comentarios o las valoraciones. La práctica estudia ambos casos por separado. |
| Regresión | Estimar el número de likes (reto normal). | Estimar la proporción de likes respecto a dislikes. |
| Segmentación | Buscar grupos de vídeos con varias técnicas, comparar su solapamiento con las categorías y aplicar reducción de dimensionalidad. | |
| Recomendación | Recomendar vídeos similares a partir de un vídeo dado. | |

La entrega desarrolla ambos retos opcionales. El de clasificación se divide en comentarios y valoraciones, lo que da un total de siete tareas de inferencia. [Enunciado de la práctica](ENUNCIADO.txt).

El [glosario](GUIA_DE_CONSULTA.md#glosario) explica los términos de la práctica. La secuencia de preparación de datos y modelo se llama **cadena de procesamiento** (`Pipeline`). La **validación cruzada** alterna partes de ajuste y comprobación; cada parte es una **partición** (`fold`). La **reserva de prueba** (`holdout`) queda separada para la evaluación final. Los nombres de funciones, argumentos y archivos se conservan para poder localizar el código.

## Ruta breve para corregir

El cuaderno se entrega **completamente ejecutado**, con tablas, gráficos y conclusiones guardados. Para leerlos en Jupyter no es necesario ejecutar celdas ni preparar el dataset. Las instalaciones y el dataset se necesitan para reproducir el proceso o consultar los modelos.

Las cifras de lectura humana en textos, tablas y ejes usan **coma decimal y punto de millares**. Los CSV, JSON, parámetros escritos en código y versiones de bibliotecas conservan sus convenciones técnicas para poder procesarse sin ambigüedad.

1. **Ver los resultados.** Abrir [exploratory_analysis.ipynb](exploratory_analysis.ipynb) y buscar **Guía de corrección de los siete retos**. La tabla reúne el modelo, su referencia y la conclusión de cada tarea. Los bloques 6 a 9 muestran las matrices de confusión, los errores de regresión, la segmentación y la evaluación del recomendador. El apartado 12, **Conclusiones globales**, cierra el cuaderno: interpreta cada resultado, indica qué decisión justifica y sus límites, y extrae conclusiones comunes a los siete retos.
2. **Reproducir la comprobación.** Preparar el entorno según la sección siguiente y ejecutar la **opción 1** de `ejecutar_practica.bat` (o `--verificar`). Para consultar un modelo con el CSV de ejemplo, elegir la **opción 3**. Para recorrer el cuaderno celda a celda, elegir la **opción 4**.
3. **Consultar el detalle.** [Resultados y decisiones](reports/results.md) resume los límites. [Informe detallado](reports/results_details.md) y `reports/` contienen las métricas, gráficos y evidencias que sustentan las conclusiones.

### Mapa de la entrega

| Elemento del enunciado | Archivo o carpeta | Qué comprobar |
|---|---|---|
| Paquete de modelos | `models/` | Una clase por tarea final, con `__init__`, `fit`, `predict`, `save` y `load`. |
| 1. Preprocesamiento común | `preprocess_data.py` | Cruza cada CSV regional con su JSON de categorías y genera el CSV unificado y la auditoría. |
| 2. Entrenamiento | `train_models.py` | Carga las recetas seleccionadas, ajusta las clases y guarda los pesos. |
| 3. Inferencia | `inference_model.py` | Carga los pesos y consulta el modelo con nuevas filas. |
| 4. Exploración y comparación | `exploratory_analysis.ipynb` | Explora los datos, compara y refina modelos e interpreta los resultados. |
| Configuración que exige el enunciado | `train.conf` | Indica al entrenador las rutas de los datos y los pesos, y guarda las recetas finales. |

El [enunciado](ENUNCIADO.txt) enumera cuatro archivos principales fuera de `models/` y exige también `train.conf` en el apartado de entrenamiento. La entrega añade archivos de apoyo para que el cuaderno y las comprobaciones puedan reproducirse desde el BAT.

| Apoyo | Para qué sirve |
|---|---|
| `ejecutar_practica.bat`, `run_practice.py`, `execute_notebook.py` y `review_notebook.py` | Iniciar el flujo automático o el repaso visual del cuaderno. |
| Módulos de experimentación y presentación | Reutilizar la lógica de partición, comparación, refinamiento, calibración y generación de resultados. |
| `requirements*.txt` y `verify_delivery.py` | Preparar el entorno y comprobar la ejecución y las siete interfaces. |
| `data/example.csv`, `artifacts/` y `reports/` | Consultar los modelos guardados y revisar las evidencias del experimento. |

El historial técnico y los PDF de estudio se conservan aparte de la entrega académica.

En inferencia, los errores de entrada se muestran en stderr con código 2. La salida stdout se reserva para JSON. Si el modelo admite continuar con columnas ausentes o contadores inválidos, se emite un aviso.

## Ejecución sencilla en Windows

El punto de entrada es `ejecutar_practica.bat`, situado junto a este README. No es necesario abrir PyCharm ni ejecutar cada script por separado: el BAT permite ejecutar el flujo completo mediante `run_practice.py` o consultar los siete modelos ya entrenados mediante `inference_model.py`. También permite abrir un repaso visual celda a celda con `review_notebook.py`. Es el único lanzador `.bat` de la práctica.

### Ubicación de la práctica: no requiere una ruta fija

La entrega puede extraerse en una carpeta de Windows con permisos de escritura y espacio para el CSV, los informes y los temporales. Se admiten nombres con espacios. Para el entrenamiento conviene una ruta corta por los límites de rutas de Windows. No hay que crear la misma ruta que se utilizó durante el desarrollo. Debe extraerse el ZIP completo conservando sus subcarpetas; no se debe ejecutar el BAT desde el interior del ZIP comprimido.

`ejecutar_practica.bat` utiliza `pushd "%~dp0"`: `%~dp0` obtiene la carpeta donde está el propio BAT y `pushd` entra en ella. Por eso el doble clic funciona sin indicar una ruta fija. `run_practice.py` obtiene su carpeta mediante `Path(__file__).resolve().parent`; `train.conf` utiliza rutas relativas para datos, pesos e informes.

El BAT se guarda en UTF-8 sin BOM. Al arrancar selecciona la página de códigos 65001 de la consola para mostrar tildes y eñes y, al salir normalmente, restaura la página anterior.

### Preparación inicial, una sola vez en el equipo del profesor

#### Paso 1. Disponer de CPython 3.12 de 64 bits

El entorno validado es **Windows x64 con CPython 3.12.14**. Las opciones 1 y 4 necesitan el parche con el que se generaron los modelos de la copia. La opción 2 permite repetir el experimento con otro parche 3.12 de 64 bits y las dependencias fijadas. La opción 3 carga los pesos ya guardados con las bibliotecas compatibles y permite una consulta con el CSV de ejemplo. El BAT utiliza rutas relativas a su carpeta. En macOS o Linux, este lanzador de Windows se sustituye por la invocación directa de Python descrita más abajo. Esos sistemas no forman parte de la comprobación de esta entrega.

Primero comprueba si ya tienes el Python exacto. Con el gestor de Python puedes listar los intérpretes instalados. Si no dispones de ese comando, comprueba el ejecutable que ya utilizas o el entorno de Conda.

```powershell
py list
py -3.12 --version
```

Si ya tienes 3.12.14, usa ese intérprete en el paso 3. [Python.org publica 3.12.14 como código fuente, sin instalador binario de Windows](https://www.python.org/downloads/release/python-31214/). Si utilizas [Python Install Manager](https://docs.python.org/3/using/windows.html), consulta su catálogo con `py list --online 3.12.14`. Ejecuta `py install 3.12.14` únicamente si el catálogo ofrece ese parche para tu arquitectura. `py -3.12` puede seleccionar otro parche si hay varios instalados.

**Si ya tienes Anaconda o Miniconda**, abre **Anaconda PowerShell Prompt** en la carpeta extraída y crea un entorno separado. [El canal conda-forge publica Python 3.12.14 para Windows x64](https://anaconda.org/conda-forge/python/files/manage?channel=main&type=conda&version=3.12.14).

```powershell
conda create --prefix ".\.conda-ml" --override-channels -c conda-forge python=3.12.14 pip
conda activate ".\.conda-ml"
python --version
python -m pip install -r requirements.txt
python -m pip check
$env:ML_PYTHON = (Get-Command python).Source
.\ejecutar_practica.bat --comprobar-entorno
```

Esta vía sustituye los pasos 3 y 4. Lanza el BAT desde la misma terminal. El BAT prefiere la `.venv` local cuando existe. Si apunta a un Python distinto o se trasladó desde otra carpeta, renómbrala antes de usar `ML_PYTHON`. Para volver a esta preparación en otra terminal, activa el entorno y fija `ML_PYTHON` de nuevo.

#### Paso 2. Abrir PowerShell en la carpeta extraída

En el Explorador de archivos, abrir la carpeta que contiene `ejecutar_practica.bat`, `run_practice.py` y este README. Escribir `powershell` en la barra de direcciones y pulsar Intro. La terminal se abrirá en esa carpeta, sea cual sea su ubicación.

Comprobar que se está en el lugar correcto:

```powershell
Get-Location
Test-Path .\ejecutar_practica.bat
Test-Path .\requirements.txt
```

Las dos últimas órdenes deben devolver `True`. Si devuelven `False`, situarse en la carpeta que realmente contiene los archivos de la práctica.

#### Paso 3. Crear el entorno virtual local .venv

Desde esa misma terminal:

```powershell
# Usa el ejecutable 3.12.14 que has comprobado en el paso 1.
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe --version
```

Si el lanzador elige otro parche, sustituye `py -3.12` por `& "C:/ruta/al/python.exe"` con la ruta del 3.12.14. La primera orden crea `.venv` dentro de la carpeta actual. Es un entorno separado para instalar los paquetes de esta práctica. La segunda comprueba qué versión de Python utiliza. Si hay varios parches 3.12 instalados, verificar que el seleccionado sea el previsto antes de continuar.

No se debe copiar `.venv` desde otro ordenador ni trasladar uno ya creado junto con la entrega: se crea en el equipo y ubicación de destino. Si se mueve la práctica después de preparar el entorno, recrearlo en la nueva carpeta. Si el entorno ya existe en esa ubicación y tiene el Python y las dependencias adecuados, no es necesario volver a crearlo. No modificar el entorno mientras esté ejecutándose la práctica.

#### Paso 4. Instalar y comprobar las dependencias

Ejecutar estas órdenes en el orden indicado; continuar sólo si la anterior termina correctamente:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
.\ejecutar_practica.bat --comprobar-entorno
```

`requirements.txt` remite a las versiones fijadas en `requirements-lock.txt`. Se necesita acceso a Internet para descargar los paquetes, salvo que ya estén disponibles localmente. `pip check` detecta incompatibilidades declaradas entre dependencias; el BAT comprueba la arquitectura de Python, todas las versiones fijadas y la importación de las bibliotecas principales. Si encuentra un problema, muestra el intérprete utilizado y los pasos de reparación. El contraste de datos, pesos y huellas se hace al ejecutar la opción 1.

Las órdenes usan el ejecutable de `.venv` explícitamente: no es necesario activar el entorno, cambiar la política de PowerShell, abrir PyCharm ni instalar los paquetes en el Python global. El BAT tampoco instala ni actualiza bibliotecas automáticamente.

#### Paso 5. Colocar el dataset original para ejecutar el flujo completo

**Para consultar modelos guardados (opción 3), se puede omitir este paso:** basta con los pesos de `artifacts/` y un CSV de consulta; se incluye `data/example.csv`. Para verificar o repetir todo el experimento (opciones 1 y 2), descargar `youtube-new.zip` del enlace del enunciado y guardarlo como `dataset.zip` junto al BAT, sin descomprimirlo. La carpeta debe tener, entre otros, estos elementos:

```text
carpeta_elegida_por_el_profesor/
    ejecutar_practica.bat
    run_practice.py
    README.md
    requirements.txt
    requirements-lock.txt
    train.conf
    dataset.zip
    .venv/
    models/
    artifacts/
    reports/
```

Comprobar desde PowerShell:

```powershell
Test-Path .\dataset.zip
```

Debe devolver `True`. No es necesario crear manualmente el CSV procesado: lo genera la propia ejecución.

#### Paso 6. Ejecutar y consultar los resultados

Hacer doble clic en el BAT y elegir uno de los modos descritos a continuación. Al terminar correctamente, abrir el cuaderno ejecutado y los informes. La práctica escribe en su propia carpeta: conviene usar una copia de trabajo para conservar intacta la entrega original.

El BAT busca primero `.venv\Scripts\python.exe` junto a él. Como alternativas admite un intérprete preparado indicado mediante `ML_PYTHON` o busca con `py -3.12` y después `python`; para una preparación nueva se recomienda el entorno local descrito en estos pasos.

### Lanzar la práctica

Hacer doble clic en `ejecutar_practica.bat`. Se muestra un menú:

| Opción | Qué ejecuta | Cuándo utilizarla |
|---|---|---|
| **1. Verificar** | Preprocesa el dataset, ejecuta todo el cuaderno reutilizando resultados y modelos sólo si sus huellas coinciden, y comprueba las siete tareas de inferencia y la integridad de sus pesos. | Corrección de una entrega final con fuentes y modelos compatibles; no repite el aprendizaje de los modelos entregados. |
| **2. Desde cero** | Preprocesa, repite comparación de familias y refinamiento, calibra los umbrales, ajusta los ganadores, ejecuta el resto del cuaderno y realiza las verificaciones. | Reproducir el experimento completo. Reemplaza modelos, recetas y resultados de esa copia; utilizar una copia de trabajo si se desean conservar los anteriores. |
| **3. Consultar modelos** | Abre un submenú con las siete tareas y carga sus pesos para consultar un CSV. No ejecuta el cuaderno, no entrena ni necesita `dataset.zip`. | Probar predicciones o recomendaciones con el ejemplo incluido o con datos propios. |
| **4. Repaso visual** | Abre una copia independiente del cuaderno en JupyterLab existente, con kernel del Python de la práctica. | Ejecutar una celda cada vez y estudiar sus gráficos, tablas y explicaciones. |
| **E. Comprobar entorno** | Revisa Python, las versiones fijadas y las importaciones principales. | Primera comprobación tras extraer el ZIP. Para el visor, usa `--diagnostico-visual`. |
| **S. Salir** | Cierra el menú sin iniciar nuevas tareas. | Terminar la sesión. |

Al finalizar desde el menú, la ventana queda abierta para leer el resultado. Sólo aparece el mensaje de éxito si el proceso termina correctamente. Consultar después `exploratory_analysis.ipynb` y `reports/results.md`; los demás informes y figuras están en `reports/`.

También se puede ejecutar desde PowerShell, sin menú:

```powershell
.\ejecutar_practica.bat --comprobar-entorno
# Alternativa interactiva: consultar los siete modelos guardados.
.\ejecutar_practica.bat --consultar
# Alternativa integral: ejecutar y verificar el cuaderno.
.\ejecutar_practica.bat --verificar
# Alternativa larga: repetir el experimento completo desde cero.
.\ejecutar_practica.bat --reentrenar
```

`--comprobar-entorno` revisa el intérprete, las versiones fijadas y las importaciones principales. Avisa si el parche difiere del registrado para reutilizar los modelos. Los datos, pesos y huellas se verifican con la opción 1. Las órdenes `--verificar` y `--reentrenar` son alternativas; no es necesario ejecutar ambas seguidas. `--consultar` abre directamente el submenú de modelos y solicita el CSV de forma interactiva. `--repaso-visual` abre el cuaderno paso a paso con gráficos. `--ayuda` muestra las opciones disponibles. Las variables del BAT se limitan a su proceso y sus hijos; no modifica la configuración permanente de Windows.

**Estado de la entrega:** experimento completo desde cero verificado el 7 de noviembre de 2026. El cuaderno tardó 67,7 minutos en esa ejecución. El ZIP se comprueba en una extracción independiente con reutilización verificada. Los PDF de estudio se conservan fuera de la entrega académica.

**Este lanzador no genera ni modifica el ZIP de entrega.** El empaquetado y la auditoría del archivo comprimido se realizan durante el desarrollo. Para corregir la práctica se usa el ZIP ya preparado.

## Entorno y reproducción

Python 3.12. Instalar `python -m pip install -r requirements.txt`; remite al entorno exacto de `requirements-lock.txt`, coherente con los pesos.

Descargar el fichero `youtube-new.zip` del enlace del enunciado y guardarlo en la carpeta de la práctica con el nombre `dataset.zip`, sin descomprimirlo. La alternativa manual al BAT es ejecutar los siguientes comandos desde esa carpeta. `run_practice.py` utiliza UTF-8 tanto en sus salidas como en las de los procesos de Python que ejecuta. Esta configuración evita incompatibilidades al redirigir texto en Windows y no modifica la configuración del sistema.

```powershell
$env:ML_REUSE_TRAINING = '1'
python run_practice.py --archive dataset.zip --verify
```

Ésta es la comprobación rápida recomendada para corregir: preprocesa los CSV/JSON regionales, ejecuta las 19 celdas de código desde un kernel limpio y comprueba las siete interfaces de inferencia descritas a continuación y que sus pesos se conservan. Reutiliza los resultados y modelos guardados únicamente tras comprobar sus huellas de compatibilidad. Esta comprobación no realiza un entrenamiento nuevo. Un kernel limpio comienza sin variables de sesiones anteriores, pero puede cargar del disco los resultados y modelos compatibles.

Como referencia, el experimento completo desde cero compara familias, refina los modelos seleccionados, calibra los umbrales de los dos clasificadores binarios y entrena los ganadores. La última ejecución completa desde cero registrada tardó 67,7 minutos; el tiempo depende del equipo y de la carga. Los presupuestos configurados son aproximadamente 6.000 filas para búsqueda, 30.000 para ajuste final y 8.000 para prueba, manteniendo separados los grupos de vídeo entre entrenamiento y prueba. Los tamaños efectivos pueden variar según la tarea y la disponibilidad de etiquetas; las curvas de aprendizaje ayudan a valorar las limitaciones de estos tamaños.

### Repaso visual: ejecutar y comprender una celda cada vez

Para **ver las salidas mientras se ejecuta cada celda**, elegir la **opción 4**. El BAT llama a `review_notebook.py`, que abre una copia de `exploratory_analysis.ipynb` en el navegador. La opción 1 llama a `run_practice.py`, que usa `execute_notebook.py` para verificar automáticamente el cuaderno y guardar sus salidas.

1. Hacer doble clic en `ejecutar_practica.bat` y elegir **4. Repaso visual**.
2. Se prepara una copia independiente en `historial_tecnico/repaso_visual/`. Incluye los datos procesados y los modelos; puede ocupar varios cientos de MB. La sesión y sus apuntes se conservan al cerrarla. Cada apertura crea una sesión nueva.
3. Se abre JupyterLab en el navegador, con una guía inicial y el cuaderno. Las salidas de esta copia comienzan vacías para distinguir lo que se ejecuta ahora. Leer los textos y seleccionar **Celda de código 1 de 19**. La numeración de la copia sigue el orden de ejecución.
4. Pulsar **Ctrl + Enter** para ejecutar la celda y permanecer en ella. `[*]` indica que está trabajando. Al terminar, leer las salidas de `print`, las tablas, los gráficos y sus comentarios. **Shift + Enter** ejecuta la celda y selecciona la siguiente.
5. Avanzar de arriba abajo: las celdas comparten variables. Una celda puede contener varios gráficos. Este recorrido ejecuta una celda completa; para detenerse en cada instrucción se utiliza un depurador.
6. Mantener abierta la ventana del BAT mientras se utiliza el visor. **Ctrl+C** en esa ventana cierra el servidor; la copia de estudio queda guardada.

Este modo ejecuta los cálculos de presentación y reutiliza comparaciones, umbrales y modelos **tras comprobar sus huellas**. No es un experimento nuevo desde cero. Trabaja sobre copias independientes: guardar el cuaderno o regenerar gráficos e informes en la sesión no sobrescribe la entrega original. Si falta el CSV procesado, se prepara dentro de la copia a partir de `dataset.zip` o de `data/raw/`.

**Se aprovecha Jupyter ya instalado**, por ejemplo el de Anaconda. No se instala ni actualiza ningún paquete al elegir esta opción. El visor puede pertenecer a Anaconda y el kernel de cálculo utiliza el `python.exe` de la práctica; así conserva las versiones de sus modelos. El kernel se llama **Python 3.12 · Práctica ML**.

El lanzador busca JupyterLab en el Python activo, Anaconda/Miniconda habituales y el PATH. Para una ubicación diferente se puede indicar el Python que contiene Jupyter, desde PowerShell:

```powershell
$env:ML_JUPYTER_PYTHON = 'C:\ProgramData\anaconda3\python.exe'
.\ejecutar_practica.bat --repaso-visual
```

La ruta es un ejemplo, no una obligación. Si no se localiza JupyterLab, aparece un mensaje que permite indicar la instalación existente. No se cambia la instalación de Anaconda ni se necesita instalar las bibliotecas de la práctica en ella. Temporales, configuración y kernel de esta sesión se guardan dentro de la copia en la unidad donde esté la práctica.

Si aún no tienes JupyterLab y quieres usar la opción 4, puedes instalarlo en un entorno separado para el visor. Utiliza un Python ya disponible para crear ese entorno. Desde PowerShell, en la carpeta de la práctica:

```powershell
py -3.12 -m venv .venv-visor
.\.venv-visor\Scripts\python.exe -m pip install jupyterlab
$env:ML_JUPYTER_PYTHON = (Resolve-Path .\.venv-visor\Scripts\python.exe).Path
.\ejecutar_practica.bat --diagnostico-visual
```

Si ya tienes JupyterLab, utiliza su instalación existente. Las opciones 1, 2 y 3 funcionan con las bibliotecas de la práctica. La opción 1 deja el cuaderno con sus salidas guardadas.

### Incidencias previstas durante la corrección

[Consultar las causas y soluciones más frecuentes](GUIA_DE_CONSULTA.md#incidencias-durante-la-corrección).

### Consultar modelos sin escribir comandos

1. Hacer doble clic en `ejecutar_practica.bat` y pulsar **3**.
2. Elegir una de las tareas del submenú:

   | Opción | Tarea | Cómo interpretar la salida |
   |---|---|---|
   | **1** | Categoría | Una categoría editorial estimada por vídeo. |
   | **2** | Comentarios deshabilitados | `1`: predice deshabilitados; `0`: habilitados. |
   | **3** | Valoraciones deshabilitadas | `1`: predice deshabilitadas; `0`: habilitadas. |
   | **4** | Likes | Número estimado con las señales disponibles; no es un pronóstico anterior a publicar. |
   | **5** | Ratio likes/dislikes | Proporción estimada de likes por dislike. |
   | **6** | Grupo | Identificador `0` o `1`; no es una categoría editorial. |
   | **7** | Recomendador | Hasta cinco vídeos por consulta, con ID, título y similitud textual. |

3. Cuando se solicite el CSV, pulsar **Intro** para usar `data/example.csv`, o escribir/pegar la ruta de un CSV propio **incluyendo el nombre del archivo y su extensión `.csv`**, por ejemplo `data/example.csv`. También se puede indicar una carpeta: se mostrarán sus archivos CSV numerados para elegir uno. En ese mismo punto se puede escribir otra ruta directamente o **S** para cancelar la consulta; **R** sigue disponible como alternativa para solicitar otra ruta. Se listan sólo los CSV de esa carpeta, sin recorrer subcarpetas. Se admiten rutas con espacios, con o sin comillas. Una ruta relativa se interpreta desde la carpeta del BAT; una absoluta puede apuntar a otra carpeta.
4. Leer los resultados explicados en la terminal: cada vídeo aparece con su título, su ID cuando está disponible y su predicción, en el orden del CSV. Las categorías conocidas muestran su traducción al español junto a la etiqueta original; los estados se explican como habilitados o deshabilitados y las cifras usan formato español. El recomendador muestra títulos, similitud textual y enlaces de los vídeos sugeridos. La similitud no es una probabilidad de satisfacción. Los datos adicionales que el modelo no utiliza generan una nota breve, sin rutas de código ni líneas internas.
5. Pulsar una tecla para volver al submenú. **V** vuelve al menú principal y **S** termina.

El CSV debe contener las columnas adecuadas para la tarea. El ejemplo incluido permite probar las siete opciones. El recomendador admite título, etiquetas o descripción, o un `video_id` que ya exista en su catálogo; no descarga información de enlaces de YouTube. Los avisos indican columnas ausentes o valores inválidos cuando el modelo permite continuar. Si se indica una carpeta desde el BAT, se ofrece elegir un CSV. Si está vacía de CSV, se permite indicar otra ruta o cancelar. El mensaje se adapta al listado: un único CSV ofrece `(1)`, varios ofrecen `(1-N)` y, si no hay ninguno, sólo se pide otra ruta o cancelar. Una opción no válida vuelve a pedir la elección y cancelar regresa al menú sin generar predicciones. Los errores de archivo ausente, vacío, sin observaciones, sin permisos, codificación o formato ilegible explican la causa y cómo corregirla; se permite volver a intentarlo. Los CSV de `reports/` contienen resultados del análisis y no son, en general, entradas de vídeos; para una primera prueba se debe usar `data/example.csv`. Los CSV regionales originales de `data/raw/` se preparan mediante `preprocess_data.py`; no se deben convertir ni sobrescribir para probar el menú. Un CSV propio debe usar UTF-8. Si la consulta no es válida, se muestra la causa; si faltan los pesos, no se entrena automáticamente.

El BAT selecciona automáticamente el archivo `.pkl` correspondiente y llama a la misma interfaz `inference_model.py` descrita a continuación. Añade `--seleccionar-csv` para habilitar la elección interactiva de archivos en carpetas y `--formato texto` para mostrar resultados comprensibles; los tres argumentos originales de inferencia mantienen su funcionamiento. Sin esa opción, una carpeta recibe un mensaje que solicita un archivo concreto. Cancelar el selector devuelve código 3 al BAT; los errores de entrada conservan el código 2. **Consultar carga los pesos sin modificarlos y no realiza búsquedas, calibración ni entrenamiento.** Las predicciones se muestran en pantalla: esta opción no actualiza los informes del experimento ni el ZIP. El menú facilita la consulta; las métricas y sus límites se estudian en el cuaderno y los informes.

**Lotes del recomendador.** La CLI y el menú conservan las filas válidas aunque otra consulta esté vacía, tenga un ID desconocido o carezca de vocabulario aprendido. El texto identifica la fila y explica cómo corregirla; en JSON esa posición contiene una lista vacía y la causa se escribe como aviso en stderr, separado de los resultados. Se conserva el orden del CSV. Una consulta única inválida o un lote sin ninguna entrada válida devuelve código 2; los fallos de pesos o estructura no se omiten. La clase del modelo mantiene su contrato estricto: predict rechaza un lote con entradas inválidas; la recuperación por filas corresponde a la CLI.

### Qué son las siete interfaces de inferencia

Una interfaz de inferencia es la forma de consultar un modelo ya entrenado para obtener una predicción o recomendación. La práctica ofrece siete tareas a través de un único script, `inference_model.py`: el primer argumento selecciona la tarea, el segundo indica su archivo de pesos y el tercero es un CSV con los vídeos que se desean consultar.

| Tarea (primer argumento) | Qué devuelve para cada vídeo consultado | Pesos (segundo argumento) |
|---|---|---|
| `category` | La categoría editorial predicha. | `artifacts/category.pkl` |
| `comments_disabled` | Si los comentarios están deshabilitados: `1` significa sí y `0`, no. | `artifacts/comments.pkl` |
| `ratings_disabled` | Si las valoraciones están deshabilitadas: `1` significa sí y `0`, no. | `artifacts/ratings.pkl` |
| `likes` | Una estimación del número de likes. | `artifacts/likes.pkl` |
| `ratio` | Una estimación de la ratio likes/dislikes. | `artifacts/ratio.pkl` |
| `cluster` | El identificador del grupo asignado por el modelo; no es una categoría editorial. | `artifacts/cluster.pkl` |
| `recommend` | Hasta cinco vídeos similares, con su identificador, título y similitud. | `artifacts/recommend.pkl` |

**Formatos de salida:** el BAT utiliza texto explicado. La invocación directa conserva JSON por defecto para programas y verificaciones; `--formato json` lo solicita explícitamente y `--formato texto` muestra la lectura por vídeo. Ambos formatos utilizan las mismas predicciones y los mismos pesos. Las notas del formato texto no modifican ni silencian los avisos del modelo: resumen las columnas adicionales y conservan los demás mensajes.

Por ejemplo, `python inference_model.py category artifacts/category.pkl data/example.csv` carga el clasificador de categorías, consulta los vídeos de `data/example.csv` y escribe las predicciones como JSON en la consola, sin volver a entrenar.

Al usar `--verify`, `verify_delivery.py` comprueba las siete tareas: carga sus pesos, verifica la inferencia individual y por lotes y ejecuta `inference_model.py` para cada una. También contrasta las respuestas de los arrays con las consultas individuales y por lotes, y comprueba que los pesos conservan su huella. Estas comprobaciones verifican el funcionamiento de las interfaces; las métricas de calidad predictiva se consultan en el cuaderno y en `reports/evaluation.json`.

## Entrenamiento e inferencia de los ganadores

```powershell
python train_models.py train.conf
python inference_model.py category artifacts/category.pkl data/example.csv
```

`train_models.py` lee rutas y las siete secciones `model:*` de `train.conf`, ajusta exclusivamente los modelos elegidos y guarda pesos. Las recetas y umbrales proceden del experimento del cuaderno. La calibración ya está incorporada en la configuración final; el entrenador no necesita una segunda orden para aplicarla. Los métodos públicos son `__init__`, `fit`, `predict`, `save`, `load`; aceptan inferencia individual o por lotes.

El recomendador admite catálogos con etiquetas o descripción sin título: devuelve el título como cadena vacía cuando falta. Exige al menos dos identificadores válidos y distintos; las consultas sin vocabulario conocido se rechazan explícitamente. El directorio de informes del entrenador se toma de `output.reports`, relativo al archivo de configuración.

Para verificar pesos y repetir todo el cuaderno sin rehacer búsquedas: establecer `ML_REUSE_TRAINING=1` antes de ejecutar el comando integral. La reutilización exige huellas coincidentes de datos, código, configuración experimental y artefactos. Para repetir las búsquedas y ajustes desde cero, ejecutar antes `Remove-Item Env:ML_REUSE_TRAINING -ErrorAction SilentlyContinue` y después el comando integral. Sin esa variable se ejecuta el experimento de nuevo.

**Almacenamiento del recomendador.** El `recommend.pkl` incluido utiliza compresión sin pérdida de los bloques numéricos y del estado completo. Los bloques repetidos se almacenan una sola vez; `load()` reconstruye las matrices originales como copias independientes, con su precisión de 64 bits (`float64`), su catálogo y su vocabulario completos. Se utilizan Python y NumPy del entorno ya documentado, sin instalar otro descompresor. La compactación no reentrena ni cambia las recomendaciones: se ha contrastado la identidad del estado reconstruido y de sus respuestas. Un nuevo `fit()`/`save()` puede generar una serialización de mayor tamaño, sin acumular ejecuciones anteriores.

**Memoria al cargar el recomendador.** La compactación ahorra espacio en disco, pero la descompresión utiliza memoria temporal. En una auditoría local se midieron picos aproximados de 380 MB de memoria Python al cargar la versión compacta, frente a 172 MB de la original. Son medidas orientativas de esa carga, no la memoria total del proceso ni un requisito universal del equipo; pueden variar con el entorno.

## Alcance de la comparación y cruce regional

La comparación ampliada arrancó desde cero. Tras una interrupción del disco, se recuperaron con huellas verificadas categoría, comentarios, valoraciones y likes, y se completaron el ratio y las etapas siguientes.

- **LDA (análisis discriminante lineal)** distingue clases a partir de sus centros y una covarianza compartida. Compite en las tres clasificaciones. Se ajustan TF-IDF (ponderación de palabras según su frecuencia y la de los documentos), SVD (descomposición en valores singulares para comprimir la representación numérica) de 30 o 60 dimensiones y escala dentro de cada partición de validación. Se comparan dos opciones de regularización de la covarianza.
- **XGBoost 3.1.3** combina árboles sucesivos para reducir errores. Compite en clasificación y en ambas regresiones. Usa las mismas particiones y métricas de selección que las otras familias. Las regresiones comparan objetivo directo y log1p. La selección y el refinamiento se deciden en entrenamiento.
- **Cruce de categorías**. Se usa el JSON local de cada región. Cuando le falta un ID, se acepta el nombre sólo si todos los otros JSON que lo contienen coinciden. El CSV final contiene la etiqueta y la región. `data/processed/videos.audit.json` guarda el recuento por ID y región. Un ID sin nombre inequívoco detiene el proceso.

## Decisiones frente a la explicación de clase

- **Comentarios y valoraciones:** en clase se sugirió una variable agregada de comentarios y valoraciones; se entregan dos clasificadores separados, uno por objetivo, para distinguir ambas decisiones y estudiar su desbalance y umbrales individualmente. Así se cubren ambos casos opcionales sin mezclar etiquetas o interpretar desactivación como contenido inapropiado.
- **Desbalance:** se comparan configuraciones con y sin ponderación de clases (`class_weight`, cuando la familia lo admite) y se selecciona un umbral OOF (puntuaciones de modelos que no se ajustaron con los vídeos evaluados) para F2 (medida que combina precisión y recuperación dando más peso a esta última). No se aplica sobremuestreo de clases (aumentar ejemplos de clases poco representadas) ni submuestreo de clases (reducir ejemplos de clases abundantes) para equilibrar clases. Los presupuestos de muestreo por vídeo y categoría limitan coste computacional, no constituyen un rebalanceo binario. La receta elegida puede conservar `class_weight=None`; no se afirma que todos los ganadores utilicen ponderación. La selección no consulta las etiquetas de prueba.
- **Preprocesamiento específico:** se encapsula en el `Pipeline` que gestiona cada modelo, en vez de un método llamado literalmente `__preprocess_data()`. El nombre del auxiliar es un ejemplo del enunciado: lo relevante es que vocabulario, imputación y escala se aprendan dentro de cada partición y se conserven al guardar. Las transformaciones comunes permanecen en `preprocess_data.py`.
- **Criterio de corrección:** se muestra una respuesta principal y se interpreta desde valores calculados; tablas de detalle, avisos y trazabilidad sirven como apoyo. Se sigue la observación del profesor en Spark ML sobre interpretar métricas y la de Basic Spark Dataframe sobre no diluir la respuesta con presentaciones duplicadas.

## Interfaces industrializadas y correspondencia con la clase

En la explicación de la práctica (aproximadamente 07:04 y 09:46) se pide nombrar los archivos por el algoritmo y crear el estimador con los parámetros elegidos en `__init__`. Los nombres incluyen además la tarea cuando dos objetivos usan SVM.

| Tarea CLI | Archivo | Clase | Familia seleccionada |
|---|---|---|---|
| `category` | `models/svm_category_classifier.py` | `SVMCategoryClassifier` | `svm` |
| `comments_disabled` | `models/svm_comments_classifier.py` | `SVMCommentsClassifier` | `svm` |
| `ratings_disabled` | `models/logistic_ratings_classifier.py` | `LogisticRatingsClassifier` | `logistic` |
| `likes` | `models/svr_likes_regressor.py` | `SVRLikesRegressor` | `svr_log_target` |
| `ratio` | `models/ridge_ratio_regressor.py` | `RidgeRatioRegressor` | `ridge_log_target` |
| `cluster` | `models/kmeans_clustering.py` | `KMeansClustering` | `kmeans` |
| `recommend` | `models/tfidf_knn_recommender.py` | `TfidfKNNRecommender` | `char_tfidf` |

Las cinco operaciones son `__init__`, `fit`, `predict`, `save` y `load`; se heredan de `BaseModel` y se sobrescriben cuando la tarea lo necesita. `__init__` crea el estimador sin entrenarlo; `fit` aprende el estado. `_create_estimator` es el auxiliar interno que las clases no supervisadas adaptan. La inyección explícita de un estimador se conserva para pruebas y experimentos y no sustituye la receta final de la entrega.

El código de presentación está en `notebook_helpers.py`: el cuaderno conserva el hilo del análisis y muestra una salida principal con explicación calculada, dejando el detalle como apoyo. `training_data.build_group_folds` centraliza las particiones de búsqueda, calibración y diagnóstico, manteniendo un protocolo compartido y reproducible.

## Regla de selección de segmentación y justificación

Regla añadida después de observar una segmentación previa del 99,5 %: cada grupo debe contener al menos el 5 % de los vídeos y tres réplicas válidas de estabilidad con ARI (índice de Rand ajustado, acuerdo entre agrupaciones descontando el azar) medio >= 0,80; se maximiza silueta entre quienes cumplen. No se usan categorías editoriales ni etiquetas de prueba. Son límites metodológicos de esta revisión, no requisitos del profesor ni criterios fijados antes del primer estudio.

Estos límites se adoptan para evitar que la silueta aislada premie una solución con casi todos los vídeos en un grupo. No garantizan géneros útiles ni segmentos comerciales. La estabilidad exigida se calcula con SVD fijo; la estabilidad al reaprender toda la cadena de procesamiento se informa aparte y no se presenta como la misma prueba. Las categorías editoriales se usan para describir el solapamiento, no para elegir la segmentación.

## Archivos que actualizan los comandos

Trabajar sobre una extracción nueva del ZIP permite conservar la entrega original. El comando integral regenera `data/processed/videos.csv`, su auditoría, el cuaderno ejecutado, figuras e informes; la comprobación consulta `data/example.csv`. Con reutilización verificada conserva los pesos, las recetas y `train.conf`. Sin reutilización también reemplaza pesos, recetas y las secciones finales de `train.conf`; la escritura con configparser no conserva los comentarios. `train_models.py train.conf` reemplaza los siete pesos, manifiestos y `reports/final_training.json`. El empaquetado de desarrollo actualiza la carpeta académica, el ZIP, su SHA y el manifiesto de archivos.

## Resultados principales de la prueba interna

| Objetivo | Resultado actual |
|---|---:|
| Categoría | Macro-F1 0,527 |
| Likes | MAE 14.202,119; R² 0,957 |
| Ratio | MAE 26,984; R² 0,132 |

**Alcance de la segmentación:** el grupo mayor concentra 51,4 % de 6.000 vídeos y el menor contiene 2914. La selección por silhouette es geométrica; no acredita por sí sola grupos temáticos ni utilidad comercial. Consultar los perfiles y las alternativas.


**Límites medidos:** en el diagnóstico por canales disjuntos, comentarios obtiene macro-F1 0,380 frente a 0,495 de referencia. Este diagnóstico reajusta las recetas con otra partición y calibración; no evalúa directamente los pesos entregados. En valoraciones, F2 pasa de 0,318 con el umbral predeterminado a 0,302 con el umbral OOF. Las conclusiones detallan si mejora o empeora y mantienen la política elegida en entrenamiento.

Las cifras corresponden al holdout interno histórico por vídeo; no acreditan validación externa prospectiva ni relevancia humana del recomendador.

## Lectura de resultados

La división y la CV conservan grupos de `video_id`; vocabulario, imputación y escala se aprenden dentro de cada partición. Las dos búsquedas seleccionan con CV, nunca con la prueba. La muestra interna ya se ha examinado durante el desarrollo, por lo que no constituye evaluación externa prospectiva.

Los clasificadores desbalanceados incluyen matrices, precisión/recuperación, PR (curva de precisión y recuperación), AP (precisión media, resumen de esa curva) y comparación de umbrales. F2 no equivale a coste de negocio. Likes y ratio son estimaciones contemporáneas, no pronósticos anteriores a publicar. La recomendación usa contenido y coseno; coincidencia editorial es un proxy. La evaluación humana de la utilidad de las recomendaciones se propone como ampliación y queda fuera de lo solicitado en el ejercicio. 

Los gráficos de clusters muestran también el espacio TF-IDF + metadatos + SVD realmente ajustado, con categorías en las mismas coordenadas, términos reconstruidos, ARI/NMI (acuerdo e información compartida entre agrupaciones) y comparación de k. Los PCA/t-SNE (proyecciones para explorar la estructura de los datos) numéricos se conservan como análisis descriptivo de contadores.

## Contenido

`models/`: siete clases y recetas. Scripts de preprocesamiento, selección experimental y entrenamiento final; módulos auxiliares reutilizados por el cuaderno. `training_data.py` centraliza la partición por vídeo y el muestreo; `metric_utils.py` comparte el cálculo de F2 entre selección, diagnósticos e informes. `artifacts/`: siete pesos y manifiestos. `reports/`: tablas necesarias, informes académicos y comprobaciones de ejecución e inferencia. Los gráficos se ven en el cuaderno y sus copias se regeneran al ejecutarlo. El dataset y el entorno se obtienen aparte por tamaño. El historial técnico se conserva fuera de la entrega académica. El registro del cuaderno distingue la ejecución actual con reutilización de la referencia histórica del entrenamiento completo. Los registros detallados externos no son necesarios para ejecutar el ZIP.

## Lectura y defensa de las decisiones

El cuaderno presenta primero la familia y la configuración elegidas, una comparación por familia y la tabla principal de los siete objetivos. El espacio de búsqueda, los avisos y las tablas por región o clase se conservan en bloques desplegables. Las curvas y matrices permanecen visibles. Si el visor no permite desplegar HTML, las tablas originales están en los JSON y CSV de `reports/`.

Para explicar la práctica: la unidad de separación es el vídeo, porque sus apariciones diarias no son independientes; cada Cadena de procesamiento aprende vocabulario, imputación y escala dentro del entrenamiento; los umbrales se eligen sin consultar etiquetas de prueba; likes y ratio son estimaciones contemporáneas; el recomendador usa similitud de contenido y se evalúa mediante coincidencia editorial, cobertura y diversidad. Las opciones de comentarios o valoraciones desactivados no equivalen a detectar contenido inapropiado. Los presupuestos de muestra se declaran y las curvas no prueban saturación.

## Cobertura del temario y convergencia

`reports/temario_coverage.md` relaciona los contenidos del módulo con experimentos y descartes. La matriz recoge SVM lineal y RBF (función de base radial para representar semejanzas no lineales), MLP, LDA y XGBoost en la comparación supervisada, y Mean Shift y GMM entre las alternativas de segmentación. La segmentación inductivo usa los mismos datos e índices; DBSCAN y aglomerativo se distinguen como exploraciones transductivas. Las categorías no eligen el ganador.

`reports/convergence_review.json` conserva los intentos, sus presupuestos y los avisos de convergencia. Se aumenta una vez el límite de iteraciones si se detectan avisos, con los mismos datos y particiones y sin consultar la prueba. Los avisos que permanezcan se declaran y el entrenamiento final mantiene la receta congelada.

### Rutas largas en Windows

La práctica puede ejecutarse desde cualquier carpeta; no depende de la unidad D: ni de la ruta del autor. Conviene extraer el ZIP en una ruta breve (por ejemplo, `C:\ML` o `D:\ML`) si Windows no tiene habilitado el soporte de rutas largas. Las cachés internas de scikit-learn y joblib añaden subcarpetas y nombres largos: una extracción muy profunda puede provocar `FileNotFoundError` al crear un archivo de caché, aunque existan los datos. Se ha comprobado también una ubicación con espacios y acentos. No se debe copiar el entorno `.venv` entre ubicaciones: se crea en la carpeta elegida siguiendo las instrucciones anteriores.

## Glosario

[Consultar el glosario de técnicas y métricas](GUIA_DE_CONSULTA.md#glosario).
