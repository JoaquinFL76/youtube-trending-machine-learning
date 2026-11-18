@echo off
setlocal EnableExtensions DisableDelayedExpansion
rem UTF-8, CP 65001. El encabezado previo al cambio usa ASCII.
set "ML_PREVIOUS_CODEPAGE="
for /f "tokens=2 delims=:" %%P in ('chcp') do for /f %%C in ("%%P") do set "ML_PREVIOUS_CODEPAGE=%%C"
chcp 65001 >nul
if errorlevel 1 (
    echo ERROR: no se pudo configurar UTF-8. Abre el BAT desde una terminal de Windows.
    if "%~1"=="" pause
    exit /b 1
)
rem echo off evita mostrar cada orden antes de ejecutarla; echo sigue mostrando mensajes.
rem El prefijo @ oculta también la propia orden echo off.
rem setlocal limita los cambios de variables al BAT y los procesos que este inicia.
rem EnableExtensions habilita funciones adicionales del intérprete de comandos.
rem DisableDelayedExpansion evita interpretar signos de admiración dentro de rutas.
rem Trabajar en la carpeta del BAT, aunque se abra desde otra ubicación.
rem La expresión %~dp0 representa la unidad y carpeta donde se encuentra este BAT.
rem pushd entra en esa carpeta y guarda la ubicación anterior para recuperarla con popd.
rem Las comillas permiten que la ruta contenga espacios.
rem >nul oculta la salida normal; 2>&1 dirige también los errores a esa misma salida.
pushd "%~dp0" >nul 2>&1
rem errorlevel consulta el código de la orden anterior: normalmente 0 significa éxito.
rem if errorlevel 1 significa código mayor o igual que 1, no exactamente igual a 1.
rem Si la carpeta no es accesible, avisar antes de que se cierre la ventana.
if errorlevel 1 (
    echo ERROR: no se puede abrir la carpeta del BAT "%~dp0".
    echo Extrae el ZIP en una carpeta local accesible y vuelve a abrir el BAT.
    if "%~1"=="" pause
    if defined ML_PREVIOUS_CODEPAGE chcp %ML_PREVIOUS_CODEPAGE% >nul
    exit /b 1
)
rem set asigna variables; las comillas exteriores evitan espacios accidentales al final.
rem EXIT_CODE empieza en 1: solo se comunicará éxito cuando se confirme expresamente.
set "EXIT_CODE=1"
rem INTERACTIVE indica si se muestra el menú y se espera una tecla antes de cerrar.
set "INTERACTIVE=0"
rem Asignar sin contenido elimina la variable. RUN_MODE guardará el modo elegido.
set "RUN_MODE="
rem PYTHON_RUNTIME guardará la ruta completa del ejecutable de Python que se utilice.
set "PYTHON_RUNTIME="
rem VISUAL_OPTIONS solo recibe opciones fijas del propio BAT, nunca texto externo.
set "VISUAL_OPTIONS="

rem Los argumentos permiten usar el lanzador sin responder al menú.
rem %~1 y %~2 son el primer y segundo argumento del BAT, sin sus comillas exteriores.
rem Se admite como máximo una opción. goto continúa desde una etiqueta del archivo.
if not "%~2"=="" goto usage_error
rem El modificador /i compara texto sin distinguir mayúsculas y minúsculas.
if /i "%~1"=="--ayuda" goto help
if /i "%~1"=="--verificar" goto mode_verify
if /i "%~1"=="--reentrenar" goto mode_train
if /i "%~1"=="--comprobar-entorno" goto mode_environment
if /i "%~1"=="--consultar" goto mode_consult
if /i "%~1"=="--repaso-visual" goto mode_visual
if /i "%~1"=="--diagnostico-visual" goto mode_visual_diagnostic
if not "%~1"=="" goto usage_error

rem El doble clic abre el menú principal; las consultas no repiten el experimento.
set "INTERACTIVE=1"
:main_menu
echo.
echo PRÁCTICA MACHINE LEARNING - YouTube Trending
echo.
echo 1. Verificar la entrega con los modelos incluidos.
echo    Comprueba sus huellas, ejecuta el cuaderno y guarda sus salidas.
echo 2. Reentrenar desde cero y reemplazar los modelos de esta carpeta.
echo    Repite las búsquedas, la calibración y las pruebas. Reemplaza también los resultados.
echo    Referencia histórica: 67,7 minutos. El tiempo depende del equipo.
echo 3. Consultar modelos: elige una de las siete tareas sin entrenar.
echo    Cuando se pida la ruta, pulsa Intro para usar data\example.csv.
echo 4. Repaso visual con los modelos incluidos.
echo    Abre una copia del cuaderno en el navegador para ejecutarlo celda a celda.
echo E. Comprobar el entorno: indica si falta Python o alguna dependencia.
echo S. Salir.
echo.
rem choice devuelve la posición de la tecla: 1 a 4, 5 para E y 6 para S.
rem Se comprueba de mayor a menor; cero o valores inesperados no ejecutan tareas.
choice /c 1234ES /n /m "Selecciona 1, 2, 3, 4, E o S: "
if errorlevel 7 goto finish
if errorlevel 6 goto cancel
if errorlevel 5 goto mode_environment
if errorlevel 4 goto mode_visual
if errorlevel 3 goto mode_consult
if errorlevel 2 goto mode_train
if errorlevel 1 goto mode_verify
goto finish

rem Las líneas que empiezan por dos puntos son etiquetas: destinos para goto.
rem No crean funciones ni variables; la ejecución continúa por las líneas siguientes.
:mode_verify
set "RUN_MODE=verify"
goto configure

:mode_train
set "RUN_MODE=train"
goto configure

:mode_environment
set "RUN_MODE=environment"
goto configure

:mode_consult
rem RUN_MODE evita pasar por el dataset y por run_practice.py al consultar pesos.
set "RUN_MODE=consult"
set "INTERACTIVE=1"
goto configure

:mode_visual_diagnostic
set "VISUAL_OPTIONS=--diagnostico"
set "RUN_MODE=visual"
goto configure

:mode_visual
set "VISUAL_OPTIONS="
rem Abre una copia independiente con el Jupyter ya instalado; no instala programas.
set "RUN_MODE=visual"
goto configure

:configure
if not exist "run_practice.py" goto missing_script
rem El gestor de Python puede instalar automáticamente al detectar que no hay intérpretes.
rem Se desactiva esa opción solo para este lanzador y sus procesos hijos.
set "PYTHON_MANAGER_AUTOMATIC_INSTALL=false"
rem Utilizar el entorno del proyecto sin tener que activarlo manualmente.
rem if exist comprueba si existe el archivo. %CD% contiene la carpeta de trabajo actual.
rem Como pushd ya entró en la carpeta del BAT, se busca el entorno local del proyecto.
if exist ".venv\Scripts\python.exe" set "PYTHON_RUNTIME=%CD%\.venv\Scripts\python.exe"
if defined PYTHON_RUNTIME goto check_python
rem ML_PYTHON permite indicar un intérprete ya preparado en otra ubicación.
rem if defined comprueba si una variable tiene valor; %ML_PYTHON% obtiene ese valor.
rem Esta alternativa permite indicar un intérprete preparado fuera del proyecto.
if defined ML_PYTHON set "PYTHON_RUNTIME=%ML_PYTHON%"
if defined PYTHON_RUNTIME goto check_python
rem Si no hay entorno local, localizar Python 3.12 con el lanzador de Windows.
rem for /f recoge la salida de una orden. delims= conserva la línea completa con espacios.
rem %%P es la variable del bucle en un BAT; recibe la ruta que imprime Python.
rem py -3.12 solicita Python 3.12; -c ejecuta el pequeño fragmento indicado entre comillas.
rem El signo ^ protege la redirección para que la interprete la orden interna del bucle.
for /f "delims=" %%P in ('py -3.12 -c "import sys; print(sys.executable)" 2^>nul') do set "PYTHON_RUNTIME=%%P"
if defined PYTHON_RUNTIME goto check_python
for /f "delims=" %%P in ('python -c "import sys; print(sys.executable)" 2^>nul') do set "PYTHON_RUNTIME=%%P"
if defined PYTHON_RUNTIME goto check_python
echo ERROR: no se encuentra CPython 3.12 de 64 bits.
echo Consulta los pasos 1 a 4 de README.md para instalarlo y preparar .venv.
echo El ZIP entregado usa Python 3.12.14 para la reutilización de sus resultados.
goto finish

:check_python
if not exist "%PYTHON_RUNTIME%" goto missing_python
rem Estas variables afectan al proceso y sus hijos; setlocal restaura el entorno al salir.
rem Los procesos Python iniciados heredarán UTF-8 para sus flujos de texto.
set "PYTHONIOENCODING=utf-8"
rem Evita crear archivos .pyc de bytecode al importar módulos Python.
set "PYTHONDONTWRITEBYTECODE=1"
rem Estas tres variables limitan hilos de bibliotecas numéricas; no todos los procesos.
rem Ayudan a controlar el uso de recursos sin modificar permanentemente Windows.
set "OMP_NUM_THREADS=2"
set "OPENBLAS_NUM_THREADS=2"
set "MKL_NUM_THREADS=2"
rem No heredar una recuperación parcial de otra sesión de trabajo.
rem Se elimina esta variable para no recuperar tareas parciales de una sesión anterior.
set "ML_RESUME_INTERRUPTED="
echo.
echo Intérprete: "%PYTHON_RUNTIME%"
rem Distinguir un Python que no arranca de un entorno que necesita paquetes.
"%PYTHON_RUNTIME%" -B -c "import sys"
if errorlevel 1 goto broken_python
rem El flujo integral comprueba el entorno en run_practice.py antes de procesar.
if /i "%RUN_MODE%"=="environment" goto environment_check
if /i "%RUN_MODE%"=="consult" goto environment_check
if /i "%RUN_MODE%"=="visual" goto environment_check
if not exist "dataset.zip" goto missing_dataset

rem Ambos modos ejecutan el flujo integral y sus comprobaciones; ninguno empaqueta.
if /i "%RUN_MODE%"=="train" goto execute_train
rem La opción 1 activa reutilización; el código Python exige huellas compatibles.
set "ML_REUSE_TRAINING=1"
echo Modo: reutilización con huellas verificadas; no entrenamiento nuevo.
echo Si las huellas no coinciden, se detendrá sin forzar su aceptación.
goto execute


:environment_check
rem El diagnóstico visual puede mostrar ambos Python aunque el parche difiera.
set "CHECK_MODE=%RUN_MODE%"
if defined VISUAL_OPTIONS set "CHECK_MODE=environment"
"%PYTHON_RUNTIME%" -B "run_practice.py" --check-environment %CHECK_MODE%
set "EXIT_CODE=%ERRORLEVEL%"
if not "%EXIT_CODE%"=="0" goto finish
if /i "%RUN_MODE%"=="environment" goto environment_ok
if /i "%RUN_MODE%"=="consult" goto query_menu

:execute_visual
if not exist "review_notebook.py" goto missing_visual_script
if defined VISUAL_OPTIONS goto call_visual
echo.
echo Repaso visual: copia de estudio con gráficos, tablas y explicaciones.
echo El BAT ejecuta review_notebook.py y abre una copia de exploratory_analysis.ipynb.
echo Selecciona una celda de código en el navegador y pulsa Ctrl + Enter.
echo Revisa debajo los mensajes, las tablas y los gráficos antes de seguir.
echo Se utilizan modelos compatibles; este modo no repite el entrenamiento desde cero.
echo No cierres esta ventana mientras uses el visor. Ctrl+C permite cerrarlo.
echo.
:call_visual
"%PYTHON_RUNTIME%" -B "review_notebook.py" %VISUAL_OPTIONS%
set "EXIT_CODE=%ERRORLEVEL%"
if not "%EXIT_CODE%"=="0" goto execution_failed
goto finish

:missing_visual_script
echo ERROR: falta review_notebook.py. Extrae la práctica completa junto al BAT.
goto finish

:execute_train
rem La opción 2 elimina la variable: el cuaderno repetirá el experimento desde cero.
rem El comando Python será el mismo, pero esta variable cambia el modo del cuaderno.
set "ML_REUSE_TRAINING="
echo Modo: experimento completo desde cero, sin reutilización de búsquedas.
echo Se actualizarán datos procesados, cuaderno, modelos, recetas e informes.

:execute
echo.
echo Ejecutando run_practice.py --archive dataset.zip --verify
echo run_practice.py usa execute_notebook.py para ejecutar exploratory_analysis.ipynb.
echo La consola muestra el avance. Las salidas se guardan en el cuaderno.
echo Para revisar las celdas y los gráficos en el navegador, elige 4 al abrir el BAT.
echo.
rem Esta es la llamada principal: el BAT entrega la ejecución a run_practice.py.
rem --archive recibe la ruta del dataset; --verify activa las comprobaciones posteriores.
rem El BAT espera a que termine Python. No se pasa --package, por lo que no crea el ZIP.
"%PYTHON_RUNTIME%" -B "run_practice.py" --archive "dataset.zip" --verify
rem Se guarda inmediatamente el resultado de Python: 0 indica finalización correcta.
rem Guardar el valor evita perderlo al ejecutar otras órdenes posteriormente.
set "EXIT_CODE=%ERRORLEVEL%"
if not "%EXIT_CODE%"=="0" goto execution_failed
echo.
echo Práctica y verificaciones completadas correctamente.
echo Consulta exploratory_analysis.ipynb y reports\results.md.
echo Este lanzador no genera ni modifica el ZIP de entrega.
goto finish

:query_menu
rem Este submenú utiliza el mismo Python comprobado; no llama a fit, save ni al cuaderno.
if not exist "inference_model.py" goto missing_inference
echo.
echo CONSULTAR MODELOS YA ENTRENADOS
echo Elige una tarea. Cuando se pida la ruta, pulsa Intro para usar data\example.csv.
echo.
echo 1. Predecir la categoría del vídeo.
echo 2. Predecir si los comentarios están deshabilitados.
echo 3. Predecir si las valoraciones están deshabilitadas.
echo 4. Estimar el número de likes.
echo 5. Estimar la proporción de likes/dislikes.
echo 6. Asignar el vídeo a un grupo.
echo 7. Recomendar hasta cinco vídeos similares.
echo V. Volver al menú principal.
echo S. Salir.
echo.
rem La posición de V es 8 y la de S es 9. Solo 1 a 7 llegan a query_N.
choice /c 1234567VS /n /m "Selecciona del 1 al 7, V o S: "
if errorlevel 10 goto finish
if errorlevel 9 goto cancel
if errorlevel 8 goto main_menu
if not errorlevel 1 goto finish
goto query_%ERRORLEVEL%

:query_1
set "QUERY_TASK=category"
set "QUERY_WEIGHTS=artifacts\category.pkl"
set "QUERY_HELP=Devuelve una categoría editorial por cada fila del CSV."
goto query_input
:query_2
set "QUERY_TASK=comments_disabled"
set "QUERY_WEIGHTS=artifacts\comments.pkl"
set "QUERY_HELP=1 predice comentarios deshabilitados; 0, habilitados."
goto query_input
:query_3
set "QUERY_TASK=ratings_disabled"
set "QUERY_WEIGHTS=artifacts\ratings.pkl"
set "QUERY_HELP=1 predice valoraciones deshabilitadas; 0, habilitadas."
goto query_input
:query_4
set "QUERY_TASK=likes"
set "QUERY_WEIGHTS=artifacts\likes.pkl"
set "QUERY_HELP=Estima likes con las señales disponibles; no pronostica antes de publicar."
goto query_input
:query_5
set "QUERY_TASK=ratio"
set "QUERY_WEIGHTS=artifacts\ratio.pkl"
set "QUERY_HELP=Estima cuántos likes corresponden a cada dislike."
goto query_input
:query_6
set "QUERY_TASK=cluster"
set "QUERY_WEIGHTS=artifacts\cluster.pkl"
set "QUERY_HELP=Asigna un grupo de vídeos similares; no es una categoría editorial."
goto query_input
:query_7
set "QUERY_TASK=recommend"
set "QUERY_WEIGHTS=artifacts\recommend.pkl"
set "QUERY_HELP=Devuelve ID, título y similitud de hasta cinco vecinos textuales por consulta."

:query_input
echo.
echo Tarea: %QUERY_TASK%
echo %QUERY_HELP%
echo Verás cada vídeo con su título, identificador y resultado explicado.
echo Deja la ruta en blanco y pulsa Intro para usar el CSV incluido data\example.csv.
echo Para usar tus datos, indica un archivo CSV o una carpeta que contenga CSV.
echo La ruta relativa se interpreta desde la carpeta del BAT. Se admiten comillas.
if /i "%QUERY_TASK%"=="recommend" echo El recomendador recibe textos o un ID conocido, no un enlace de YouTube.
rem Se borra QUERY_INPUT antes de set /p: Intro no reutiliza el CSV de la consulta anterior.
set "QUERY_INPUT="
set /p "QUERY_INPUT=Ruta del CSV [Intro = data\example.csv]: "
if not exist "%QUERY_WEIGHTS%" goto query_missing_weights
echo.
rem La entrada se transmite por el entorno, nunca se inserta como código de cmd o Python.
rem subprocess recibe una lista de argumentos sin shell: espacios, ampersand y porcentajes
rem en el nombre del CSV se tratan como parte de la ruta, sin ejecutar órdenes adicionales.
rem sys.executable conserva el intérprete preparado. Se invoca la CLI existente sin modificarla.
"%PYTHON_RUNTIME%" -B -c "import os, subprocess, sys; source = os.environ.get('QUERY_INPUT', '').strip().strip(chr(34)) or 'data/example.csv'; print('Ruta de entrada: ' + source, flush=True); sys.exit(subprocess.call([sys.executable, '-B', 'inference_model.py', os.environ['QUERY_TASK'], os.environ['QUERY_WEIGHTS'], source, '--seleccionar-csv', '--formato', 'texto']))"
rem Guardar inmediatamente el resultado permite distinguir errores de consultas correctas.
set "QUERY_EXIT_CODE=%ERRORLEVEL%"
rem El código 3 significa cancelación voluntaria: no se anuncia una consulta correcta.
if "%QUERY_EXIT_CODE%"=="3" goto query_return
if not "%QUERY_EXIT_CODE%"=="0" goto query_failed
echo.
echo Consulta completada. No se han entrenado ni guardado modelos.
goto query_return

:query_failed
echo.
rem El código 2 ya incluye una causa y una solución legibles desde la CLI.
rem Otros fallos conservan su mensaje anterior y una indicación breve para el usuario.
if "%QUERY_EXIT_CODE%"=="2" goto query_retry
echo No se ha podido completar la consulta. Revisa el mensaje anterior.
:query_retry
echo Puedes repetir la consulta; pulsa Intro en la ruta para usar el CSV de ejemplo.
goto query_return

:query_missing_weights
echo.
echo ERROR: falta el modelo guardado %QUERY_WEIGHTS%.
echo Restaura los pesos de la entrega o prepara los modelos desde una copia de trabajo.
echo Esta consulta no iniciará un entrenamiento automáticamente.

:query_return
echo.
pause
goto query_menu

:missing_inference
echo ERROR: falta inference_model.py junto al BAT. Extrae la práctica completa.
goto finish

:execution_failed
echo.
echo La tarea se ha detenido. Código de salida: %EXIT_CODE%.
echo Sigue la indicación del mensaje anterior. E en el menú comprueba el entorno.
echo Para un fallo del visor, usa --diagnostico-visual y consulta README.md.
goto finish

:environment_ok
echo Entorno localizado. No se han procesado datos ni ejecutado modelos.
echo Esta comprobación no abre JupyterLab ni valida los datos, pesos o huellas.
echo Opciones 1 y 2: coloca dataset.zip junto al BAT. Opción 4: comprueba el visor con --diagnostico-visual.
set "EXIT_CODE=0"
goto finish

:missing_python
echo ERROR: la ruta configurada de Python no existe. Revisa .venv o ML_PYTHON.
goto finish
:broken_python
echo ERROR: existe "%PYTHON_RUNTIME%", pero no se puede iniciar.
echo Si moviste la práctica, recrea .venv en esta carpeta. Si usas ML_PYTHON, comprueba su ruta.
echo Consulta los pasos 1 a 4 de README.md para reparar el entorno.
goto finish
:missing_script
echo ERROR: falta run_practice.py. Coloca el BAT en la carpeta de la práctica.
goto finish
:missing_dataset
echo ERROR: falta dataset.zip en la carpeta de la práctica.
echo Descarga youtube-new.zip del enunciado y renómbralo como dataset.zip.
echo Conserva el archivo comprimido, sin descomprimirlo.
goto finish
:usage_error
echo ERROR: opción no reconocida. Utiliza --ayuda para consultar las opciones.
goto finish

:help
echo Uso: ejecutar_practica.bat [opción]
echo Sin opción: verificar, repetir el experimento, consultar modelos o repasar con gráficos.
echo --verificar          Flujo integral con reutilización verificada y controles.
echo --reentrenar         Búsquedas, calibración, ajustes y controles desde cero.
echo --comprobar-entorno   Comprueba Python, versiones fijadas e importaciones principales.
echo --consultar          Abre las siete consultas sobre modelos guardados.
echo --repaso-visual      Abre una copia del cuaderno en Jupyter para ir celda a celda.
echo --diagnostico-visual Muestra el visor, el kernel y sus versiones sin ejecutar celdas.
echo --ayuda              Muestra esta ayuda.
echo Requiere CPython 3.12 de 64 bits y los paquetes de requirements.txt.
echo El ZIP entregado usa Python 3.12.14. Opciones 1 y 2 requieren dataset.zip.
echo Consultar usa los pesos guardados. Pulsa Intro para usar data\example.csv.
echo No instala dependencias ni genera el ZIP de entrega.
set "EXIT_CODE=0"
goto finish

:cancel
echo Menú cerrado. No se iniciarán más tareas.
set "EXIT_CODE=0"

:finish
rem Mantener visible el resultado cuando se ha abierto mediante doble clic.
rem pause espera una tecla para que se pueda leer el resultado tras abrir el menú.
if "%INTERACTIVE%"=="1" pause
rem popd recupera la carpeta anterior que pushd había guardado al comienzo.
popd
rem Restaurar la página de códigos al volver a la terminal llamante.
if defined ML_PREVIOUS_CODEPAGE chcp %ML_PREVIOUS_CODEPAGE% >nul
rem endlocal restaura las variables anteriores y exit /b devuelve el código guardado.
rem El símbolo & separa dos órdenes; el valor se sustituye antes de cerrar el entorno.
endlocal & exit /b %EXIT_CODE%
