"""Contrato de entrada, transformaciones y persistencia de los modelos finales."""
# json: lee y escribe datos estructurados, configuraciones y evidencias en JSON.
import json
# pickle: serializa y recupera el estado de modelos propios y de confianza.
import pickle
# platform: consulta versiones e información del entorno de ejecución.
import platform
# warnings: emite, captura o filtra avisos sin confundirlos con errores.
import warnings
# datetime: representa fechas, horas y zonas horarias para registrar la ejecución.
# Se importa: datetime, timezone.
from datetime import datetime, timezone
# pathlib: maneja rutas de archivos y carpetas mediante objetos, como Path.
# Se importa: Path.
from pathlib import Path
# numpy: realiza cálculos numéricos y operaciones con vectores y matrices.
import numpy as np
# pandas: organiza, limpia y analiza datos tabulares mediante DataFrame y Series.
import pandas as pd
# sklearn.base: expone la interfaz que permite clonar y evaluar un clasificador.
# Se importa: BaseEstimator, ClassifierMixin.
from sklearn.base import BaseEstimator, ClassifierMixin
# sklearn: aporta algoritmos de aprendizaje automático y su versión de biblioteca.
import sklearn
# sklearn.compose: combina transformaciones de columnas o transforma el objetivo de regresión.
# Se importa: ColumnTransformer, TransformedTargetRegressor.
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
# sklearn.pipeline: encadena transformaciones y estimadores en un único flujo de ajuste.
# Se importa: Pipeline.
from sklearn.pipeline import Pipeline
# sklearn.preprocessing: codifica categorías, escala atributos y aplica transformaciones.
# Se importa: OneHotEncoder, StandardScaler, FunctionTransformer.
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
# sklearn.impute: rellena valores ausentes con estadísticas aprendidas en entrenamiento.
# Se importa: SimpleImputer.
from sklearn.impute import SimpleImputer
# sklearn.feature_extraction.text: convierte texto en atributos numéricos mediante TF-IDF.
# Se importa: TfidfVectorizer.
from sklearn.feature_extraction.text import TfidfVectorizer
# sklearn.neighbors: busca vecinos próximos y aporta modelos basados en su proximidad.
# Se importa: NearestNeighbors.
from sklearn.neighbors import NearestNeighbors
# sklearn.cluster: aporta algoritmos de agrupación de observaciones sin etiquetas.
# Se importa: KMeans.
from sklearn.cluster import KMeans
# sklearn.decomposition: reduce dimensiones mediante técnicas como PCA y SVD.
# Se importa: TruncatedSVD.
from sklearn.decomposition import TruncatedSVD

# NUMERIC: atributos numéricos candidatos; cada tarea restringe su uso.
NUMERIC = ['views', 'dislikes', 'comment_count', 'title_length', 'publish_hour', 'publish_weekday']
# CATEGORICAL: atributos nominales codificados dentro de cada ajuste.
CATEGORICAL = ['region']
# TEXT: campos textuales combinados para representar contenido.
TEXT = ['title', 'tags', 'description']


class IgnoredColumnsWarning(UserWarning):
    """Aviso de atributos recibidos que el contrato de inferencia no utiliza."""


def warn_ignored_columns(data, expected):
    """Avisa de columnas extra sin alterarlas ni incorporarlas como predictores.

    data es un DataFrame; expected es el esquema guardado. Devuelve None.
    Permite detectar etiquetas o nombres erróneos enviados por accidente.
    """
    # ignored: nombres descartados, ordenados para emitir mensajes reproducibles.
    ignored = sorted(set(data.columns) - set(expected), key=str)
    if ignored:
        warnings.warn('Columnas ignoradas en predict: ' + ', '.join(map(str, ignored)),
                      IgnoredColumnsWarning, stacklevel=3)

def frame(data, columns=None):
    """Normaliza Series, DataFrame o array a un DataFrame independiente.

    Para arrays, columns fija nombres y orden; un vector representa una fila.
    Rechaza entradas vacías, columnas duplicadas, anchura incoherente o ausencia
    de atributos reconocibles con ValueError. Nunca modifica el objeto original."""
    if isinstance(data, pd.Series):
        # data: entradas recibidas; se copian y normalizan antes de transformar.
        data = data.to_frame().T
    if not isinstance(data, pd.DataFrame):
        # values: matriz sin nombres que se valida contra columns antes de crear un DataFrame.
        values = np.asarray(data)
        if values.ndim == 1:
            # values: matriz sin nombres que se valida contra columns antes de crear un DataFrame.
            values = values.reshape(1, -1)
        if values.ndim != 2 or columns is None:
            raise ValueError('Para arrays indique feature_names en fit y conserve ese orden en predict')
        if values.shape[1] != len(columns):
            raise ValueError(f'Se esperan {len(columns)} columnas: {columns}')
        # data: entradas recibidas; se copian y normalizan antes de transformar.
        data = pd.DataFrame(values, columns=columns)
    if data.empty or not data.columns.is_unique:
        raise ValueError('Se requieren filas y nombres de columna únicos')
    if not set(data.columns) & set(TEXT + ['views', 'dislikes', 'comment_count']):
        raise ValueError('Faltan entradas reconocibles: título, etiquetas, descripción o contadores')
    return data.copy()

def features(data, columns=None):
    """Genera atributos deterministas sin aprender del conjunto recibido.

    Une texto, calcula longitud de título y hora/día UTC, y convierte contadores
    negativos o no finitos en ausentes. La imputación se aprende después en el fold.
    Devuelve una copia enriquecida y conserva el índice y el número de filas."""
    # out: copia enriquecida de las entradas; no contiene transformadores ajustados.
    out = frame(data, columns)
    for col in TEXT:
        out[col] = out[col].fillna('').astype(str) if col in out else ''
    # out['text']: contenido combinado sin aprender un vocabulario.
    out['text'] = out[TEXT].agg(' '.join, axis=1)
    # out['title_length']: longitud del título en caracteres.
    out['title_length'] = out.title.str.len()
    # dates: fechas convertidas con ausentes explícitos cuando falla el parseo.
    dates = pd.to_datetime(out.get('publish_time', pd.Series(pd.NaT, index=out.index)), errors='coerce', utc=True)
    # out['publish_hour']: hora UTC de publicación, sin usar la fecha de tendencia; out['publish_weekday']:
    # día de publicación UTC, de lunes 0 a domingo 6.
    out['publish_hour'], out['publish_weekday'] = dates.dt.hour, dates.dt.dayofweek
    for col in NUMERIC:
        if col not in out:
            out[col] = np.nan
        out[col] = pd.to_numeric(out[col], errors='coerce')
        out.loc[(out[col] < 0) | ~np.isfinite(out[col]), col] = np.nan
    # out['region']: región de aparición conservada como atributo nominal.
    out['region'] = out.region.fillna('unknown').astype(str) if 'region' in out else 'unknown'
    return out

def transformer(task, ratio_counters=True):
    # Moderación: excluir contadores contemporáneos que revelarían la etiqueta.
    """Construye el preprocesador sin ajustarlo y según las señales permitidas.

    TF-IDF aprende vocabulario; la rama numérica imputa mediana, aplica log1p y
    escala sin centrar matrices dispersas. One-hot admite regiones desconocidas.
    Moderación/categoría excluyen contadores; ratio nunca usa likes/dislikes.
    ratio_counters permite la ablación controlada de views/comment_count."""
    nums = NUMERIC if task in ['likes', 'cluster'] else NUMERIC[3:]
    if task == 'ratio' and ratio_counters:
        nums = ['views', 'comment_count'] + NUMERIC[3:]
    return ColumnTransformer([
        ('text', TfidfVectorizer(max_features=8000, ngram_range=(1, 2), min_df=2, sublinear_tf=True), 'text'),
        ('num', Pipeline([('impute', SimpleImputer(strategy='median', keep_empty_features=True)),
                          ('log', FunctionTransformer(np.log1p, feature_names_out='one-to-one')),
                          ('scale', StandardScaler(with_mean=False))]), nums),
        ('cat', OneHotEncoder(handle_unknown='ignore'), CATEGORICAL)
    ], sparse_threshold=1.0)

def recipe(task):
    """Lee la receta seleccionada para task desde recipes.json, junto al módulo.

    Devuelve un diccionario vacío si no existe archivo o receta; un JSON inválido
    produce error explícito. No entrena ni modifica parámetros persistidos."""
    path = Path(__file__).with_name('recipes.json')
    return json.loads(path.read_text(encoding='utf-8')).get(task, {}) if path.exists() else {}

class _EncodedXGBClassifier(ClassifierMixin, BaseEstimator):
    """Ajusta XGBoost con etiquetas consecutivas y devuelve las etiquetas originales."""

    def __init__(self, max_depth=4, n_estimators=80, learning_rate=0.1):
        """Conserva los hiperparámetros que GridSearchCV puede clonar y ajustar."""
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate

    def fit(self, X, y):
        """Codifica las clases presentes en este ajuste, incluso si falta alguna clase global."""
        # xgboost: implementa árboles potenciados por gradiente.
        # Se importa: XGBClassifier.
        from xgboost import XGBClassifier
        self.classes_, labels = np.unique(np.asarray(y), return_inverse=True)
        self.estimator_ = XGBClassifier(max_depth=self.max_depth, n_estimators=self.n_estimators,
                                        learning_rate=self.learning_rate, tree_method='hist',
                                        n_jobs=2, random_state=42)
        self.estimator_.fit(X, labels)
        self.n_features_in_ = self.estimator_.n_features_in_
        return self

    def predict(self, X):
        """Devuelve las etiquetas originales en el mismo orden que las filas."""
        return self.classes_[self.estimator_.predict(X).astype(int)]

    def predict_proba(self, X):
        """Devuelve probabilidades con columnas alineadas con classes_."""
        return self.estimator_.predict_proba(X)


def estimator_from_name(name):
    """Crea una instancia nueva de la familia identificada por name.

    El sufijo _log_target envuelve regresión con log1p/expm1 para volver a unidades
    originales. Las semillas y límites computacionales son explícitos. Devuelve
    un estimador sin ajustar; un nombre no registrado provoca KeyError."""
    # sklearn.linear_model: aporta modelos lineales de clasificación y regresión.
    # Se importa: LogisticRegression, Ridge, SGDClassifier.
    from sklearn.linear_model import LogisticRegression, Ridge, SGDClassifier
    # sklearn.discriminant_analysis: aporta el análisis discriminante lineal supervisado.
    # Se importa: LinearDiscriminantAnalysis.
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    # sklearn.svm: aporta máquinas de vectores soporte para clasificación y regresión.
    # Se importa: LinearSVC, LinearSVR, SVC, SVR.
    from sklearn.svm import LinearSVC, LinearSVR, SVC, SVR
    # sklearn.naive_bayes: aporta clasificadores probabilísticos Naive Bayes.
    # Se importa: MultinomialNB.
    from sklearn.naive_bayes import MultinomialNB
    # sklearn.neighbors: busca vecinos próximos y aporta modelos basados en su proximidad.
    # Se importa: KNeighborsClassifier, KNeighborsRegressor.
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
    # sklearn.tree: aporta árboles de decisión para clasificación y regresión.
    # Se importa: DecisionTreeClassifier, DecisionTreeRegressor.
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    # sklearn.ensemble: aporta modelos que combinan árboles u otros estimadores.
    # Se importa: RandomForestClassifier, RandomForestRegressor, ExtraTreesRegressor.
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, ExtraTreesRegressor
    # sklearn.neural_network: aporta redes neuronales multicapa para clasificación y regresión.
    # Se importa: MLPClassifier, MLPRegressor.
    from sklearn.neural_network import MLPClassifier, MLPRegressor
    # sklearn.dummy: crea referencias simples para comparar el rendimiento de los modelos.
    # Se importa: DummyClassifier, DummyRegressor.
    from sklearn.dummy import DummyClassifier, DummyRegressor
    # log: indicador de transformación logarítmica del objetivo.
    log = name.endswith('_log_target')
    # key: identificador de familia o entrada de metadatos del bloque.
    key = name.removesuffix('_log_target')
    # estimators: factorías que crean un estimador independiente por nombre.
    def xgb_regressor():
        """Crea un regresor XGBoost de CPU para objetivos directos o logarítmicos."""
        # xgboost: implementa árboles potenciados por gradiente.
        # Se importa: XGBRegressor.
        from xgboost import XGBRegressor
        return XGBRegressor(max_depth=4, n_estimators=80, learning_rate=0.1,
                            tree_method='hist', n_jobs=2, random_state=42)

    estimators = {
        'logistic': lambda: LogisticRegression(max_iter=3000, class_weight='balanced', random_state=42),
        'svm': lambda: LinearSVC(class_weight='balanced', dual=False, max_iter=30000, random_state=42),
        'svm_rbf': lambda: SVC(kernel='rbf', class_weight='balanced', cache_size=256, max_iter=30000, random_state=42),
        'svr_rbf': lambda: SVR(kernel='rbf', cache_size=256, max_iter=30000),
        'naive_bayes': lambda: MultinomialNB(),
        'lda_classifier': lambda: LinearDiscriminantAnalysis(solver='lsqr'),
        'xgb_classifier': _EncodedXGBClassifier,
        'xgb_regressor': xgb_regressor,
        'sgd': lambda: SGDClassifier(loss='log_loss', class_weight='balanced', random_state=42),
        'knn_classifier': lambda: KNeighborsClassifier(weights='distance', metric='cosine', algorithm='brute', n_jobs=1),
        'knn_regressor': lambda: KNeighborsRegressor(weights='distance', metric='cosine', algorithm='brute', n_jobs=1),
        'tree_classifier': lambda: DecisionTreeClassifier(max_features='sqrt', class_weight='balanced', random_state=42),
        'tree_regressor': lambda: DecisionTreeRegressor(max_features='sqrt', random_state=42),
        'forest_classifier': lambda: RandomForestClassifier(n_estimators=80, max_depth=20, class_weight='balanced', n_jobs=1, random_state=42),
        'forest_regressor': lambda: RandomForestRegressor(n_estimators=80, max_depth=20, max_features='sqrt', n_jobs=1, random_state=42),
        'extra_trees': lambda: ExtraTreesRegressor(n_estimators=80, max_depth=20, max_features='sqrt', n_jobs=1, random_state=42),
        'mlp_classifier': lambda: MLPClassifier(hidden_layer_sizes=(32,), max_iter=240, early_stopping=False, random_state=42),
        'mlp_regressor': lambda: MLPRegressor(hidden_layer_sizes=(32,), max_iter=240, early_stopping=True, random_state=42),
        'ridge': lambda: Ridge(solver='lsqr'),
        'svr': lambda: LinearSVR(dual='auto', max_iter=30000, random_state=42),
        'baseline_classifier': lambda: DummyClassifier(strategy='most_frequent'),
        'baseline_regressor': lambda: DummyRegressor(strategy='median'),
    }
    # estimator: estimador concreto de la familia, distinto del preprocesador.
    estimator = estimators[key]()
    return TransformedTargetRegressor(regressor=estimator, func=np.log1p, inverse_func=np.expm1, check_inverse=False) if log else estimator

class BaseModel:
    """Contrato común de ajuste, predicción, esquema y persistencia de las tareas."""
    # task: identificador de la tarea; determina atributos, métricas y compatibilidad de pesos.
    task = None

    def __init__(self, estimator=None, params=None, feature_names=None):
        """Crea el estimador configurado, sin entrenar ni cargar pesos.

        Si estimator y params se omiten, utiliza la receta seleccionada de esta tarea.
        feature_names fija el esquema de arrays. fit interpreta params: rutas de
        Pipeline en modelos supervisados, opciones de SVD/clustering o TF-IDF en
        las otras clases. Conserva el umbral OOF de la receta para predecir."""
        # defaults: valores de la receta usados sólo si no se proporcionan parámetros explícitos.
        defaults = recipe(self.task) if estimator is None and params is None else {}
        # self.estimator: estimador inyectado o construido con la receta seleccionada.
        self.estimator = estimator
        # self.params: copia de hiperparámetros para no modificar el diccionario recibido.
        self.params = dict(defaults.get('params', {}) if params is None else params)
        # self.feature_names: esquema de nombres y orden exigido a los arrays de entrada.
        self.feature_names = feature_names
        # self.recipe_name: familia ganadora usada por el constructor sin argumentos.
        self.recipe_name = defaults.get('model')
        # self.decision_threshold_: umbral binario OOF; None conserva la decisión predeterminada.
        self.decision_threshold_ = defaults.get('threshold')
        # Crear el objeto sin ajustar: pipeline_ sólo existirá tras fit o load.
        if self.estimator is None:
            self.estimator = self._create_estimator()
            estimator_params = {key.removeprefix('estimator__'): value
                                for key, value in self.params.items()
                                if key.startswith('estimator__')}
            if estimator_params:
                self.estimator.set_params(**estimator_params)

    def _create_estimator(self):
        """Construye la familia de la receta sin aprender pesos ni transformaciones.

        Las clases no supervisadas sobrescriben este auxiliar. Un estimador
        recibido explícitamente se respeta y no pasa por esta factoría.
        """
        name = self.recipe_name or ('logistic' if self.task in
            ['category', 'comments_disabled', 'ratings_disabled'] else 'ridge_log_target')
        return estimator_from_name(name)

    def _validate_classification_text(self, data):
        """Rechaza filas sin texto en las tres tareas de clasificación."""
        if self.task in ['category', 'comments_disabled', 'ratings_disabled']:
            text = data.reindex(columns=TEXT).fillna('').astype(str).agg(' '.join, axis=1)
            if not text.str.strip().ne('').all():
                raise ValueError('Cada observación de clasificación necesita contenido textual')

    def fit(self, X, y=None):
        """Ajusta el Pipeline de esta tarea con X y una etiqueta y por fila.

        Acepta DataFrame, Series o array con feature_names; conserva un esquema
        permitido sin columnas objetivo. Aprende vocabulario, imputación y escala
        sólo aquí. Devuelve self; rechaza etiquetas ausentes, longitudes incoherentes
        y texto vacío en clasificación mediante ValueError."""
        X = frame(X, self.feature_names)
        self._validate_classification_text(X)
        if y is None or len(y) != len(X) or pd.isna(np.asarray(y)).any():
            raise ValueError('y debe tener una etiqueta válida por fila')
        # allowed: columnas admitidas para evitar incluir etiquetas como predictores.
        allowed = TEXT + ['publish_time', 'region']
        if self.task == 'likes':
            # allowed: columnas admitidas para evitar incluir etiquetas como predictores.
            allowed += ['views', 'dislikes', 'comment_count']
        elif self.task == 'ratio':
            # allowed: columnas admitidas para evitar incluir etiquetas como predictores.
            allowed += ['views', 'comment_count']
        # self.columns_: esquema permitido que excluye etiquetas y fija el orden de los arrays.
        self.columns_ = [c for c in X.columns if c in allowed]
        if self.estimator is None:
            name = self.recipe_name or ('logistic' if self.task in ['category', 'comments_disabled', 'ratings_disabled'] else 'ridge_log_target')
            # self.estimator: estimador inyectado o construido con la receta seleccionada.
            self.estimator = estimator_from_name(name)
        # self.pipeline_: preprocesador y estimador aprendidos que se reutilizan al predecir.
        # La reducción de RBF/MLP aprende dentro de cada ajuste, también al predecir desde pickle.
        steps = [('features', transformer(self.task))]
        if any(key.startswith('reduce__') for key in self.params):
            steps.append(('reduce', TruncatedSVD(n_components=30, random_state=42)))
            steps.append(('latent_scale', StandardScaler()))
        steps.append(('estimator', self.estimator))
        self.pipeline_ = Pipeline(steps)
        self.pipeline_.set_params(**{k: tuple(v) if k.endswith('ngram_range') else v for k, v in self.params.items()})
        self.pipeline_.fit(features(X), y)
        return self

    def predict(self, X):
        """Predice una fila o lote con los transformadores ya aprendidos.

        Los arrays deben respetar columns_; devuelve un vector, incluso para una fila.
        En moderación aplica el umbral OOF si existe. No reajusta transformadores ni
        modifica las entradas. Exige fit/load previo y rechaza esquemas inválidos."""
        if not hasattr(self, 'pipeline_'):
            raise RuntimeError('Debe ejecutar fit o load primero')
        X = frame(X, self.columns_)
        warn_ignored_columns(X, self.columns_)
        self._validate_classification_text(X)
        # prepared: atributos deterministas preparados para el Pipeline aprendido.
        prepared = features(X)
        if self.decision_threshold_ is not None and self.task in ['comments_disabled', 'ratings_disabled']:
            return (positive_scores(self.pipeline_, prepared) >= self.decision_threshold_).astype(int)
        return self.pipeline_.predict(prepared)

    def save(self, path):
        """Escribe un pickle propio y su manifiesto JSON de compatibilidad.

        path identifica el archivo; crea sus directorios padre. Devuelve self.
        Rechaza modelos sin ajuste con RuntimeError. El manifiesto conserva tarea,
        esquema, versiones y procedencia; no hace seguro cargar pickles ajenos."""
        if not any(hasattr(self, attr) for attr in ['pipeline_', 'index_']):
            raise RuntimeError('No se puede guardar un modelo sin entrenar')
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # manifest: metadatos de procedencia y compatibilidad, o hashes de contenido en el empaquetado.
        manifest = {'format': 2, 'task': self.task, 'class': type(self).__name__,
                    'created_utc': datetime.now(timezone.utc).isoformat(),
                    'python': platform.python_version(), 'sklearn': sklearn.__version__,
                    'numpy': np.__version__, 'pandas': pd.__version__, 'schema': self.columns_,
                    'training': getattr(self, 'training_manifest_', {})}
        with path.open('wb') as stream:
            pickle.dump({'manifest': manifest, 'state': self.__dict__}, stream, protocol=pickle.HIGHEST_PROTOCOL)
        path.with_suffix('.manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
        return self

    def load(self, path):
        # Sólo pickles propios y de confianza: el formato no evita ejecución de código.
        """Carga pesos propios y de confianza tras validar su manifiesto.

        path debe existir y contener formato 2, misma tarea/clase y versión sklearn.
        Devuelve self; archivo ausente produce FileNotFoundError y formato, corrupción
        o incompatibilidad detectada producen ValueError antes de reemplazar el estado.
        Pickle puede ejecutar código: esta API presupone procedencia de confianza."""
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f'No se encuentran los pesos: {path}')
        try:
            with path.open('rb') as stream:
                # payload: contenedor persistido con manifiesto y estado aprendido.
                payload = pickle.load(stream)
        except (pickle.UnpicklingError, EOFError) as exc:
            raise ValueError(f'Archivo pickle incompleto o corrupto: {path}') from exc
        if not isinstance(payload, dict) or payload.get('manifest', {}).get('format') != 2:
            raise ValueError('Formato sin manifiesto compatible; regenere los pesos con train_models.py')
        # manifest: metadatos de procedencia y compatibilidad, o hashes de contenido en el empaquetado.
        manifest = payload['manifest']
        if manifest.get('task') != self.task or manifest.get('class') != type(self).__name__:
            raise ValueError(f"Pesos de {manifest.get('task')} incompatibles con {self.task}")
        if manifest.get('sklearn') != sklearn.__version__:
            raise ValueError(f"Versión sklearn incompatible: {manifest.get('sklearn')} frente a {sklearn.__version__}")
        if not isinstance(payload.get('state'), dict):
            raise ValueError('Estado del modelo inválido')
        self.__dict__.update(payload['state'])
        # self.manifest_: manifiesto validado que acredita tarea, versión y procedencia del archivo.
        self.manifest_ = manifest
        return self

def positive_scores(pipeline, prepared):
    """Obtiene puntuaciones de la clase positiva 1 sin aplicar un nuevo umbral.

    Prioriza predict_proba y después decision_function; el respaldo convierte
    predicciones a float. Las puntuaciones SVM no se presentan como probabilidades.
    Devuelve un vector alineado con prepared."""
    if hasattr(pipeline, 'predict_proba'):
        return pipeline.predict_proba(prepared)[:, list(pipeline.classes_).index(1)]
    if hasattr(pipeline, 'decision_function'):
        return pipeline.decision_function(prepared)
    return np.asarray(pipeline.predict(prepared), dtype=float)


def cluster_estimator(name, k, params):
    """Construye la familia de segmentación congelada sin ajustar datos.

    k fija centros o componentes; Mean Shift conserva bandwidth en params y
    determina sus centros al ajustar. Todas estas familias admiten predict.
    """
    # sklearn.cluster: aporta algoritmos de agrupación de observaciones sin etiquetas.
    # Se importa: MiniBatchKMeans, MeanShift.
    from sklearn.cluster import MiniBatchKMeans, MeanShift
    # sklearn.mixture: aporta mezclas gaussianas para modelar grupos probabilísticos.
    # Se importa: GaussianMixture.
    from sklearn.mixture import GaussianMixture
    seed = params.get('seed', 42)
    if name == 'gmm':
        return GaussianMixture(n_components=k, covariance_type=params.get('covariance_type', 'diag'),
                               n_init=3, max_iter=300, reg_covar=1e-5, random_state=seed)
    if name == 'meanshift':
        return MeanShift(bandwidth=params['bandwidth'], bin_seeding=True,
                         cluster_all=True, max_iter=300, n_jobs=1)
    if name not in ['kmeans', 'minibatch']:
        raise ValueError(f'Familia de clustering desconocida: {name}')
    cls = MiniBatchKMeans if name == 'minibatch' else KMeans
    return cls(n_clusters=k, n_init=10, random_state=seed)
