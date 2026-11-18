"""Segmentación inductiva con reducción ajustada únicamente al catálogo de ajuste."""
# .core: comparte BaseModel y auxiliares de preparación, algoritmos y persistencia.
# Se importan: BaseModel, frame, features, transformer, Pipeline, TruncatedSVD, recipe y cluster_estimator.
from .core import BaseModel, frame, features, transformer, Pipeline, TruncatedSVD, recipe, cluster_estimator

class KMeansClustering(BaseModel):
    """Segmentación inductiva con preprocesamiento, SVD y estimador persistidos."""
    # task: identificador de la tarea; determina atributos, métricas y compatibilidad de pesos.
    task = 'cluster'

    def _create_estimator(self):
        """Crea el algoritmo de la receta; fit limita sus grupos al tamaño disponible."""
        config = recipe(self.task)
        return cluster_estimator(config.get('model', 'kmeans'), config.get('k', 5),
                                 dict(config.get('params', {}), **self.params))

    def fit(self, X, y=None):
        """Aprende preprocesamiento, SVD y la familia de segmentación sobre X; ignora y.

        Limita dimensiones al rango disponible; exige dos filas y atributos útiles.
        Devuelve self para encadenar operaciones. KMeans, MiniBatch, Mean Shift y GMM
        admiten predict sobre nuevas filas con el mismo preprocesador persistido."""
        # X: entradas normalizadas conservando el orden y esquema de atributos.
        X = frame(X, self.feature_names)
        # self.columns_: esquema permitido que excluye etiquetas y fija el orden de los arrays.
        self.columns_ = [c for c in X.columns if c in ['title','tags','description','region','publish_time','views','dislikes','comment_count']]
        # prep: transformador que aprende exclusivamente de las filas de ajuste.
        prep = transformer(self.task)
        # matrix: representación dispersa aprendida sobre el conjunto de ajuste.
        matrix = prep.fit_transform(features(X))
        # dimensions: componentes efectivos, limitados por filas y atributos disponibles.
        dimensions = min(self.params.get('dimensions', 30), matrix.shape[0]-1, matrix.shape[1]-1)
        if dimensions < 1:
            raise ValueError('Se necesitan al menos dos observaciones y dos atributos')
        # reducer: SVD ajustado sólo con las filas de entrenamiento.
        reducer = TruncatedSVD(n_components=dimensions, random_state=self.params.get('seed', 42))
        # reduced: coordenadas SVD que se usan para entrenar el clustering.
        reduced = reducer.fit_transform(matrix)
        # Se respeta el estimador recibido o se reconstruye con la misma receta del constructor.
        estimator = self.estimator if self.estimator is not None else self._create_estimator()
        # Limitar centros/componentes sin cambiar el algoritmo elegido.
        for key in ['n_clusters', 'n_components']:
            if key in estimator.get_params():
                estimator.set_params(**{key: min(estimator.get_params()[key], len(X))})
        estimator.fit(reduced)
        # self.pipeline_: preprocesador y estimador aprendidos que se reutilizan al predecir.
        self.pipeline_ = Pipeline([('features', prep), ('reduce', reducer), ('estimator', estimator)])
        return self
