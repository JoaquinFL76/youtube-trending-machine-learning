"""Modelo público de la tarea; hereda el contrato documentado de BaseModel."""
# .svm_category_classifier: expone la clase que predice la categoría editorial del vídeo.
# Se importa: SVMCategoryClassifier.
from .svm_category_classifier import SVMCategoryClassifier
# .svm_comments_classifier: expone la clase que predice si los comentarios están deshabilitados.
# Se importa: SVMCommentsClassifier.
from .svm_comments_classifier import SVMCommentsClassifier
# .logistic_ratings_classifier: expone la clase que predice si las valoraciones están deshabilitadas.
# Se importa: LogisticRatingsClassifier.
from .logistic_ratings_classifier import LogisticRatingsClassifier
# .svr_likes_regressor: expone la clase que estima el número de likes.
# Se importa: SVRLikesRegressor.
from .svr_likes_regressor import SVRLikesRegressor
# .ridge_ratio_regressor: expone la clase que estima la ratio likes/dislikes.
# Se importa: RidgeRatioRegressor.
from .ridge_ratio_regressor import RidgeRatioRegressor
# .kmeans_clustering: expone la clase que aprende grupos y asigna nuevos vídeos a ellos.
# Se importa: KMeansClustering.
from .kmeans_clustering import KMeansClustering
# .tfidf_knn_recommender: expone la clase que recupera vídeos con contenido similar.
# Se importa: TfidfKNNRecommender.
from .tfidf_knn_recommender import TfidfKNNRecommender

# REGISTRY: relación entre identificador de tarea y clase pública.
REGISTRY = {'category': SVMCategoryClassifier, 'comments_disabled': SVMCommentsClassifier, 'ratings_disabled': LogisticRatingsClassifier, 'likes': SVRLikesRegressor, 'ratio': RidgeRatioRegressor, 'cluster': KMeansClustering, 'recommend': TfidfKNNRecommender}
