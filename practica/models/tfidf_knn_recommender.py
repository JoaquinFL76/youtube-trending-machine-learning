"""Recuperación por contenido con control de identidad y texto repetitivo."""
# re: busca y transforma texto mediante expresiones regulares.
import re
# numpy: realiza cálculos numéricos y operaciones con vectores y matrices.
import numpy as np
# pandas: organiza, limpia y analiza datos tabulares mediante DataFrame y Series.
import pandas as pd
# .core: comparte BaseModel y auxiliares de preparación, algoritmos y persistencia.
# Se importa: BaseModel, frame, TfidfVectorizer, NearestNeighbors, warn_ignored_columns.
from .core import BaseModel, frame, TfidfVectorizer, NearestNeighbors, warn_ignored_columns


class TfidfKNNRecommender(BaseModel):
    """Recupera vecinos textuales; su similitud no acredita relevancia humana."""

    # task: identifica la clase y su receta al guardar y cargar el modelo.
    task = 'recommend'

    def _create_estimator(self):
        """Crea vecinos coseno sin índice aprendido; index_ se define sólo en fit."""
        return NearestNeighbors(metric='cosine', algorithm='brute')

    def _content(self, data):
        """Compone texto según opciones aprendidas, sin usar categorías ni región.

        legacy concatena tres campos. focused pondera título/etiquetas, elimina
        URL y limita caracteres de descripción para reducir plantillas. Devuelve
        una Series alineada. Los pesos repiten campos; no son probabilidades.
        """
        # fields: texto por campo; las ausencias se representan como cadenas vacías.
        fields = {name: data[name].fillna('').astype(str) if name in data
                  else pd.Series('', index=data.index)
                  for name in ['title', 'tags', 'description']}
        # options: configuración persistida; los pesos antiguos admiten modo legacy.
        options = getattr(self, 'content_options_', {'mode': 'legacy'})
        if options['mode'] == 'legacy':
            return fields['title'] + ' ' + fields['tags'] + ' ' + fields['description']
        for name in fields:
            fields[name] = fields[name].str.replace(
                r'https?://\S+|www\.\S+', '', regex=True, flags=re.IGNORECASE)
        fields['tags'] = fields['tags'].mask(fields['tags'].str.lower().isin(['[none]', 'none']), '')
        # description: prefijo acotado; cero excluye la descripción por completo.
        description = fields['description'].str.slice(0, options['description_chars'])
        return ((fields['title'] + ' ') * options['title_weight']
                + (fields['tags'] + ' ') * options['tags_weight'] + description)

    def fit(self, X, y=None):
        """Aprende vocabulario e índice coseno sobre al menos dos IDs distintos.

        X admite DataFrame o array con feature_names; y se ignora. params admite
        opciones TF-IDF, content_mode (legacy/focused), description_chars,
        title_weight y tags_weight. Los límites/pesos son enteros no booleanos.
        Devuelve self; catálogo, esquema o vocabulario inválido producen ValueError.
        El título es opcional: los resultados sin título lo representan con una cadena vacía.
        Valida identidad y opciones antes de reemplazar el catálogo del objeto.
        """
        # data: copia independiente para normalizar IDs sin modificar la entrada.
        data = frame(X, self.feature_names)
        if 'video_id' not in data:
            raise ValueError('El catálogo necesita video_id')
        data = data.loc[data.video_id.notna()].copy()
        data['video_id'] = data.video_id.astype(str).str.strip()
        data = data.loc[~data.video_id.isin(['#NAME?', '', 'nan', 'None'])]
        # catalog: el mínimo se comprueba DESPUÉS de deduplicar la identidad.
        catalog = data.drop_duplicates('video_id').reset_index(drop=True)
        if len(catalog) < 2:
            raise ValueError('El catálogo necesita al menos dos vídeos distintos y válidos')
        # settings: opciones TF-IDF, separadas de las opciones de contenido.
        settings = {'max_features': 15000, 'ngram_range': (1, 2), 'sublinear_tf': True}
        settings.update(self.params)
        # options: tratamiento de campos que se conservará con el estado aprendido.
        options = {'mode': settings.pop('content_mode', 'legacy'),
                   'description_chars': settings.pop('description_chars', 400),
                   'title_weight': settings.pop('title_weight', 3),
                   'tags_weight': settings.pop('tags_weight', 2)}
        if options['mode'] not in ['legacy', 'focused']:
            raise ValueError('content_mode debe ser legacy o focused')
        for name in ['description_chars', 'title_weight', 'tags_weight']:
            if type(options[name]) is not int or options[name] < (0 if name == 'description_chars' else 1):
                raise ValueError(f'{name} debe ser un entero válido, sin booleanos')
        if 'ngram_range' in settings:
            settings['ngram_range'] = tuple(settings['ngram_range'])
        self.content_options_ = options
        self.columns_ = [name for name in data.columns if name in ['video_id', 'title', 'tags', 'description']]
        self.catalog_ = catalog.copy()
        self.vectorizer_ = TfidfVectorizer(**settings)
        self.matrix_ = self.vectorizer_.fit_transform(self._content(self.catalog_))
        self.index_ = self.estimator.fit(self.matrix_)
        return self

    def _encode_queries(self, data, *, require_content=False):
        """Vectoriza texto con el vocabulario aprendido y devuelve una matriz dispersa.

        require_content rechaza consultas vacías antes de transformar. La evaluación
        conserva las filas vacías para diagnosticar consultas sin vocabulario conocido.
        Requiere fit/load previo. La resolución de IDs se realiza en predict.
        """
        if not hasattr(self, 'vectorizer_'):
            raise RuntimeError('Debe ejecutar fit o load primero')
        content = self._content(frame(data, self.columns_))
        if require_content and not content.str.strip().ne('').all():
            raise ValueError('Consulta vacía: aporte contenido o un ID conocido')
        return self.vectorizer_.transform(content)

    def predict(self, X):
        """Devuelve hasta cinco vecinos textuales, excluyendo la propia identidad.

        Admite texto nuevo, array con el esquema guardado o ID conocido sin texto.
        Ordena por similitud coseno positiva y devuelve una lista de listas incluso
        para una fila. ID desconocido y consultas vacías/OOV producen ValueError;
        la ausencia de fit/load produce RuntimeError. Avisa de columnas ignoradas.
        """
        if not hasattr(self, 'index_'):
            raise RuntimeError('Debe ejecutar fit o load primero')
        if isinstance(X, pd.Series):
            X = X.to_frame().T
        if isinstance(X, pd.DataFrame):
            X = X.copy()
            warn_ignored_columns(X, self.columns_)
            if 'video_id' in X:
                X['video_id'] = X.video_id.astype(str).str.strip()
            if 'video_id' in X and not set(X.columns) & {'title', 'tags', 'description'}:
                # lookup: resolución por ID normalizado, sin inventar contenido.
                lookup = self.catalog_.set_index('video_id')
                if not X.video_id.isin(lookup.index).all():
                    raise ValueError('ID desconocido: aporte título, etiquetas o descripción')
                X = lookup.loc[X.video_id].reset_index()
        # query: filas normalizadas; los arrays conservan el orden de columns_.
        query = frame(X, self.columns_)
        vectors = self._encode_queries(query, require_content=True)
        if np.any(np.asarray(vectors.getnnz(axis=1)) == 0):
            raise ValueError('Consulta sin vocabulario conocido; no se pueden justificar recomendaciones')
        # Se recuperan seis vecinos para poder retirar la propia identidad.
        distances, indices = self.index_.kneighbors(vectors, n_neighbors=min(6, len(self.catalog_)))
        result = []
        for position, (_, row) in enumerate(query.iterrows()):
            records = []
            for distance, index in zip(distances[position], indices[position]):
                item = self.catalog_.iloc[index]
                if item.video_id != row.get('video_id') and distance < 1 - 1e-12:
                    records.append({'video_id': str(item.video_id), 'title': '' if pd.isna(item.get('title', '')) else str(item.get('title', '')),
                                    'similarity': float(1 - distance)})
            result.append(records[:5])
        return result
