"""Clustering con cobertura comparable y recuperación con validación y prueba separadas."""
# json: lee y escribe datos estructurados, configuraciones y evidencias en JSON.
import json
# numpy: realiza cálculos numéricos y operaciones con vectores y matrices.
import numpy as np
# pandas: organiza, limpia y analiza datos tabulares mediante DataFrame y Series.
import pandas as pd
# sklearn.base: clona estimadores sin reutilizar el estado aprendido de otro ajuste.
# Se importa: clone.
from sklearn.base import clone
# sklearn.cluster: aporta algoritmos de agrupación de observaciones sin etiquetas.
# Se importa: KMeans, MiniBatchKMeans, AgglomerativeClustering, DBSCAN.
from sklearn.cluster import KMeans, MiniBatchKMeans, AgglomerativeClustering, DBSCAN
# sklearn.decomposition: reduce dimensiones mediante técnicas como PCA y SVD.
# Se importa: TruncatedSVD.
from sklearn.decomposition import TruncatedSVD
# sklearn.model_selection: organiza particiones, validación cruzada y búsquedas de parámetros.
# Se importa: train_test_split.
from sklearn.model_selection import train_test_split
# sklearn.metrics: calcula métricas para evaluar predicciones, grupos y similitudes.
# Se importa: silhouette_score, adjusted_rand_score, normalized_mutual_info_score.
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
# models: expone las siete clases finales y el registro que relaciona tareas con clases.
# Se importa: REGISTRY.
from models import REGISTRY
# models.core: comparte validación, transformaciones, factorías y persistencia de modelos.
# Se importa: transformer, features.
from models.core import transformer, features
# cluster_diagnostics: exporta perfiles, términos y proyecciones del clustering.
# Se importa: export_cluster_diagnostics.
from cluster_diagnostics import export_cluster_diagnostics
# recommendation_analysis: evalúa el proxy de recuperación y prepara la revisión humana.
# Se importa: evaluate_retrieval, export_diagnostics, export_human_review.
from recommendation_analysis import evaluate_retrieval, export_diagnostics, export_human_review

def retrieval_metrics(model, queries, catalog):
    """Delega la evaluación en una única representación compartida con predict.

    Devuelve métricas proxy y listas; las consultas OOV cuentan como vacías.
    Exige identidades disjuntas y no sustituye la evaluación humana.
    """
    return evaluate_retrieval(model, queries, catalog)


def run_unsupervised(final,base,conf,seed,manifest):
    """Compara segmentaciones y representaciones de recuperación sólo con final.

    Selecciona por silhouette entre soluciones con tamaño y estabilidad admitidos; informa
    estabilidad condicionada a SVD fijo y cobertura de DBSCAN. Selecciona recuperación
    en validación y evalúa consultas de prueba disjuntas antes del reajuste final.
    Guarda pesos/tablas y devuelve resultados y recetas para los dos problemas."""
    # out: destino de informes o resultado intermedio según la operación local.
    out = base/conf['output']['reports']
    # unique: una fila por vídeo para no ponderar apariciones repetidas.
    unique = final.drop_duplicates('video_id').reset_index(drop=True)
    # sample: submuestra de observaciones usada por la operación local.
    sample = unique.sample(n=min(conf.getint('training','cluster_rows'),len(unique)),random_state=seed).reset_index(drop=True)
    # matrix: representación dispersa aprendida sobre el conjunto de ajuste.
    matrix = transformer('cluster').fit_transform(features(sample))
    # ix: índices comunes de evaluación para comparar segmentaciones en las mismas filas.
    ix = np.random.default_rng(seed).choice(len(sample),min(1000,len(sample)),replace=False)
    # trials: ensayos acumulados con puntuaciones, tiempos y avisos; winning: mejor segmentación inductiva
    # según silhouette comparable.
    # clustering_experiments: compara familias de clustering y calcula estabilidad.
    # Se importa: compare_clusterings, cluster_count.
    from clustering_experiments import compare_clusterings, cluster_count, SELECTION_POLICY
    trials, winning = compare_clusterings(matrix, sample.category, seed)
    # cluster: modelo inductivo de segmentación con preprocesador y reducción persistidos.
    cluster = REGISTRY['cluster'](estimator=clone(winning[2]),params=winning[6]).fit(sample)
    cluster.training_manifest_ = dict(manifest,fit_rows=len(sample),selected=winning[1])
    cluster.save(base/conf['output']['cluster'])
    # labels: orden de etiquetas que fija matrices y medias macro.
    labels = cluster.predict(sample)
    diagnostics = export_cluster_diagnostics(cluster, sample, trials, out)
    pd.crosstab(pd.Series(labels,name='cluster'),sample.category).to_csv(out/'cluster_categories.csv')
    # profiled: vídeos con su etiqueta de cluster para describir perfiles.
    profiled = sample.assign(cluster=labels)
    # profiles: resúmenes por cluster; describen actividad relativa, sin validar segmentos comerciales.
    profiles = profiled.groupby('cluster').agg(n=('video_id','size'),views_median=('views','median'),likes_median=('likes','median'),
        dislikes_median=('dislikes','median'),comments_median=('comment_count','median'),title_length_median=('title',lambda s:s.fillna('').str.len().median()),
        top_category=('category',lambda s:s.mode().iloc[0]),top_region=('region',lambda s:s.mode().iloc[0]))
    # order: rango de audiencia mediana usado para interpretar perfiles relativos.
    order = profiles.views_median.rank(method='first').astype(int)
    profiles['interpretation'] = [f'Nivel relativo de audiencia {order.loc[i]}/{len(profiles)}; {row.views_median:,.0f} vistas medianas, {row.comments_median:,.0f} comentarios. Categoría modal: {row.top_category}.' for i,row in profiles.iterrows()]
    profiles.to_csv(out/'cluster_profiles.csv')
    profiled[['video_id','category','cluster','views','likes','dislikes','comment_count']].to_csv(out/'cluster_sample.csv',index=False)
    # catalog_all: catálogo ampliado usado sólo después de evaluar la recuperación.
    catalog_all = unique.sample(n=min(conf.getint('training','recommend_rows'),len(unique)),random_state=seed).reset_index(drop=True)
    # catalog_idx: índices del catálogo de recuperación, disjuntos de las consultas; query_idx: índices
    # reservados para consultas de validación y prueba.
    catalog_idx, query_idx = train_test_split(np.arange(len(catalog_all)),test_size=.4,random_state=seed)
    # val_idx: índices de consultas para elegir la representación; test_idx: índices del conjunto reservado de
    # prueba.
    val_idx,test_idx = train_test_split(query_idx,test_size=.5,random_state=seed)
    # catalog: catálogo de categorías o vídeos consultable por identificador en este bloque.
    catalog = catalog_all.iloc[catalog_idx]
    # val: consultas de validación para seleccionar la representación de recuperación.
    val = catalog_all.iloc[val_idx].sample(n=min(300,len(val_idx)),random_state=seed)
    # test: muestra de evaluación reservada y ajena a la selección del bloque.
    test = catalog_all.iloc[test_idx].sample(n=min(300,len(test_idx)),random_state=seed)
    if not (not set(val.video_id)&set(test.video_id)):
        raise ValueError('Comprobación incumplida en unsupervised_analysis.py: not set(val.video_id)&set(test.video_id)')
    # retrieval_trials: comparación de representaciones sobre consultas de validación; chosen: representación
    # elegida por el proxy de validación.
    retrieval_trials, chosen = [],None
    for name, params in [
        ('tfidf', {}),
        ('binary_cosine', {'use_idf': False, 'binary': True}),
        ('char_tfidf', {'analyzer': 'char_wb', 'ngram_range': (3, 5)}),
        ('focused_words', {'content_mode': 'focused'}),
        ('focused_chars', {'content_mode': 'focused', 'analyzer': 'char_wb', 'ngram_range': (3, 5)}),
        ('focused_metadata', {'content_mode': 'focused', 'description_chars': 0,
                              'analyzer': 'char_wb', 'ngram_range': (3, 5)}),
    ]:
        # model: instancia de la tarea que se ajusta, carga o comprueba.
        model = REGISTRY['recommend'](params=params).fit(catalog)
        # metrics: métricas del conjunto que corresponde a esta fase de evaluación.
        metrics,_ = retrieval_metrics(model,val,catalog)
        retrieval_trials.append({'model':name,**metrics})
        if chosen is None or metrics['category_agreement_at_5_PROXY'] > chosen[0]:
            # chosen: representación elegida por el proxy de validación.
            chosen = (metrics['category_agreement_at_5_PROXY'],name,params,model)
    # test_metrics: métricas calculadas una vez sobre consultas de prueba disjuntas; recommendations: listas
    # de candidatos ordenadas por similitud.
    test_metrics, recommendations = retrieval_metrics(chosen[3],test,catalog)
    # controls: referencia aleatoria, intervalos por consulta y errores por escritura/región.
    controls = export_diagnostics(test, catalog, recommendations, out, seed)
    # human_review: hoja ciega con controles; las puntuaciones quedan pendientes.
    human_review = export_human_review(test, catalog, recommendations, out, seed)
    # final_rec: recomendador reajustado al catálogo ampliado tras la evaluación.
    final_rec = REGISTRY['recommend'](params=chosen[2]).fit(catalog_all)
    final_rec.training_manifest_ = dict(manifest,fit_rows=len(catalog_all),selected=chosen[1])
    final_rec.save(base/conf['output']['recommend'])
    # recommendation: informe de recuperación y sus limitaciones de relevancia.
    recommendation = {'controls': controls, 'human_review': human_review, 'selected':chosen[1],'trials':retrieval_trials,'independent_proxy_test':test_metrics,
        'catalog_videos':len(catalog),'validation_queries':len(val),'test_queries':len(test),'final_catalog_videos':len(catalog_all),
        'method':'Contenido + coseno; selección en validación, proxy final en consultas disjuntas; catálogo final reajustado después',
        'limitation':'Categoría no equivale a relevancia. Sin etiquetas humanas ni interacciones no hay Precision@5/NDCG reales. La hoja de revisión humana se entrega sin inventar valoraciones.'}
    return {'clustering_selection_policy':SELECTION_POLICY, 'cluster_interpretation':diagnostics,'clustering':trials,'cluster_selected':{'name':winning[1],'k':cluster_count(winning[2]),'dimensions':winning[3],'silhouette':winning[0]},
            'recommendation':recommendation}, {'cluster':{'model':winning[1],'k':cluster_count(winning[2]),'params':winning[6]},
                                            'recommend':{'model':chosen[1],'params':chosen[2]}}
