"""Proyecciones e interpretación del espacio real del clustering."""
# numpy: realiza cálculos numéricos y operaciones con vectores y matrices.
import numpy as np
# pandas: organiza, limpia y analiza datos tabulares mediante DataFrame y Series.
import pandas as pd
# sklearn.decomposition: reduce dimensiones mediante técnicas como PCA y SVD.
# Se importa: PCA.
from sklearn.decomposition import PCA
# sklearn.metrics: calcula métricas para evaluar predicciones, grupos y similitudes.
# Se importa: adjusted_rand_score, normalized_mutual_info_score.
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
# models.core: comparte validación, transformaciones, factorías y persistencia de modelos.
# Se importa: features.
from models.core import features


def export_cluster_diagnostics(model, sample, trials, directory):
    """Proyecta el SVD aprendido y reconstruye términos de los centroides.

    No selecciona con categorías ni reajusta el preprocesamiento. Los términos
    reconstruidos son aproximaciones descriptivas, no efectos causales.
    """
    # matplotlib.pyplot: crea, guarda y cierra figuras y ejes de los análisis.
    import matplotlib.pyplot as plt
    # matplotlib.colors: define mapas y opciones de color para las figuras.
    # Se importa: BoundaryNorm, ListedColormap.
    from matplotlib.colors import BoundaryNorm, ListedColormap
    pipe = model.pipeline_
    matrix = pipe.named_steps['features'].transform(features(sample))
    reduced = pipe.named_steps['reduce'].transform(matrix)
    labels = pipe.named_steps['estimator'].predict(reduced)
    projection = PCA(n_components=2, random_state=42).fit_transform(reduced)
    categories = pd.Categorical(sample.category)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    # Cada etiqueta dispone de un color y un intervalo; la barra muestra IDs enteros.
    estimator = pipe.named_steps['estimator']
    # GMM tiene medias; las otras familias inductivas tienen centros.
    raw_centers = estimator.means_ if hasattr(estimator, 'means_') else estimator.cluster_centers_
    cluster_ids = np.arange(len(raw_centers))
    palette = plt.get_cmap('tab20').resampled(len(cluster_ids))
    boundaries = np.arange(len(cluster_ids) + 1) - .5
    first = axes[0].scatter(*projection.T, c=labels, cmap=palette,
                            norm=BoundaryNorm(boundaries, palette.N), s=9, alpha=.6)
    fig.colorbar(first, ax=axes[0], ticks=cluster_ids, boundaries=boundaries, label='Cluster')
    second = axes[1].scatter(*projection.T, c=categories.codes, cmap='tab20', s=9, alpha=.5)
    handles, _ = second.legend_elements(num=list(range(len(categories.categories))))
    axes[1].legend(handles, categories.categories, loc='center left', bbox_to_anchor=(1, .5), fontsize=7)
    for ax, title in zip(axes, ['Clusters en su espacio SVD', 'Categorías en el mismo espacio']):
        ax.set(title=title, xlabel='PCA 1 del espacio SVD', ylabel='PCA 2 del espacio SVD')
    fig.tight_layout()
    fig.savefig(directory/'cluster_model_space.png', dpi=140)
    plt.close(fig)
    names = pipe.named_steps['features'].get_feature_names_out()
    centers = raw_centers @ pipe.named_steps['reduce'].components_
    text_indices = np.flatnonzero(np.char.startswith(names.astype(str), 'text__'))
    rows = []
    for cluster, center in enumerate(centers):
        for index in text_indices[np.argsort(center[text_indices])[-12:][::-1]]:
            rows.append({'cluster':cluster, 'term':str(names[index]).removeprefix('text__'), 'reconstructed_weight':float(center[index])})
    pd.DataFrame(rows).to_csv(directory/'cluster_terms.csv', index=False)
    pd.crosstab(labels, sample.category, normalize='index').to_csv(directory/'cluster_category_proportions.csv')
    ari = adjusted_rand_score(sample.category, labels)
    nmi = normalized_mutual_info_score(sample.category, labels)
    frame = pd.DataFrame([row for row in trials if row['name'] in ['kmeans', 'minibatch', 'meanshift', 'gmm'] and row.get('silhouette') is not None])
    fig, ax = plt.subplots(figsize=(9, 4))
    for (name, dimensions), block in frame.groupby(['name', 'dimensions']):
        block = block.sort_values('k')
        ax.plot(block.k, block.silhouette, marker='o', label=f'{name}, SVD={dimensions}')
    ax.set(xlabel='Número de clusters', ylabel='Silhouette en índices comunes', title='Selección de k y dimensiones')
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(directory/'cluster_selection.png', dpi=140)
    plt.close(fig)
    interpretation = ('La coincidencia con categorías editoriales es débil: los grupos no recuperan géneros de forma fiable.'
                      if ari < .1 else 'Existe asociación con categorías, pero no equivalencia entre particiones.')
    # clustering_experiments: declara el origen posterior de la regla de selección.
    from clustering_experiments import SELECTION_POLICY
    lines = ['# Interpretación de la segmentación', '', SELECTION_POLICY, '',
             f'ARI frente a categoría: **{ari:.4f}**; NMI: **{nmi:.4f}**. {interpretation}', '',
             'ARI=0 corresponde aproximadamente al acuerdo esperado al azar; NMI mide información compartida. Las categorías no eligen k.', '',
             'Las proyecciones comparten coordenadas del espacio TF-IDF + metadatos + SVD del modelo. La proyección 2D pierde información y no sustituye silhouette.', '',
             'cluster_category_proportions.csv muestra porcentajes por grupo. cluster_terms.csv contiene términos reconstruidos desde los centroides; pueden reflejar plantillas y sistemas de escritura.', '',
             'Los términos y perfiles de audiencia se interpretan conjuntamente. Estabilidad alta y silhouette bajo pueden coexistir; estabilidad no acredita utilidad comercial.']
    (directory/'cluster_interpretation.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return {'ARI':float(ari), 'NMI':float(nmi), 'space':'TF-IDF + metadatos + SVD ajustado'}
