import os
import numpy as np
import scanpy as sc
import pandas as pd
import anndata as ad
import umap
from annoy import AnnoyIndex
from scmodal.model import *
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist


def acquire_pairs(X, Y, k=30, metric='angular'):
    f = X.shape[1]
    t1 = AnnoyIndex(f, metric)
    t2 = AnnoyIndex(f, metric)
    for i in range(len(X)):
        t1.add_item(i, X[i])
    for i in range(len(Y)):
        t2.add_item(i, Y[i])
    t1.build(10)
    t2.build(10)

    mnn_mat = np.bool_(np.zeros((len(X), len(Y))))
    sorted_mat = np.array([t2.get_nns_by_vector(item, k) for item in X])
    for i in range(len(sorted_mat)):
        mnn_mat[i,sorted_mat[i]] = True
    _ = np.bool_(np.zeros((len(X), len(Y))))
    sorted_mat = np.array([t1.get_nns_by_vector(item, k) for item in Y])
    for i in range(len(sorted_mat)):
        _[sorted_mat[i],i] = True
    mnn_mat = np.logical_and(_, mnn_mat).astype(int)
    return mnn_mat
     
def annotate_by_nn(vec_tar, vec_ref, label_ref, k=20, metric='cosine'):
    dist_mtx = cdist(vec_tar, vec_ref, metric=metric)
    idx = dist_mtx.argsort()[:, :k]
    labels = [max(list(label_ref[i]), key=list(label_ref[i]).count) for i in idx]
    return labels

def compute_umap(adata, rep=None):
    import umap

    reducer = umap.UMAP(n_neighbors=30,
                        n_components=2,
                        metric="correlation",
                        n_epochs=None,
                        learning_rate=1.0,
                        min_dist=0.3,
                        spread=1.0,
                        set_op_mix_ratio=1.0,
                        local_connectivity=1,
                        repulsion_strength=1,
                        negative_sample_rate=5,
                        a=None,
                        b=None,
                        random_state=1234,
                        metric_kwds=None,
                        angular_rp_forest=False,
                        verbose=True)
    if rep is None:
        X_umap = reducer.fit_transform(adata.X)
    else:
        X_umap = reducer.fit_transform(adata.obsm[rep])

    adata.obsm['X_umap'] = X_umap

