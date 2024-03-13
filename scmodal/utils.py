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


def integrate_datasets(lowdim_list, # list of low-dimensional representations
                       search_cos=False, # searching for an optimal lambdacos
                       lambda_cos=20.0,
                       training_steps=2000,
                       space=None, # None or "reference" or "latent"
                       data_path="data",
                       mixingmetric_subsample=True
                       ):

    if space == None:
        if len(lowdim_list) == 2:
            space = "latent"
        else:
            space = "reference"

    print("Incrementally integrating %d datasets..." % len(lowdim_list))

    if not search_cos:
        # if not search hyperparameter lambdacos
        if isinstance(lambda_cos, float) or isinstance(lambda_cos, int):
            lambda_cos_tmp = lambda_cos

        for i in range(len(lowdim_list) - 1):

            if isinstance(lambda_cos, list):
                lambda_cos_tmp = lambda_cos[i]

            print("Integrating the %d-th dataset to the 1-st dataset..." % (i + 2))
            model = Model(lambdacos=lambda_cos_tmp,
                          training_steps=training_steps, 
                          data_path=os.path.join(data_path, "preprocess"), 
                          model_path="models/%d_datasets" % (i + 2), 
                          result_path="results/%d_datasets" % (i + 2))
            if i == 0:
                model.emb_A = lowdim_list[0]
            else:
                model.emb_A = emb_total
            model.emb_B = lowdim_list[i + 1]
            model.train()
            model.eval()
            emb_total = model.data_Aspace
        if space == "reference":
            return emb_total
        elif space == "latent":
            return model.latent
        else:
            raise ValueError("Space should be either 'reference' or 'latent'.")
    else:
        for i in range(len(lowdim_list) - 1):
            print("Integrating the %d-th dataset to the 1-st dataset..." % (i + 2))
            for lambda_cos in [15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0]:
                model = Model(lambdacos=lambda_cos,
                              training_steps=training_steps, 
                              data_path=os.path.join(data_path, "preprocess"), 
                              model_path="models/%d_datasets" % (i + 2), 
                              result_path="results/%d_datasets" % (i + 2))
                if i == 0:
                    model.emb_A = lowdim_list[0]
                else:
                    model.emb_A = emb_total
                model.emb_B = lowdim_list[i + 1]
                model.train()
                model.eval()
                meta = pd.DataFrame(index=np.arange(model.emb_A.shape[0] + model.emb_B.shape[0]))
                meta["method"] = ["A"] * model.emb_A.shape[0] + ["B"] * model.emb_B.shape[0]
                mixing = calculate_mixing_metric(model.latent, meta, k=5, max_k=300, methods=list(set(meta.method)), subsample=mixingmetric_subsample)
                print("lambda_cos: %f, mixing metric: %f \n" % (lambda_cos, mixing))
                if lambda_cos == 15.0:
                    model_opt = model
                    mixing_metric_opt = mixing
                elif mixing < mixing_metric_opt:
                    model_opt = model
                    mixing_metric_opt = mixing
            emb_total = model_opt.data_Aspace
        if space == "reference":
            return emb_total
        elif space == "latent":
            return model_opt.latent
        else:
            raise ValueError("Space should be either 'reference' or 'latent'.")


def integrate_recover_expression(lowdim_list, # list of low-dimensional representations
                                 mean, std, pca, # information for recovering expression
                                 search_cos=False, # searching for an optimal lambdacos
                                 lambda_cos=20.0,
                                 training_steps=2000,
                                 data_path="data",
                                 mixingmetric_subsample=True
                                 ):

    print("Incrementally integrating %d datasets..." % len(lowdim_list))

    if not search_cos:
        # if not search hyperparameter lambdacos
        if isinstance(lambda_cos, float) or isinstance(lambda_cos, int):
            lambda_cos_tmp = lambda_cos

        for i in range(len(lowdim_list) - 1):

            if isinstance(lambda_cos, list):
                lambda_cos_tmp = lambda_cos[i]

            print("Integrating the %d-th dataset to the 1-st dataset..." % (i + 2))
            model = Model(lambdacos=lambda_cos_tmp,
                          training_steps=training_steps, 
                          data_path=os.path.join(data_path, "preprocess"), 
                          model_path="models/%d_datasets" % (i + 2), 
                          result_path="results/%d_datasets" % (i + 2))
            if i == 0:
                model.emb_A = lowdim_list[0]
            else:
                model.emb_A = emb_total
            model.emb_B = lowdim_list[i + 1]
            model.train()
            model.eval()
            emb_total = model.data_Aspace
    else:
        for i in range(len(lowdim_list) - 1):
            print("Integrating the %d-th dataset to the 1-st dataset..." % (i + 2))
            for lambda_cos in [15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0]:
                model = Model(lambdacos=lambda_cos,
                              training_steps=training_steps, 
                              data_path=os.path.join(data_path, "preprocess"), 
                              model_path="models/%d_datasets" % (i + 2), 
                              result_path="results/%d_datasets" % (i + 2))
                if i == 0:
                    model.emb_A = lowdim_list[0]
                else:
                    model.emb_A = emb_total
                model.emb_B = lowdim_list[i + 1]
                model.train()
                model.eval()
                meta = pd.DataFrame(index=np.arange(model.emb_A.shape[0] + model.emb_B.shape[0]))
                meta["method"] = ["A"] * model.emb_A.shape[0] + ["B"] * model.emb_B.shape[0]
                mixing = calculate_mixing_metric(model.latent, meta, k=5, max_k=300, methods=list(set(meta.method)), subsample=mixingmetric_subsample)
                print("lambda_cos: %f, mixing metric: %f \n" % (lambda_cos, mixing))
                if lambda_cos == 15.0:
                    model_opt = model
                    mixing_metric_opt = mixing
                elif mixing < mixing_metric_opt:
                    model_opt = model
                    mixing_metric_opt = mixing
            emb_total = model_opt.data_Aspace

    expression_scaled = pca.inverse_transform(emb_total)
    expression_log_normalized = expression_scaled * std + mean

    return expression_scaled, expression_log_normalized

     
def annotate_by_nn(vec_tar, vec_ref, label_ref, k=20, metric='cosine'):
    dist_mtx = cdist(vec_tar, vec_ref, metric=metric)
    idx = dist_mtx.argsort()[:, :k]
    labels = [max(list(label_ref[i]), key=list(label_ref[i]).count) for i in idx]
    return labels

def plot_UMAP(data, meta, space="latent", score=None, colors=["method"], subsample=False,
              save=False, result_path=None, filename_suffix=None):
    if filename_suffix is not None:
        filenames = [os.path.join(result_path, "%s-%s-%s.pdf" % (space, c, filename_suffix)) for c in colors]
    else:
        filenames = [os.path.join(result_path, "%s-%s.pdf" % (space, c)) for c in colors]

    if subsample:
        if data.shape[0] >= 1e5:
            np.random.seed(1234)
            subsample_idx = np.random.choice(data.shape[0], 50000, replace=False)
            data = data[subsample_idx]
            meta = meta.iloc[subsample_idx]
            if score is not None:
                score = score[subsample_idx]
    
    adata = ad.AnnData(X=data)
    adata.obs.index = meta.index
    adata.obs = pd.concat([adata.obs, meta], axis=1)
    adata.var.index = "dim-" + adata.var.index
    adata.obsm["latent"] = data
    
    # run UMAP
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
    embedding = reducer.fit_transform(adata.obsm["latent"])
    adata.obsm["X_umap"] = embedding

    n_cells = embedding.shape[0]
    if n_cells >= 10000:
        size = 120000 / n_cells
    else:
        size = 12

    for i, c in enumerate(colors):
        groups = sorted(set(adata.obs[c].astype(str)))
        if "nan" in groups:
            groups.remove("nan")
        palette = "rainbow"
        if save:
            fig = sc.pl.umap(adata, color=c, palette=palette, groups=groups, return_fig=True, size=size)
            fig.savefig(filenames[i], bbox_inches='tight', dpi=300)
        else:
            sc.pl.umap(adata, color=c, palette=palette, groups=groups, size=size)

    if space == "Aspace":
        method_set = pd.unique(meta["method"])
        adata.obs["score"] = score
        adata.obs["margin"] = (score < -5.0) * 1
        fig = sc.pl.umap(adata[meta["method"]==method_set[1]], color="score", palette=palette, groups=groups, return_fig=True, size=size)
        fig.savefig(os.path.join(result_path, "%s-score.pdf" % space), bbox_inches='tight', dpi=300)
        fig = sc.pl.umap(adata[meta["method"]==method_set[1]], color="margin", palette=palette, groups=groups, return_fig=True, size=size)
        fig.savefig(os.path.join(result_path, "%s-margin.pdf" % space), bbox_inches='tight', dpi=300)
    if space == "Bspace":
        method_set = pd.unique(meta["method"])
        adata.obs["score"] = score
        adata.obs["margin"] = (score < -5.0) * 1
        fig = sc.pl.umap(adata[meta["method"]==method_set[0]], color="score", palette=palette, groups=groups, return_fig=True, size=size)
        fig.savefig(os.path.join(result_path, "%s-score.pdf" % space), bbox_inches='tight', dpi=300)
        fig = sc.pl.umap(adata[meta["method"]==method_set[0]], color="margin", palette=palette, groups=groups, return_fig=True, size=size)
        fig.savefig(os.path.join(result_path, "%s-margin.pdf" % space), bbox_inches='tight', dpi=300)

