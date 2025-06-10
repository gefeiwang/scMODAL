# scMODAL
[![DOI](https://zenodo.org/badge/771573139.svg)](https://doi.org/10.5281/zenodo.15304075)

*scMODAL: A general deep learning framework for **s**ingle-**c**ell **M**ulti-**O**mics **D**ata **A**lignment with feature **L**inks*

We introduce scMODAL, a deep learning framework tailored for single-cell multi-omics data alignment using feature links. scMODAL integrates datasets with limited known positively correlated features, leveraging neural networks and generative adversarial networks to align cell embeddings and preserve feature topology. Our experiments demonstrate scMODAL's effectiveness in removing unwanted variation, preserving biological information, and accurately identifying cell subpopulations across diverse datasets. scMODAL not only advances integration tasks but also supports downstream analyses such as feature imputation and inference of feature relationships, offering a robust solution for advancing single-cell multi-omics research.

![scMODAL_overview](https://github.com/gefeiwang/scMODAL/blob/main/demos/Overview.png)

## Installation
scMODAL can be installed from from GitHub:
```bash
git clone https://github.com/gefeiwang/scMODAL.git
cd scMODAL
conda env update --f environment.yml
conda activate scmodal
```
Normally the installation time is less than 5 minutes.

## Quick Start

### Basic Usage
If the datasets are preprocessed as AnnData objects whose first `n_shared` columns contain linked features from different modalities, scMODAL can be ran using the following code:
```python
import scmodal

model = scmodal.model.Model()
model.preprocess(adata1, adata2, shared_gene_num=n_shared)
model.train() # train the model
model.eval() # get integrated latent representation of cells
```
`model.latent` stores the integrated latent representation of cells, enabling downstream integrative analysis.

Alternatively, scMODAL also takes the inputs with linked features and all features in separate matrices. For example, three datasets can be integrated using 
```python
model.integrate_datasets_feats(input_feats=[adata1.X, adata2.X, adata3.X],
                               paired_input_MNN=[[X1_12, X2_12], [X2_23, X3_23]])
```
where `[X1_12, X2_12]` represents the pair of linked features between datasets 1 and 2, and `[X2_23, X3_23]` represents the pair of linked features between datasets 2 and 3.

## Vignettes
We provide source codes for using scMODAL and reproducing the experiments. Please check the [tutorial website](https://scmodal-tutorial.readthedocs.io/en/latest/index.html) for more details.
### Note on find feature correspondence between protein and RNA modalities
Currently, we follow [MaxFuse](https://github.com/shuxiaoc/maxfuse/tree/main#vignettes) to establish feature correspondence between protein and RNA modalities with a given .csv file. Users can also retrieve the correspondence from online resources, such as [mygene](https://docs.mygene.info/projects/mygene-py/en/latest/):

```python
import mygene
mg = mygene.MyGeneInfo()

# List of cell surface proteins
proteins = ["Ki-67", "CD3", "CD4", "CD8", "CD38"]

# Query the database
result = mg.querymany(proteins, scopes="name", fields="symbol,name", species="human")

# Extract gene symbols
for entry in result:
    print(f"Protein: {entry.get('name')}, Gene Symbol: {entry.get('symbol')}")
```
However, due to differences in nomenclature conversions, the output may be incomplete. Therefore, we recommend that users gather the aliases of proteins and gene symbols in order to refine the RNA-protein correspondences in the output.

## Citation
Gefei Wang, Jia Zhao, Yingxin Lin, Tianyu Liu, Yize Zhao, Hongyu Zhao. scMODAL: A general deep learning framework for comprehensive single-cell multi-omics data alignment with feature links. Nature Communications 16, 4994 (2025).
