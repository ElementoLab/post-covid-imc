#!/usr/bin/env python

"""
Visualize the clinical data/metadata of the patient cohort.
"""
from functools import partial

import numpy as np
import pandas as pd
from anndata import AnnData
import scanpy as sc
import matplotlib
from matplotlib import rcParams
import seaborn as sns
from seaborn_extensions import swarmboxenplot, clustermap
from src.types import Path

# clustermap = partial(clustermap, dendrogram_ratio=0.1, square=True)


rcParams["font.family"] = "Arial"


metadata_dir = Path("metadata")
output_dir = (Path("results") / "clinical").mkdir()
figkws = dict(bbox_inches="tight", dpi=300)
colors = np.asarray(sns.color_palette())[[1, 3, 6, 4]]

meta = pd.read_csv(metadata_dir / "clinical_metadata.2023-01-23.csv", index_col=0)
meta.index = meta.index.astype(str).str.zfill(2)
meta["Group"] = pd.Categorical(
    meta["Group"],
    categories=["COVID-19-early", "COVID-19-late", "PC-neg", "PC-pos"],
    ordered=True,
)
meta = meta.dropna(subset="Group")

# Get
# # For continuous variables
cont_vars = meta.columns[
    list(map(lambda x: x.name.lower() in ["float64", "int64"], meta.dtypes))
].tolist()
cont_meta = meta.loc[:, cont_vars].astype(float)

# # For categoricals
cat_vars = meta.columns[
    list(
        map(
            lambda x: x.name.lower() in ["category", "bool", "boolean"],
            meta.dtypes,
        )
    )
].tolist()
cat_meta = meta.loc[:, cat_vars]

# # # convert categoricals
cat_meta = pd.DataFrame(
    {
        x: cat_meta[x].astype(float)
        if cat_meta[x].dtype.name in ["bool", "boolean"]
        else cat_meta[x].cat.codes
        for x in cat_meta.columns
    }
)
cat_meta = cat_meta.loc[:, cat_meta.nunique() > 1].drop("Group", axis=1)

x = cont_meta.fillna(-1)
x -= x.mean()
x /= x.std()
grid = clustermap(
    x,
    mask=cont_meta.isnull(),
    # config="z",
    cmap="coolwarm",
    row_colors=meta["Group"],
    figsize=(4.3, 6),
    dendrogram_ratio=0.1,
)
grid.savefig(output_dir / "continuous.clustermap.svg", **figkws)

grid = clustermap(
    cat_meta,
    mask=cat_meta.isnull(),
    vmin=-0.25,
    vmax=1.25,
    cmap="BrBG",
    row_colors=meta["Group"],
    figsize=(3.5, 6),
    dendrogram_ratio=0.1,
)
grid.savefig(output_dir / "categorical.clustermap.svg", **figkws)


x = cont_meta.fillna(-1).join(cat_meta.fillna(-1))
grid = clustermap(
    x,
    mask=cont_meta.isnull().join(cat_meta.isnull()),
    config="z",
    cmap="coolwarm",
    row_colors=meta["Group"],
)
grid.savefig(output_dir / "joint.clustermap.svg", **figkws)

a = AnnData(x, obs=meta)
a.uns["Group_colors"] = [matplotlib.colors.rgb2hex(c) for c in colors]
sc.pp.scale(a)
sc.pp.pca(a)
fig = sc.pl.pca(a, color="Group", show=False, size=250).figure
fig.savefig(output_dir / "joint.pca.svg", **figkws)

# Difference in continuous variables by group
fig, stats = swarmboxenplot(
    data=meta, y=cont_vars[:-1], x="Group", plot_kws=dict(palette=colors)
)
stats.to_csv(output_dir / "continous.swarmboxenplot.stats.csv", index=False)
fig.savefig(output_dir / "continous.swarmboxenplot.svg", **figkws)

# Stratify lung weight by disease and gender (Fig1b)
fig, stats = swarmboxenplot(
    data=meta,
    x="Group",
    y="LUNG PATHOLOGY: Lung Weight",
    hue="GENDER (M/F)",
    test_kws=dict(parametric=False),
    plot_kws=dict(palette=colors),
)
fig.savefig(output_dir / "continous.swarmboxenplot.stratified.svg", **figkws)

# Difference in categorical variable co-occurence with group
add_cat_vars = [
    "GENDER (M/F)",
    "Self-Identified Racial Background",
    "SMOKE (Y/N)",
    "Fever (Tmax)",
    "Cough",
    "Shortness of breath",
]
_res = list()
for cat in add_cat_vars + cat_meta.columns.tolist():
    res = (
        pg.contingency.chi2_independence(data=meta, x="Group", y=cat)[2]
        .query("test == 'pearson'")
        .squeeze()
        .rename(cat)
    )
    _res.append(res)
res = pd.DataFrame(_res).rename(columns={"pval": "p-unc"})
res["p-cor"] = pg.multicomp(res["p-unc"])[1]
