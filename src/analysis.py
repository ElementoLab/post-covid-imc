#!/usr/bin/env python

"""
Analysis description.
"""

from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from seaborn_extensions import clustermap, swarmboxenplot, volcano_plot
import scanpy as sc

from imc.types import AnnData
from imc.graphics import rasterize_scanpy

from src._config import prj, config


def main() -> int:
    assert all(r.get_input_filename("stack").exists() for r in prj.rois)
    assert all(r.get_input_filename("cell_mask").exists() for r in prj.rois)

    illustrate()
    qc()
    phenotype()

    return 0


def illustrate(overwrite: bool = False):
    output_dir = (config.results_dir / "illustration").mkdir()
    # # Mean channels for all ROIs
    f = output_dir / prj.name + ".all_rois.mean.pdf"
    if not f.exists() or overwrite:
        fig = prj.plot_channels(["mean"])
        fig.savefig(f, **config.figkws)
        plt.close(fig)

    # # Signal per channel for all ROIs
    for c in tqdm(config.channels_include[14:]):
        f = output_dir / prj.name + f".all_rois.{c}.pdf"
        if not f.exists() or overwrite:
            fig = prj.plot_channels([c])
            fig.savefig(f, **config.figkws)
            plt.close(fig)
            import gc

            gc.collect()

    # # All channels for each ROI
    for roi in tqdm(prj.rois):
        f = output_dir / roi.name + ".all_channels.pdf"
        if not f.exists() or overwrite:
            fig = roi.plot_channels()
            fig.savefig(f, **config.figkws)
            plt.close(fig)
            import gc

            gc.collect()


def qc():
    channel_means, fig = prj.channel_summary(channel_exclude=config.channels_exclude)
    prj.channel_correlation()


def cohort_characteristics():
    ...
    # N. ROIs per disease group

    # N. cells per disease group

    # Area profiled per disease group


def phenotype() -> None:
    output_dir = config.results_dir / "phenotyping"

    a = sc.read(config.results_dir / "phenotyping" / "processed.h5ad")
    disease = (
        pd.Series({r.name: r.disease for r in prj.rois}, name="disease")
        .rename_axis("roi")
        .reset_index()
    )
    a.obs = a.obs.reset_index().merge(disease).set_index("index")

    chs = a.var.index[~a.var.index.isin(config.channels_exclude)].tolist()
    vmax = pd.Series([np.percentile(a.raw[:, c].X, 95) for c in chs], index=chs)
    vmax = vmax.replace(0, 0.1)

    axs = sc.pl.pca(a, color=chs, vmax=vmax.tolist(), show=False)
    fig = axs[0].figure
    rasterize_scanpy(fig)
    fig.savefig(output_dir / "pca.marker_expression.svg", **config.figkws)
    plt.close(fig)

    axs = sc.pl.umap(a, color=chs, vmax=vmax.tolist(), show=False)
    fig = axs[0].figure
    rasterize_scanpy(fig)
    fig.savefig(output_dir / "umap.marker_expression.svg", **config.figkws)
    plt.close(fig)

    clusters = a.obs.columns[a.obs.columns.str.startswith("cluster_")].tolist()
    fig = sc.pl.umap(a, color=["disease"] + clusters, show=False)[0].figure
    rasterize_scanpy(fig)
    fig.savefig(output_dir / "umap.disease_clusters.svg", **config.figkws)
    plt.close(fig)

    for cluster in clusters:
        means = a.to_df().groupby(a.obs[cluster]).mean()
        means = means.loc[:, means.std() > 0]
        sizes = a.to_df().groupby(a.obs[cluster]).size().rename("cell_count")
        grid = clustermap(means, config="z", row_colors=sizes.to_frame())
        grid.fig.savefig(
            output_dir / f"clustering.{cluster}.clustermap.svg", **config.figkws
        )
        plt.close(grid.fig)

    cell_type_assignments = {
        1: "Macrophages",
        11: "Macrophages",
        14: "Mast",
        2: "Epithelial",
        8: "Epithelial",
        4: "Endothelial",
        6: "Endothelial",
        12: "Endothelial",
        3: "Fibroblasts",
        5: "Neutrophils",
        13: "Neutrophils",
        7: "Smooth muscle",
        9: "CD4 T",
        10: "CD8 T",
    }
    a.obs["cell_type_label"] = (
        a.obs["cluster_1.0"].astype(int).replace(cell_type_assignments)
    )

    fig = sc.pl.umap(a, color=["cluster_1.0", "cell_type_label"], show=False)[0].figure
    rasterize_scanpy(fig)
    fig.savefig(output_dir / "umap.clusters.svg", **config.figkws)
    plt.close(fig)


def abundance_comparison(a: AnnData):
    cts = a.obs.set_index(["sample", "roi", "obj_id"])["cell_type_label"].rename(
        "cluster"
    )
    prj.set_clusters(cts)
    prj.sample_comparisons(
        channel_exclude=config.channels_exclude, sample_attributes=config.attribute_order
    )


if __name__ == "__main__" and "get_ipython" not in locals():
    import sys

    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit()
