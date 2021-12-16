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


def add_microanatomy_context(a: AnnData):
    from gatdu.segmentation import segment_slide as utag

    output_dir = (config.results_dir / "domains").mkdir()

    # Run domain discovery
    b = a[:, ~a.var.index.isin(config.channels_exclude)]
    b = b.raw.to_adata()[:, b.var.index]
    b = b[:, b.var.index != "Ki67(Er168)"]

    s = utag(
        b,
        save_key="utag_cluster",
        slide_key="roi",
        max_dist=20,
        clustering_method=["leiden"],
        resolutions=[0.3],
    )
    # clusts = s.obs.columns[s.obs.columns.str.contains("utag_cluster")].tolist()
    # sc.pl.pca(s, color=["sample"] + clusts)
    # sc.pl.umap(s, color=["sample"] + clusts)
    # sc.pl.spatial(s, spot_size=20, color=s.var.index.tolist() + clusts)

    # Observe the cell type composition of domains
    clust = "gatdu_cluster_leiden_0.3"
    c = s.obs.groupby(clust)["cell_type_label"].value_counts()
    cf = c / c.groupby(level=0).sum()
    cfp = cf.reset_index().pivot_table(
        index=clust, columns="level_1", values="cell_type_label"
    )
    grid = clustermap(cfp, config="abs")
    grid.fig.savefig(output_dir / "domain_composition.clustermap.svg")

    domain_assignments = {
        8: "Airway",
        13: "Airway",
        3: "Alveolar",
        9: "Alveolar",
        4: "Alveolar",
        12: "Alveolar",
        0: "Alveolar",
        1: "Vessel",
        11: "Vessel",
        2: "Connective",
        7: "Immune",
        10: "Immune",
        5: "Immune",
        6: "Immune",
    }
    s.obs["domain"] = pd.Categorical(s.obs[clust].astype(int).replace(domain_assignments))
    s.write(output_dir / "utag.h5ad")

    # this is just to have consistent colors across ROIs for 'domain'
    _ = sc.pl.umap(s, color="domain", show=False)
    plt.close("all")

    # Illustrate domains for each ROI
    for sample in tqdm(prj.samples):
        n = len(sample.rois)
        fig, axes = plt.subplots(n, 3, figsize=(3 * 4, n * 4))
        for axs, roi_name in zip(axes, [r.name for r in sample.rois]):
            s1 = s[s.obs["roi"] == roi_name, :].copy()
            for ax, c in zip(axs, [clust, "cell_type_label", "domain"]):
                sc.pl.spatial(s1, spot_size=20, color=c, show=False, ax=ax)
        rasterize_scanpy(fig)
        fig.savefig(output_dir / f"domain_composition.illustration.{sample.name}.svg")
        plt.close(fig)


# UTAG improvements:
# - Default max_dist == 20
# - Warn when std nor present in .var or calculate it
# - Keep .var from original object
# - Cluster PARC vs Leiden: homogenize int vs str
# - PARC clustering, reuse neighbor graph


if __name__ == "__main__" and "get_ipython" not in locals():
    import sys

    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit()
