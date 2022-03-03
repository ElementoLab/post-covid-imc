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
from src.domains import get_topological_domain_annotation


# TODO:
# - [x] segment lacunae
# - [x] segment/quantify fibrosis
# - [ ] segment vessels (or use manual annotations)
# - [ ] segment NETs
# - [ ] quantify CitH3 in vessels
# - [x] label domains
# - [x] illustrate domains
# - [ ] domain interaction
# - [x] phenotype in relation to domains
# - [ ] cell type distance to domains
# - [x] cell type interaction
# - [ ] subcellular quantification (nuclear vs rest)
# - [ ] cleanup signal in cell state markers, observe in light of cell type identity
# - [ ] sub-cluster T cell, myeloid compartments


def main() -> int:
    assert all(r.get_input_filename("stack").exists() for r in prj.rois)
    assert all(r.get_input_filename("cell_mask").exists() for r in prj.rois)

    topo_annots, topo_sc = get_topological_domain_annotation()

    cohort_characteristics()
    illustrate()
    qc()
    phenotype()
    cellular_interactions()
    unsupervised()

    return 0


def illustrate(overwrite: bool = False):
    import gc

    output_dir = (config.results_dir / "illustration").mkdir()
    # # Mean channels for all ROIs
    f = output_dir / prj.name + ".all_rois.mean.png"
    if not f.exists() or overwrite:
        fig = prj.plot_channels(["mean"])
        fig.savefig(f, bbox_inches="tight", dpi=75)
        plt.close(fig)

    # # Signal per channel for all ROIs
    for c in tqdm(config.channels_include):
        f = output_dir / prj.name + f".all_rois.{c}.png"
        if not f.exists() or overwrite:
            fig = prj.plot_channels([c])
            for ax, roi in zip(fig.axes, prj.rois):
                attrs = "; ".join(str(getattr(roi, attr)) for attr in config.attributes)
                ax.set_title(f"{roi.name}\n{attrs}")
            fig.savefig(f, bbox_inches="tight", dpi=75)
            plt.close(fig)
            gc.collect()

    # # All channels for each ROI
    for roi in tqdm(prj.rois):
        f = output_dir / roi.name + ".all_channels.png"
        if not f.exists() or overwrite:
            fig = roi.plot_channels()
            fig.savefig(f, **config.figkws)
            plt.close(fig)
            gc.collect()


def qc():
    channel_means, fig = prj.channel_summary(channel_exclude=config.channels_exclude)
    prj.channel_correlation()


def cohort_characteristics():
    output_dir = (config.results_dir / "cohort").mkdir()

    for attr in config.categorical_attributes:
        # N. samples/ROIs per disease group
        n = config.sample_attributes.groupby(attr).size()
        fig, ax = plt.subplots(figsize=(3, 3))
        sns.barplot(x=n, y=n.index, palette=config.colors[attr])
        ax.set(xlabel="Number of samples", ylabel="Sample group")
        fig.savefig(output_dir / f"samples_per_group.{attr}.svg", **config.figkws)
        plt.close(fig)

        n = config.roi_attributes.groupby(attr).size()

        fig, ax = plt.subplots(figsize=(3, 3))
        sns.barplot(x=n, y=n.index, palette=config.colors[attr])
        ax.set(xlabel="Number of ROIs", ylabel="ROI group")
        fig.savefig(output_dir / f"rois_per_group.{attr}.svg", **config.figkws)
        plt.close(fig)

        # Area profiled per disease group
        fig, stats = swarmboxenplot(
            data=(config.roi_areas / 1e6).to_frame().join(config.roi_attributes),
            x=attr,
            y="area_mm2",
            plot_kws=dict(palette=config.colors[attr]),
        )
        fig.savefig(output_dir / f"area_mm2_per_group.{attr}.svg", **config.figkws)
        plt.close(fig)

    # N. cells per disease group
    cabs = pd.Series(
        {r.name: len(np.unique(r.mask)) for r in prj.rois}, name="cells"
    ).rename_axis("roi")
    cmm2 = pd.Series(
        {r.name: r.cells_per_area_unit() for r in prj.rois}, name="cells_mm2"
    ).rename_axis("roi")

    for attr in config.categorical_attributes:
        fig, stats = swarmboxenplot(
            data=cabs.to_frame().join(config.roi_attributes),
            x=attr,
            y="cells",
            plot_kws=dict(palette=config.colors[attr]),
        )
        fig.savefig(output_dir / f"cells_per_group.{attr}.svg", **config.figkws)
        plt.close(fig)

        fig, stats = swarmboxenplot(
            data=(cmm2 * 1e6).to_frame().join(config.roi_attributes),
            x=attr,
            y="cells_mm2",
            plot_kws=dict(palette=config.colors[attr]),
        )
        fig.savefig(output_dir / f"cells_mm2_per_group.{attr}.svg", **config.figkws)
        plt.close(fig)


def phenotype() -> None:
    output_dir = config.results_dir / "phenotyping"

    a = sc.read(output_dir / "processed.h5ad")

    # Add sample attributes
    for attr in config.attributes:
        if attr not in a.obs.columns:
            attr_vals = (
                pd.Series({r.name: getattr(r, attr) for r in prj.rois}, name=attr)
                .rename_axis("roi")
                .reset_index()
            )
            a.obs = a.obs.reset_index().merge(attr_vals).set_index("index")

    # Add attribute colors
    for attr in config.colors:
        if isinstance(config.roi_attributes[attr].dtype, pd.CategoricalDtype):
            a.obs[attr] = pd.Categorical(
                a.obs[attr],
                ordered=config.roi_attributes[attr].cat.ordered,
                categories=config.roi_attributes[attr].cat.categories,
            )
        a.uns[attr + "_colors"] = list(
            map(matplotlib.colors.to_hex, config.colors[attr])
        )

    # Add topological domains
    _, topo_sc = get_topological_domain_annotation()
    a.obs = a.obs.join(topo_sc["topological_domain"])

    # Plot
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
    fig = sc.pl.umap(a, color=clusters, show=False)[0].figure
    rasterize_scanpy(fig)
    fig.savefig(output_dir / "umap.clusters.svg", **config.figkws)
    plt.close(fig)

    _a = a[a.obs.sample(frac=1).index, :]
    fig = sc.pl.umap(
        _a, color="topological_domain", show=False, s=5, alpha=0.25, palette="tab20"
    ).figure
    rasterize_scanpy(fig)
    fig.savefig(output_dir / "umap.topological_domain.svg", **config.figkws)
    plt.close(fig)

    _a = a[a.obs.sample(frac=1).index, :]
    fig = sc.pl.umap(_a, color=config.attributes, show=False)[0].figure
    rasterize_scanpy(fig)
    fig.savefig(output_dir / "umap.attributes.svg", **config.figkws)
    plt.close(fig)
    del _a

    clusters = a.obs.columns[a.obs.columns.str.startswith("cluster_")].tolist()
    cluster = "cluster_3.5"
    for cluster in clusters:
        means = a.to_df().groupby(a.obs[cluster]).mean()
        means = means.loc[:, means.std() > 0]
        sizes = a.to_df().groupby(a.obs[cluster]).size().rename("cell_count")

        grid = clustermap(
            means.loc[:, means.columns.isin(config.channels_include)],
            config="z",
            row_colors=sizes.to_frame(),
        )
        grid.fig.savefig(
            output_dir / f"clustering.{cluster}.clustermap.z.clean.svg", **config.figkws
        )
        plt.close(grid.fig)

        grid = clustermap(
            means.loc[:, means.columns.isin(config.cell_type_markers)],
            config="z",
            # figsize=(5, 10),
            # square=True,
            row_colors=sizes.to_frame(),
        )
        grid.fig.savefig(
            output_dir / f"clustering.{cluster}.clustermap.z.clean_celltypemarkers.svg",
            **config.figkws,
        )
        plt.close(grid.fig)

        grid2 = clustermap(
            means.loc[:, means.columns.isin(config.cell_type_markers)],
            config="z",
            col_linkage=grid.dendrogram_col.linkage,
            row_linkage=grid.dendrogram_row.linkage,
            row_colors=sizes.to_frame(),
        )
        grid2.fig.savefig(
            output_dir
            / f"clustering.{cluster}.clustermap.z.clean_celltypemarkers.post_cleanup.svg",
            **config.figkws,
        )
        plt.close(grid2.fig)

        grid3 = clustermap(
            means.loc[:, means.columns.isin(config.cell_type_markers)],
            config="abs",
            row_linkage=grid.dendrogram_row.linkage,
            # figsize=(5, 10),
            # square=True,
            row_colors=sizes.to_frame(),
        )
        grid3.fig.savefig(
            output_dir
            / f"clustering.{cluster}.clustermap.abs.clean_celltypemarkers.svg",
            **config.figkws,
        )
        plt.close(grid3.fig)

        grid2 = clustermap(
            means.loc[:, means.columns.isin(config.cell_state_markers)],
            config="abs",
            cmap="PuOr_r",
            row_linkage=grid.dendrogram_row.linkage,
            figsize=(3, grid.fig.get_figheight()),
            robust=True,
        )
        grid2.fig.savefig(
            output_dir
            / f"clustering.{cluster}.clustermap.abs.clean_cellstatemarkers.cleanup.svg",
            **config.figkws,
        )
        plt.close(grid2.fig)

        o = means.iloc[grid.dendrogram_row.reordered_ind].index
        gc = a.obs.groupby("disease_subgroup")[cluster].value_counts()
        gc /= gc.groupby(level=0).sum()

        gcp = gc.reset_index().pivot_table(
            index="level_1", columns="disease_subgroup", values=cluster
        )
        gcp = gcp[config.attribute_order["disease_subgroup"]]

        grid4 = clustermap(
            gcp,
            # config="z",
            cmap="PiYG_r",
            col_cluster=False,
            row_linkage=grid.dendrogram_row.linkage,
            figsize=(3, grid.fig.get_figheight()),
        )
        grid4.fig.savefig(
            output_dir / f"clustering.{cluster}.clustermap.abs.disease_subgroup.svg",
            **config.figkws,
        )
        plt.close(grid4.fig)

        grid4 = clustermap(
            gcp,
            config="z",
            z_score=0,
            cmap="PiYG_r",
            col_cluster=False,
            row_linkage=grid.dendrogram_row.linkage,
            figsize=(3, grid.fig.get_figheight()),
        )
        grid4.fig.savefig(
            output_dir / f"clustering.{cluster}.clustermap.z.disease_subgroup.svg",
            **config.figkws,
        )
        plt.close(grid4.fig)

        grid4 = clustermap(
            gcp.drop("Mixed", axis=1),
            config="z",
            z_score=0,
            cmap="PiYG_r",
            col_cluster=False,
            row_linkage=grid.dendrogram_row.linkage,
            figsize=(3, grid.fig.get_figheight()),
        )
        grid4.fig.savefig(
            output_dir
            / f"clustering.{cluster}.clustermap.z.disease_subgroup.no_mixed.svg",
            **config.figkws,
        )
        plt.close(grid4.fig)

        c = a.obs.groupby(cluster)[["topological_domain"]].value_counts()
        c = c[c > 50].rename("count")

        pc = c.reset_index().pivot_table(
            index=cluster,
            columns="topological_domain",
            fill_value=0,
            values="count",
        )
        pt = (pc / pc.sum(0)) * 100
        pt = (pt.T / pt.sum(1)).T * 100
        # pt = pt.loc[:, ~pt.columns.str.contains("-")]

        order = [
            "L-A",
            "A",
            "A-AW",
            "AW",
            "background",
            "AW-M",
            "M",
            "AR-M",
            "AR",
            "AR-V",
            "V",
        ]

        grid5 = clustermap(
            pt.iloc[grid.dendrogram_row.reordered_ind].loc[:, order],
            row_cluster=False,
            robust=True,
            figsize=(3, grid.fig.get_figheight()),
            cbar_kws=dict(label="Percent of cluster"),
            col_cluster=False,
        )
        grid5.fig.savefig(
            output_dir / f"clustering.{cluster}.clustermap.z.topological_domain.svg",
            **config.figkws,
        )
        plt.close(grid5.fig)

    fig = sc.pl.umap(a, color=clusters, show=False)[0].figure
    rasterize_scanpy(fig)
    fig.savefig(output_dir / "umap.clusters.svg", **config.figkws)
    plt.close(fig)

    # See relationship between cluster resolutions
    from scipy.optimize import linear_sum_assignment

    n = len(clusters) - 1
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6))
    for ax, (c1, c2) in zip(axes.T, zip(clusters[:-1], clusters[1:])):
        # count-based
        c = (
            a.obs[[c1, c2]]
            .value_counts()
            .reset_index()
            .pivot_table(index=c2, columns=c1, values=0, fill_value=0, aggfunc=sum)
        )
        c /= c.max(0)
        c.index = "M2-" + c.index.astype(str)
        c.columns = "M1-" + c.columns.astype(str)

        # expression based
        cf1 = a.to_df().groupby(a.obs[c1]).mean()
        cf1.index = "M1-" + cf1.index.astype(str)
        cf2 = a.to_df().groupby(a.obs[c2]).mean()
        cf2.index = "M2-" + cf2.index.astype(str)
        s = cf2.T.join(cf1.T).corr()
        np.fill_diagonal(s.values, -1)

        # combine both
        comb = (s + c.reindex(index=s.index, columns=s.columns).fillna(0)) / 2
        # comb = 100 ** comb

        idx = linear_sum_assignment(comb, maximize=True)
        ns = comb.iloc[idx[0], idx[1]]
        ns = ns.loc[ns.index.str.startswith("M2"), ns.columns.str.startswith("M1")]
        # sns.heatmap(ns, xticklabels=True, yticklabels=True, cmap='RdBu_r', center=0, ax=ax[0])

        nc = c.loc[ns.index, ns.columns].rename_axis(
            index=c.index.name, columns=c.columns.name
        )
        nc.index = nc.index.str.replace("M2-", "")
        nc.columns = nc.columns.str.replace("M1-", "")

        sns.heatmap(nc, xticklabels=True, yticklabels=True, ax=ax, square=True)
    fig.savefig(
        output_dir / f"clustering.cluster_frequency_connection.heatmap.svg",
        **config.figkws,
    )
    plt.close(fig)

    cell_type_assignments = json.load(
        (config.metadata_dir / "cell_type_assignments.json").open()
    )
    resolution = "3.5"
    for resolution in cell_type_assignments:
        a.obs[f"cell_type_label_{resolution}"] = (
            a.obs[f"cluster_{resolution}"]
            .astype(str)
            .replace(cell_type_assignments[resolution])
        )

        m = (
            a.to_df()
            .groupby(a.obs[f"cell_type_label_{resolution}"])
            .mean()
            .rename_axis(columns="marker")
            .drop(config.channels_exclude, axis=1, errors="ignore")
        )
        m = m.loc[:, m.var() > 0]
        grid = clustermap(m.T, config="z", square=True)
        grid.fig.savefig(
            output_dir
            / f"clustering.cell_type_label_{resolution}.phenotype.clustermap.svg",
            **config.figkws,
        )
        plt.close(grid.fig)

        grid = clustermap(m, config="z", square=True)
        grid.fig.savefig(
            output_dir
            / f"clustering.cell_type_label_{resolution}.phenotype.clustermap.T.svg",
            **config.figkws,
        )
        plt.close(grid.fig)

        # plot cell abundance
        for grouping in ["roi", "sample"]:
            counts = (
                a.obs.groupby(grouping)[f"cell_type_label_{resolution}"]
                .value_counts()
                .rename("count")
            )
            counts_mm2 = (
                ((counts / getattr(config, f"{grouping}_areas") * 1e6))
                .rename("cells_per_mm2")
                .to_frame()
                .pivot_table(
                    index=grouping,
                    columns=f"cell_type_label_{resolution}",
                    values="cells_per_mm2",
                    fill_value=0,
                )
            )
            p = counts_mm2.join(getattr(config, f"{grouping}_attributes"))

            for attr in config.colors:
                fig, stats = swarmboxenplot(
                    data=p,
                    x=attr,
                    y=counts_mm2.columns,
                    plot_kws=dict(palette=config.colors[attr]),
                )
                fig.savefig(
                    output_dir
                    / f"clustering.cell_type_label_{resolution}.swarmboxenplot.by_{grouping}_and_{attr}.area.svg",
                    **config.figkws,
                )
                plt.close(fig)

            p = p.query("disease != 'Mixed'")
            p["disease"] = p["disease"].cat.remove_unused_categories()
            p["disease_subgroup"] = p["disease_subgroup"].cat.remove_unused_categories()
            # color not matched!
            for attr in config.colors:
                fig, stats = swarmboxenplot(
                    data=p,
                    x=attr,
                    y=counts_mm2.columns,
                    plot_kws=dict(palette=config.colors[attr]),
                    # fig_kws=dict(figsize=(4, 12)),
                )
                fig.savefig(
                    output_dir
                    / f"clustering.cell_type_label_{resolution}.swarmboxenplot.by_{grouping}_and_{attr}.area.no_mixed.svg",
                    **config.figkws,
                )
                plt.close(fig)

        labs = pd.Series(
            [
                cell_type_assignments[resolution][c]
                for c in a.obs[f"cluster_{resolution}"]
            ],
            index=a.obs.index,
        )
        a.obs[f"cluster_{resolution}_label"] = pd.Categorical(
            a.obs[f"cluster_{resolution}"].astype(str).str.zfill(2) + " - " + labs
        )
        m = (
            a.to_df()
            .groupby(a.obs[f"cluster_{resolution}_label"])
            .mean()
            .rename_axis(columns="marker")
        )
        m = m.loc[:, m.var() > 0]
        grid = clustermap(m.T, config="z", square=True)
        grid.fig.savefig(
            output_dir / f"clustering.cluster_{resolution}.phenotype.clustermap.svg",
            **config.figkws,
        )
        plt.close(grid.fig)

        grid = clustermap(
            m.loc[:, m.columns.isin(config.cell_type_markers)],
            config="z",
        )
        grid.fig.savefig(
            output_dir
            / f"clustering.{cluster}.clustermap.z.clean_celltypemarkers.labeled.svg",
            **config.figkws,
        )
        plt.close(grid.fig)

        grid = clustermap(m, config="z", square=True)
        grid.fig.savefig(
            output_dir / f"clustering.cluster_{resolution}.phenotype.clustermap.T.svg",
            **config.figkws,
        )
        plt.close(grid.fig)

        # plot cell abundance
        for grouping in ["roi", "sample"]:
            counts = (
                a.obs.groupby(grouping)[f"cluster_{resolution}_label"]
                .value_counts()
                .rename("count")
            )
            counts_mm2 = (
                ((counts / getattr(config, f"{grouping}_areas") * 1e6))
                .rename("cells_per_mm2")
                .rename_axis(index=[grouping, f"cluster_{resolution}"])
                .to_frame()
                .pivot_table(
                    index=grouping,
                    columns=f"cluster_{resolution}",
                    values="cells_per_mm2",
                    fill_value=0,
                )
            )
            p = counts_mm2.join(getattr(config, f"{grouping}_attributes"))

            for attr in config.colors:
                fig, stats = swarmboxenplot(
                    data=p,
                    x=attr,
                    y=counts_mm2.columns,
                    plot_kws=dict(palette=config.colors[attr]),
                )
                fig.savefig(
                    output_dir
                    / f"clustering.cluster_{resolution}.swarmboxenplot.by_{grouping}_and_{attr}.area.svg",
                    **config.figkws,
                )
                plt.close(fig)

            p = p.query("disease != 'Mixed'")
            p["disease"] = p["disease"].cat.remove_unused_categories()
            p["disease_subgroup"] = p["disease_subgroup"].cat.remove_unused_categories()
            # color not matched!
            for attr in config.colors:
                fig, stats = swarmboxenplot(
                    data=p,
                    x=attr,
                    y=counts_mm2.columns,
                    plot_kws=dict(palette=config.colors[attr]),
                    # fig_kws=dict(figsize=(4, 12))
                )
                fig.savefig(
                    output_dir
                    / f"clustering.cluster_{resolution}.swarmboxenplot.by_{grouping}_and_{attr}.area.no_mixed.svg",
                    **config.figkws,
                )
                plt.close(fig)

    # Plot all clusters and cell type annotations on UMAP
    cell_type_labels = a.obs.columns[
        a.obs.columns.str.startswith("cell_type_label")
    ].tolist()
    fig = sc.pl.umap(a, color=clusters + cell_type_labels, show=False)[0].figure
    rasterize_scanpy(fig)
    fig.savefig(output_dir / "umap.clusters_cell_type_labels.svg", **config.figkws)
    plt.close(fig)

    # #
    # grouping = 'roi'
    # resolution = 1.0
    # counts = (
    #     a.obs.groupby(grouping)[f"cell_type_label_{resolution}"]
    #     .value_counts()
    #     .rename("count")
    # )
    # counts_mm2 = (
    #     ((counts / getattr(config, f"{grouping}_areas") * 1e6))
    #     .rename("cells_per_mm2")
    #     .to_frame()
    #     .pivot_table(
    #         index=grouping,
    #         columns=f"cell_type_label_{resolution}",
    #         values="cells_per_mm2",
    #         fill_value=0,
    #     )
    # )

    # roi_name = counts_mm2['_Empty'].sort_values().index[-1]
    # roi = [r for r in prj.rois if r.name == roi_name][0]

    # sample = [s for s in  prj if s.name == 'A19_15_A24'][0]
    # ct = a.obs.query(f"roi == '{roi_name}'").set_index(['obj_id'])[f"cell_type_label_{resolution}"].rename("cluster")
    # roi.set_clusters(ct)
    # roi.plot_cell_types(ct)

    # a.obs[f"cell_type_label_{resolution}"] = a.obs[f"cell_type_label_{resolution}"].replace("_Empty", "Empty")

    # a2 = a[a.obs['roi'] == roi_name]
    # sc.pl.spatial(a2, color=f'cell_type_label_{resolution}', spot_size=20)

    a.write(output_dir / "processed.labeled.h5ad")


def cellular_interactions():
    output_dir = (config.results_dir / "interactions").mkdir()

    a = sc.read(config.results_dir / "phenotyping" / "processed.labeled.h5ad")
    c = a.obs.set_index(["sample", "roi", "obj_id"])[f"cell_type_label_3.5"].rename(
        "cluster"
    )

    prj.set_clusters(c, write_to_disk=True)
    adjs = prj.measure_adjacency()

    adjs = adjs.merge(config.roi_attributes, how="left")

    for attr in config.categorical_attributes:
        n = len(config.attribute_order[attr])
        fig = get_grid_dims(
            n, return_fig=True, gridspec_kw=dict(wspace=0.85, hspace=0.85)
        )
        for group, ax in zip(config.attribute_order[attr], fig.axes):
            adj = adjs.query(f"{attr} == '{group}'")
            adj_m = (
                adj.groupby(["index", "variable"])["value"]
                .mean()
                .reset_index()
                .pivot_table(index="index", columns="variable", values="value")
            )
            v = adj_m.abs().mean().mean()
            v += adj_m.abs().std().std() * 3
            sns.heatmap(
                adj_m,
                xticklabels=True,
                yticklabels=True,
                cmap="RdBu_r",
                center=0,
                ax=ax,
                vmin=-v,
                vmax=v,
                cbar_kws=dict(label=f"Interaction strength (log(odds))"),
            )
            ax.set(title=group)
        fig.suptitle(f"Cellular interactions per {attr}")
        fig.savefig(
            output_dir / f"cellular_interactions.per_{attr}.heatmap.svg",
            **config.figkws,
        )
        plt.close(fig)

        # # Plot as differnece to baseline
        base = config.attribute_order[attr][0]
        base_adj = adjs.query(f"{attr} == '{base}'")
        base_adj = (
            base_adj.groupby(["index", "variable"])["value"]
            .mean()
            .reset_index()
            .pivot_table(index="index", columns="variable", values="value")
        )

        fig = get_grid_dims(
            n - 1, return_fig=True, gridspec_kw=dict(wspace=0.85, hspace=0.85)
        )
        for group, ax in zip(config.attribute_order[attr][1:], fig.axes):
            adj = adjs.query(f"{attr} == '{group}'")
            adj_m = (
                adj.groupby(["index", "variable"])["value"]
                .mean()
                .reset_index()
                .pivot_table(index="index", columns="variable", values="value")
            )
            sns.heatmap(
                adj_m - base_adj,
                xticklabels=True,
                yticklabels=True,
                cmap="RdBu_r",
                center=0,
                ax=ax,
                cbar_kws=dict(label=f"Difference over {base}"),
            )
            ax.set(title=group)
        fig.suptitle(f"Cellular interactions per {attr}")
        fig.savefig(
            output_dir
            / f"cellular_interactions.per_{attr}.heatmap.difference_over_{base}.svg",
            **config.figkws,
        )
        plt.close(fig)

        # Differential testing
        adjs["interaction"] = (
            adjs["index"].astype(str) + " <-> " + adjs["variable"].astype(str)
        )

        tt = adjs.pivot_table(index="roi", columns="interaction", values="value")
        stats = swarmboxenplot(
            tt.join(config.roi_attributes), x=attr, y=tt.columns, plot=False
        )
        stats = stats.drop_duplicates(
            subset=stats.columns.to_series().filter(regex="[^Variable]")
        )
        stats = stats.join(
            stats["Variable"]
            .str.split(" <-> ")
            .apply(pd.Series)
            .rename(columns={0: "Cell type 1", 1: "Cell type 2"})
        )
        stats.to_csv(output_dir / f"cellular_interactions.per_{attr}.csv", index=False)

        # # Volcano plots
        base = config.attribute_order[attr][0]

        res = stats.query(f"`A` == '{base}'")
        fig = volcano_plot(
            stats=res,
            n_top=15,
            diff_threshold=None,
            fig_kws=dict(gridspec_kw=dict(wspace=3, hspace=1)),
        )
        fig.savefig(
            output_dir / f"cellular_interactions.per_{attr}.volcano_plot.svg",
            **config.figkws,
        )
        plt.close(fig)

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


def unsupervised(a: AnnData):
    output_dir = (config.results_dir / "unsupervised").mkdir()
    resolution = 3.5

    from sklearn.decomposition import PCA

    c = a.obs.groupby("roi")[f"cell_type_label_{resolution}"].value_counts()
    c = c.reset_index().pivot_table(
        index="roi", columns="level_1", values=f"cell_type_label_{resolution}"
    )
    cp = (c.T / c.sum(1)).T * 100
    ca = (c.T / config.roi_areas).T * 1e6

    gcp = ca.join(config.roi_attributes).groupby("disease_subgroup").mean().T

    fig, ax = plt.subplots()
    sns.heatmap(gcp.drop("Mixed", axis=1).corr(), cmap="RdBu_r", ax=ax)
    fig.savefig(
        output_dir / f"corrmap.{cluster}.by_disease_subgroup.heatmap.svg",
        **config.figkws,
    )

    grid = clustermap(
        gcp.drop("Mixed", axis=1).corr(),
        cmap="RdBu_r",
        rasterized=True,
    )
    grid.fig.savefig(
        output_dir / f"corrmap.{cluster}.by_disease_subgroup.clustermap.svg",
        **config.figkws,
    )

    grid = clustermap(
        cp.T.corr(),
        cmap="RdBu_r",
        center=0,
        row_colors=config.roi_attributes,
        rasterized=True,
    )
    grid.fig.savefig(
        output_dir / f"corrmap.{cluster}.per_roi.by_disease_subgroup.clustermap.svg",
        **config.figkws,
    )

    cps = cp.join(config.roi_attributes["sample"]).groupby("sample").mean()

    grid = clustermap(
        cps.T.corr(),
        cmap="RdBu_r",
        center=0,
        row_colors=config.sample_attributes,
        rasterized=True,
    )
    grid.fig.savefig(
        output_dir / f"corrmap.{cluster}.per_sample.by_disease_subgroup.clustermap.svg",
        **config.figkws,
    )

    cp = (cp - cp.mean()) / cp.std()
    ca = (ca - ca.mean()) / ca.std()
    for df, label in [(ca, "area"), (cp, "percentage")]:
        pca = PCA(2)
        rep = pd.DataFrame(pca.fit_transform(df), index=df.index).join(
            config.roi_attributes
        )

        fig, ax = plt.subplots(1)
        for i, g in enumerate(config.attribute_order["disease_subgroup"]):
            x = rep.loc[rep["disease_subgroup"] == g]
            color = config.colors["disease_subgroup"][i]
            ax.scatter(x[0], x[1], color=color, label=g, alpha=0.5)
            ax.scatter(
                x[0].mean(),
                x[1].mean(),
                marker="^",
                color=color,
                s=200,
                edgecolor="black",
                linewidth=2,
            )

            xp = x.groupby("sample").mean()
            ax.scatter(
                xp[0],
                xp[1],
                marker=".",
                color=color,
                s=100,
                edgecolor="black",
                linewidth=2,
            )
            for p in xp.index:
                ax.text(xp.loc[p, 0], xp.loc[p, 1], s=p)
        ax.set(xlabel="PCA1", ylabel="PCA2")
        ax.legend()
        fig.savefig(
            output_dir / f"pca.{cluster}.{label}.by_disease_subgroup.svg",
            **config.figkws,
        )

        df = df.join(config.roi_attributes).query("disease != 'Mixed'")[df.columns]

        pca = PCA(2)
        rep = pd.DataFrame(pca.fit_transform(df), index=df.index).join(
            config.roi_attributes
        )

        fig, ax = plt.subplots(1)
        for i, g in enumerate(config.attribute_order["disease_subgroup"]):
            x = rep.loc[rep["disease_subgroup"] == g]
            color = config.colors["disease_subgroup"][i]
            if x.empty:
                continue
            ax.scatter(x[0], x[1], color=color, label=g, alpha=0.5)
            ax.scatter(
                x[0].mean(),
                x[1].mean(),
                marker="^",
                color=color,
                s=200,
                edgecolor="black",
                linewidth=2,
            )
            ax.scatter(
                x.groupby("sample")[0].mean(),
                x.groupby("sample")[1].mean(),
                marker=".",
                color=color,
                s=100,
                edgecolor="black",
                linewidth=2,
            )

            xp = x.groupby("sample").mean()
            for p in xp.index:
                ax.text(xp.loc[p, 0], xp.loc[p, 1], s=p)
        ax.set(xlabel="PCA1", ylabel="PCA2")
        ax.legend()
        fig.savefig(
            output_dir / f"pca.{cluster}.{label}.by_disease_subgroup.no_Mixed.svg",
            **config.figkws,
        )

        df = df.join(config.roi_attributes).query(
            "disease != 'UIP/IPF' & disease != 'Mixed'"
        )[df.columns]

        pca = PCA(2)
        rep = pd.DataFrame(pca.fit_transform(df), index=df.index).join(
            config.roi_attributes
        )

        fig, ax = plt.subplots(1)
        for i, g in enumerate(config.attribute_order["disease_subgroup"]):
            x = rep.loc[rep["disease_subgroup"] == g]
            color = config.colors["disease_subgroup"][i]
            if x.empty:
                continue
            ax.scatter(x[0], x[1], color=color, label=g, alpha=0.5)
            ax.scatter(
                x[0].mean(),
                x[1].mean(),
                marker="^",
                color=color,
                s=200,
                edgecolor="black",
                linewidth=2,
            )
            ax.scatter(
                x.groupby("sample")[0].mean(),
                x.groupby("sample")[1].mean(),
                marker=".",
                color=color,
                s=100,
                edgecolor="black",
                linewidth=2,
            )

            xp = x.groupby("sample").mean()
            for p in xp.index:
                ax.text(xp.loc[p, 0], xp.loc[p, 1], s=p)
        ax.set(xlabel="PCA1", ylabel="PCA2")
        ax.legend()
        fig.savefig(
            output_dir
            / f"pca.{cluster}.{label}.by_disease_subgroup.no_IPF_no_Mixed.svg",
            **config.figkws,
        )


if __name__ == "__main__" and "get_ipython" not in locals():
    import sys

    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit()
