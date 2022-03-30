#!/usr/bin/env python

"""
Analysis of multiplexed images from patients of post-acute sequalae of COVID-19.
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


# - [ ] segment periostin, quantify cell types in Periostin, quantify distance to Periostin per cell type


def main() -> int:
    assert all(r.get_input_filename("stack").exists() for r in prj.rois)
    assert all(r.get_input_filename("cell_mask").exists() for r in prj.rois)

    topo_annots, topo_sc = get_topological_domain_annotation()

    # filter_noise()

    cohort_characteristics()
    illustrate()
    qc()
    phenotype()
    cellular_interactions()
    unsupervised()

    return 0


def filter_noise():
    from src.utils import _fix
    import parmap

    output_dir = (config.results_dir / "denoising").mkdir()

    chs = [
        (("K818", "CD45", "aSMA"), "SFTPC"),
        (("K818", "aSMA", "SFTPC"), "CD11b"),
        (("aSMA",), "CD31"),
        (("iNOS",), "cKIT"),
        (("SFTPA",), "CC3"),
        (("CC16",), "CD8a"),
        (("CC16",), "CD11c"),
        (
            (
                "K818",
                "aSMA",
                "CitH3",
            ),
            "SARS",
        ),
    ]
    chs += [(("_EMPTY_(Pt194)",), c) for c in config.channels_include]

    # for r in tqdm(prj.rois):
    #     _quant.append(_fix(r, plot=True))
    parmap.map(
        _fix,
        prj.rois,
        chs,
        output_dir,
        save=True,
        plot=True,
        release=True,
        pm_processes=6,
        pm_pbar=True,
    )

    _quant = list()
    for r in tqdm(prj.rois):
        new_stack_f = r.get_input_filename("stack").replace_(".tiff", ".denoised.tiff")

        r._stack = tifffile.imread(new_stack_f)
        q = r.quantify_cell_intensity()
        q2 = r.quantify_cell_morphology()
        _quant.append(q.join(q2).assign(roi=r.name, sample=r.sample.name))
        r._stack = None

    quant = pd.concat(_quant)
    quant["axis_ratio"] = quant["major_axis_length"] / quant["minor_axis_length"]
    quant.to_csv(config.results_dir / "quantification.denoised.csv.gz")

    x = quant[config.cell_type_markers + config.cell_state_markers]
    obs = quant[quant.columns[~quant.columns.isin(x.columns)]]
    f = len(str(obs.index.max()))
    x.index = obs["roi"] + "-" + obs.index.astype(str).str.zfill(f)
    obs.index = x.index

    a = AnnData(x * 100, obs=obs)
    a.raw = a
    sc.pp.normalize_per_cell(a)
    sc.pp.log1p(a)
    sc.pp.scale(a)
    sc.pp.pca(a)

    # sc.external.pp.bbknn(a, 'sample')
    sc.external.pp.harmony_integrate(a, "sample")
    sc.pp.neighbors(a, use_rep="X_pca_harmony")
    sc.tl.umap(a)

    markers = [
        "K818(Yb174)",
        "aSMA(Pr141)",
        "CD45(Sm152)",
        "CD3(Er170)",
        "CD8a(Dy162)",
        "CD4(Gd156)",
        # 'SFTPC(Lu175)', 'CC16(Dy163)',
        "Periostin(Dy161)",
        "CD31(Eu151)",
        "SARSSpikeS1(Eu153)",
    ]

    # vmax = [np.percentile(a.raw[:, c].X, 95) for c in markers]
    sc.pl.pca(a, color=markers, vmax=vmax)

    markers = a.var.index
    vmax = [np.percentile(a[:, c].X, 95) for c in markers]
    sc.pl.umap(a, color=markers, vmax=vmax, use_raw=False)
    a.write(config.results_dir / "quantification.denoised.all_markers.harmony.h5ad")
    a.write(
        config.results_dir / "quantification.denoised.cell_type_markers.harmony.h5ad"
    )
    # a.write(config.results_dir / "quantification.denoised.bbknn.h5ad")

    a.to_df().join(a.obs).groupby("sample")[["SARSSpikeS1(Eu153)"]].mean().join(
        config.sample_attributes
    )

    sc.tl.leiden(a)

    # sc.pl.pca(a, color=a.var.index)
    sc.pl.umap(a, color=a.var.index.tolist() + ["leiden"])

    # roi_name = "S19_6699_B9-01"
    # roi_name = "A18_19_A19-04"
    # roi_name = "A18_19_A19-06"
    # roi_name = "A19_33_A26-05"
    # roi_name = "S19_6699_B9-05"
    # roi_name = 'S15_38419_B1-03'
    # r = [r for r in prj.rois if r.name == roi_name][0]

    # # Weigh signal by tech channels
    # back = min_max_scale(
    #     r._get_channels(["80ArAr", "129Xe", "190BCKG"])[1]
    # ).mean(0)
    # back[back < 0] = 0
    # back[back > 1] = 1
    # back = min_max_scale(back)
    # back = skimage.filters.gaussian(back, 10)
    # back = skimage.filters.median(back, skimage.morphology.disk(10))
    # back = skimage.filters.gaussian(back, 10)

    # ch = r._get_channels(["CC16"])[1].squeeze()
    # t = ch * (1 - back)
    # ts = t * (ch.max() / t.max())

    # fig, axes = plt.subplots(1, 2, sharex=True, sharey=True)
    # axes[0].imshow(ch, vmax=1)
    # axes[1].imshow(ts, vmax=1)

    # fig, axes = plt.subplots(1, 3, sharex=True, sharey=True)
    # axes[0].imshow(ch1, vmax=1)
    # axes[1].imshow(ch2, vmax=1)
    # axes[2].imshow(d, vmax=1)
    # axes[3].imshow(res - ch1, vmin=-1, vmax=1, cmap='RdBu_r')

    # # Try to denoise channels
    # ch = np.log1p(r._get_channels(["CD206"])[1].squeeze())
    # chn = min_max_scale(ch, left_perc=90, right_perc=98)
    # chn[chn > 1] = 1
    # print(ch.min(), ch.max())
    # print(chn.min(), chn.max())

    # fig, axes = plt.subplots(1, 2, sharex=True, sharey=True)
    # axes[0].imshow(min_max_scale(ch, right_perc=98), vmax=1)
    # axes[1].imshow(chn, vmax=1)

    # #
    # chs = [
    #     (("K818", "CD45", "aSMA"), "SFTPC"),
    #     (("K818", "aSMA", "SFTPC"), "CD11b"),
    #     (("aSMA",), "CD31"),
    #     (("iNOS",), "cKIT"),
    #     (("SFTPA",), "CC3"),
    #     (("CC16",), "CD8a"),
    #     (
    #         (
    #             "K818",
    #             "aSMA",
    #             "CitH3",
    #         ),
    #         "SARS",
    #     ),
    # ]
    # for cha, chb in chs:
    #     if isinstance(cha, str):
    #         cha = (cha,)

    #     x = min_max_scale(r._get_channels(cha + (chb,))[1], right_perc=98)
    #     to_rem = x[:-1]
    #     target = x[-1]
    #     new = target.copy()
    #     for i, rem in enumerate(to_rem):
    #         new = new - rem
    #         new[new < 0] = 0

    #     new = skimage.filters.gaussian(new, 0.25)
    #     new = skimage.filters.median(new, skimage.morphology.disk(1))
    #     new = skimage.filters.gaussian(new, 0.25)

    #     sel = r.channel_labels.str.contains(chb)
    #     r._stack[sel] = new

    #     fig, axes = plt.subplots(1, 2 + i + 1, sharex=True, sharey=True)
    #     for j, (ax, name) in enumerate(zip(axes, cha)):
    #         ax.imshow(to_rem[j], vmax=1)
    #         ax.set(title=name)
    #     axes[-2].imshow(target, vmax=1)
    #     axes[-2].set(title=chb)
    #     axes[-1].imshow(new, vmax=1)
    #     axes[-1].set(title=chb + " denoised")
    #     for ax in axes:
    #         ax.axis("off")

    #     [(3, 360), (415, 115)]

    # fig = r.plot_channels()
    # fig.savefig(r.name + "_stack.pdf", dpi=150)

    #

    #

    # # Quant
    # q = r.quantify_cell_intensity()
    # q2 = r.quantify_cell_morphology()

    # q.index = q.index.astype(str)
    # q2.index = q2.index.astype(str)
    # a = AnnData(
    #     q.drop(
    #         config.channels_exclude + ["HH3(In113)", "DNA1(Ir191)", "DNA2(Ir193)"],
    #         axis=1,
    #     ),
    #     obs=q2,
    # )
    # sc.pp.log1p(a)
    # sc.pp.scale(a)
    # sc.pp.pca(a)

    # sc.pp.neighbors(a)
    # sc.tl.umap(a)
    # sc.tl.leiden(a)

    # # sc.pl.pca(a, color=a.var.index)
    # sc.pl.umap(a, color=a.var.index.tolist() + ["leiden"])


def filter_noise_sc():
    output_dir = (config.results_dir / "denoising").mkdir()

    chs = [
        (("CitH3",), "SARS"),
        (("K818", "CD45", "aSMA"), "SFTPC"),
        (("K818", "aSMA", "SFTPC"), "CD11b"),
        (("aSMA",), "CD31"),
        (("iNOS",), "cKIT"),
        (("SFTPA",), "CC3"),
        (("CD8a",), "CC16"),
        (("CC16",), "CD8a"),
        (("CC16",), "CD11c"),
        (
            (
                "K818",
                "aSMA",
                "CitH3",
            ),
            "SARS",
        ),
    ]
    chs += [(("_EMPTY_(Pt194)",), c) for c in config.channels_include]

    for cha, chb in chs:
        a.X[:, a.var.index.isin([chb])] = (
            a.X[:, a.var.index.isin([chb])] - a.X[:, a.var.index.isin(cha)]
        )
        # # For a df:
        # df.loc[:, df.columns.isin([chb])] = (
        #     df.loc[:, df.columns.isin([chb])].values - df.loc[:, df.columns.isin(cha)].values
        # )


    a.X[a.X < 0] = 0

    sc.pp.scale(a)
    sc.pp.pca(a)
    sc.pp.neighbors(a)
    sc.tl.umap(a)

    sc.pl.pca(a, color="cluster_1.5")
    sc.pl.umap(a, color="cluster_2.5")

    m = a.to_df().groupby(a.obs["cluster_3.5"]).mean()


def _illustrate_nets():
    from csbdeep.utils import normalize

    roi_names = ["A21_76_A16-01", "A20_47_A37-01"]
    for roi_name in roi_names:

        r = [r for r in prj.rois if r.name == roi_name][0]

        chs = r._get_channels(["CitH3", "Periostin", "DNA1"])[1]
        x = np.moveaxis(np.asarray(list(map(normalize, chs))), 0, -1)

        fig, ax = plt.subplots()
        ax.imshow(x / 2)
        ax.axis("off")
        ax.set(title=f"{roi_name}")
        fig.savefig(f"vessels_{roi_name}.pdf", **config.figkws)

        chs = r._get_channels(["CitH3", "SARS", "DNA1"])[1]
        x = np.moveaxis(np.asarray(list(map(normalize, chs))), 0, -1)

        fig, ax = plt.subplots()
        ax.imshow(x / 2)
        ax.axis("off")
        ax.set(title=f"{roi_name}")
        fig.savefig(f"sars_{roi_name}.pdf", **config.figkws)

    roi_name = "S19_6699_B9-01"
    r = [r for r in prj.rois if r.name == roi_name][0]
    fig = r.plot_channels(["CitH3", "Periostin", "SARSSpikeS1", "DNA1"])


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


def illustrations_specific():

    sel_markers = {
        "general": [
            ("K818", "AQ1", "Periostin"),
            ("ColTypeI", "CC16", "IRF2BP2"),
            ("SFTPA", "SFTPC", "CC3"),
            ("CD68", "CD3(", "MPO"),
        ],
        "inflammation": [
            ("CD68", "CD163", "CD206"),
            ("IRF2BP2", "IL6", "pNFkbp65"),
            ("uPAR", "pSTAT3Tyr705", "IL1beta"),
            ("KIT", "CC3", "p16"),
        ],
    }
    selected = {
        "Normal_airway_1": "A18_19_A19-06",
        "Normal_vessel_1": "A18_19_A19-04",
        "Normal_alveolar_1": "A19_33_A26-03",
        "Normal_alveolar_2": "A19_33_A26-07",
        "UIP/IPF_1": "S15_38419_B1-02",
        "UIP/IPF_2": "S15_38419_B1-01",
        "COVID_early_1": "A20_47_A37-02",
        "COVID_early_2": "A20_47_A37-06",
        "COVID_airway_1": "A20_79_A23-05",
        "COVID_vessel_1": "A20_79_A23-08",
        "COVID_alveolar_1": "A20_79_A23-01",
        "COVID_alveolar_2": "A20_79_A23-04",
        "COVID_alveolar_3": "A20_79_A23-02",
        "PASC-neg_vessel_1": "A20_100_A28-01",
        "PASC-neg_alveolar_1": "A20_100_A28-09",
        "PASC-pos_1": "A21_76_A16-01",
        "PASC-pos_2": "A21_76_A16-05",
    }

    for marker_type, markers in sel_markers.items():
        for desc, roi_name in selected.items():
            roi = [r for r in prj.rois if r.name == roi_name][0]
            attrs = "; ".join(str(getattr(roi, attr)) for attr in config.attributes)

            o = (
                config.results_dir / "illustration" / roi.name
                + f".{marker_type}_markers.svg"
            )
            if not o.exists():
                all_markers = [y for x in markers for y in x]
                fig = roi.plot_channels(all_markers)
                fig.savefig(o, bbox_inches="tight", dpi=300)

            o = (
                config.results_dir / "illustration" / roi.name
                + f".{marker_type}_markers.merged.svg"
            )
            if not o.exists():
                fig, axes = plt.subplots(
                    1,
                    len(markers),
                    figsize=(len(markers) * 3, 3),
                    gridspec_kw=dict(wspace=0),
                )
                for ax, marker_set in zip(axes, markers):
                    roi.plot_channels(marker_set, merged=True, axes=[ax])
                fig.suptitle(f"{roi.name}; {attrs}")
                fig.savefig(o, bbox_inches="tight", dpi=300)

    prj.set_clusters(
        a.obs.set_index(["sample", "roi", "obj_id"])["cluster_2.5"].rename("cluster")
    )

    fig = roi.plot_cell_types()

    counts_n = (
        counts.rename_axis(index=["roi", f"cluster_{resolution}"])
        .rename("cells")
        .to_frame()
        .pivot_table(
            index=grouping,
            columns=f"cluster_{resolution}",
            values="cells",
            fill_value=0,
        )
    )

    counts_n["16"].sort_values()
    counts_mm2["16"].sort_values()

    roi_name = "A20_100_A28-02"


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

    # Add cell type labels
    cell_type_assignments = json.load(
        (config.metadata_dir / "cell_type_assignments.json").open()
    )
    for resolution in cell_type_assignments:
        a.obs[f"cell_type_label_{resolution}"] = (
            a.obs[f"cluster_{resolution}"]
            .astype(str)
            .replace(cell_type_assignments[resolution])
        )

    # For each resolution, plot
    resolution = "3.5"
    for resolution in cell_type_assignments:
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




def infected_cells(a: AnnData) -> AnnData:
    output_dir = (config.results_dir / "positivity").mkdir()

    a = sc.read(config.results_dir / "phenotyping" / "processed.labeled.h5ad")
    a.obs['covid'] = a.obs['disease'].isin(['COVID-19', 'Mixed', 'Convalescent'])

    # Conditional scalling approach
    # # Shrink SARS signal in non-COVID samples, find a threshold which maximizes separation
    df = np.log1p(a.raw.to_adata().to_df())
    # df = a.to_df()
    col = 'SARSSpikeS1(Eu153)'
    _res = list()
    for f in tqdm(np.linspace(1, 0, 50)):
        df2 = df.copy()
        df2.loc[~a.obs['covid'], col] *= f
        for t in np.linspace(0, 5, 50):
            pos = df2[col] > t
            p = pos.groupby(a.obs['covid']).sum()
            n = pos.groupby(a.obs['covid']).size()
            _res.append([f, t, p[True], p[False], n[True], n[False]])
    res = pd.DataFrame(_res, columns=['shrinkage_factor', 'threshold', 'pos_COVID', 'pos_NON', 'n_COVID', 'n_NON'])
    res['frac_COVID'] = res['pos_COVID'] / res['n_COVID']
    res['frac_NON'] = res['pos_NON'] / res['n_NON']
    res['log_ratio'] = np.log((res['pos_COVID'] / res['n_COVID']) / (res['pos_NON'] / res['n_NON']))

    vmax = res[res != np.inf]['log_ratio'].max() * 0.95
    _p = res.pivot_table(index='shrinkage_factor', columns='threshold', values='log_ratio').iloc[1:, 1:]
    _p.index = _p.index.to_series().round(2)
    _p.columns = _p.columns.to_series().round(2)
    fig, ax = plt.subplots()
    sns.heatmap(_p, vmax=vmax, ax=ax, cbar_kws=dict(label='log(COVID / non-COVID) SARSSpikeS1+ cells'))
    fig.savefig(output_dir / 'SARSSpikeS1_thresholds.svg')

    # Choose parameters: shrink 0.8, threshold 1.0
    df.loc[~a.obs['covid'], col] *= 0.8
    pos = df[col] > thresholds[col]
    a.obs[col + "_pos"] = pos

    a.obs.groupby(pos)['covid'].value_counts()

    counts = a.obs.groupby(['roi', 'cell_type_label_3.5'])[col + "_pos"].value_counts()
    counts_mm2 = (counts / config.roi_areas) * 1e6
    counts_perc = (counts / a.obs.groupby('roi').size()) * 100

    for group in ['disease', 'disease_subgroup']:
        for df, label in [(counts_mm2, 'mm2'), (counts_perc, 'perc')]:
            p = df[:, :, True].rename(col + "_pos").to_frame()
            pp = p.pivot_table(index='roi', columns='cell_type_label_3.5', values=col + "_pos", fill_value=0)
            fig, stats = swarmboxenplot(data=pp.join(config.roi_attributes), x=group, y=pp.columns, plot_kws=dict(palette=config.colors[group]))
            fig.savefig(output_dir / f"SARSSpikeS1_positivity.by_{group}.{label}.swarmboxenplot.svg", **config.figkws)
            plt.close(fig)

            ppp = pp.join(config.roi_attributes[[group]]).groupby(group).mean().T
            grid = clustermap(ppp, col_cluster=False, square=True, cbar_kws=dict(label=label))
            grid.savefig(output_dir / f"SARSSpikeS1_positivity.by_{group}.{label}.clustermap.svg", **config.figkws)
            plt.close(grid.fig)

    q = counts_perc[:, 'Alveolar type 2', True].to_frame("pos").join(config.roi_attributes).query("disease == 'Convalescent'")
    q['disease_subgroup'] = q['disease_subgroup'].cat.remove_unused_categories()
    fig, stats = swarmboxenplot(data=q, x='disease_subgroup', y="pos")

    # Plot
    q = a.obs.groupby(pos)['cell_type_label_3.5'].value_counts()[True] / a.obs['cell_type_label_3.5'].value_counts()

    ct = a.obs.groupby(pos)['cell_type_label_3.5'].value_counts() / a.obs.groupby(pos)['cell_type_label_3.5'].size() # / a.obs['cell_type_label_3.5'].value_counts()
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=False)
    o = ct[True].sort_values(ascending=False).index
    sns.barplot(x=ct[False].reindex(o), y=o, ax=axes[0], orient='horiz')
    sns.barplot(x=ct[True], y=o, ax=axes[1], orient='horiz')
    axes[0].set(title='SARS-CoV-2 Spike neg')
    axes[1].set(title='SARS-CoV-2 Spike pos')
    fig.savefig(output_dir / 'SARSSpikeS1_infected.per_cell_type.svg', **config.figkws)

    ct_cov = a.obs.query("covid").groupby(pos)['cell_type_label_3.5'].value_counts() / a.obs.query("covid").groupby(pos)['cell_type_label_3.5'].size()
    ct_non = a.obs.query("~covid").groupby(pos)['cell_type_label_3.5'].value_counts() / a.obs.query("~covid").groupby(pos)['cell_type_label_3.5'].size()

    _res2 = list()
    for t in np.linspace(0, 5, 50):
        pos = val > t
        ct = a.obs.groupby(pos)['cell_type_label_3.5'].value_counts() / a.obs.groupby(pos)['cell_type_label_3.5'].size()
        if True in ct.index:
            _res2.append(ct[True].rename(t))
    res2 = pd.DataFrame(_res2)

    res2.index = res2.index.to_series().round(2)
    fig, ax = plt.subplots()
    sns.heatmap(res2, ax=ax)
    fig.savefig(output_dir / 'SARSSpikeS1_thresholds_per_cell_type.svg', **config.figkws)

    fig, ax = plt.subplots()
    sns.heatmap((res2 - res2.mean()) / res2.std(), ax=ax)
    fig.savefig(output_dir / 'SARSSpikeS1_thresholds_per_cell_type.zscore.svg', **config.figkws)


    # compare phenotype of infected vs non-infected per cell type
    _v = [col + "_pos",'cell_type_label_3.5']
    df = a.raw.to_adata().to_df()
    df = np.log1p((df.T / df.sum(1)).T * 1e4)
    df2 = df
    # df2 = df.loc[a.obs['disease'] == 'Convalescent', :]
    
    aggs = df2.join(a.obs[_v]).groupby(_v).mean()
    diff = (aggs.loc[True, :, :] - aggs.loc[False, :, :])[config.cell_state_markers]# .drop(col, axis=1)
    # diff = diff[[x for x in config.channels_include if x in diff.columns]]

    ns = df2.join(a.obs[_v]).groupby(_v).size()

    diff = diff.loc[:, diff.mean(0).sort_values().index]

    grid = clustermap(diff, cmap='RdBu_r', center=0, col_cluster=True, cbar_kws=dict(label="log fold change (SARSSpikeS1+/SARSSpikeS1-)"), row_colors=ns[False].to_frame("neg").join(ns[True].rename('pos')), config='abs', figsize=(2, 4))
    grid.savefig(output_dir / 'SARSSpikeS1_positivity.change_in_expression.clustermap.svg', **config.figkws)

    for ct in sorted(a.obs['cell_type_label_3.5'].unique()):
        df3 = df.loc[(a.obs['cell_type_label_3.5'] == ct) & ~a.obs['disease_subgroup'].isin(['Normal', 'UIP/IPF'])]
        p = df3.join(a.obs[['disease_subgroup', col + "_pos"]])
        p['disease_subgroup'] = p['disease_subgroup'].cat.remove_unused_categories()

        fig, axes = plt.subplots(3, 4, figsize=(4 * 2, 3 * 4), sharex=True, sharey=True, gridspec_kw=dict(wspace=0, hspace=0))
        for ax, marker in zip(axes.flat, [x for x in config.cell_state_markers if x != col]):
            sns.violinplot(data=p, x='disease_subgroup', y=marker, hue=col + "_pos", ax=ax)
            ax.set(xlabel='', ylabel='')
            ax.set_title(marker, y=0.9)
        for ax in axes[-1, :]:
            ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        for ax in axes.flatten()[1:]:
            l = ax.get_legend()
            if l:
                l.set_visible(False)
        t = p.groupby(['disease_subgroup', col + "_pos"]).size()
        fig.suptitle(ct + "\nn = " + '; '.join(t.astype(str)))
        fig.savefig(output_dir / f"SARSSpikeS1_positivity.change_in_expression.{ct.replace(' ', '_')}.violinplot.svg", **config.figkws)
        plt.close(fig)


def marker_positivity(a: AnnData) -> AnnData:
    output_dir = (config.results_dir / "positivity").mkdir()

    a = sc.read(config.results_dir / "phenotyping" / "processed.labeled.h5ad")
    a.obs['covid'] = a.obs['disease'].isin(['COVID-19', 'Mixed', 'Convalescent'])

    # df = np.log1p(a.raw.to_adata().to_df())
    df = a.raw.to_adata().to_df()
    df = np.log1p((df.T / df.sum(1)).T * 1e4)

    col = 'SARSSpikeS1(Eu153)'
    df.loc[~a.obs['covid'], col] *= 0.8

    thresholds = dict()
    for col in tqdm(sorted(config.cell_state_markers)):
        _res2 = list()
        for t in np.linspace(0, df[col].max(), 50):
            pos = df[col] > t
            ct = a.obs.groupby(pos)['cell_type_label_3.5'].value_counts() / a.obs.groupby(pos)['cell_type_label_3.5'].size()
            if True in ct.index:
                _res2.append(ct[True].rename(t))
        res2 = pd.DataFrame(_res2)

        res2.index = res2.index.to_series().round(2)
        fig, ax = plt.subplots()
        sns.heatmap(res2, ax=ax)
        fig.savefig(output_dir / f'{col}_thresholds_per_cell_type.svg', **config.figkws)

        fig, ax = plt.subplots()
        sns.heatmap((res2 - res2.mean()) / res2.std(), ax=ax)
        fig.savefig(output_dir / f'{col}_thresholds_per_cell_type.zscore.svg', **config.figkws)

        # threshold
        from imc.ops.mixture import get_population
        pos = get_population(df[col])
        thresholds[col] = df.loc[pos, col].min()
        # pos = (df[col] > 1)
        a.obs[col + "_pos"] = pos

        counts = a.obs.groupby(['roi', 'cell_type_label_3.5'])[col + "_pos"].value_counts()
        counts_mm2 = (counts / config.roi_areas) * 1e6
        counts_perc = (counts / a.obs.groupby('roi').size()) * 100

        for group in ['disease', 'disease_subgroup']:
            for df2, label in [(counts_mm2, 'mm2'), (counts_perc, 'perc')]:
                if True not in df2.index.levels[2]: continue
                p = df2[:, :, True].rename(col + "_pos").to_frame()
                pp = p.pivot_table(index='roi', columns='cell_type_label_3.5', values=col + "_pos", fill_value=0)
                fig, stats = swarmboxenplot(data=pp.join(config.roi_attributes), x=group, y=pp.columns, plot_kws=dict(palette=config.colors[group]))
                fig.savefig(output_dir / f"{col}_positivity.by_{group}.{label}.swarmboxenplot.svg", **config.figkws)
                plt.close(fig)

                ppp = pp.join(config.roi_attributes[[group]]).groupby(group).mean().T.fillna(0)
                if ppp.shape[0] < 2: continue
                grid = clustermap(ppp, col_cluster=False, square=True, cbar_kws=dict(label=label))
                grid.savefig(output_dir / f"{col}_positivity.by_{group}.{label}.clustermap.svg", **config.figkws)
                plt.close(grid.fig)

                order = pp.mean().sort_values(ascending=False)
                _tp = pp.reset_index().melt(id_vars='roi')
                fig, ax = plt.subplots(figsize=(2, 4))
                sns.barplot(data=_tp, x='value', y='cell_type_label_3.5', ax=ax, order=order.index, color=sns.color_palette()[0])
                fig.savefig(output_dir / f"{col}_positivity.by_{group}.{label}.barplot.svg", **config.figkws)
                plt.close(fig)


    pd.Series(thresholds).to_csv(output_dir / 'positivity_thresholds.csv')

    # Positivity across all cell types and markers
    pos = pd.DataFrame(index=df.index, columns=thresholds)
    for m, t in thresholds.items():
        pos[m] = df[m] > t
    g = pos.join(a.obs[['roi', 'cell_type_label_3.5']]).groupby(['roi', 'cell_type_label_3.5'])
    p = g.sum()
    c = g.size()
    perc = pd.DataFrame([(p[col] / c).rename(col) for col in pos.columns]).T.fillna(0) * 100
    mm2 = pd.DataFrame([(p[col] / config.roi_areas).rename(col) for col in pos.columns]).T.fillna(0) * 1e6

    for group in ['disease', 'disease_subgroup']:
        for df2, label in [(mm2, 'mm2'), (perc, 'perc')]:
            p = df2.join(config.roi_attributes[group]).reset_index().groupby([group, 'cell_type_label_3.5']).mean().reset_index()
            pp = p.pivot_table(index='cell_type_label_3.5', columns=group)
            grid = clustermap(pp, cbar_kws=dict(label=label), col_cluster=False, square=True, config='abs')
            grid.savefig(output_dir / f"all_markers_positivity.by_{group}.{label}.clustermap.abs.svg", **config.figkws)
            plt.close(grid.fig)

            grid = clustermap(pp + np.random.random(pp.shape) * 1e-10, cbar_kws=dict(label=label), col_cluster=False, square=True, config='z')
            grid.savefig(output_dir / f"all_markers_positivity.by_{group}.{label}.clustermap.z.svg", **config.figkws)
            plt.close(grid.fig)

            ppp = pp.copy()
            for col in ppp.columns.levels[0]:
                ppp[col] = (ppp[col] - ppp[col].values.mean()) / ppp[col].values.std()

            ppp += np.random.random(ppp.shape) * 1e-10
            grid = clustermap(ppp.fillna(0), cbar_kws=dict(label=label), col_cluster=False, square=True, config='abs', vmin=0, vmax=5)
            grid.savefig(output_dir / f"all_markers_positivity.by_{group}.{label}.clustermap.std.svg", **config.figkws)
            plt.close(grid.fig)


    # Compare with expression
    df = a[:, config.cell_state_markers].to_df()
    col = 'SARSSpikeS1(Eu153)'
    # df.loc[~a.obs['covid'], col] = df.loc[~a.obs['covid'], col] * 0.8
    mean = df.join(a.obs[['roi', 'cell_type_label_3.5']]).groupby(['roi', 'cell_type_label_3.5']).mean().fillna(0)

    label = 'expression'
    for group in ['disease', 'disease_subgroup']:
        p = mean.join(config.roi_attributes[group]).reset_index().groupby([group, 'cell_type_label_3.5']).mean().reset_index()
        pp = p.pivot_table(index='cell_type_label_3.5', columns=group)

        pp = ((pp.T - pp.mean(1))).T

        grid = clustermap(pp, cbar_kws=dict(label=label), col_cluster=False, square=True, config='abs', vmin=0)
        grid.savefig(output_dir / f"all_markers_positivity.by_{group}.{label}.clustermap.abs.svg", **config.figkws)
        plt.close(grid.fig)

        grid = clustermap(pp + np.random.random(pp.shape) * 1e-10, cbar_kws=dict(label=label), col_cluster=False, square=True, config='z', vmin=-1, vmax=5)
        grid.savefig(output_dir / f"all_markers_positivity.by_{group}.{label}.clustermap.z.svg", **config.figkws)
        plt.close(grid.fig)


def periostin_interactions(a: AnnData) -> AnnData:
    a = sc.read(config.results_dir / "phenotyping" / "processed.labeled.h5ad")
    a.obs['covid'] = a.obs['disease'].isin(['COVID-19', 'Mixed', 'Convalescent'])

    # Check extent of periostin
    from src.pathology import quantify_fibrosis

    df = a.raw.to_adata().to_df()
    col = 'Periostin(Dy161)'

    fib = quantify_fibrosis(col)
    fib.to_csv(config.results_dir / 'pathology' / f'{col}_score.csv')
    fig, stats = swarmboxenplot(data=fib.join(config.roi_attributes), x='disease', y=fib.columns)
    fig.savefig(config.results_dir / 'pathology' / f'{col}_score.disease.svg', **config.figkws)
    fig, stats = swarmboxenplot(data=fib.join(config.roi_attributes), x='disease_subgroup', y=fib.columns)
    fig.savefig(config.results_dir / 'pathology' / f'{col}_score.disease_subgroup.svg', **config.figkws)

    # Quantify cell type distance to periostin
    from imc.ops.quant import quantify_cell_intensity
    _dists = list()
    for roi in tqdm(prj.rois):
        x = np.log1p(roi._get_channel(marker)[1].squeeze())
        mask = skimage.filters.gaussian(x, 2) > skimage.filters.threshold_otsu(x)
        dist = scipy.ndimage.morphology.distance_transform_edt((~mask).astype(int), sampling=[2, 2])
        q = quantify_cell_intensity(dist[np.newaxis, ...], roi.mask).rename(columns={0: 'distance'})
        _dists.append(q.assign(sample=roi.sample.name, roi=roi.name))

        # fig, axes = plt.subplots(1, 3)
        # axes[0].imshow(mask)
        # axes[1].imshow(~mask)
        # axes[2].imshow(dist)

    dists = pd.concat(_dists)
    index = dists.index
    dists = dists.merge(config.roi_areas.reset_index())
    dists.index = index
    dists['distance_norm'] = (dists['distance'] / dists['area_mm2']) * 1e6
    dists.to_csv(output_dir / f'{col}_cell_distance.csv')

    # plt.scatter(dists['distance'], dists['distance_norm'], alpha=0.1, s=2)

    a.obs[f'distance_to_{col}'] = dists['distance'].values
    a.obs[f'distance_to_{col}_norm'] = dists['distance_norm'].values
    diff_dists = a.obs.groupby(['disease', 'cell_type_label_3.5'])[f'distance_to_{col}_norm'].mean()

    (diff_dists['COVID-19'] - diff_dists['Normal']).sort_values()
    (diff_dists['Convalescent'] - diff_dists['Normal']).sort_values()
    (diff_dists['UIP/IPF'] - diff_dists['Normal']).sort_values()

    dists_p = diff_dists.to_frame().pivot_table(index='cell_type_label_3.5', columns='disease', values=f'distance_to_{col}')
    grid = clustermap(dists_p, cmap='RdBu_r', center=0)
    grid.savefig(output_dir / f'{col}_cell_distance.aggregated_cell_type.heatmap.svg', **config.figkws)
    grid = clustermap((dists_p.T - dists_p['Normal']).T.drop(['Normal'], axis=1), cmap='RdBu_r', center=0)
    grid.savefig(output_dir / f'{col}_cell_distance.aggregated_cell_type.relative_normal.heatmap.svg', **config.figkws)

    # Now account for cell type composition of ROIs
    c = a.obs[['roi', 'disease_subgroup', 'cell_type_label_3.5', f'distance_to_{col}']].copy()
    _res = list()
    for n in tqdm(range(100)):
        d = c.copy()
        for roi in c['roi'].unique():
            d.loc[d['roi'] == roi, 'cell_type_label_3.5'] = d.loc[d['roi'] == roi, 'cell_type_label_3.5'].sample(frac=1).values
        e = d.groupby(['disease_subgroup', 'cell_type_label_3.5'])[f'distance_to_{col}'].mean()
        _res.append(e)

    random_dists = pd.concat(_res).groupby(level=[0, 1]).mean()
    obs_dists = a.obs.groupby(['disease_subgroup', 'cell_type_label_3.5'])[f'distance_to_{col}'].mean()
    norm_dists = np.log(obs_dists / random_dists).rename(f"distance_to_{col}")

    norm_dists_agg = norm_dists.to_frame().pivot_table(index='cell_type_label_3.5', columns='disease_subgroup', values=f'distance_to_{col}')

    dist_diff = (norm_dists_agg.T - norm_dists_agg['Normal']).T.drop(['Normal'], axis=1)
    order = dist_diff.mean(1).sort_values().index

    grid = clustermap(norm_dists_agg.loc[order, :], figsize=(4, 5), center=0, row_cluster=False, col_cluster=False, cmap='PuOr_r', cbar_kws=dict(label="Relative distance to Periostin patch\nlog(observed / expected)"))
    grid.savefig(output_dir / f'{col}_cell_distance.aggregated_cell_type.norm.heatmap.svg', **config.figkws)

    grid = clustermap(dist_diff.loc[order, :], figsize=(4, 5), cmap='RdBu_r', center=0, row_cluster=False, col_cluster=False, cbar_kws=dict(label="Difference to 'Normal'\nin distance to Periostin patch"))
    grid.savefig(output_dir / f'{col}_cell_distance.aggregated_cell_type.norm.relative_normal.heatmap.svg', **config.figkws)




def cellular_interactions():
    output_dir = (config.results_dir / "interactions").mkdir()

    a = sc.read(config.results_dir / "phenotyping" / "processed.labeled.h5ad")
    c = a.obs.set_index(["sample", "roi", "obj_id"])[f"cell_type_label_3.5"].rename(
        "cluster"
    )
    f = prj.results_dir / "single_cell" / prj.name + ".adjacency_frequencies.csv"
    if not f.exists():
        prj.set_clusters(c, write_to_disk=True)
        adjs = prj.measure_adjacency()
    adjs = pd.read_csv(f, index_col=0)
    adjs = adjs.merge(config.roi_attributes.reset_index(), how="left")

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


    # Dimres based on interactions
    f = prj.results_dir / "single_cell" / prj.name + ".adjacency_frequencies.csv"
    adjs = pd.read_csv(f, index_col=0)

    adjs['interaction'] = adjs['index'] + " - " + adjs['variable']

    piv = adjs.pivot_table(index='roi', columns='interaction', values='value')

    grid = clustermap(piv, config='z')
    grid = clustermap(piv, config='abs', center=0, cmap='RdBu_r', robust=False)

    ai = AnnData(6 ** piv.values, obs=config.roi_attributes, var=piv.columns.to_frame())
    sc.pp.log1p(ai)
    sc.pp.highly_variable_genes(ai)
    sc.pl.highly_variable_genes(ai)
    sc.pp.scale(ai)
    sc.pp.pca(ai)

    sc.pp.neighbors(ai)
    sc.tl.umap(ai, gamma=10)
    sc.tl.diffmap(ai)

    _ai = ai[ai.obs.sample(frac=1).index, :]
    sc.pl.pca(_ai, color=config.attributes)
    sc.pl.umap(_ai, color=config.attributes)
    sc.pl.diffmap(_ai, color=config.attributes)



def normalize_background(a: AnnData):
    from tqdm.auto import tqdm
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    quant = pd.read_csv("processed/quantification.csv.gz", index_col=[0, 1])
    # quant.columns = quant.columns.str.split("(").map(lambda x: x[0])
    quant.columns = "C_" + quant.columns.str.replace(
        "(", "__", regex=False
    ).str.replace(")", "__", regex=False)
    tech_channels = ["C_80ArAr__ArAr80__", "C_129Xe__Xe129__"]
    tech = " + ".join(tech_channels)
    adj_channels = pd.Series(
        [
            "X_centroid",
            "Y_centroid",
            "area",
            "perimeter",
            "major_axis_length",
            "eccentricity",
            "solidity",
            "sample",
        ]
    )

    _regout = list()
    for roi in tqdm(quant.index.levels[0], position=0):
        x = quant.loc[roi, quant.columns.isin(tech_channels)]
        y = quant.loc[
            roi, ~quant.columns.isin(("C_" + adj_channels).tolist() + tech_channels)
        ]
        y = y.loc[:, y.var() > 0]
        data = x.join(y)

        _resid = list()
        for channel in tqdm(y.columns, position=1):
            model = smf.glm(
                f"{channel} ~ {tech}", family=sm.families.Poisson(), data=data
            ).fit()
            model.summary2().tables[1]
            _resid.append(
                pd.Series(model.resid_pearson, index=data.index, name=channel)
            )
        _regout.append(pd.DataFrame(_resid).T.assign(roi=roi))

    regout = pd.concat(_regout)
    # regout.columns = regout.columns.str.replace("C_", "").str.split("__").map(lambda x: x[0])

    m = regout.groupby(["roi"]).quantile(0.99).fillna(0)
    fig, stats = swarmboxenplot(
        data=m.join(config.roi_attributes), x="disease_subgroup", y=m.columns
    )

    regout.index = a.obs.index
    m = (
        regout.join(a.obs["cell_type_label_2.5"])
        .reset_index()
        .groupby(["roi", "cell_type_label_2.5"])
        .quantile(0.99)
    )

    m.columns = m.columns.str.replace(r"^C_", "", regex=True)
    c = m.columns.str.extract(r"(.*)__(.*)__")
    m.columns = c[0] + "(" + c[1] + ")"

    p = (
        m.join(config.roi_attributes)
        .reset_index()
        .groupby(["cell_type_label_2.5", "disease_subgroup"])
        .mean()
        .drop("age", axis=1)
    )

    grid = clustermap(p.T, config="z", z_score=0)

    grid = clustermap(p.T, config="z", z_score=0, col_cluster=False)
    grid.fig.savefig(output_dir / "regressed_out.clustermap.svg", **config.figkws)

    grid = clustermap(
        p[config.cell_type_markers].T, config="z", z_score=0, col_cluster=False
    )
    grid.fig.savefig(
        output_dir / "regressed_out.clustermap.cell_type_markers.svg", **config.figkws
    )

    grid = clustermap(
        p[config.cell_state_markers].T, config="z", z_score=0, col_cluster=False
    )
    grid.fig.savefig(
        output_dir / "regressed_out.clustermap.cell_state_markers.svg", **config.figkws
    )

    m = m.join(config.roi_attributes).query("disease != 'Mixed'")
    m["disease_subgroup"] = m["disease_subgroup"].cat.remove_unused_categories()

    for ct in m.index.levels[1]:
        fig, stats = swarmboxenplot(
            data=m.loc[:, ct, :],
            x="disease_subgroup",
            y=config.cell_state_markers + ["ColTypeI(Tm169)"],
            fig_kws=dict(figsize=(8, 8)),
            plot_kws=dict(palette=config.colors["disease_subgroup"]),
        )
        fig.savefig(
            output_dir
            / f"regressed_out.per_roi.swarmboxenplot.{ct}.cell_state_markers.svg",
            **config.figkws,
        )

    #
    #
    #

    raw = a.raw.to_adata().to_df()[a.var.index]

    y = raw.copy()  # .sample(n=50_000)
    y = y.drop(y.columns[y.var() == 0], axis=1)
    # y = ((y - y.mean()) / y.std())

    attrs = ["sample", "gender"]  # 'disease'
    x = (
        y.index.str.split("-")
        .map(lambda x: x[0] + "-" + x[1])
        .rename("roi")
        .to_frame()
        .join(config.roi_attributes)
    )
    x["sample"] = x["roi"].str.split("-").map(lambda x: x[0])

    for attr in attrs:
        x = pd.concat([x, pd.get_dummies(x[attr])], axis=1)
    x.index = y.index
    x["intercept"] = 1

    x = x.drop(attrs + ["roi", "disease_subgroup", "disease"], axis=1)
    x["age"] = x["age"].astype(int)

    _params = list()
    _res = list()
    for col in tqdm(y.columns):
        # model = sm.OLS(y[col], x, family=sm.families.Poisson()).fit()
        # _res.append(model.resid.rename(col))
        model = sm.GLM(y[col], x, family=sm.families.Poisson()).fit_regularized()
        _res.append(model.resid_pearson.rename(col))
        _params.append(model.summary2().tables[1].assign(col=col))
    params = pd.concat(_params)
    res = pd.DataFrame(_res).T
    res.to_csv(config.results_dir / "regression.poisson.residuals.csv")
    res = pd.read_csv(
        config.results_dir / "regression.poisson.residuals.csv", index_col=0
    )

    # sns.heatmap(res.groupby('roi').mean() - nres.groupby('roi').mean(), xticklabels=True, yticklabels=True, center=0, robust=True)

    ## how about now removing background noise from other channels
    background_channels = [
        "_EMPTY_(In115)",
        "129Xe(Xe129)",
        "_EMPTY_(Pt194)",
        "_EMPTY_(Pt198)",
    ]
    x2 = res[background_channels]
    y2 = res[res.columns[~res.columns.isin(background_channels)]].drop("roi", axis=1)

    _params2 = list()
    _res2 = list()
    for col in tqdm(y2.columns):
        model = sm.OLS(y2[col], x2).fit()
        _res2.append(model.resid.rename(col))
        _params2.append(model.summary2().tables[1].assign(col=col))
    params2 = pd.concat(_params2)
    res2 = pd.DataFrame(_res2).T

    res["roi"] = res.index.str.split("-").map(lambda x: x[0] + "-" + x[1])
    # res['sample'] = res.index.str.split("-").map(lambda x: x[0])
    m = (
        res.merge(config.roi_attributes.reset_index())
        .groupby("disease_subgroup")
        .mean()
    )
    sns.heatmap(m.T, xticklabels=True, yticklabels=True)

    res["roi"] = res.index.str.split("-").map(lambda x: x[0] + "-" + x[1])
    m = res.groupby("roi").mean().join(config.roi_attributes)
    fig, stats = swarmboxenplot(
        data=m, x="disease_subgroup", y=res.columns.drop(["roi"])
    )
    fig.savefig(
        config.results_dir / "regression.poisson.residuals.swarmboxenplot.svg",
        **config.figkws,
    )

    res2["roi"] = res2.index.str.split("-").map(lambda x: x[0] + "-" + x[1])
    m = res2.groupby("roi").mean().join(config.roi_attributes)
    fig, stats = swarmboxenplot(
        data=m, x="disease_subgroup", y=res2.columns.drop(["roi"])
    )
    fig.savefig(
        config.results_dir / "regression.double.residuals.swarmboxenplot.svg",
        **config.figkws,
    )


def marker_comparison(a: AnnData):

    # Spike expression

    ## Mean of channels
    mean = pd.read_csv(
        config.results_dir / "single_cell" / "covid-pasc-imc.channel_mean.csv"
    )
    mean = (
        mean.set_index("roi")
        .join(config.roi_attributes[["disease_subgroup"]])
        .reset_index()
    )

    # mean = mean.groupby(['sample', 'channel']).mean().join(config.sample_attributes).reset_index()

    i = mean.query("channel == 'SARSSpikeS1(Eu153)'")
    fig, stats = swarmboxenplot(data=i, x="disease_subgroup", y="value")
    fig.savefig(
        config.results_dir
        / "phenotyping"
        / "SARSSpikeS1_expression.sample.whole_image.svg",
        **config.figkws,
    )

    # Raw values
    m = (
        a.raw.to_adata()
        .to_df()[a.var.index]
        .join(a.obs)
        .groupby(["cell_type_label_2.5", "roi"])
        .mean()
    )
    ## All cell types
    i = m["SARSSpikeS1(Eu153)"].reset_index().merge(config.roi_attributes.reset_index())
    # fig, stats = swarmboxenplot(data=i, x='disease_subgroup', y='SARSSpikeS1(Eu153)')

    i = (
        m.loc["Epithelial"]["SARSSpikeS1(Eu153)"]
        .reset_index()
        .merge(config.roi_attributes.reset_index())
    )
    fig, stats = swarmboxenplot(data=i, x="disease_subgroup", y="SARSSpikeS1(Eu153)")
    fig.savefig(
        config.results_dir / "phenotyping" / "SARSSpikeS1_expression.roi.raw.svg",
        **config.figkws,
    )

    # Normalized values
    m = a.to_df().join(a.obs).groupby(["cell_type_label_2.5", "roi"]).mean()
    ## All cell types
    i = m["SARSSpikeS1(Eu153)"].reset_index().merge(config.roi_attributes.reset_index())
    # fig, stats = swarmboxenplot(data=i, x='disease_subgroup', y='SARSSpikeS1(Eu153)')

    i = (
        m.loc["Epithelial"]["SARSSpikeS1(Eu153)"]
        .reset_index()
        .merge(config.roi_attributes.reset_index())
    )
    fig, stats = swarmboxenplot(data=i, x="disease_subgroup", y="SARSSpikeS1(Eu153)")
    fig.savefig(
        config.results_dir / "phenotyping" / "SARSSpikeS1_expression.roi.norm.svg",
        **config.figkws,
    )

    # Remove tech: 129Xe

    ## Astir probability
    ### all cell types
    # i = m['infected'].reset_index().merge(config.roi_attributes.reset_index())
    # fig, stats = swarmboxenplot(data=i, x='disease_subgroup', y='infected')

    ### epithelial
    i = (
        m.loc["Epithelial"]["infected"]
        .reset_index()
        .merge(config.roi_attributes.reset_index())
    )
    fig, stats = swarmboxenplot(data=i, x="disease_subgroup", y="infected")
    fig.savefig(
        config.results_dir
        / "phenotyping"
        / "SARSSpikeS1_expression.roi.astir_prob.svg",
        **config.figkws,
    )


def abundance_comparison(a: AnnData):
    output_dir = (config.results_dir / "abundance").mkdir()

    clusters = a.obs.columns[a.obs.columns.str.startswith("cluster")].tolist()
    cell_type_labels = a.obs.columns[
        a.obs.columns.str.startswith("cell_type_label")
    ].tolist()

    for ctl in cell_type_labels:
        cts = a.obs.set_index(["sample", "roi", "obj_id"])[ctl].rename("cluster")
        prj.set_clusters(cts)
        prj.sample_comparisons(
            output_prefix=output_dir / ctl + ".",
            channel_exclude=config.channels_exclude,
            sample_attributes=config.attribute_order,
        )


def gating(a: AnnData):
    from imc.ops.mixture import (
        get_best_mixture_number,
        get_threshold_from_gaussian_mixture,
    )

    output_dir = (config.results_dir / "gating").mkdir()

    quant = a[:, a.var.index.isin(config.channels_include)].to_df()

    # Univariate gating of each channel
    thresholds_file = output_dir / "thresholds.json"
    mixes_file = output_dir / "mixes.json"
    if not (thresholds_file.exists() and thresholds_file.exists()):
        mixes = dict()
        thresholds = dict()
        for m in quant.columns:
            if m not in thresholds:
                mixes[m] = get_best_mixture_number(quant[m], 2, 8)
                thresholds[m] = get_threshold_from_gaussian_mixture(
                    quant[m], None, mixes[m]
                ).to_dict()
                json.dump(thresholds, open(thresholds_file, "w"))
                json.dump(mixes, open(mixes_file, "w"))
    thresholds = json.load(open(output_dir / "thresholds.json"))
    mixes = json.load(open(output_dir / "mixes.json"))

    # Make dataframe with population for each marker
    gating_file = output_dir / "gating.csv"
    ids = ["sample", "roi"]
    if not gating_file.exists():
        pos = pd.DataFrame(
            index=quant.index, columns=quant.columns, dtype=bool
        ).rename_axis(index="obj_id")
        for m in pos.columns:
            pos[m] = False
            name = m.split("(")[0]
            o = sorted(thresholds[m])
            pos[m] = quant[m] > thresholds[m][o[-1]]
            # if mixes[m] == 2:
            #     pos[m] = (quant[m] > thresholds[m][o[0]])
            #     # .replace({False: name + "-", True: name + "+"})
            # else:
            #     pos[m] = (quant[m] > thresholds[m][o[-1]])
            #     # .replace({True: name + "+"})
            #     sel = pos[m] == False
            #     pos.loc[sel, m] = (
            #         quant.loc[sel, m] > thresholds[m][o[-2]]
            #     )
            #     # .replace({True: name + "dim", False: name + "-"})
        pos.to_csv(gating_file)
    pos = pd.read_csv(gating_file, index_col=0)

    cell_type_labels = a.obs.columns[
        a.obs.columns.str.startswith("cell_type_label")
    ].tolist()

    p = pos.groupby(a.obs["cell_type_label_2.5"]).sum().T / a.obs.shape[0] * 100
    grid = clustermap(p.loc[config.cell_state_markers])

    p = pos.groupby(a.obs["cell_type_label_2.5"]).sum() / pos.sum(0) * 100
    grid = clustermap(p)

    p = pos.join(a.obs[["roi", "cell_type_label_2.5"]]).query(
        "`cell_type_label_2.5` == 'Epithelial'"
    )
    count = p.groupby("roi").sum()
    count = (count.T / count.sum(1)).T * 100

    fig, stats = swarmboxenplot(
        data=count.join(config.roi_attributes),
        x="disease_subgroup",
        y=config.cell_state_markers,
    )

    p = (
        pos.join(a.obs[["cell_type_label_2.5", "disease_subgroup"]])
        .groupby(["cell_type_label_2.5", "disease_subgroup"])
        .mean()
    )

    grid = clustermap(
        p[config.cell_state_markers], row_cluster=False, row_colors=p.index.to_frame()
    )


def add_microanatomical_context(a: AnnData):
    from utag import utag

    output_dir = (config.results_dir / "domains").mkdir()

    # Run domain discovery
    b = a[:, ~a.var.index.isin(config.channels_exclude)]
    b = b.raw.to_adata()[:, b.var.index]
    b = b[:, b.var.index != "Ki67(Er168)"]

    s = utag(
        b,
        save_key="utag_cluster",
        slide_key="roi",
        max_dist=10,
        clustering_method=["kmeans"],
        resolutions=[0.5, 1.0, 1.5],
    )
    # clusts = s.obs.columns[s.obs.columns.str.contains("utag_cluster")].tolist()
    # sc.pl.pca(s, color=["sample"] + clusts)
    # sc.pl.umap(s, color=["sample"] + clusts)
    # sc.pl.spatial(s, spot_size=20, color=s.var.index.tolist() + clusts)

    # Observe the cell type composition of domains
    clust = "cell_type_label_1.0"
    domain = "utag_cluster_kmeans_1.0"
    c = s.obs.groupby(domain)[clust].value_counts()
    cf = (c / c.groupby(level=0).sum()).rename("count")
    cfp = cf.reset_index().pivot_table(
        index=clust, columns=domain, values="count", fill_value=0
    )
    grid = clustermap(cfp, config="abs")
    grid.fig.savefig(output_dir / "domain_composition.clustermap.svg")

    domain_assignments = {}
    s.obs["domain"] = pd.Categorical(
        s.obs[clust].astype(int).replace(domain_assignments)
    )
    s.write(output_dir / "utag.h5ad")
    s = sc.read(output_dir / "utag.h5ad")

    # Illustrate markers, cell types, and domains for each ROI
    clusters = s.obs.columns[s.obs.columns.str.startswith("cluster")].tolist()
    cell_type_labels = s.obs.columns[
        s.obs.columns.str.startswith("cell_type_label")
    ].tolist()
    domains = s.obs.columns[s.obs.columns.str.startswith("utag")].tolist()

    combs = [
        ["K818", "ColTypeI(", "CD31"],
        ["CD45", "CD3(", "CD68"],
        ["CD206", "MPO", "CD14"],
        ["SC5b9", "CitH3", "SARS"],
        ["IL1beta", "SFTPC", "KIT"],
    ]
    to_plot = cell_type_labels + domains  # + clusters

    tc = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    tc = [
        sns.color_palette("tab10")[3],
        sns.color_palette("tab10")[2],
        sns.color_palette("tab10")[0],
    ]

    # # this is just to have consistent colors across ROIs for 'domain'
    for c in to_plot:
        _ = sc.pl.umap(s, color=c, show=False)
        plt.close("all")

    m = len(combs) + len(to_plot)
    for sample in tqdm(prj.samples):
        f = output_dir / f"domain_composition.illustration.{sample.name}.pdf"
        if f.exists():
            continue
        n = len(sample.rois)
        fig, axes = plt.subplots(
            n, m, figsize=(m * 4, n * 4), gridspec_kw=dict(hspace=0, wspace=0.1)
        )
        fig.suptitle(f"Sample: {sample.name}, group: {sample.disease}")
        for axs, roi in zip(axes, sample.rois):
            for ax, comb in zip(axs, combs):
                roi.plot_channels(comb, merged=True, target_colors=tc, axes=[ax])

            s1 = s[s.obs["roi"] == roi.name, :].copy()
            for ax, c in zip(axs[len(combs) :], to_plot):
                sc.pl.spatial(s1, spot_size=20, color=c, show=False, ax=ax)
                leg = ax.get_legend()
                if leg:
                    leg.set_bbox_to_anchor((0.5, -0.85))
                    leg.set(visible=True if ax in axes[-1] else False)
        rasterize_scanpy(fig)
        fig.savefig(f, **config.figkws)
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
