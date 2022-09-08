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


def main() -> int:
    assert all(r.get_input_filename("stack").exists() for r in prj.rois)
    assert all(r.get_input_filename("cell_mask").exists() for r in prj.rois)

    # filter_noise()

    cohort_characteristics()
    illustrate()
    qc()
    phenotype()

    periostin_interactions()
    cith3_ammount()
    marker_positivity()
    infected_cells()

    increased_diversity(
        prefix="myeloid",
        cois=[
            "Monocytes",
            "Macrophages",
            "Peribronchial Macrophages",
            "CD16+ inflammatory monocytes",
            "Dendritic cells",
        ],
        resolution=3.0,
    )
    increased_diversity(
        prefix="tcell",
        cois=["Peribronchial CD4 T", "Peribronchial CD8 T", "CD8 T"],
        resolution=3.0,
    )
    increased_diversity(
        prefix="stroma",
        cois=["Fibroblasts", "Basal", "Mesenchymal"],
        resolution=3.0,
    )

    cellular_interactions()

    # unsupervised(
    #     add_expression=False,
    #     regress_out=False,
    #     scale=False,
    #     prefix="noscale.noregout.abundance",
    #     corrmaps=True,
    # )
    unsupervised(
        add_expression=False,
        regress_out=True,
        scale=True,
        prefix="scale.regout.abundance",
        corrmaps=True,
    )
    unsupervised(
        add_expression=True,
        regress_out=True,
        scale=True,
        prefix="scale.regout.mixed_abundance_expression",
        corrmaps=True,
    )

    contrast_coefficients()

    return 0


def get_domain_areas(as_percentage: bool = True):
    from shapely.geometry import Polygon

    topo_annots, topo_sc = get_topological_domain_annotation()
    _areas = dict()
    for roi in topo_annots:
        top_areas = {
            dom["label"]: Polygon(dom["points"]).area for dom in topo_annots[roi]
        }
        top_areas["total"] = config.roi_areas[roi]
        _areas[roi] = top_areas

    areas = (
        pd.DataFrame(_areas)
        .fillna(0)
        .drop("total")
        .T.rename_axis(index="roi")
        .sort_index()
    )
    if as_percentage:
        areas = (areas.T / areas.sum(1)).T * 100
    return areas


def qc_signal():
    _metrics = list()
    for r in tqdm(prj.rois):
        df = pd.DataFrame(
            [
                r.stack.mean((1, 2)).tolist(),
                r.stack.var((1, 2)).tolist(),
                r.channel_labels.tolist(),
                [r.name] * r.channel_labels.shape[0],
            ],
            index=["mean", "var", "channel", "roi"],
        ).T
        _metrics.append(df)
    metrics = pd.concat(_metrics)
    for col in ["mean", "var"]:
        metrics[col] = metrics[col].astype(float)
    metrics.to_csv(config.results_dir / "roi_channel_stats.csv", index=False)
    # metrics = metrics.join(config.roi_attributes)

    fig, axes = plt.subplots(6, 8, figsize=(3 * 8, 3 * 6), sharex=False, sharey=False)
    for i, (ax, ch) in enumerate(zip(fig.axes, r.channel_labels)):
        p = metrics.query(f"channel == '{ch}'").set_index("roi")[["mean", "var"]]
        ax.scatter(p["mean"], p["var"], alpha=0.75, s=3)
        # for s in p.index:
        #     ax.text(p.loc[s, 'mean'], p.loc[s, 'var'], s=s)
        v = p["mean"].max()
        b = p["mean"].min()
        for cv in [1, 10, 100]:
            ax.plot((b, v), (b * cv, v * cv), linestyle="--", color="grey")
        ax.set(title=ch)
        ax.loglog()
    for ax in fig.axes[i + 1 :]:
        ax.axis("off")

    metrics["cv"] = np.sqrt(metrics["var"]) / metrics["mean"]
    metricsp = metrics.pivot_table(index="roi", columns="channel", values="cv")

    grid = clustermap(np.log1p(metricsp))

    _res = list()
    for r in tqdm(prj.rois):
        r.channel_labels.iloc[23]
        ch = np.log1p(r.stack[23])
        ft = np.fft.fftshift(np.fft.fft2(ch))
        _res.append(ft)

        fig, axes = plt.subplots(1, 2)
        axes[0].imshow(r.stack[23])
        axes[1].imshow(np.log(abs(ft)))

    res = np.asarray(_res)
    plt.scatter(metricsp.iloc[:, 23], [abs(x).mean() for x in res])

    metricsp["fft"] = [abs(x).mean() for x in res]

    fig, stats = swarmboxenplot(
        data=metricsp.join(config.roi_attributes), x="disease_subgroup", y="fft"
    )


def sars_detection_and_pathology():
    from imc.graphics import get_grid_dims

    a = sc.read(config.results_dir / "phenotyping" / "processed.labeled.h5ad")
    a.obs["covid"] = a.obs["disease"].isin(["COVID-19", "Mixed", "Convalescent"])

    output_dir = config.results_dir / "pathology"
    output_prefix = output_dir / "overlap_IMC_pathology."
    cell_type_label = "cell_type_label_3.5"
    grouping = "sample"

    exp = (
        a.to_df()
        .join(a.obs[[cell_type_label, grouping]])
        .groupby([cell_type_label, grouping])
        .mean()
    )
    path = pd.read_csv("metadata/samples.pathology_tests.csv", index_col=0).replace(
        [None], np.nan, regex=False
    )
    cols = path.columns[path.columns.str.startswith("lung")]
    path["lung_positive"] = False
    path.loc[path.loc[:, cols].isnull().all(1), "lung_positive"] = np.nan
    path.loc[path.loc[:, cols].any(1), "lung_positive"] = True

    p = (
        exp.reset_index()
        .pivot_table(
            index="sample", columns=cell_type_label, values="SARSSpikeS1(Eu153)"
        )
        .join(getattr(config, f"{grouping}_attributes"))
        .query("disease == 'Convalescent'")
        .join(path["lung_positive"])
    )

    fig, stats = swarmboxenplot(data=p, x="lung_positive", y=exp.index.levels[0])
    for ax in fig.axes:
        ax.set(ylabel="SARSSpikeS1(Eu153)", xlabel="SARS Spike positivity (ISH, IHC)")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    fig.savefig(output_prefix + "swarmboxenplot.yaxis_flexible.svg", **config.figkws)

    fig, stats = swarmboxenplot(
        data=p, x="lung_positive", y=exp.index.levels[0], fig_kws=dict(sharey=True)
    )
    for ax in fig.axes:
        ax.set(
            ylabel="SARSSpikeS1(Eu153)",
            xlabel="SARS Spike positivity (ISH, IHC)",
            ylim=(0, 1.8),
        )
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    fig.savefig(output_prefix + "swarmboxenplot.yaxis_fixed.svg", **config.figkws)

    stats.to_csv(output_prefix + "stats.csv")
    stats = pd.read_csv(output_prefix + "stats.csv")

    fig, ax = plt.subplots()
    stats["mean"] = stats[["median_A", "median_B"]].mean(1)
    sns.regplot(x=stats["mean"], y=stats["hedges"] * -1, scatter=False, ax=ax)
    res = pg.corr(x=stats["mean"], y=stats["hedges"] * -1).squeeze()
    ax.scatter(stats["mean"], stats["hedges"] * -1)
    v = stats["hedges"].abs().max()
    v += v * 0.1
    ax.set(
        xlabel="Mean SARSSpikeS1 expression",
        ylabel="Agreement IMC vs IHC/ISH (fold enrichment)",
        ylim=(-v, v),
        title=f"r = {res['r']}; p = {res['p-val']}",
    )
    ax.axhline(0, linestyle="--", color="grey")
    for _, row in stats.iterrows():
        ax.text(row["mean"], row["hedges"] * -1, s=row["Variable"])
    ax.axvline(1, linestyle="--", color="grey")
    fig.savefig(output_prefix + "agreement_hedges.scatterplot.svg", **config.figkws)

    perc = (a.obs["cell_type_label_3.5"].value_counts() / a.obs.shape[0]) * 100
    sel = perc[perc >= 1].index

    stats = stats.loc[stats["Variable"].isin(sel), :]
    fig, ax = plt.subplots()
    stats["mean"] = stats[["median_A", "median_B"]].mean(1)
    sns.regplot(x=stats["mean"], y=stats["hedges"] * -1, scatter=False, ax=ax)
    res = pg.corr(x=stats["mean"], y=stats["hedges"] * -1).squeeze()
    ax.scatter(stats["mean"], stats["hedges"] * -1)
    v = stats["hedges"].abs().max()
    v += v * 0.1
    ax.set(
        xlabel="Mean SARSSpikeS1 expression",
        ylabel="Agreement IMC vs IHC/ISH (fold enrichment)",
        ylim=(-v, v),
        title=f"r = {res['r']}; p = {res['p-val']}",
    )
    ax.axhline(0, linestyle="--", color="grey")
    for _, row in stats.iterrows():
        ax.text(row["mean"], row["hedges"] * -1, s=row["Variable"])
    ax.axvline(1, linestyle="--", color="grey")
    ax.set(xlim=(-0.05, 2.05))
    fig.savefig(
        output_prefix + "agreement_hedges.scatterplot.filtered_1%.svg", **config.figkws
    )

    p = (
        exp.reset_index()
        .pivot_table(
            index="sample", columns=cell_type_label, values="SARSSpikeS1(Eu153)"
        )
        .join(getattr(config, f"{grouping}_attributes"))
        .join(path["lung_positive"])
    )
    p["infected"] = False
    p.loc[p["Alveolar type 2"] > 1.4, "infected"] = True
    classification = p[["disease_subgroup", "infected"]].copy()
    classification["infected"] = pd.Categorical(
        classification["infected"], ordered=True
    )

    a = sc.read(config.results_dir / "phenotyping" / "processed.labeled.h5ad")
    a.obs["covid"] = a.obs["disease"].isin(["COVID-19", "Mixed", "Convalescent"])

    output_dir = config.results_dir / "pathology"
    output_prefix = output_dir / "overlap_IMC_pathology."
    cell_type_label = "cell_type_label_3.5"
    grouping = "roi"

    counts = a.obs.groupby([grouping])[[cell_type_label]].value_counts().rename("count")
    counts_mm2 = (
        ((counts / getattr(config, f"{grouping}_areas") * 1e6))
        .rename("cells_per_mm2")
        .to_frame()
        .pivot_table(
            index=grouping,
            columns=cell_type_label,
            values="cells_per_mm2",
            fill_value=0,
        )
    )
    p = counts_mm2.join(
        getattr(config, f"{grouping}_attributes")
    )  # .query("disease_subgroup.str.startswith('COVID-19-long')", engine='python')
    # fig, ax = plt.subplots()
    # ax.scatter(p['disease_subgroup'].cat.codes, p['Alveolar type 2'])
    p = p.merge(classification[["infected"]], left_on="sample", right_index=True)
    fig, stats = swarmboxenplot(data=p, x="infected", y=counts_mm2.columns)
    fig.savefig(
        output_prefix + "regrouped_by_SARS_IMC_intensity.swarmboxenplot.by_sample.svg",
        **config.figkws,
    )

    # p.index = p.index.str.extract(r"(.*)-\d+")[0].rename("sample")

    fig, stats = swarmboxenplot(data=p, x="infected", y=counts_mm2.columns)
    fig.savefig(
        output_prefix + "regrouped_by_SARS_IMC_intensity.swarmboxenplot.by_roi.svg",
        **config.figkws,
    )

    p["disease_subgroup_reordered"] = p["disease_subgroup"]
    p.loc[
        p["disease_subgroup"].str.startswith("COVID-19-long") & (p["infected"] == True),
        "disease_subgroup_reordered",
    ] = "COVID-19-long-neg"
    p.loc[
        p["disease_subgroup"].str.startswith("COVID-19-long")
        & (p["infected"] == False),
        "disease_subgroup_reordered",
    ] = False
    fig, stats = swarmboxenplot(
        data=p, x="disease_subgroup_reordered", y=counts_mm2.columns
    )
    fig.savefig(
        output_prefix
        + "regrouped_by_SARS_IMC_intensity.swarmboxenplot.by_roi.all_subgroups.svg",
        **config.figkws,
    )


def abundance_and_pathology():
    a = sc.read(config.results_dir / "phenotyping" / "processed.labeled.h5ad")
    a.obs["covid"] = a.obs["disease"].isin(["COVID-19", "Mixed", "Convalescent"])

    output_dir = config.results_dir / "pathology"
    output_prefix = output_dir / "overlap_IMC_pathology."
    cell_type_label = "cell_type_label_3.5"
    grouping = "roi"

    counts = a.obs.groupby([grouping])[[cell_type_label]].value_counts().rename("count")
    counts_mm2 = (
        ((counts / getattr(config, f"{grouping}_areas") * 1e6))
        .rename("cells_per_mm2")
        .to_frame()
        .pivot_table(
            index=grouping,
            columns=cell_type_label,
            values="cells_per_mm2",
            fill_value=0,
        )
    )
    p = counts_mm2.join(getattr(config, f"{grouping}_attributes")).query(
        "disease_subgroup.str.startswith('COVID-19-long')", engine="python"
    )

    path = pd.read_csv("metadata/samples.pathology_tests.csv", index_col=0).replace(
        [None], np.nan, regex=False
    )
    cols = path.columns[path.columns.str.startswith("lung")]
    path["lung_positive"] = False
    path.loc[path.loc[:, cols].isnull().all(1), "lung_positive"] = np.nan
    path.loc[path.loc[:, cols].any(1), "lung_positive"] = True
    cols = path.columns[path.columns.str.startswith("trachea")]
    path["trachea_positive"] = False
    path.loc[path.loc[:, cols].isnull().all(1), "trachea_positive"] = np.nan
    path.loc[path.loc[:, cols].any(1), "trachea_positive"] = True

    p = p.merge(path, left_index=True, right_index=True, how="left")

    fig, stats = swarmboxenplot(data=p, x="lung_positive", y=counts_mm2.columns)
    for ax in fig.axes:
        ax.set(ylabel="cells per mm2", xlabel="SARS Spike positivity (ISH, IHC)")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    fig.savefig(
        output_prefix
        + "reassignment_based_on_pathology.cell_abundance.swarmboxenplot.svg",
        **config.figkws,
    )

    # # Macs
    # roi_name = 'A20_77_A27-02'
    # roi_name = 'A20_185_A33_v1-07'
    # roi = [r for r in prj.rois if r.name == roi_name][0]
    # fig = roi.plot_channels(['CD163', 'CD206', 'ColTypeI', 'K818'], equalize=False, minmax=False, log=True, merged=True, smooth=0.5)
    # fig.savefig(config.results_dir / 'illustration' / roi.name + ".markers.merged.svg", **config.figkws)


def contrast_coefficients():
    from matplotlib.patches import Circle
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    output_dir = (config.results_dir / "temporal").mkdir()

    a = sc.read(config.results_dir / "phenotyping" / "processed.labeled.h5ad")
    a.obs["covid"] = a.obs["disease"].isin(["COVID-19", "Mixed", "Convalescent"])

    meta = pd.read_csv(config.metadata_dir / "samples.csv", index_col=0).rename_axis(
        "sample"
    )

    cell_type_label = "cell_type_label_3.5"
    var = "days_since_first_infection"
    # var = "age"
    for grouping in ["sample", "roi"]:
        attrs = getattr(config, grouping + "_attributes")
        attrs = (
            attrs.reset_index()
            .merge(meta.reset_index()[["sample", var]], how="left")
            .set_index(grouping)
        )
        counts = (
            a.obs.groupby([grouping])[[cell_type_label]].value_counts().rename("count")
        )
        counts_mm2 = (
            ((counts / getattr(config, f"{grouping}_areas") * 1e6))
            .rename("cells_per_mm2")
            .to_frame()
            .pivot_table(
                index=grouping,
                columns=cell_type_label,
                values="cells_per_mm2",
                fill_value=0,
            )
        )
        data = counts_mm2.join(attrs[[var]]).dropna()
        x = data.drop(var, axis=1)
        y = data[[var]]

        x = (x - x.mean()) / x.std()
        y = (y - y.mean()) / y.std()
        model = sm.GLM(y, x.assign(Intercept=1)).fit()
        model.summary2()
        res = pd.DataFrame([model.params, model.pvalues], index=["hedges", "p-unc"]).T
        res.to_csv(
            output_dir / f"regression.{var}.cells_per_mm2.GLM.per_{grouping}.csv"
        )
        res = pd.read_csv(
            output_dir / f"regression.{var}.cells_per_mm2.GLM.per_{grouping}.csv",
            index_col=0,
        )

        res = res.sort_values("hedges").drop("Intercept")

        fig, ax = plt.subplots(figsize=(4 * 1.1, 4))
        v = res["hedges"].abs().max()
        v += v * 0.1
        ax.scatter(
            res["hedges"],
            res.index,
            s=5 + 5 ** (-np.log10(res["p-unc"])),
            c=res["hedges"],
            cmap="coolwarm",
            vmin=-v,
            vmax=v,
            edgecolor="black",
            alpha=0.75,
        )
        ax.axvline(0, linestyle="--", color="grey")
        ps = [1e-0, 1e-1, 1e-2, 1e-3][::-1]
        ax.scatter(
            [0] * len(ps),
            [0] * len(ps),
            s=5 + 5 ** (-np.log10(ps)),
            edgecolor="black",
            alpha=0.75,
            color="orange",
        )
        fig.savefig(
            output_dir
            / f"regression.{var}.cells_per_mm2.GLM.per_{grouping}.scatter.rank_vs_value.svg",
            **config.figkws,
        )

        # counts_perc = (
        #     ((counts / counts.groupby(level=0).sum()))
        #     .rename("cells_percent")
        #     .to_frame()
        #     .pivot_table(
        #         index=grouping,
        #         columns=cell_type_label,
        #         values="cells_percent",
        #         fill_value=0,
        #     )
        # )
        # data = counts_perc.join(meta[[var]]).dropna()
        # x = data.drop(var, axis=1)
        # y = data[[var]] / data[[var]].sum()

        # model = sm.GLM(y, x, family=sm.families.Gamma(sm.families.links.Log())).fit()

    # Using coefficients from comparing groups (phenotype function)
    fr = (
        config.results_dir
        / "phenotyping"
        / "clustering.cell_type_label_3.5.swarmboxenplot.by_roi_and_disease.area.no_mixed.stats.csv"
    )
    stats1 = pd.read_csv(fr)
    fs = (
        config.results_dir
        / "phenotyping"
        / "clustering.cell_type_label_3.5.swarmboxenplot.by_roi_and_disease_subgroup.area.no_mixed.stats.csv"
    )
    stats2 = pd.read_csv(fs)
    stats = stats1.append(stats2)

    pairs = [
        [
            ("Normal", "UIP/IPF"),
            ("Normal", "COVID-19"),
            (-1, -1),
        ],
        [
            ("Normal", "COVID-19"),
            ("Normal", "Convalescent"),
            (-1, -1),
        ],
        [
            ("Normal", "COVID-19"),
            ("COVID-19", "Convalescent"),
            (-1, -1),
        ],
        [
            ("COVID-19", "Convalescent"),
            ("COVID-19-long-pos", "COVID-19-long-neg"),
            (-1, 1),
        ],
        [
            ("COVID-19-early", "COVID-19-late"),
            ("COVID-19-long-pos", "COVID-19-long-neg"),
            (1, 1),
        ],
        [
            ("Normal", "COVID-19"),
            ("COVID-19-early", "COVID-19-long-pos"),
            (-1, -1),
        ],
        [
            ("Normal", "COVID-19"),
            ("COVID-19-late", "COVID-19-long-pos"),
            (-1, -1),
        ],
        [
            ("Normal", "COVID-19"),
            ("COVID-19-early", "COVID-19-long-neg"),
            (-1, -1),
        ],
        [
            ("Normal", "COVID-19"),
            ("COVID-19-late", "COVID-19-long-neg"),
            (-1, -1),
        ],
        [
            ("Normal", "COVID-19"),
            ("COVID-19", "Convalescent"),
            (-1, -1),
        ],
        [
            ("Normal", "COVID-19-long-pos"),
            ("COVID-19-long-pos", "COVID-19-long-neg"),
            (-1, 1),
        ],
        [
            ("Normal", "COVID-19-long-neg"),
            ("COVID-19-long-pos", "COVID-19-long-neg"),
            (-1, -1),
        ],
    ]

    fig, axes = plt.subplots(
        1, len(pairs), figsize=(4 * len(pairs) * 1.1, 4), squeeze=False
    )
    for ax, (pair1, pair2, changes) in zip(fig.axes, pairs):
        a1 = (
            stats.query(f"A == '{pair1[0]}' & B == '{pair1[1]}'")
            .set_index("Variable")
            .groupby(level=0)
            .mean()
        )
        a1["hedges"] *= changes[0]
        b1 = (
            stats.query(f"A == '{pair2[0]}' & B == '{pair2[1]}'")
            .set_index("Variable")
            .groupby(level=0)
            .mean()
        )
        b1["hedges"] *= changes[1]

        ps = pd.concat([a1["p-unc"], b1["p-unc"]], axis=1).min(1)
        # v = pd.concat([a1["hedges"], b1["hedges"]]).abs().max()
        v = 2.2
        v += v * 0.1
        ax.scatter(
            a1["hedges"],
            b1["hedges"],
            s=5 + 3 ** (-np.log10(ps)),
            c=(a1["hedges"] * b1["hedges"]),
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
            edgecolor="black",
            alpha=0.75,
        )
        ax.axhline(0, linestyle="--", color="grey")
        ax.axvline(0, linestyle="--", color="grey")
        # ax.set(xlim=(-v, v), ylim=(-v, v))
        for s in a1.index:
            ax.text(a1.loc[s, "hedges"], b1.loc[s, "hedges"], s=s)
        ax.set(
            xlabel=f"Fold change ({pair1[0]} vs {pair1[1]})"
            if changes[0] == 1
            else f"Fold change ({pair1[1]} vs {pair1[0]})",
            ylabel=f"Fold change ({pair2[0]} vs {pair2[1]})"
            if changes[0] == 1
            else f"Fold change ({pair2[1]} vs {pair2[0]})",
        )
    ps = [1e-0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6][::-1]
    fig.axes[-1].scatter(
        [0] * len(ps),
        [0] * len(ps),
        s=5 + 3 ** (-np.log10(ps)),
        edgecolor="black",
        alpha=0.75,
        color="orange",
    )
    fig.savefig(
        # output_dir / "coefficient_comparison.pairs.scatter.fixed_axes.svg",
        output_dir / "coefficient_comparison.pairs.scatter.flexible_axes.svg",
        **config.figkws,
    )

    # Compare coefficient from regressing time vs comparing groups
    fig, ax = plt.subplots(figsize=(4 * 1 * 1.1, 4))
    a1 = (
        stats.query(f"A == 'COVID-19-long-pos' & B == 'COVID-19-long-neg'")
        .set_index("Variable")
        .groupby(level=0)
        .mean()
    )
    b1 = res.sort_index()
    ps = pd.concat([a1["p-unc"], b1["p-unc"]], axis=1).min(1)
    v = 2.2
    v += v * 0.1
    ax.scatter(
        a1["hedges"],
        b1["hedges"],
        s=5 + 3 ** (-np.log10(ps)),
        c=(a1["hedges"] * b1["hedges"]),
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        edgecolor="black",
        alpha=0.75,
    )
    sns.regplot(a1["hedges"], b1["hedges"], ax=ax, scatter=False, color="grey")
    ax.axhline(0, linestyle="--", color="grey")
    ax.axvline(0, linestyle="--", color="grey")
    # ax.set(xlim=(-v, v), ylim=(-v, v))
    for s in a1.index:
        ax.text(a1.loc[s, "hedges"], b1.loc[s, "hedges"], s=s)
    ax.set(
        xlabel=f"Fold change (PASC-neg vs PASC-pos)",
        ylabel=f"Fold change (time since onset)",
    )
    ps = [1e-0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6][::-1]
    fig.axes[-1].scatter(
        [0] * len(ps),
        [0] * len(ps),
        s=5 + 3 ** (-np.log10(ps)),
        edgecolor="black",
        alpha=0.75,
        color="orange",
    )
    fig.savefig(
        output_dir / "coefficient_comparison.group_vs_time.scatter.flexible_axes.svg",
        **config.figkws,
    )

    # TODO: regress out time per cell type in expression of markers


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


def cohort_temporal_visualization():
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    import matplotlib.patches as patches

    output_dir = (config.results_dir / "cohort").mkdir()

    df = pd.read_csv(config.metadata_dir / "samples.csv", index_col=0)
    df = df.query(
        "disease.isin(['Convalescent', 'Mixed'])", engine="python"
    ).sort_values("date_first_infection")

    for col in df.columns[df.columns.str.startswith("date_")]:
        if df[col].str.contains("-").any():
            continue
        df[col] = pd.to_datetime(df[col])

    # colors = config.colors["disease_subgroup"][-2:]

    s = plt.rcParams["lines.markersize"] ** 2
    fig, ax = plt.subplots(figsize=(8, 3))
    i = 0
    # for subgroup, color in zip(df['disease_subgroup'].unique(), colors):
    # df2 = df.query(f"disease_subgroup == '{subgroup}'")
    for (
        pid,
        row,
    ) in df.iterrows():
        ax.plot(
            (row["date_first_infection"], row["date_death"]),
            (i, i),
            color="grey",
            linestyle="--",
            zorder=-1000,
        )
        ax.scatter(
            row["date_first_infection"],
            i - 0,
            s=s + s * 0.1,
            color="purple",
            marker="s",
        )
        ax.scatter(row["date_death"], i + 0, s=s + s * 0.1, color="red", marker="s")

        # Negative tests
        for date in pd.to_datetime(row["dates_neg_tests"].split(",")):
            ax.scatter(date, i, s=s, color="green", edgecolors="black", alpha=0.8)
        # Positive tests
        for date in pd.to_datetime(row["dates_pos_tests"].split(",")):
            ax.scatter(date, i, s=s, color="orange", edgecolors="black", alpha=0.8)
        # Additional events
        if row["dates_events"] != "":
            for event in row["dates_events"].split(","):
                date, desc = event.split("-")
                ax.scatter(pd.to_datetime(date), i, color="grey", marker="x", s=s / 2)
                ax.text(pd.to_datetime(date), i + 0.1, s=desc)
        i += 1
    ax.set_yticks(range(df.shape[0]))
    ax.set_yticklabels(df["alternative_id"])
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y-%m"))

    # rect = patches.Rectangle((pd.to_datetime(""), 100), 40, 30, linewidth=1, edgecolor='r', facecolor='none')
    # ax.add_patch(rect)
    x, y = (
        pd.to_datetime("2021/03/01"),
        pd.to_datetime("2021/07/01"),
        pd.to_datetime("2021/07/01"),
        pd.to_datetime("2021/03/01"),
        pd.to_datetime("2021/03/01"),
    ), (6 + 0.5, 6 + 0.5, 10 + 0.5, 10 + 0.5, 6 + 0.5)
    ax.plot(x, y, linestyle="--", color="black", alpha=0.5, linewidth=0.5)

    ax2 = inset_axes(ax, "50%", "35%", loc="upper left")
    for (
        pid,
        row,
    ) in df.iloc[-4:].iterrows():
        ax2.plot(
            (row["date_first_infection"], row["date_death"]),
            (i, i),
            color="grey",
            linestyle="--",
            zorder=-1000,
        )
        ax2.scatter(
            row["date_first_infection"],
            i - 0,
            s=s + s * 0.1,
            color="purple",
            marker="s",
        )
        ax2.scatter(row["date_death"], i + 0, s=s + s * 0.1, color="red", marker="s")

        # Negative tests
        for date in pd.to_datetime(row["dates_neg_tests"].split(",")):
            ax2.scatter(date, i, s=s, color="green", edgecolors="black", alpha=0.8)
        # Positive tests
        for date in pd.to_datetime(row["dates_pos_tests"].split(",")):
            ax2.scatter(date, i, s=s, color="orange", edgecolors="black", alpha=0.8)
        # Additional events
        if row["dates_events"] != "":
            for event in row["dates_events"].split(","):
                date, desc = event.split("-")
                ax2.scatter(pd.to_datetime(date), i, color="grey", marker="x", s=s / 2)
                ax2.text(pd.to_datetime(date), i + 0.1, s=desc)
        i += 1
    ax2.set_xticks(
        [
            pd.to_datetime("2021/04/01"),
            pd.to_datetime("2021/05/01"),
            pd.to_datetime("2021/06/01"),
        ]
    )
    ax2.set_xticklabels(ax2.get_xticklabels(), fontsize=8)
    ax2.set_yticklabels([])
    ax2.margins(0.05, 0.15)
    ax2.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y-%m"))
    for axis in ["top", "bottom", "left", "right"]:
        ax2.spines[axis].set(linewidth=0.5, linestyle="--", alpha=0.5)
    fig.savefig(
        output_dir / "cohort_timeline.convalescent_patients.svg", **config.figkws
    )


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

    a.uns["disease_subgroup_colors"] = config.colors["disease_subgroup"]

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

    _a = a[a.obs.sample(frac=1).index, :]
    clusters = a.obs.columns[a.obs.columns.str.startswith("cluster_")].tolist()
    fig = sc.pl.umap(_a, color=clusters, ncols=1, show=False)[0].figure
    rasterize_scanpy(fig)
    fig.savefig(output_dir / "umap.clusters.svg", **config.figkws)
    plt.close(fig)

    fig = sc.pl.umap(
        _a,
        color="disease_subgroup",
        ncols=1,
        show=False,
        s=5,
        alpha=0.25,
    ).figure
    rasterize_scanpy(fig)
    fig.savefig(output_dir / "umap.disease_subgroup.svg", **config.figkws)
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
        b = a[a.obs["disease_subgroup"] == "Normal", :]
        means = b.to_df().groupby(a.obs[cluster]).mean()
        means = means.loc[:, means.std() > 0]
        sizes = b.to_df().groupby(a.obs[cluster]).size().rename("cell_count")

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

        c = b.obs.groupby(cluster)[["topological_domain"]].value_counts()
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
                a.obs.groupby(grouping)[[f"cell_type_label_{resolution}"]]
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
                stats.to_csv(
                    output_dir
                    / f"clustering.cell_type_label_{resolution}.swarmboxenplot.by_{grouping}_and_{attr}.area.stats.csv",
                    index=False,
                )

            p = p.query("disease != 'Mixed'")
            p["disease"] = p["disease"].cat.remove_unused_categories()
            p["disease_subgroup"] = p["disease_subgroup"].cat.remove_unused_categories()
            # color not matched!
            for attr in config.colors:
                colors = config.colors[attr].copy()
                if attr == "disease_subgroup":
                    colors = np.delete(colors, 4, axis=0)
                fig, stats = swarmboxenplot(
                    data=p,
                    x=attr,
                    y=counts_mm2.columns,
                    plot_kws=dict(palette=colors),
                    # fig_kws=dict(figsize=(4, 12)),
                )
                fig.savefig(
                    output_dir
                    / f"clustering.cell_type_label_{resolution}.swarmboxenplot.by_{grouping}_and_{attr}.area.no_mixed.svg",
                    **config.figkws,
                )
                plt.close(fig)

                stats.to_csv(
                    output_dir
                    / f"clustering.cell_type_label_{resolution}.swarmboxenplot.by_{grouping}_and_{attr}.area.no_mixed.stats.csv",
                    index=False,
                )

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


def differential_location():
    output_dir = (config.results_dir / "domains" / "differential_location").mkdir()
    a = sc.read(config.results_dir / "phenotyping" / "processed.labeled.h5ad")
    cluster = "cell_type_label_3.5"

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

    c = (
        a.obs.groupby(["disease_subgroup", cluster, "roi"])[["topological_domain"]]
        .value_counts()
        .rename("count")
        .reorder_levels([0, 2, 3, 1])
    )
    t = (
        a.obs.groupby(["disease_subgroup", "roi"])[["topological_domain"]]
        .value_counts()
        .rename("total")
    )
    c = (c / t).rename("fraction")  # .reorder_levels([0, 2, 1])

    pc = c.reset_index().pivot_table(
        index=["roi", "disease_subgroup", cluster],
        columns="topological_domain",
        fill_value=0,
        values="fraction",
        aggfunc="mean",
    )
    pt = (pc.T / pc.sum(1)).T.fillna(0) * 100

    grid = clustermap(pt.groupby(level=cluster).mean(), config="abs")
    grid = clustermap(pt.groupby(level=cluster).mean(), config="z")
    grid.fig.savefig(
        output_dir / f"domain_proportions.clustermap.by_celltype.svg", **config.figkws
    )

    ptg = pt.groupby(level=[cluster, "disease_subgroup"]).mean()
    ptg = (
        ptg.loc[:, config.attribute_order["disease_subgroup"], :]
        .drop("Mixed", level=1)
        .sort_index(level=0, sort_remaining=False)
        .loc[:, order]
    )
    grid = clustermap(
        ptg,
        config="abs",
        row_cluster=False,
        col_cluster=False,
        row_colors=ptg.index.to_frame(),
        yticklabels=True,
    )
    ptgz = ptg / ptg.sum()
    ptgz = (ptgz.T / ptgz.sum(1)).T
    grid = clustermap(
        ptgz,
        config="abs",
        row_cluster=False,
        col_cluster=False,
        row_colors=ptg.index.to_frame(),
        yticklabels=True,
    )
    grid.fig.savefig(
        output_dir / f"domain_proportions.clustermap.sorted.svg", **config.figkws
    )

    # ptz = (pt - pt.mean()) / pt.std()
    ptz = pt

    _stats = list()
    for ct in tqdm(pt.index.levels[2]):
        d = ptz.loc[:, :, ct, :].reset_index()
        d["disease_subgroup"] = pd.Categorical(
            d["disease_subgroup"],
            ordered=True,
            categories=config.attribute_order["disease_subgroup"],
        )
        fig, stats = swarmboxenplot(
            data=d,
            x="disease_subgroup",
            y=pt.columns,
            plot_kws=dict(palette=config.colors["disease_subgroup"]),
        )
        fig.savefig(
            output_dir / f"domain_proportions.{ct}.swarmboxenmplot.svg", **config.figkws
        )
        _stats.append(stats.assign(cell_type=ct))
    stats = pd.concat(_stats)

    q = (
        stats.sort_values("hedges")
        .dropna()
        .query(
            "A != 'Mixed' & B != 'Mixed' & A != 'UIP/IPF' & B != 'UIP/IPF' & Variable.isin(['A', 'AR', 'V']).values"
        )
    )
    q.query("cell_type == 'CD16+ inflammatory monocytes'")
    q.query("cell_type == 'Alveolar type 2'")

    grid = clustermap(pt, config="z", row_colors=pt.index.to_frame())


def illustrate_CC16_secretion_bleeding():
    # "A18_19_A19-02"
    rois = ["S19_6699_B9-01", "S20_1842_A7-05", "A20_79_A23-08", "A20_100_A28-09"]
    for roi_name in rois:
        roi = [r for r in prj.rois if r.name == roi_name][0]
        fig = roi.plot_channels(
            ["AQ1", "CC16", "ColTypeI"],
            equalize=True,
            minmax=False,
            log=True,
            merged=True,
            smooth=1,
        )
        fig.axes[0].set_title(roi.disease_subgroup)
        fig.savefig(
            config.results_dir
            / "illustration"
            / f"CC16_secretion_bleeding.{roi.name}.svg",
            **config.figkws,
        )


def illustrate_CitH3_NETs():
    rois = ["A20_77_A27-04", "A20_77_A27-03", "A20_185_A33_v1-01"]
    for roi_name in rois:
        roi = [r for r in prj.rois if r.name == roi_name][0]
        fig = roi.plot_channels(
            # ["K818", "MPO", "ColTypeI", "CitH3"],
            # ["MPO", "CitH3", "ColTypeI", 'uPAR', 'CD31'],
            ["MPO", "CitH3", "ColTypeI", "CD31"],
            target_colors=["yellow", "red", "green", "blue"],
            equalize=True,
            minmax=False,
            log=True,
            merged=True,
            smooth=1,
        )
        fig.axes[0].set_title(roi.disease_subgroup)
        fig.savefig(
            config.results_dir / "illustration" / f"CitH3_NETs.{roi.name}.svg",
            **config.figkws,
        )


def infected_cells() -> AnnData:
    output_dir = (config.results_dir / "positivity").mkdir()

    a = sc.read(config.results_dir / "phenotyping" / "processed.labeled.h5ad")
    a.obs["covid"] = a.obs["disease"].isin(["COVID-19", "Mixed", "Convalescent"])

    col = "SARSSpikeS1(Eu153)"
    a.X[~a.obs["covid"], a.var.index == col] *= 0.8
    a.raw.X[~a.obs["covid"], a.raw.var.index == col] *= 0.8
    a.write(config.results_dir / "phenotyping" / "processed.labeled.signalfix.h5ad")

    # Conditional scalling approach
    # # Shrink SARS signal in non-COVID samples, find a threshold which maximizes separation
    df = np.log1p(a.raw.to_adata().to_df())
    # df = a.to_df()
    col = "SARSSpikeS1(Eu153)"
    _res = list()
    for f in tqdm(np.linspace(1, 0, 50)):
        df2 = df.copy()
        df2.loc[~a.obs["covid"], col] *= f
        for t in np.linspace(0, 5, 50):
            pos = df2[col] > t
            p = pos.groupby(a.obs["covid"]).sum()
            n = pos.groupby(a.obs["covid"]).size()
            _res.append([f, t, p[True], p[False], n[True], n[False]])
    res = pd.DataFrame(
        _res,
        columns=[
            "shrinkage_factor",
            "threshold",
            "pos_COVID",
            "pos_NON",
            "n_COVID",
            "n_NON",
        ],
    )
    res["frac_COVID"] = res["pos_COVID"] / res["n_COVID"]
    res["frac_NON"] = res["pos_NON"] / res["n_NON"]
    res["log_ratio"] = np.log(
        (res["pos_COVID"] / res["n_COVID"]) / (res["pos_NON"] / res["n_NON"])
    )

    vmax = res[res != np.inf]["log_ratio"].max() * 0.95
    _p = res.pivot_table(
        index="shrinkage_factor", columns="threshold", values="log_ratio"
    ).iloc[1:, 1:]
    _p.index = _p.index.to_series().round(2)
    _p.columns = _p.columns.to_series().round(2)
    fig, ax = plt.subplots()
    sns.heatmap(
        _p,
        vmax=vmax,
        ax=ax,
        cbar_kws=dict(label="log(COVID / non-COVID) SARSSpikeS1+ cells"),
    )
    fig.savefig(output_dir / "SARSSpikeS1_thresholds.svg")

    # Choose parameters: shrink 0.8, threshold 1.0
    df.loc[~a.obs["covid"], col] *= 0.8
    pos = df[col] > thresholds[col]
    a.obs[col + "_pos"] = pos

    a.obs.groupby(pos)["covid"].value_counts()

    counts = a.obs.groupby(["roi", "cell_type_label_3.5"])[col + "_pos"].value_counts()
    counts_mm2 = (counts / config.roi_areas) * 1e6
    counts_perc = (counts / a.obs.groupby("roi").size()) * 100

    for group in ["disease", "disease_subgroup"]:
        for df, label in [(counts_mm2, "mm2"), (counts_perc, "perc")]:
            p = df[:, :, True].rename(col + "_pos").to_frame()
            pp = p.pivot_table(
                index="roi",
                columns="cell_type_label_3.5",
                values=col + "_pos",
                fill_value=0,
            )
            fig, stats = swarmboxenplot(
                data=pp.join(config.roi_attributes),
                x=group,
                y=pp.columns,
                plot_kws=dict(palette=config.colors[group]),
            )
            fig.savefig(
                output_dir
                / f"SARSSpikeS1_positivity.by_{group}.{label}.swarmboxenplot.svg",
                **config.figkws,
            )
            plt.close(fig)

            ppp = pp.join(config.roi_attributes[[group]]).groupby(group).mean().T
            grid = clustermap(
                ppp, col_cluster=False, square=True, cbar_kws=dict(label=label)
            )
            grid.savefig(
                output_dir
                / f"SARSSpikeS1_positivity.by_{group}.{label}.clustermap.svg",
                **config.figkws,
            )
            plt.close(grid.fig)

    q = (
        counts_perc[:, "Alveolar type 2", True]
        .to_frame("pos")
        .join(config.roi_attributes)
        .query("disease == 'Convalescent'")
    )
    q["disease_subgroup"] = q["disease_subgroup"].cat.remove_unused_categories()
    fig, stats = swarmboxenplot(data=q, x="disease_subgroup", y="pos")

    # Plot
    q = (
        a.obs.groupby(pos)["cell_type_label_3.5"].value_counts()[True]
        / a.obs["cell_type_label_3.5"].value_counts()
    )

    ct = (
        a.obs.groupby(pos)["cell_type_label_3.5"].value_counts()
        / a.obs.groupby(pos)["cell_type_label_3.5"].size()
    )  # / a.obs['cell_type_label_3.5'].value_counts()
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=False)
    o = ct[True].sort_values(ascending=False).index
    sns.barplot(x=ct[False].reindex(o), y=o, ax=axes[0], orient="horiz")
    sns.barplot(x=ct[True], y=o, ax=axes[1], orient="horiz")
    axes[0].set(title="SARS-CoV-2 Spike neg")
    axes[1].set(title="SARS-CoV-2 Spike pos")
    fig.savefig(output_dir / "SARSSpikeS1_infected.per_cell_type.svg", **config.figkws)

    ct_cov = (
        a.obs.query("covid").groupby(pos)["cell_type_label_3.5"].value_counts()
        / a.obs.query("covid").groupby(pos)["cell_type_label_3.5"].size()
    )
    ct_non = (
        a.obs.query("~covid").groupby(pos)["cell_type_label_3.5"].value_counts()
        / a.obs.query("~covid").groupby(pos)["cell_type_label_3.5"].size()
    )

    _res2 = list()
    for t in np.linspace(0, 5, 50):
        pos = val > t
        ct = (
            a.obs.groupby(pos)["cell_type_label_3.5"].value_counts()
            / a.obs.groupby(pos)["cell_type_label_3.5"].size()
        )
        if True in ct.index:
            _res2.append(ct[True].rename(t))
    res2 = pd.DataFrame(_res2)

    res2.index = res2.index.to_series().round(2)
    fig, ax = plt.subplots()
    sns.heatmap(res2, ax=ax)
    fig.savefig(
        output_dir / "SARSSpikeS1_thresholds_per_cell_type.svg", **config.figkws
    )

    fig, ax = plt.subplots()
    sns.heatmap((res2 - res2.mean()) / res2.std(), ax=ax)
    fig.savefig(
        output_dir / "SARSSpikeS1_thresholds_per_cell_type.zscore.svg", **config.figkws
    )

    # compare phenotype of infected vs non-infected per cell type
    _v = [col + "_pos", "cell_type_label_3.5"]
    df = a.raw.to_adata().to_df()
    df = np.log1p((df.T / df.sum(1)).T * 1e4)
    df2 = df
    # df2 = df.loc[a.obs['disease'] == 'Convalescent', :]

    aggs = df2.join(a.obs[_v]).groupby(_v).mean()
    diff = (aggs.loc[True, :, :] - aggs.loc[False, :, :])[
        config.cell_state_markers
    ]  # .drop(col, axis=1)
    # diff = diff[[x for x in config.channels_include if x in diff.columns]]

    ns = df2.join(a.obs[_v]).groupby(_v).size()

    diff = diff.loc[:, diff.mean(0).sort_values().index]

    grid = clustermap(
        diff,
        cmap="RdBu_r",
        center=0,
        col_cluster=True,
        cbar_kws=dict(label="log fold change (SARSSpikeS1+/SARSSpikeS1-)"),
        row_colors=ns[False].to_frame("neg").join(ns[True].rename("pos")),
        config="abs",
        figsize=(2, 4),
    )
    grid.savefig(
        output_dir / "SARSSpikeS1_positivity.change_in_expression.clustermap.svg",
        **config.figkws,
    )

    for ct in sorted(a.obs["cell_type_label_3.5"].unique()):
        df3 = df.loc[
            (a.obs["cell_type_label_3.5"] == ct)
            & ~a.obs["disease_subgroup"].isin(["Normal", "UIP/IPF"])
        ]
        p = df3.join(a.obs[["disease_subgroup", col + "_pos"]])
        p["disease_subgroup"] = p["disease_subgroup"].cat.remove_unused_categories()

        fig, axes = plt.subplots(
            3,
            4,
            figsize=(4 * 2, 3 * 4),
            sharex=True,
            sharey=True,
            gridspec_kw=dict(wspace=0, hspace=0),
        )
        for ax, marker in zip(
            axes.flat, [x for x in config.cell_state_markers if x != col]
        ):
            sns.violinplot(
                data=p, x="disease_subgroup", y=marker, hue=col + "_pos", ax=ax
            )
            ax.set(xlabel="", ylabel="")
            ax.set_title(marker, y=0.9)
        for ax in axes[-1, :]:
            ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        for ax in axes.flatten()[1:]:
            l = ax.get_legend()
            if l:
                l.set_visible(False)
        t = p.groupby(["disease_subgroup", col + "_pos"]).size()
        fig.suptitle(ct + "\nn = " + "; ".join(t.astype(str)))
        fig.savefig(
            output_dir
            / f"SARSSpikeS1_positivity.change_in_expression.{ct.replace(' ', '_')}.violinplot.svg",
            **config.figkws,
        )
        plt.close(fig)


def marker_positivity() -> AnnData:
    output_dir = (config.results_dir / "positivity").mkdir()

    a = sc.read(config.results_dir / "phenotyping" / "processed.labeled.h5ad")
    a.obs["covid"] = a.obs["disease"].isin(["COVID-19", "Mixed", "Convalescent"])

    # df = np.log1p(a.raw.to_adata().to_df())
    df = a.raw.to_adata().to_df()
    df = np.log1p((df.T / df.sum(1)).T * 1e4)

    col = "SARSSpikeS1(Eu153)"
    df.loc[~a.obs["covid"], col] *= 0.8

    thresholds = dict()
    for col in tqdm(sorted(config.cell_state_markers)):
        _res2 = list()
        for t in np.linspace(0, df[col].max(), 50):
            pos = df[col] > t
            ct = (
                a.obs.groupby(pos)["cell_type_label_3.5"].value_counts()
                / a.obs.groupby(pos)["cell_type_label_3.5"].size()
            )
            if True in ct.index:
                _res2.append(ct[True].rename(t))
        res2 = pd.DataFrame(_res2)

        res2.index = res2.index.to_series().round(2)
        fig, ax = plt.subplots()
        sns.heatmap(res2, ax=ax)
        fig.savefig(output_dir / f"{col}_thresholds_per_cell_type.svg", **config.figkws)

        fig, ax = plt.subplots()
        sns.heatmap((res2 - res2.mean()) / res2.std(), ax=ax)
        fig.savefig(
            output_dir / f"{col}_thresholds_per_cell_type.zscore.svg", **config.figkws
        )

        # threshold
        from imc.ops.mixture import get_population

        pos = get_population(df[col])
        thresholds[col] = df.loc[pos, col].min()
        # pos = (df[col] > 1)
        a.obs[col + "_pos"] = pos

        counts = a.obs.groupby(["roi", "cell_type_label_3.5"])[
            col + "_pos"
        ].value_counts()
        counts_mm2 = (counts / config.roi_areas) * 1e6
        counts_perc = (counts / a.obs.groupby("roi").size()) * 100

        for group in ["disease", "disease_subgroup"]:
            for df2, label in [(counts_mm2, "mm2"), (counts_perc, "perc")]:
                if True not in df2.index.levels[2]:
                    continue
                p = df2[:, :, True].rename(col + "_pos").to_frame()
                pp = p.pivot_table(
                    index="roi",
                    columns="cell_type_label_3.5",
                    values=col + "_pos",
                    fill_value=0,
                )
                fig, stats = swarmboxenplot(
                    data=pp.join(config.roi_attributes),
                    x=group,
                    y=pp.columns,
                    plot_kws=dict(palette=config.colors[group]),
                )
                fig.savefig(
                    output_dir
                    / f"{col}_positivity.by_{group}.{label}.swarmboxenplot.svg",
                    **config.figkws,
                )
                plt.close(fig)

                ppp = (
                    pp.join(config.roi_attributes[[group]])
                    .groupby(group)
                    .mean()
                    .T.fillna(0)
                )
                if ppp.shape[0] < 2:
                    continue
                grid = clustermap(
                    ppp, col_cluster=False, square=True, cbar_kws=dict(label=label)
                )
                grid.savefig(
                    output_dir / f"{col}_positivity.by_{group}.{label}.clustermap.svg",
                    **config.figkws,
                )
                plt.close(grid.fig)

                order = pp.mean().sort_values(ascending=False)
                _tp = pp.reset_index().melt(id_vars="roi")
                fig, ax = plt.subplots(figsize=(2, 4))
                sns.barplot(
                    data=_tp,
                    x="value",
                    y="cell_type_label_3.5",
                    ax=ax,
                    order=order.index,
                    color=sns.color_palette()[0],
                )
                fig.savefig(
                    output_dir / f"{col}_positivity.by_{group}.{label}.barplot.svg",
                    **config.figkws,
                )
                plt.close(fig)

    pd.Series(thresholds).to_csv(output_dir / "positivity_thresholds.csv")

    thresholds = (
        pd.read_csv(output_dir / "positivity_thresholds.csv", index_col=0)
        .squeeze()
        .to_dict()
    )

    # Positivity across all cell types and markers
    pos = pd.DataFrame(index=df.index, columns=thresholds)
    for m, t in thresholds.items():
        pos[m] = df[m] > t
    g = pos.join(a.obs[["roi", "cell_type_label_3.5"]]).groupby(
        ["roi", "cell_type_label_3.5"]
    )
    p = g.sum()
    c = g.size()
    perc = (
        pd.DataFrame([(p[col] / c).rename(col) for col in pos.columns]).T.fillna(0)
        * 100
    )
    mm2 = (
        pd.DataFrame(
            [(p[col] / config.roi_areas).rename(col) for col in pos.columns]
        ).T.fillna(0)
        * 1e6
    )

    perc.to_csv(output_dir / "positivity_per_cell_type.percentage.csv")
    mm2.to_csv(output_dir / "positivity_per_cell_type.area.csv")

    # # Global view
    # m = perc.reset_index().melt(id_vars=["roi", "cell_type_label_3.5"])
    # p = m.pivot_table(
    #     index=["cell_type_label_3.5", "variable"], columns=["roi"], values="value"
    # )
    # p = ((p.T - p.mean(1)) / p.std(1)).T
    # p = (p - p.mean()) / p.std()
    # grid = clustermap(p.corr(), row_colors=config.roi_attributes[['disease_subgroup', 'sample']], cmap='coolwarm', vmin=-1, vmax=1)
    # grid.savefig(output_dir / "positivity.similarity.roi.clustermap.svg", **config.figkws)

    # pg = p.T.join(config.roi_attributes[['disease_subgroup', 'sample']]).groupby('disease_subgroup').mean().T
    # grid = clustermap(pg.corr(), cmap='coolwarm', vmin=-1, vmax=1)
    # grid.savefig(output_dir / "positivity.similarity.disease_subgroup.clustermap.svg", **config.figkws)

    # # Illustrations
    # p = perc.loc[:, 'Alveolar type 2', :]['SARSSpikeS1(Eu153)']
    # c = config.roi_attributes.join(p).query("disease_subgroup == 'COVID-19-long-pos'").sort_values("SARSSpikeS1(Eu153)")
    # 'A20_137_A26'
    # roi_name = c.index[-2]
    # roi_name = 'A21_63_A14-05'
    # roi = [r for r in prj.rois if r.name == roi_name][0]
    # fig = roi.plot_channels(['aSMA', 'AQ1', 'Ki67', 'Periostin', 'ColTypeI'], equalize=False, minmax=False, log=True, merged=True, smooth=0.5)
    # fig.savefig(config.results_dir / 'illustration' / roi.name + ".markers.merged.svg", **config.figkws)

    # fig = roi.plot_channels(['ColTypeI', 'CD31', 'K818'], equalize=True, merged=False)
    # fig = roi.plot_channels(['ColTypeI', 'CD31', 'K818'], equalize=True, merged=True)
    # fig = roi.plot_channels(['CD68', 'IL6', 'IRF'], equalize=True, merged=True)
    # fig = roi.plot_channels(['CD68', 'IL6', 'IRF'], equalize=True, merged=False)
    # # fig.savefig(config.results_dir / 'illustration' / roi.name + ".markers.svg", **config.figkws)

    # roi_name = 'A21_67_A30-04'
    # pos = [(360, 270), (450, 350)]

    # fig = roi.plot_channels(['SARS', 'aSMA', 'CD45', 'SFTPC', 'SFTPA', 'K818', 'IL6', 'MPO', 'CD206', 'AQ1', 'CC3', 'Col', 'CD31'], equalize=True, position=pos)
    # # fig.savefig(config.results_dir / 'illustration' / roi.name + ".markers.svg", **config.figkws)

    # fig = roi.plot_channels(['DNA1', 'SARS', 'SFTPC'], position=pos, equalize=False, minmax=False, log=False, smooth=0)
    # fig.savefig(config.results_dir / 'illustration' / roi.name + ".markers.zoom.separate.svg", **config.figkws)

    # fig = roi.plot_channels(['DNA1', 'SARS', 'SFTPC'], position=pos, equalize=False, minmax=False, log=False, merged=True, smooth=0)
    # fig.savefig(config.results_dir / 'illustration' / roi.name + ".markers.zoom.merged.svg", **config.figkws)

    # pal = sns.color_palette('tab10')
    # target = [pal[3], pal[2], pal[0]]
    # for ch, color in zip(['SARS', 'SFTPC', 'K818'], target):
    #     fig = roi.plot_channels([ch], position=pos, equalize=False, minmax=False, log=False, merged=True, smooth=1)
    #     fig.savefig(config.results_dir / 'illustration' / roi.name + f".markers.zoom.{ch}.svg", **config.figkws)

    # fig, axes = plt.subplots(2, 2)
    # pal = sns.color_palette('tab10')
    # target = [pal[3], pal[2], pal[0]]
    # roi.plot_channels(['SARS', 'SFTPC', 'DNA1',], position=pos, equalize=False, minmax=False, log=False, merged=True, smooth=0, target_colors=target, axes=[fig.axes[0]])
    # for ax, ch, color in zip(fig.axes[1:], ['SARS', 'SFTPC', 'DNA1',], target):
    #     colors = [(0,0,0), color]
    #     newcmp = matplotlib.colors.LinearSegmentedColormap.from_list('', colors)
    #     # newcmp = matplotlib.colors.LinearSegmentedColormap('testCmap', segmentdata=cdict, N=256)
    #     roi.plot_channel(ch, position=pos, equalize=False, minmax=False, log=False, smooth=1, ax=ax, cmap=newcmp)
    # fig.savefig(config.results_dir / 'illustration' / roi.name + f".markers.zoom.separate.svg", **config.figkws)

    # fig = roi.plot_channels(['SARS', 'SFTPC', 'K818'], merged=True, target_colors=target, equalize=False, log=False, minmax=False)  # target_colors=['red', 'green', 'blue']
    # fig.savefig(config.results_dir / 'illustration' / roi.name + ".markers.merged.svg", **config.figkws)

    # for ch, color in zip(['SARS', 'SFTPC', 'K818'], target):
    #     fig = roi.plot_channels([ch], merged=True, target_colors=[color], equalize=False, log=False, minmax=False)  # target_colors=['red', 'green', 'blue']
    #     fig.savefig(config.results_dir / 'illustration' / roi.name + f".markers.{ch}.svg", **config.figkws)

    for group in ["disease", "disease_subgroup"]:
        for df2, label in [(mm2, "mm2"), (perc, "perc")]:
            p = (
                df2.join(config.roi_attributes[group])
                .reset_index()
                .groupby([group, "cell_type_label_3.5"])
                .mean()
                .reset_index()
            )
            pp = p.pivot_table(index="cell_type_label_3.5", columns=group)
            grid = clustermap(
                pp,
                cbar_kws=dict(label=label),
                col_cluster=False,
                square=True,
                config="abs",
            )
            grid.savefig(
                output_dir
                / f"all_markers_positivity.by_{group}.{label}.clustermap.abs.svg",
                **config.figkws,
            )
            plt.close(grid.fig)

            grid = clustermap(
                pp + np.random.random(pp.shape) * 1e-10,
                cbar_kws=dict(label=label),
                col_cluster=False,
                square=True,
                config="z",
            )
            grid.savefig(
                output_dir
                / f"all_markers_positivity.by_{group}.{label}.clustermap.z.svg",
                **config.figkws,
            )
            plt.close(grid.fig)

            ppp = pp.copy()
            for col in ppp.columns.levels[0]:
                ppp[col] = (ppp[col] - ppp[col].values.mean()) / ppp[col].values.std()

            ppp += np.random.random(ppp.shape) * 1e-10
            grid = clustermap(
                ppp.fillna(0),
                cbar_kws=dict(label=label),
                col_cluster=False,
                square=True,
                config="abs",
                vmin=0,
                vmax=5,
            )
            grid.savefig(
                output_dir
                / f"all_markers_positivity.by_{group}.{label}.clustermap.std.svg",
                **config.figkws,
            )
            plt.close(grid.fig)

    # Compare with expression
    df = a[:, config.cell_state_markers].to_df()
    col = "SARSSpikeS1(Eu153)"
    # df.loc[~a.obs['covid'], col] = df.loc[~a.obs['covid'], col] * 0.8
    mean = (
        df.join(a.obs[["roi", "cell_type_label_3.5"]])
        .groupby(["roi", "cell_type_label_3.5"])
        .mean()
        .fillna(0)
    )

    label = "expression"
    for group in ["disease", "disease_subgroup"]:
        p = (
            mean.join(config.roi_attributes[group])
            .reset_index()
            .groupby([group, "cell_type_label_3.5"])
            .mean()
            .reset_index()
        )
        pp = p.pivot_table(index="cell_type_label_3.5", columns=group)

        pp = ((pp.T - pp.mean(1))).T

        grid = clustermap(
            pp,
            cbar_kws=dict(label=label),
            col_cluster=False,
            square=True,
            config="abs",
            vmin=0,
        )
        grid.savefig(
            output_dir
            / f"all_markers_positivity.by_{group}.{label}.clustermap.abs.svg",
            **config.figkws,
        )
        plt.close(grid.fig)

        grid = clustermap(
            pp + np.random.random(pp.shape) * 1e-10,
            cbar_kws=dict(label=label),
            col_cluster=False,
            square=True,
            config="z",
            vmin=-1,
            vmax=5,
        )
        grid.savefig(
            output_dir / f"all_markers_positivity.by_{group}.{label}.clustermap.z.svg",
            **config.figkws,
        )
        plt.close(grid.fig)


def cith3_ammount() -> AnnData:
    # Check extent of periostin
    from src.pathology import quantify_fibrosis

    col = "CitH3(Sm154)"
    res = quantify_fibrosis(col)
    res.to_csv(config.results_dir / "pathology" / f"{col}_score.csv")

    fig, stats = swarmboxenplot(
        data=res.join(config.roi_attributes), x="disease", y=res.columns
    )
    fig.savefig(
        config.results_dir / "pathology" / f"{col}_score.disease.svg", **config.figkws
    )
    fig, stats = swarmboxenplot(
        data=res.join(config.roi_attributes), x="disease_subgroup", y=res.columns
    )
    fig.savefig(
        config.results_dir / "pathology" / f"{col}_score.disease_subgroup.svg",
        **config.figkws,
    )

    res2 = res.loc[res.sort_values("score").index[:-1]]

    fig, stats = swarmboxenplot(
        data=res2.join(config.roi_attributes), x="disease", y=res2.columns
    )
    fig.savefig(
        config.results_dir / "pathology" / f"{col}_score.disease.no_out.svg",
        **config.figkws,
    )
    fig, stats = swarmboxenplot(
        data=res2.join(config.roi_attributes), x="disease_subgroup", y=res2.columns
    )
    fig.savefig(
        config.results_dir / "pathology" / f"{col}_score.disease_subgroup.no_out.svg",
        **config.figkws,
    )


def periostin_interactions() -> AnnData:
    a = sc.read(config.results_dir / "phenotyping" / "processed.labeled.h5ad")
    a.obs["covid"] = a.obs["disease"].isin(["COVID-19", "Mixed", "Convalescent"])

    # Check extent of periostin
    from src.pathology import quantify_fibrosis

    df = a.raw.to_adata().to_df()
    col = "Periostin(Dy161)"

    fib = quantify_fibrosis(col)
    fib.to_csv(config.results_dir / "pathology" / f"{col}_score.csv")
    fig, stats = swarmboxenplot(
        data=fib.join(config.roi_attributes), x="disease", y=fib.columns
    )
    fig.savefig(
        config.results_dir / "pathology" / f"{col}_score.disease.svg", **config.figkws
    )
    fig, stats = swarmboxenplot(
        data=fib.join(config.roi_attributes), x="disease_subgroup", y=fib.columns
    )
    fig.savefig(
        config.results_dir / "pathology" / f"{col}_score.disease_subgroup.svg",
        **config.figkws,
    )

    # Quantify cell type distance to periostin
    from imc.ops.quant import quantify_cell_intensity

    _dists = list()
    for roi in tqdm(prj.rois):
        x = np.log1p(roi._get_channel(marker)[1].squeeze())
        mask = skimage.filters.gaussian(x, 2) > skimage.filters.threshold_otsu(x)
        dist = scipy.ndimage.morphology.distance_transform_edt(
            (~mask).astype(int), sampling=[2, 2]
        )
        q = quantify_cell_intensity(dist[np.newaxis, ...], roi.mask).rename(
            columns={0: "distance"}
        )
        _dists.append(q.assign(sample=roi.sample.name, roi=roi.name))

        # fig, axes = plt.subplots(1, 3)
        # axes[0].imshow(mask)
        # axes[1].imshow(~mask)
        # axes[2].imshow(dist)

    dists = pd.concat(_dists)
    index = dists.index
    dists = dists.merge(config.roi_areas.reset_index())
    dists.index = index
    dists["distance_norm"] = (dists["distance"] / dists["area_mm2"]) * 1e6
    dists.to_csv(output_dir / f"{col}_cell_distance.csv")

    # plt.scatter(dists['distance'], dists['distance_norm'], alpha=0.1, s=2)

    a.obs[f"distance_to_{col}"] = dists["distance"].values
    a.obs[f"distance_to_{col}_norm"] = dists["distance_norm"].values
    diff_dists = a.obs.groupby(["disease", "cell_type_label_3.5"])[
        f"distance_to_{col}_norm"
    ].mean()

    (diff_dists["COVID-19"] - diff_dists["Normal"]).sort_values()
    (diff_dists["Convalescent"] - diff_dists["Normal"]).sort_values()
    (diff_dists["UIP/IPF"] - diff_dists["Normal"]).sort_values()

    dists_p = diff_dists.to_frame().pivot_table(
        index="cell_type_label_3.5", columns="disease", values=f"distance_to_{col}"
    )
    grid = clustermap(dists_p, cmap="RdBu_r", center=0)
    grid.savefig(
        output_dir / f"{col}_cell_distance.aggregated_cell_type.heatmap.svg",
        **config.figkws,
    )
    grid = clustermap(
        (dists_p.T - dists_p["Normal"]).T.drop(["Normal"], axis=1),
        cmap="RdBu_r",
        center=0,
    )
    grid.savefig(
        output_dir
        / f"{col}_cell_distance.aggregated_cell_type.relative_normal.heatmap.svg",
        **config.figkws,
    )

    # Now account for cell type composition of ROIs
    c = a.obs[
        ["roi", "disease_subgroup", "cell_type_label_3.5", f"distance_to_{col}"]
    ].copy()
    _res = list()
    for n in tqdm(range(100)):
        d = c.copy()
        for roi in c["roi"].unique():
            d.loc[d["roi"] == roi, "cell_type_label_3.5"] = (
                d.loc[d["roi"] == roi, "cell_type_label_3.5"].sample(frac=1).values
            )
        e = d.groupby(["disease_subgroup", "cell_type_label_3.5"])[
            f"distance_to_{col}"
        ].mean()
        _res.append(e)

    random_dists = pd.concat(_res).groupby(level=[0, 1]).mean()
    obs_dists = a.obs.groupby(["disease_subgroup", "cell_type_label_3.5"])[
        f"distance_to_{col}"
    ].mean()
    norm_dists = np.log(obs_dists / random_dists).rename(f"distance_to_{col}")

    norm_dists_agg = norm_dists.to_frame().pivot_table(
        index="cell_type_label_3.5",
        columns="disease_subgroup",
        values=f"distance_to_{col}",
    )

    dist_diff = (norm_dists_agg.T - norm_dists_agg["Normal"]).T.drop(["Normal"], axis=1)
    order = dist_diff.mean(1).sort_values().index

    grid = clustermap(
        norm_dists_agg.loc[order, :],
        figsize=(4, 5),
        center=0,
        row_cluster=False,
        col_cluster=False,
        cmap="PuOr_r",
        cbar_kws=dict(
            label="Relative distance to Periostin patch\nlog(observed / expected)"
        ),
    )
    grid.savefig(
        output_dir / f"{col}_cell_distance.aggregated_cell_type.norm.heatmap.svg",
        **config.figkws,
    )

    grid = clustermap(
        dist_diff.loc[order, :],
        figsize=(4, 5),
        cmap="RdBu_r",
        center=0,
        row_cluster=False,
        col_cluster=False,
        cbar_kws=dict(label="Difference to 'Normal'\nin distance to Periostin patch"),
    )
    grid.savefig(
        output_dir
        / f"{col}_cell_distance.aggregated_cell_type.norm.relative_normal.heatmap.svg",
        **config.figkws,
    )


def increased_diversity(
    prefix: str, cois: list[str], resolution: float, assemble_figure: bool = True
) -> AnnData:
    a = sc.read(config.results_dir / "phenotyping" / "processed.labeled.h5ad")
    a.obs["covid"] = a.obs["disease"].isin(["COVID-19", "Mixed", "Convalescent"])

    output_dir = config.results_dir / "phenotyping"
    cell_type_label = "cell_type_label_3.5"
    cluster = f"{prefix}_{resolution}"
    label = f"{prefix}_diversity.leiden_{str(resolution).replace('.', '')}"

    a = a[a.obs[cell_type_label].isin(cois), :]

    fig = sc.pl.umap(a, color=[cell_type_label, "cluster_3.5"], show=False, ncols=1)[
        0
    ].figure
    fig.suptitle(f"Original UMAP, {int(a.shape[0])} cells")
    rasterize_scanpy(fig)
    fig.savefig(output_dir / f"{label}.1-umap_1_original.svg", **config.figkws)

    new_h5ad_f = output_dir / f"{label}.h5ad"
    if not new_h5ad_f.exists():
        sc.tl.umap(a)
        sc.tl.leiden(a, resolution=resolution, key_added=cluster)
        a.obs[cluster] = f"{prefix}_cluster_" + (a.obs[cluster].astype(int) + 1).astype(
            str
        )
        a.write(new_h5ad_f)
    a = sc.read(new_h5ad_f)

    fig = sc.pl.umap(a, color=[cell_type_label, cluster], show=False, ncols=1)[0].figure
    fig.suptitle(f"New UMAP, {int(a.shape[0])} cells")
    rasterize_scanpy(fig)
    fig.savefig(output_dir / f"{label}.1-umap_2_new.svg", **config.figkws)

    cats = a.obs[cluster].cat.categories[a.obs[cluster].value_counts() >= 100]
    a = a[a.obs[cluster].isin(cats), :]

    cmean = (
        a.to_df()[[x for x in config.channels_include if x in a.var.index]]
        .groupby(a.obs[cluster])
        .mean()
    )

    origc = a.obs[cluster].groupby(a.obs[cell_type_label]).value_counts()
    s = a.obs[cluster].groupby(a.obs[cell_type_label]).size()
    origc = (origc / s) * 100
    cts = origc.reset_index().pivot_table(
        index="level_1", columns=cell_type_label, values=cluster
    )
    cts = (cts.T / cts.sum(1)).T

    top = a.obs[cluster].groupby(a.obs["topological_domain"]).value_counts()
    s = a.obs[cluster].groupby(a.obs["topological_domain"]).size()
    top = (top / s) * 100
    top = top.reset_index().pivot_table(
        index="level_1", columns="topological_domain", values=cluster
    )
    top = (top.T / top.sum(1)).T
    top.columns = top.columns.to_series().replace(config.topo_labels)
    top = top[config.topo_labels.values()]

    cov = a.obs[cluster].groupby(a.obs["covid"]).value_counts()
    s = a.obs[cluster].groupby(a.obs["covid"]).size()
    cov = (cov / s) * 100
    cov = cov.reset_index().pivot_table(
        index="level_1", columns="covid", values=cluster
    )
    cov = np.log(cov[True] / cov[False]).rename("log(COVID / non-COVID)")

    abu = np.log10(a.obs[cluster].value_counts().to_frame("abundance"))

    grid = clustermap(
        cmean,
        cmap="RdBu_r",
        center=0,
        row_colors=abu.join(cts).join(cov),
        config="abs",
        cbar_kws=dict(label="Marker intensity"),
        metric="euclidean",
    )
    # grid.ax_heatmap.set_yticklabels(grid.ax_heatmap.get_yticklabels(), rotation=90)
    grid.savefig(output_dir / f"{label}.2-clustermap.svg", **config.figkws)
    grid2 = clustermap(
        top,
        row_linkage=grid.dendrogram_row.linkage,
        col_cluster=False,
        config="abs",
        cbar_kws=dict(label="Fraction in compartment"),
        figsize=(2, 4),
        cmap="inferno",
        square=True,
    )
    # grid2.ax_heatmap.set_yticklabels(grid2.ax_heatmap.get_yticklabels(), rotation=90)
    grid2.savefig(
        output_dir / f"{label}.3-clustermap.topo_domains.svg",
        **config.figkws,
    )

    cov = cov.iloc[grid.dendrogram_row.reordered_ind[::-1]]

    fig, ax = plt.subplots(figsize=(2, 8))
    ax.scatter(cov, cov.index)
    ax.axvline(0, linestyle="--", color="grey")
    ax.set(xlabel="log(COVID / non-COVID)", title="Ordered as heatmap")
    fig.savefig(
        output_dir / f"{label}.4-log_foldchange.ordered.svg",
        **config.figkws,
    )

    cov = cov.sort_values()
    fig, ax = plt.subplots(figsize=(2, 8))
    ax.scatter(cov, cov.index)
    ax.axvline(0, linestyle="--", color="grey")
    ax.set(xlabel="log(COVID / non-COVID)", title="Sorted")
    fig.savefig(
        output_dir / f"{label}.5-log_foldchange.sorted.svg",
        **config.figkws,
    )

    counts = (
        a.obs.groupby("roi")[cluster]
        .value_counts()
        .reset_index()
        .pivot_table(index="roi", columns="level_1", values=cluster)
        .rename_axis(columns=cluster)
    )
    counts_mm2 = (counts.T / config.roi_areas).T * 1e6
    fig, stats = swarmboxenplot(
        data=counts_mm2.join(config.roi_attributes),
        x="disease_subgroup",
        y=counts.columns,
        plot_kws=dict(palette=config.colors["disease_subgroup"]),
    )
    fig.savefig(
        output_dir / f"{label}.6-swarmboxenplot.svg",
        **config.figkws,
    )

    # Assemble sharable figure
    ## Dependencies: inkscape, https://github.com/astraw/svg_stack
    if assemble_figure:
        cmd = f"svg_stack.py --direction=h --margin=20 {output_dir}/{label}*.svg > {output_dir}/{label}.all.svg"
        os.system(cmd)
        cmd = f"inkscape --export-background=white -d 300 -o {output_dir}/{label}.all.png {output_dir}/{label}.all.svg"
        os.system(cmd)
        cmd = f"inkscape --export-background=white -d 300 -o {output_dir}/{label}.all.pdf {output_dir}/{label}.all.svg"
        os.system(cmd)


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

        # # Plot as difference to baseline
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

        stats = pd.read_csv(output_dir / f"cellular_interactions.per_{attr}.csv")

        # # Volcano plots
        base = config.attribute_order[attr][0]

        res = stats.query(f"`A` == '{base}'")
        fig = volcano_plot(
            stats=res,
            n_top=15,
            diff_threshold=None,
            fig_kws=dict(gridspec_kw=dict(wspace=3, hspace=1), figsize=(18, 9)),
        )
        fig.savefig(
            output_dir / f"cellular_interactions.per_{attr}.volcano_plot.svg",
            **config.figkws,
        )
        plt.close(fig)

        if attr == "disease":
            res = stats.query(f"`A` == 'COVID-19' & B == 'Convalescent'")
            fig = volcano_plot(
                stats=res,
                n_top=15,
                diff_threshold=None,
                fig_kws=dict(gridspec_kw=dict(wspace=3, hspace=1)),
            )
            fig.savefig(
                output_dir
                / f"cellular_interactions.per_{attr}.PASC-COVID.volcano_plot.svg",
                **config.figkws,
            )
            plt.close(fig)

        if attr == "disease_subgroup":
            res = stats.query(f"`A` == 'COVID-19-long-pos' & B == 'COVID-19-long-neg'")
            fig = volcano_plot(
                stats=res,
                n_top=15,
                diff_threshold=None,
                fig_kws=dict(gridspec_kw=dict(wspace=3, hspace=1)),
            )
            fig.savefig(
                output_dir
                / f"cellular_interactions.per_{attr}.PASC-neg-pos.volcano_plot.svg",
                **config.figkws,
            )
            plt.close(fig)

    # Examples
    ct1 = "Airway wall vascular endothelial"
    ct2 = "Alveolar type 2"
    a = adjs.query(f"disease == 'COVID-19' & index == '{ct1}' & variable == '{ct2}'")
    b = adjs.query(
        f"disease == 'Convalescent' & index == '{ct1}' & variable == '{ct2}'"
    )

    a.groupby("sample")["value"].mean()
    b.groupby("sample")["value"].mean()

    roi_name = b.sort_values("value").iloc[-1]["roi"]
    r = [r for r in prj.rois if r.name == roi_name][0]
    c = r.clusters.drop_duplicates().reset_index(drop=True)
    c.index += 1
    to_rep = {v: str(k).zfill(2) + " - " + v for k, v in c.items()}

    fig = plot_cell_types(r, cell_type_assignments=r.clusters.replace(to_rep))
    fig = r.plot_cell_type(ct1)
    fig = r.plot_cell_type(ct2)

    roi_name = "A21_63_A14-05"
    r = [r for r in prj.rois if r.name == roi_name][0]

    target_colors = sns.color_palette("tab10")
    rgb = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    fig = r.plot_channels(
        ["CD31", "AQ1", "aSMA"],
        merged=True,
        smooth=0.5,
        log=False,
        minmax=True,
        equalize=True,
        target_colors=rgb,
    )  # , target_colors=target_colors[:3])
    fig = r.plot_channels(
        ["K818", "SFTPC", "SFTPA"],
        merged=True,
        smooth=0.5,
        log=False,
        minmax=True,
        equalize=True,
        target_colors=rgb,
    )  # , target_colors=target_colors[3:6])
    fig = r.plot_channels(
        ["ColTypeI", "Periostin", "Vimentin"],
        merged=True,
        smooth=0.5,
        log=False,
        minmax=True,
        equalize=True,
        target_colors=rgb,
    )  # , target_colors=target_colors[6:9])
    fig = r.plot_channels(
        ["IL6", "pNFkbp65", "pSTAT3Tyr705"],
        merged=True,
        smooth=0.5,
        log=False,
        minmax=True,
        equalize=True,
        target_colors=rgb,
    )  # , target_colors=target_colors[6:9])

    fig = r.plot_channels(
        r.channel_include, smooth=0.5, log=False, minmax=True, equalize=True
    )
    fig.savefig(
        output_dir / roi_name + ".all_channels.illustration.pdf", **config.figkws
    )

    #

    #

    # Dimres based on interactions
    f = prj.results_dir / "single_cell" / prj.name + ".adjacency_frequencies.csv"
    adjs = pd.read_csv(f, index_col=0)
    adjs = adjs.loc[adjs["index"] != adjs["variable"]]

    # adjs["interaction2"] = adjs["variable"] + " <-> " + adjs["index"]
    # adjs = adjs.loc[adjs['interaction'] != adjs['interaction2']]

    for i, row in tqdm(adjs.iterrows(), total=adjs.shape[0]):
        adjs.loc[i, "comb"] = "-".join(sorted([row["index"], row["variable"]]))
    adjs = adjs.drop_duplicates(subset=["comb", "value"])
    adjs["interaction"] = adjs["index"] + " <-> " + adjs["variable"]

    piv = adjs.pivot_table(index="roi", columns="interaction", values="value")
    piv = piv.loc[:, ~piv.isnull().any()]

    grid = clustermap(piv, config="z", row_colors=config.roi_attributes)
    grid = clustermap(
        piv,
        config="abs",
        center=0,
        cmap="RdBu_r",
        robust=False,
        row_colors=config.roi_attributes,
    )

    piv2 = piv.copy()
    piv2[piv2 < 0] = 0
    # piv2[piv2 < 2] = 0
    # piv2[piv2 >= 2] = 1
    piv2 = piv2.loc[:, piv2.var() > 0]
    grid = clustermap(
        np.log1p(piv2),
        metric="correlation",
        row_colors=config.roi_attributes,
        figsize=(6, 6),
        dendrogram_ratio=0.1,
        xticklabels=False,
    )
    grid.ax_heatmap.set(rasterized=True)
    grid.fig.savefig(
        output_dir / f"cellular_interactions.all_rois.clustermap.svg",
        **config.figkws,
    )
    plt.close(grid.fig)

    ai = AnnData(piv2.values, obs=config.roi_attributes, var=piv2.columns.to_frame())
    ai = ai[~ai.obs["disease"].isin(["Mixed"]), :]
    # ai = ai[ai.obs['disease'].isin(['Normal', 'UIP/IPF']), :]
    # res = stats.query(f"`A` == 'Normal' & B == 'UIP/IPF'").sort_values('hedges').set_index("Variable")
    # sel = res.head(10).index.tolist() + res.tail(10).index.tolist()
    # ai = ai[:, ai.var.index.isin(sel)]
    sc.pp.log1p(ai)
    sc.pp.highly_variable_genes(ai)
    fig = sc.pl.highly_variable_genes(ai, show=False).figure
    fig.savefig(
        output_dir / f"cellular_interactions.highly_variable.svg", **config.figkws
    )
    plt.close(fig)

    # sc.pp.scale(ai)
    sc.pp.pca(ai)

    ai.uns["pca"]["variance_ratio"]

    sc.pp.neighbors(ai)
    sc.tl.umap(ai, gamma=3)
    sc.tl.diffmap(ai)

    _ai = ai[ai.obs.sample(frac=1).index, :]
    fig = sc.pl.pca(
        _ai,
        alpha=0.8,
        color=config.attributes,
        components=["1,2", "1,3", "2,3", "2,6"],
        ncols=4,
        show=False,
    )[0].figure
    rasterize_scanpy(fig)
    fig.savefig(output_dir / f"cellular_interactions.pca.svg", **config.figkws)
    plt.close(fig)
    fig = sc.pl.umap(_ai, alpha=0.8, color=config.attributes, show=False)[0].figure
    rasterize_scanpy(fig)
    fig.savefig(output_dir / f"cellular_interactions.umap.svg", **config.figkws)
    plt.close(fig)
    fig = sc.pl.diffmap(_ai, alpha=0.8, color=config.attributes, show=False)[0].figure
    rasterize_scanpy(fig)
    fig.savefig(output_dir / f"cellular_interactions.diffmap.svg", **config.figkws)
    plt.close(fig)

    pca = pd.DataFrame(ai.obsm["X_pca"], index=ai.obs.index)
    pcag = pca.groupby(ai.obs["disease_subgroup"]).mean()

    (pcag.iloc[-1] - pcag.iloc[-2]).sort_values()

    pcs = pd.DataFrame(ai.varm["PCs"], index=ai.var.index)
    pcs.sort_values(2)
    pcs.sort_values(3)

    d = (
        adjs.query(
            "index == 'CD16+ inflammatory monocytes' & variable == 'Airway wall vascular endothelial'"
        )
        .drop(["index", "variable", "interaction", "comb", "sample"], axis=1)
        .set_index("roi")
    )
    fig, stats = swarmboxenplot(
        data=d.join(config.roi_attributes), x="disease_subgroup", y="value"
    )

    d = (
        adjs.query("index == 'Vascular endothelial' & variable == 'Macrophages'")
        .drop(["index", "variable", "interaction", "comb", "sample"], axis=1)
        .set_index("roi")
    )
    fig, stats = swarmboxenplot(
        data=d.join(config.roi_attributes), x="disease_subgroup", y="value"
    )


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


def characterize_microanatomical_context():
    a = sc.read(config.results_dir / "phenotyping" / "processed.labeled.h5ad")

    c = (
        a.obs.groupby("topological_domain")[["cell_type_label_3.5"]]
        .value_counts()
        .rename("count")
    )
    c = c.reset_index().pivot_table(
        index="cell_type_label_3.5", columns="topological_domain", values="count"
    )

    order = [
        "L-A",
        "A",
        "A-AW",
        "AW",
        # "background",
        "AW-M",
        "M",
        "AR-M",
        "AR",
        "AR-V",
        "V",
    ]
    fig, axes = plt.subplots(
        2,
        5,
        figsize=(5 * 3, 2 * 3),
        sharey=True,
        sharex=True,
        gridspec_kw=dict(wspace=0, hspace=0),
    )
    for ax, top in zip(fig.axes, order):
        sns.barplot(
            x=c[top] / c[top].sum() * 100,
            y=c.index,
            orient="horiz",
            ax=ax,
            palette="rainbow",
        )
        ax.set(xlabel="", ylabel="")
        ax.set_title(label=top, loc="center", y=0.85)
    fig.axes[7].set_xlabel("% cells")

    grid = clustermap(c)

    c = (
        a.obs.groupby(["roi", "topological_domain"])[["cell_type_label_3.5"]]
        .value_counts()
        .rename("count")
    )
    cp = (c / c.groupby(level=(0, 1)).sum()).fillna(0) * 100

    q = cp.reset_index().set_index("roi").join(config.roi_attributes)

    m = q.groupby(["disease_subgroup", "topological_domain", "cell_type_label_3.5"])[
        "count"
    ].mean()
    m = m.reset_index().pivot_table(
        index=["topological_domain", "disease_subgroup"],
        columns="cell_type_label_3.5",
        values="count",
    )
    grid = clustermap(np.log1p(m), row_cluster=False)

    q["label"] = (
        q["topological_domain"].astype(str)
        + " - "
        + q["cell_type_label_3.5"].astype(str)
    )
    for top in order:
        qq = q.query(f"topological_domain == '{top}'")
        fig, stats = swarmboxenplot(
            data=qq, x="cell_type_label_3.5", y="count", hue="disease_subgroup"
        )


def unsupervised(
    grouping: str = "roi",
    resolution: float = 3.5,
    scale: bool = True,
    regress_out: bool = False,
    corrmaps: bool = True,
    add_expression: bool = True,
    expression_weight: float = 1.0,
    prefix: str = "",
):
    from sklearn.decomposition import PCA
    from sklearn.manifold import SpectralEmbedding, Isomap, MDS

    output_dir = (config.results_dir / "unsupervised").mkdir()
    if prefix != "":
        if not prefix.endswith("."):
            prefix += "."

    a = sc.read(config.results_dir / "phenotyping" / "processed.labeled.h5ad")

    areas = getattr(config, "roi" + "_areas")

    meta = pd.read_csv(config.metadata_dir / "samples.csv", index_col=0)
    cols = ["days_since_first_infection", "days_since_positive"]
    meta = meta[cols]

    c = (
        a.obs.groupby(["roi"])[[f"cell_type_label_{resolution}"]]
        .value_counts()
        .rename("count")
    )
    c = c.reset_index().pivot_table(
        index="roi", columns=f"cell_type_label_{resolution}", values="count"
    )
    cp = (c.T / c.sum(1)).T * 100
    ca = (c.T / areas).T * 1e6

    # stats = pd.concat([ca.mean(), ca.std()], axis=1)
    # stats.columns = ["mean", "std"]
    # stats["cv"] = stats["std"] / stats["mean"]

    # # fig, axes = plt.subplots(2, 1)
    # # axes[0].scatter(stats["mean"], stats["std"])
    # # axes[1].scatter(stats["mean"], stats["cv"])
    # # for s in stats.index:
    # #     axes[0].text(stats.loc[s, "mean"], stats.loc[s, "std"], s=s)
    # #     axes[1].text(stats.loc[s, "mean"], stats.loc[s, "cv"], s=s)

    # sel = stats.loc[stats["cv"] < (1.5 if 'roi' == "roi" else 1)].index

    # cp = cp.loc[:, sel]
    # ca = ca.loc[:, sel]

    if add_expression:
        perc = pd.read_csv(
            config.results_dir
            / "positivity"
            / "positivity_per_cell_type.percentage.csv",
            index_col=0,
        )
        mm2 = pd.read_csv(
            config.results_dir / "positivity" / "positivity_per_cell_type.area.csv",
            index_col=0,
        )

        perc = (
            perc.reset_index()
            .melt(id_vars=["roi", "cell_type_label_3.5"])
            .pivot_table(
                index="roi", columns=["cell_type_label_3.5", "variable"], values="value"
            )
        )
        perc.columns = perc.columns.map(
            lambda x: x if isinstance(x, str) else " - ".join(x)
        )
        mm2 = (
            mm2.reset_index()
            .melt(id_vars=["roi", "cell_type_label_3.5"])
            .pivot_table(
                index="roi", columns=["cell_type_label_3.5", "variable"], values="value"
            )
        )
        mm2.columns = mm2.columns.map(
            lambda x: x if isinstance(x, str) else " - ".join(x)
        )

        cp = cp.join(perc * expression_weight)
        ca = ca.join(mm2 * expression_weight)

    if scale:
        cp = (cp - cp.mean()) / cp.std()
        ca = (ca - ca.mean()) / ca.std()

    if regress_out:
        ap_perc = pd.read_csv(
            config.results_dir / "domains" / "domain_distribution.per_roi.perc.csv",
            index_col=0,
        )
        ap_mm2 = pd.read_csv(
            config.results_dir / "domains" / "domain_distribution.per_roi.mm2.csv",
            index_col=0,
        )
        an = AnnData(cp, obs=ap_perc.reindex(cp.index))
        sc.pp.scale(an)
        sc.pp.regress_out(an, ap_perc.columns.tolist(), n_jobs=12)

        cp = pd.DataFrame(an.X, index=cp.index, columns=cp.columns)

        an = AnnData(cp, obs=ap_mm2.reindex(cp.index))
        sc.pp.scale(an)
        sc.pp.regress_out(an, ap_mm2.columns.tolist(), n_jobs=12)

        ca = pd.DataFrame(an.X, index=ca.index, columns=ca.columns)

    cf = ca.join(
        cp.rename(columns=dict(zip(cp.columns, cp.columns.astype(str) + "___")))
    )

    if grouping == "sample":
        attrs = getattr(config, "roi_attributes")
        cp = cp.join(attrs["sample"]).groupby("sample").mean()
        ca = ca.join(attrs["sample"]).groupby("sample").mean()
        cf = cf.join(attrs["sample"]).groupby("sample").mean()

    attrs = getattr(config, grouping + "_attributes")
    attrs = attrs.merge(meta, how="left", left_on="sample", right_index=True)

    gcp = cp.join(attrs["disease_subgroup"]).groupby("disease_subgroup").mean().T
    gca = ca.join(attrs["disease_subgroup"]).groupby("disease_subgroup").mean().T
    gcf = cf.join(attrs["disease_subgroup"]).groupby("disease_subgroup").mean().T

    # gcp = gcp.loc[sel, :]
    # gca = gca.loc[sel, :]
    # gcf = gcf.loc[sel, :]

    gca = gca.drop("Mixed", axis=1)
    gcp = gcp.drop("Mixed", axis=1)
    gcf = gcf.drop("Mixed", axis=1)

    ca = ca.loc[attrs["disease"] != "Mixed"]
    cp = cp.loc[attrs["disease"] != "Mixed"]
    cf = cf.loc[attrs["disease"] != "Mixed"]

    if corrmaps:
        for df1, df2, label in [
            (cp, gcp, "percentage"),
            (ca, gca, "area"),
            (cf, gcf, "combined"),
        ]:
            fig, ax = plt.subplots()
            sns.heatmap(
                df2.corr(),
                cmap="coolwarm",
                vmin=-1,
                vmax=1,
                ax=ax,
            )
            fig.savefig(
                output_dir
                / f"{prefix}corrmap.{resolution}.per_{grouping}.by_disease_subgroup.{label}.heatmap.svg",
                **config.figkws,
            )

            grid = clustermap(
                df2.corr(),
                cmap="coolwarm",
                center=0,
                dendrogram_ratio=0.1,
            )
            grid.fig.savefig(
                output_dir
                / f"{prefix}corrmap.{resolution}.per_{grouping}.by_disease_subgroup.{label}.clustermap.svg",
                **config.figkws,
            )

            grid = clustermap(
                df1.T.corr(),
                cmap="coolwarm",
                center=0,
                row_colors=attrs,
                rasterized=False,
                dendrogram_ratio=0.1,
            )
            grid.ax_heatmap.get_children()[0].set(rasterized=True)
            grid.fig.savefig(
                output_dir
                / f"{prefix}corrmap.{resolution}.per_{grouping}.{label}.clustermap.svg",
                **config.figkws,
            )

    for algo, name in [
        (PCA, "PCA"),
        (SpectralEmbedding, "SpectralEmbedding"),
        (Isomap, "Isomap"),
        (MDS, "MDS"),
    ]:
        for df, label in [
            (cp, "percentage"),
            (ca, "area"),
            (cf, "combined"),
        ]:

            # fig = _plot_lat(
            #     df,
            #     algo,
            #     attributes=["disease_subgroup", "days_since_first_infection"],
            #     attr_df=attrs,
            # )
            # fig.savefig(
            #     output_dir
            #     / f"{prefix}{name}.{resolution}.{label}.per_{grouping}.by_disease_subgroup.svg",
            #     **config.figkws,
            # )

            df = df.join(attrs).query("disease != 'Mixed'")[df.columns]
            fig = _plot_lat(
                df,
                algo,
                attributes=["disease_subgroup", "days_since_first_infection"],
                attr_df=attrs,
            )
            fig.savefig(
                output_dir
                / f"{prefix}{name}.{resolution}.{label}.per_{grouping}.by_disease_subgroup.no_Mixed.svg",
                **config.figkws,
            )

            # df = df.join(attrs).query("disease != 'UIP/IPF' & disease != 'Mixed'")[
            #     df.columns
            # ]
            # fig = _plot_lat(
            #     df,
            #     algo,
            #     attributes=["disease_subgroup", "days_since_first_infection"],
            #     attr_df=attrs,
            # )
            # fig.savefig(
            #     output_dir
            #     / f"{prefix}{name}.{resolution}.{label}.per_{grouping}.by_disease_subgroup.no_IPF_no_Mixed.svg",
            #     **config.figkws,
            # )
            plt.close("all")


import typing as tp
from src.types import DataFrame


def _plot_lat(
    df: DataFrame, algo: tp.Callable, attributes: list[str], attr_df: DataFrame
):
    from imc.utils import is_numeric
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    model = algo()
    rep = pd.DataFrame(model.fit_transform(df), index=df.index).join(attr_df)

    n = len(attributes)
    fig, axes = plt.subplots(1, n, figsize=(4 * n * 1.1, 4), sharex=True, sharey=True)
    for ax, attr in zip(fig.axes, attributes):
        if not is_numeric(rep[attr]):
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

        else:
            x = rep.dropna(subset=[attr])
            s = ax.scatter(x[0], x[1], c=x[attr], alpha=0.85)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(s, cax=cax, orientation="vertical", label=attr)

        # ax.set(xlabel=f"{name}1", ylabel=f"{name}2")
        ax.legend(bbox_to_anchor=(0, -0.1), loc="upper left")
    return fig


if __name__ == "__main__" and "get_ipython" not in locals():
    import sys

    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit()
