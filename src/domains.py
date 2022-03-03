#!/usr/bin/env python

"""
Annotation, quantification and interpretation of lung tissue at the topological domain level.
"""

import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from seaborn_extensions import clustermap, swarmboxenplot

from imc.types import Path, Array
from imc.graphics import close_plots
from imc.ops.domain import (
    label_domains,
    collect_domains,
    illustrate_domains,
    get_domain_areas,
    get_domains_per_cell,
    get_domain_mask,
)

from src._config import prj, config


output_dir = (config.results_dir / "domains").mkdir()


def main() -> int:

    # Get domains
    topo_annots, topo_sc = get_topological_domain_annotation()

    # Illustrate
    # # simplify vessels
    to_repl = {"Vl": "V", "Vs": "V"}
    for roi, shapes in topo_annots.items():
        for shape in shapes:
            shape["label"] = to_repl.get(shape["label"], shape["label"])
    illustrate_domains(
        topo_annots,
        prj.rois,
        output_dir=output_dir / "illustration",
        channels=["K818", "CC16", "ColTypeI", "aSMA", "CD31"],
        cleanup=True,
    )

    # Stats
    # # Area
    areas = get_domain_areas(topo_annots, per_domain=True)
    areas["topological_domains"] = areas["topological_domain"]
    areas["topological_domain"] = (
        areas["topological_domain"]
        .str.replace("Vl", "V")
        .str.replace("Vs", "V")
        .str.split("-")
        .apply(set)
        .apply(lambda x: "-".join(x))
    )

    ap = areas.pivot_table(
        index="roi", columns="topological_domain", values="area", fill_value=0
    )

    ap = (ap.T / config.roi_areas).T * 100

    grid = clustermap(
        ap,
        config="abs",
        row_colors=config.roi_attributes[config.colors.keys()],
        row_colors_cmaps=config.colors.values(),
        square=False,
    )
    grid.fig.savefig(
        output_dir / "domain_distribution.clustermap.abs.svg", **config.figkws
    )
    grid = clustermap(
        ap,
        config="z",
        row_colors=config.roi_attributes[config.colors.keys()],
        row_colors_cmaps=config.colors.values(),
        square=False,
    )
    grid.fig.savefig(
        output_dir / "domain_distribution.clustermap.z.svg", **config.figkws
    )

    roi = prj.rois[0]

    get_domain_mask(topo_annots[roi.name], roi, per_domain=True)

    # # Abundance

    topo_sc["roi"] = list(
        map((lambda x: "-".join(x[:-1])), topo_sc.index.str.split("-"))
    )
    cu = topo_sc.groupby("roi")["domain_id"].nunique()
    cun = ((cu / config.roi_areas) * 1e6).rename("domain_id")

    fig, stats = swarmboxenplot(
        config.roi_areas.to_frame().join(config.roi_attributes),
        x="disease_subgroup",
        y="area_mm2",
    )
    fig.savefig(output_dir / "roi_areas.svg", **config.figkws)

    # # Distribution across sample attributes
    fig, stats = swarmboxenplot(
        cu.to_frame().join(config.roi_attributes), x="disease_subgroup", y="domain_id"
    )
    fig.savefig(output_dir / "domain_count.raw.svg", **config.figkws)
    fig, stats = swarmboxenplot(
        cun.to_frame().join(config.roi_attributes), x="disease_subgroup", y="domain_id"
    )
    fig.savefig(output_dir / "domain_count.per_mm2.svg", **config.figkws)

    # # Intensity per marker

    # Interactions

    # Tissue architechture reconstruction

    return 0


def get_topological_domain_annotation():
    labeling_dir = output_dir / "manual"

    polygon_f = config.metadata_dir / "topological_domains.json"
    domain_f = config.metadata_dir / "topological_domain_assignment.csv"

    if not domain_f.exists():

        label_domains(
            prj.rois,
            output_dir=labeling_dir,
            channels=["AQ1", "ColTypeI", "aSMA"],
        )

        #
        topo_annots = collect_domains(labeling_dir)
        json.dump(topo_annots, polygon_f.open("w"))

        #
        topo_sc = get_domains_per_cell(topo_annots, prj.rois)
        topo_sc.index = (
            topo_sc.index.get_level_values(1)
            + "-"
            + topo_sc.index.get_level_values(2).astype(str).str.zfill(4)
        )
        topo_sc = topo_sc[["domain_id", "topological_domain"]]
        topo_sc.to_csv(domain_f.replace_(".csv", ".original.csv"))

        # simplify domains and remove some redundancies
        topo_sc["topological_domains"] = topo_sc["topological_domain"]
        topo_sc["topological_domain"] = (
            topo_sc["topological_domain"]
            .str.replace("Vl", "V")
            .str.replace("Vs", "V")
            .str.split("-")
            .apply(set)
            .apply(lambda x: "-".join(x))
        )

        c = topo_sc["topological_domain"].value_counts()
        keep = c[c >= 100].index
        topo_sc.loc[
            ~topo_sc["topological_domain"].isin(keep), "topological_domain"
        ] = np.nan
        topo_sc.loc[topo_sc["topological_domain"] == "", "topological_domain"] = np.nan
        topo_sc.drop("topological_domains", axis=1).rename_axis(index="obj_id").to_csv(
            domain_f
        )

    topo_annots = json.load(polygon_f.open("r"))
    topo_sc = pd.read_csv(domain_f, index_col=0).replace({"": np.nan})
    return topo_annots, topo_sc
    # topo_sc["topological_domain"].value_counts()


def quantify_domains(topo_annots, rois):
    import parmap

    labels = list(set(geom["label"] for n, j in topo_annots.items() for geom in j))
    label_order = dict(zip(labels, range(len(labels))))

    _quant = parmap.map(
        _quantify_domains_roi,
        prj.rois,
        topo_annots=topo_annots,
        label_order=label_order,
        pm_pbar=True,
    )
    quant = pd.concat([y for x in _quant for y in x])
    quant.to_csv(output_dir / "domain_expression.csv")

    quant["topological_domain"] = (
        quant.index.to_series()
        .str.replace(r"\d+", "", regex=True)
        .replace({"Vs": "V", "Vl": "V"})
    )

    mean_s = (
        quant.groupby(["roi", "topological_domain"]).mean().join(config.roi_attributes)
    )

    for dom in mean_s.index.levels[1]:
        fig, stats = swarmboxenplot(
            data=mean_s.loc[:, dom, :], x="disease_subgroup", y=config.channels_include
        )
        fig.savefig(output_dir / f"domain_expression.{dom}.swarmboxenmplot.svg")

    mean_t = quant.groupby(["topological_domain"]).mean()[config.channels_include]
    grid = clustermap((mean_t - mean_t.mean()) / mean_t.std(), config="abs")


def _quantify_domains_roi(roi, topo_annots, label_order):
    import scipy
    from imc.ops.quant import quantify_cell_intensity

    str_arr = get_domain_mask(topo_annots[roi.name], roi, per_domain=True)
    mask_stack = str_array_to_multi_bool(str_arr, label_order)
    stack = roi.stack

    _quant = list()
    for dom, order in label_order.items():
        if mask_stack[order].sum() == 0:
            continue

        dom_mask = scipy.ndimage.label(mask_stack[order])[0]
        q = quantify_cell_intensity(stack, dom_mask, border_objs=True)
        q.index = dom + q.index.astype(str)
        q.columns = roi.channel_labels
        _quant.append(q.assign(roi=roi.name))
    return _quant


def str_array_to_multi_bool(str_arr: Array, label_order: dict[str, int]):
    output = np.zeros((len(label_order),) + str_arr.shape, dtype="uint8")
    for dom, order in label_order.items():
        output[order, ...] = np.asarray(str_arr == dom).astype(int)
    return output


if __name__ == "__main__" and "get_ipython" not in locals():
    import sys

    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit()
