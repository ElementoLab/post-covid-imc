#!/usr/bin/env python

"""
A module to provide the boilerplate needed for all the analysis.
"""

import typing as tp

import numpy as np  # type: ignore[import]
import pandas as pd  # type: ignore[import]
import matplotlib  # type: ignore[import]
import matplotlib.pyplot as plt  # type: ignore[import]
import seaborn as sns  # type: ignore[import]

from imc import Project  # type: ignore[import]
from imc.types import Path, DataFrame  # type: ignore[import]


class config:
    # constants
    channels_exclude_strings: tp.Final[tp.List[str]] = [
        "<EMPTY>",
        "_EMPTY_",
        "190BCKG",
        "80ArAr",
        "129Xe",
    ]
    roi_exclude_strings: tp.Final[tp.List[str]] = [
        "A20_105_A25-06",
        "A19_15_A24-01",
        "A19_15_A24-03",
        "A19_15_A24-04",
        "A19_15_A24-05",
    ]

    channels_exclude: tp.List[str]
    channels_include: tp.List[str]

    ## Major attributes to contrast when comparing observation groups
    attributes: tp.Final[tp.List[str]] = ["disease"]
    attribute_order: tp.Dict[str, tp.List[str]] = dict(
        disease=["Normal", "Mixed", "UIP/IPF", "COVID-19", "Convalescent"]
    )
    roi_attributes: DataFrame
    sample_attributes: DataFrame

    figkws: tp.Final[tp.Dict] = dict(
        dpi=300, bbox_inches="tight", pad_inches=0, transparent=False
    )

    # directories
    metadata_dir: tp.Final[Path] = Path("metadata")
    data_dir: tp.Final[Path] = Path("data")
    processed_dir: tp.Final[Path] = Path("processed")
    results_dir: tp.Final[Path] = Path("results")

    # Color codes
    colors: tp.Final[tp.Dict[str, np.ndarray]] = dict(
        cat1=np.asarray(sns.color_palette())[[2, 0, 1, 3]],
        cat2=np.asarray(sns.color_palette())[[2, 0, 1, 5, 4, 3]],
    )

    # Output files
    metadata_file: tp.Final[Path] = metadata_dir / "clinical_annotation.pq"
    quantification_file: tp.Final[Path] = results_dir / "cell_type" / "quantification.pq"
    gating_file: tp.Final[Path] = results_dir / "cell_type" / "gating.pq"
    h5ad_file: tp.Final[Path] = (
        results_dir / "cell_type" / "anndata.all_cells.processed.h5ad"
    )
    counts_file: tp.Final[Path] = results_dir / "cell_type" / "cell_type_counts.pq"
    roi_areas_file: tp.Final[Path] = results_dir / "roi_areas.csv"
    sample_areas_file: tp.Final[Path] = results_dir / "sample_areas.csv"


# Initialize project
prj = Project(metadata=config.metadata_dir / "samples.csv", name="covid-pasc-imc")

# Filter channels and ROIs
channels = prj.channel_labels.stack().drop_duplicates().reset_index(level=1, drop=True)
config.channels_exclude = channels.loc[
    channels.str.contains(r"^\d")
    | channels.str.contains("|".join(config.channels_exclude_strings))
].tolist()
config.channels_include = channels[~channels.isin(config.channels_exclude)]

for roi in prj.rois:
    roi.set_channel_exclude(config.channels_exclude)

# Remove ROIs to be excluded
to_rem = [r for r in prj.rois if r.name in config.roi_exclude_strings]
for r in to_rem:
    for ft in ["stack", "channel_labels", "probabilities", "cell_mask"]:
        r.get_input_filename(ft).unlink()
for s in prj:
    s.rois = [r for r in s if r.name not in config.roi_exclude_strings]


# # ROIs
roi_names = [x.name for x in prj.rois]
config.roi_attributes = (
    pd.DataFrame(
        np.asarray(
            [getattr(r.sample, attr) for r in prj.rois for attr in config.attributes]
        ).reshape((-1, len(config.attributes))),
        index=roi_names,
        columns=config.attributes,
    )
    .rename_axis(index="roi")
    .rename(columns={"name": "sample"})
)

# # Samples
sample_names = [x.name for x in prj.samples]
config.sample_attributes = pd.DataFrame(
    np.asarray(
        [getattr(s, attr) for s in prj.samples for attr in config.attributes]
    ).reshape((-1, len(config.attributes))),
    index=sample_names,
    columns=config.attributes,
).rename_axis(index="sample")


for df in [config.roi_attributes, config.sample_attributes]:
    for cat, order in config.attribute_order.items():
        df[cat] = pd.Categorical(df[cat], categories=order, ordered=True)
