#!/usr/bin/env python

"""
A module to provide the boilerplate needed for all the analysis.
"""

import typing as tp
from functools import partial

import numpy as np  # type: ignore[import]
import pandas as pd  # type: ignore[import]
from matplotlib import rcParams
import seaborn as sns  # type: ignore[import]

from imc import Project  # type: ignore[import]
from imc.types import Path, DataFrame  # type: ignore[import]


rcParams["font.family"] = "Arial"


pd.read_csv = partial(pd.read_csv, engine="pyarrow")


class config:
    # constants
    channels_exclude_strings: tp.Final[tp.List[str]] = [
        "<EMPTY>",
        "_EMPTY_",
        "190BCKG",
        "80ArAr",
        "129Xe",
        "Ki67",
    ]
    roi_exclude_strings: tp.Final[tp.List[str]] = [
        "A19_15_A24-01",
        "A19_15_A24-02",
        "A19_15_A24-03",
        "A19_15_A24-04",
        "A19_15_A24-05",
        "A19_33_A26-01",
        "A20_105_A25-06",
        "A21_22_A38-03",
        "A21_22_A38-06",
    ]

    channels_exclude: tp.List[str]
    channels_include: tp.List[str]

    ## Major attributes to contrast when comparing observation groups
    attributes: tp.Final[tp.List[str]] = [
        "disease",
        "disease_subgroup",
        "age",
        "gender",
    ]
    numeric_attributes = ["age"]

    attribute_order: tp.Dict[str, tp.List[str]] = dict(
        disease=[
            "Normal",
            "UIP/IPF",
            "COVID-19",
            "Mixed",
            "Convalescent",
        ],
        disease_subgroup=[
            "Normal",
            "UIP/IPF",
            "COVID-19-early",
            "COVID-19-late",
            "Mixed",
            "COVID-19-long-pos",
            "COVID-19-long-neg",
        ],
        gender=[
            "F",
            "M",
        ],
    )
    roi_attributes: DataFrame
    sample_attributes: DataFrame

    figkws: tp.Final[tp.Dict] = dict(
        dpi=300, bbox_inches="tight", pad_inches=0.1, transparent=False
    )

    # directories
    metadata_dir: tp.Final[Path] = Path("metadata")
    data_dir: tp.Final[Path] = Path("data")
    processed_dir: tp.Final[Path] = Path("processed")
    results_dir: tp.Final[Path] = Path("results")

    # Color codes
    colors: tp.Final[tp.Dict[str, np.ndarray]] = dict(
        disease=np.asarray(sns.color_palette())[[2, 0, 3, 5, 4]],
        disease_subgroup=np.asarray(sns.color_palette())[[2, 0, 1, 3, 5, 6, 4]],
        gender=np.asarray(sns.color_palette())[[2, 4]],
    )

    # Output files
    metadata_file: tp.Final[Path] = metadata_dir / "clinical_annotation.pq"
    quantification_file: tp.Final[Path] = (
        results_dir / "cell_type" / "quantification.pq"
    )
    gating_file: tp.Final[Path] = results_dir / "cell_type" / "gating.pq"
    h5ad_file: tp.Final[Path] = (
        results_dir / "cell_type" / "anndata.all_cells.processed.h5ad"
    )
    counts_file: tp.Final[Path] = results_dir / "cell_type" / "cell_type_counts.pq"

    # Areas of images/samples
    roi_areas_file: tp.Final[Path] = metadata_dir / "roi_areas.csv"
    sample_areas_file: tp.Final[Path] = metadata_dir / "sample_areas.csv"
    roi_areas: tp.Final[DataFrame]
    sample_areas: tp.Final[DataFrame]

    cell_type_markers = [
        "aSMA(Pr141)",
        "cKIT(Nd143)",
        "CD206(Nd144)",
        "CD16(Nd146)",
        "CD163(Sm147)",
        "CD14(Nd148)",
        "CD11b(Sm149)",
        "CD31(Eu151)",
        "CD45(Sm152)",
        "CD4(Gd156)",
        "Periostin(Dy161)",
        "CD8a(Dy162)",
        "CC16(Dy163)",
        "AQ1(Dy164)",
        "CD123(Er167)",
        "ColTypeI(Tm169)",
        "CD3(Er170)",
        "SFTPA(Yb171)",
        "MPO(Yb173)",
        "K818(Yb174)",
        "SFTPC(Lu175)",
        "CD11c(Yb176)",
        "Vimentin(Pt195)",
        "CD68(Pt196)",
    ]

    config.cell_state_markers = [
        "SARSSpikeS1(Eu153)",
        "pSTAT3Tyr705(Gd158)",
        "pNFkbp65(Ho165)",
        "IRF2BP2(Tb159)",
        "IL6(Gd160)",
        "IL1beta(Er166)",
        "iNOS(Nd142)",
        "CitH3(Sm154)",
        "SC5b9(Gd155)",
        "CC3(Yb172)",
        "Periostin(Dy161)",
        "p16(Nd150)",
        "uPAR(Nd145)",
        # "Ki67(Er168)",
    ]


# Initialize project
prj = Project(metadata=config.metadata_dir / "samples.csv", name="covid-pasc-imc")

if not prj.rois:
    raise ValueError("No ROIs in project. Check directories exist!")

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
config.roi_attributes["sample"] = config.roi_attributes.index.str.split("-").map(
    lambda x: x[0]
)
for attr in config.numeric_attributes:
    config.roi_attributes[attr] = config.roi_attributes[attr].astype(float)
for roi in prj.rois:
    roi.attributes = config.attributes
config.roi_attributes.to_csv(config.metadata_dir / "roi_attributes.csv")

# # Samples
sample_names = [x.name for x in prj.samples]
config.sample_attributes = pd.DataFrame(
    np.asarray(
        [getattr(s, attr) for s in prj.samples for attr in config.attributes]
    ).reshape((-1, len(config.attributes))),
    index=sample_names,
    columns=config.attributes,
).rename_axis(index="sample")
for attr in config.numeric_attributes:
    config.sample_attributes[attr] = config.sample_attributes[attr].astype(float)

for df in [config.roi_attributes, config.sample_attributes]:
    for cat, order in config.attribute_order.items():
        df[cat] = pd.Categorical(df[cat], categories=order, ordered=True)
for sample in prj:
    sample.attributes = config.attributes
config.sample_attributes.to_csv(config.metadata_dir / "sample_attributes.csv")

# Calculate area space
if not config.roi_areas_file.exists():
    roi_areas = pd.Series(
        [r.area for r in prj.rois], index=[r.name for r in prj.rois], name="area_mm2"
    ).rename_axis(index="roi")
    roi_areas.to_csv(config.roi_areas_file)

    sample_areas = pd.Series(
        [sum([r.area for r in s.rois]) for s in prj],
        index=[s.name for s in prj],
        name="area_mm2",
    ).rename_axis(index="sample")
    sample_areas.to_csv(config.sample_areas_file)

config.roi_areas = pd.read_csv(config.roi_areas_file, index_col=0).squeeze()
config.sample_areas = pd.read_csv(config.sample_areas_file, index_col=0).squeeze()


config.categorical_attributes = [
    a for a in config.attributes if a not in config.numeric_attributes
]
