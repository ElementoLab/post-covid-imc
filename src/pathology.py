#!/usr/bin/env python

"""
Image level analysis of broad pathological features of IMC samples.
"""

import typing as tp

from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import skimage
import parmap

from imc.types import Path, Array, DataFrame
from imc.graphics import close_plots

from src._config import prj, config


output_dir = (config.results_dir / "pathology").mkdir()


def main() -> int:

    # Lacunarity
    lac = quantify_lacunar_space()
    interpret_metric(lac, "lacunarity")

    # Fibrosis
    fib1 = score_marker(channel="ColTypeI(Tm169)")
    interpret_metric(fib1, "fibrosis_collagen")

    fib2 = score_marker(channel="Periostin(Dy161)")
    interpret_metric(fib2, "fibrosis_periostin")

    fib3 = score_marker(channel="CC16(Dy163)")
    interpret_metric(fib3, "CC16_bleeding")
    # Check CC16 abundance only in airway ROIS
    score_compartment_specific(channel="CC16(Dy163)")

    fib4 = score_marker(channel="CitH3(Sm154)")
    interpret_metric(fib4, "CitH3")

    # Combine both
    # # weighted by the relative mean of the channels across images
    chs = ["ColTypeI(Tm169)", "Periostin(Dy161)"]
    metrics = pd.read_csv(
        config.results_dir / "roi_channel_stats.csv", index_col=["roi", "channel"]
    )
    m = metrics.loc[:, chs, :].groupby("channel")["mean"].mean()
    r = m[chs[0]] / m[chs[1]]

    fib = pd.concat([fib1, fib2 * r]).groupby(level=0).mean()
    interpret_metric(fib, "fibrosis_joint")

    # Vessels
    # # (CD31, AQP1, aSMA, fill holes)

    return 0


def score_compartment_specific(
    channel: str = "CC16(Dy163)",
    attribute_name: str = "CC16_bleeding_airways",
    compartment: str = "A",
    compartment_name="airways",
):
    from src.analysis import get_domain_areas

    areas = get_domain_areas()

    f = output_dir / f"extent_and_intensity.{channel}_quantification.csv"
    if not f.exists():
        fib = score_marker(channel=channel)
    fib = pd.read_csv(f, index_col=0)

    interpret_metric(
        fib.loc[(areas[compartment] > 0)], f"{attribute_name}_{compartment_name}"
    )
    interpret_metric(
        fib.loc[(areas[compartment] == 0)], f"{attribute_name}_non{compartment_name}"
    )


def quantify_lacunar_space(overwrite: bool = False):
    f = output_dir / "lacunarity.quantification.csv"

    if not f.exists() or overwrite:
        _res = parmap.map(get_lacunae, prj.rois, pm_pbar=True)
        res = pd.DataFrame(
            [(x > 0).sum() for x in _res],
            index=[r.name for r in prj.rois],
            columns=["lacunar_space"],
        )
        res["area"] = [r.area for r in prj.rois]
        res["lacunar_fraction"] = res["lacunar_space"] / res["area"]
        res.to_csv(f)
    res = pd.read_csv(f, index_col=0)

    return res


def score_marker(channel: str = "ColTypeI(Tm169)", overwrite: bool = False):
    f = output_dir / f"extent_and_intensity.{channel}_quantification.csv"

    if not f.exists() or overwrite:
        _res = parmap.map(get_extent_and_mean, prj.rois, marker=channel, pm_pbar=True)
        res = pd.DataFrame(
            _res, columns=["extent", "intensity"], index=[r.name for r in prj.rois]
        )
        res["score"] = res.apply(lambda x: (x - x.mean()) / x.std()).mean(1)
        res.to_csv(f)
    res = pd.read_csv(f, index_col=0)

    return res


@close_plots
def interpret_metric(res: DataFrame, metric):

    # get mean per sample
    res_sample = (
        res.join(config.roi_attributes["sample"])
        .groupby("sample")
        .mean()
        .join(config.sample_attributes)
    )

    for attr in config.categorical_attributes:
        fig, stats = swarmboxenplot(
            data=res.join(config.roi_attributes),
            x=attr,
            y=res.columns,
            plot_kws=dict(palette=config.colors.get(attr)),
        )
        fig.savefig(
            output_dir / f"{metric}.roi.by_{attr}.svg",
            **config.figkws,
        )
        stats.to_csv(output_dir / f"{metric}.roi.by_{attr}.csv", index=False)

        fig, stats = swarmboxenplot(
            data=res_sample,
            x=attr,
            y=res.columns,
            plot_kws=dict(palette=config.colors.get(attr)),
        )
        fig.savefig(
            output_dir / f"{metric}.sample.by_{attr}.svg",
            **config.figkws,
        )
        stats.to_csv(output_dir / f"{metric}.sample.by_{attr}.csv", index=False)


def get_lacunae(
    roi: "ROI",
    selem_diam: int = 5,
    min_diam: int = 25,
    max_area_percentage: float = 50,
    fill_holes: bool = False,
) -> Array:
    from csbdeep.utils import normalize
    import skimage as ski

    image = roi.stack[~roi.channel_exclude, ...]
    image = np.asarray([normalize(np.log1p(x)) for x in image]).mean(0)

    # threshold, close
    img = image > ski.filters.threshold_otsu(image)
    # img = image > ski.filters.threshold_multiotsu(image)[1]
    img = ski.morphology.binary_dilation(img, footprint=ski.morphology.disk(selem_diam))
    img = ski.morphology.closing(img, ski.morphology.disk(5))

    # clean up small objects inside
    if fill_holes:
        img = ~ndi.binary_fill_holes(~img)
        img = ~ski.morphology.remove_small_objects(~img, min_size=min_diam**2)

    lac = ndi.label(~img)[0]

    # remove objects too large
    remove = [
        i
        for i in np.unique(lac)
        if ((lac == i).sum() / img.size) * 100 > max_area_percentage
    ]
    if remove:
        for i in remove:
            lac[lac == i] = 0
    return lac
    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.contour(lac, levels=3, cmap="Reds")


def get_vessels(
    roi: "ROI",
    min_diam: int = 25,
) -> Array:

    raise NotImplementedError

    from csbdeep.utils import normalize
    import skimage as ski

    image = roi._get_channel("AQ1")[1]
    image = np.asarray([normalize(np.log1p(x)) for x in image]).mean(0)

    # threshold, close
    img = image > ski.filters.threshold_otsu(image)
    img = ski.morphology.remove_small_objects(img, min_size=min_diam**2)


def get_extent_and_mean(roi: "ROI", marker: str) -> tp.Tuple[float, float]:
    x = np.log1p(roi._get_channel(marker)[1].squeeze())
    area = np.multiply(*roi.shape[1:])
    mask = skimage.filters.gaussian(x, 2) > skimage.filters.threshold_otsu(x)
    return mask.sum() / area, x.mean()


if __name__ == "__main__" and "get_ipython" not in locals():
    import sys

    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit()
