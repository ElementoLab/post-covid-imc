#!/usr/bin/env python

"""
High-level analysis of IMC samples.
"""

from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import skimage

from imc.types import Path, Array
from imc.graphics import close_plots

from src._config import prj, config


output_dir = (config.results_dir / "pathology").mkdir()


def main() -> int:

    # Lacunarity
    lac = quantify_lacunar_space()
    interpret_metric(lac, "lacunarity")

    # Fibrosis
    fib = quantify_fibrosis()
    interpret_metric(fib, "fibrosis")

    # Vessels
    # # (CD31, AQP1, aSMA, fill holes)

    return 0


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


def quantify_fibrosis(
    collagen_channel: str = "ColTypeI(Tm169)", overwrite: bool = False
):
    f = output_dir / "fibrosis.extent_and_intensity.quantification.csv"

    if not f.exists() or overwrite:
        _res = parmap.map(
            get_extent_and_mean, prj.rois, marker=collagen_channel, pm_pbar=True
        )
        res = pd.DataFrame(
            _res, columns=["extent", "intensity"], index=[r.name for r in prj.rois]
        )
        res["score"] = res.apply(lambda x: (x - x.mean()) / x.std()).mean(1)
        res.to_csv(f)
    res = pd.read_csv(f, index_col=0)

    return res


@close_plots
def interpret_metric(res: DataFrame, metric: str):

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
        img = ~ski.morphology.remove_small_objects(~img, min_size=min_diam ** 2)

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
    img = ski.morphology.remove_small_objects(img, min_size=min_diam ** 2)


def get_extent_and_mean(roi: "ROI", marker: str) -> tp.Tuple[float, float]:
    x = np.log1p(roi._get_channel(marker)[1].squeeze())
    area = np.multiply(*roi.shape[1:])
    return (x > skimage.filters.threshold_otsu(x)).sum() / area, x.mean()


if __name__ == "__main__" and "get_ipython" not in locals():
    import sys

    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit()
