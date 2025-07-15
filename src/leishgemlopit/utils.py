from __future__ import annotations
from collections import defaultdict
from collections.abc import Hashable, Mapping
import json
import pathlib
import io
from typing import TYPE_CHECKING

from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

if TYPE_CHECKING:
    from .tsne import TSNEAnalysis
from .constants import FIXTURE_PATH

import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle
import pandas as pd


class LabelMixin(Mapping[str, Hashable]):
    def get_presentation_label(self, label: Hashable):
        return label

    def was_assigned(self, gene_id: str) -> bool:
        return True


class PNGMixin:
    def _figure(self) -> Figure:
        raise NotImplementedError

    def to_png(self, output_file: pathlib.Path | str | io.FileIO):
        if (
            isinstance(output_file, pathlib.Path) or
            isinstance(output_file, str)
        ):
            output_file = pathlib.Path(output_file).open("wb")
        with output_file:
            output_file.write(self._repr_png_())

    def _repr_png_(self):
        figure = self._figure()
        png_bytes = io.BytesIO()
        with png_bytes:
            figure.savefig(png_bytes, format="png", bbox_inches="tight")
            ret_value = png_bytes.getvalue()
        plt.close(figure)
        return ret_value


class TSNEPlotMixin(PNGMixin, LabelMixin):
    tsne_analysis: TSNEAnalysis = None

    def _figure(self):
        if self.tsne_analysis is None:
            raise ValueError("Need a TSNEAnalysis to produce figure.")
        fig, ax = plt.subplots(1, 1, figsize=(10, 6), dpi=150)
        fig.subplots_adjust(left=0, bottom=0, right=0.6, top=1)

        df = (
            pd.Series(
                {
                    geneid: label
                    for geneid, label in self.items()
                    if self.was_assigned(geneid)
                },
                name="label"
            )
            .to_frame()
            .join(self.tsne_analysis.to_dataframe(), how="right")
        )

        nolabels = df[df.label.isna()]
        labels = df[df.label.notna()]
        ax.scatter(
            nolabels.x, nolabels.y,
            s=1,
            marker=MarkerStyle("o", fillstyle="none"),
            linewidth=0.5,
            color="grey",
        )

        styles = MarkerPlotStyles()

        groups = defaultdict(list)
        for label, g in labels.groupby("label"):
            groups[self.get_presentation_label(label)].append(g)

        component_list = list(_CELL_COMPONENTS) + ["unknown"]
        nocolors = set(groups).difference(set(component_list))
        if nocolors:
            print(
                "WARNING: The following components do not have a colour "
                f"defined: {', '.join(nocolors)}. They will not be shown."
            )

        for label in component_list:
            for g in groups[label]:
                style = styles(label)
                ax.scatter(
                    g.x, g.y,
                    s=10,
                    marker=MarkerStyle(style["marker"], fillstyle="none"),
                    linewidth=1,
                    label=label,
                    color=style["color"],
                )

        ax.axis("off")
        ax.legend(
            *zip(
                *[
                    (Rectangle((0, 0), 1, 1, facecolor=_R_COLORS[color]), label)
                    for label, color in _CELL_COMPONENTS.items()
                    if groups[label]
                ] +
                [
                    (Rectangle((0, 0), 1, 1, facecolor="black"), "unknown")
                ]),
            frameon=False,
            bbox_to_anchor=(1, 1),
            loc="upper left",
        )

        return fig


with (FIXTURE_PATH / "component_colors.json").open("r") as f:
    _CELL_COMPONENTS: dict[str, str] = json.load(f)

with (FIXTURE_PATH / "rcolors.json").open("r") as f:
    _R_COLORS: dict[str, tuple[int, int, int]] = json.load(f)


class MarkerPlotStyles:
    markers = [
        "o", "x", "v", "^", "<", ">",
        "s", "D", "P", "X", "d", "8",
        "H", "*", "1", "2", "3", "4"
    ]

    def _style_generator(self, label):
        if label == "unknown":
            colors = ["black"]
        else:
            colors = [_CELL_COMPONENTS[label]]
        while True:
            for color in colors:
                for marker in self.markers:
                    yield {
                        "color": _R_COLORS[color],
                        "marker": marker,
                    }
            print(f"WARNING: Recycling {label} markers!")

    def __init__(self):
        self._component_styles = {
            label: self._style_generator(label)
            for label in list(_CELL_COMPONENTS) + ["unknown"]
        }

    def __call__(self, label):
        return next(self._component_styles[label])
