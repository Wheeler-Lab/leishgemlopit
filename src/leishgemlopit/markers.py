import io
import pathlib
from typing import Mapping

from cycler import cycler
from matplotlib import pyplot as plt
from matplotlib.markers import MarkerStyle
import pandas as pd

from leishgemlopit.constants import ANNOTATIONS_BACKGROUND_LIKE
from leishgemlopit.tsne import TSNEAnalysis


class MarkerGenerator:
    def __call__(self, gene_id: str) -> set[str]:
        raise NotImplementedError


class MarkerFactory:
    def __init__(self, generators: list[MarkerGenerator]):
        self.generators = generators

    def __call__(self, gene_id):
        terms = set()
        for generator in self.generators:
            terms |= generator(gene_id)
        return self._filter_multiple(
            self._filter_background_like(terms))

    @staticmethod
    def _filter_background_like(terms: set[str]):
        if ANNOTATIONS_BACKGROUND_LIKE.issuperset(terms):
            return terms
        return terms.difference(ANNOTATIONS_BACKGROUND_LIKE)

    def _filter_multiple(self, terms: set[str]):
        if len(terms) > 1:
            return set()
        return terms


class Markers(Mapping[str, set[str]]):
    def __init__(
        self,
        marker_factory: MarkerFactory,
        gene_ids: list[str],
        tsne_analysis: TSNEAnalysis | None = None,
    ):
        self._data = {
            gene_id: marker_factory(gene_id)
            for gene_id in gene_ids
        }

        self.tsne_analysis = tsne_analysis

    def __getitem__(self, gene_id: str):
        return self._data[gene_id]

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def to_dataframe(self):
        entries: dict[str, str] = []
        for gene_id, markers in self.items():
            for marker in markers:
                entries.append({
                    "geneid": gene_id,
                    "marker": marker,
                })
        df = pd.DataFrame(entries).set_index("geneid")
        if self.tsne_analysis is not None:
            return df.join(self.tsne_analysis.to_dataframe(), how="outer")
        return df

    def _figure(self):
        if self.tsne_analysis is None:
            raise ValueError("Need a TSNEAnalysis to produce figure.")
        fig, ax = plt.subplots(1, 1, figsize=(10, 6), dpi=150)
        fig.subplots_adjust(left=0, bottom=0, right=0.6, top=1)

        df = self.to_dataframe()
        nomarkers = df[df.marker.isna()]
        markers = df[df.marker.notna()]
        ax.scatter(
            nomarkers.x, nomarkers.y,
            s=1,
            marker=MarkerStyle("o", fillstyle="none"),
            linewidth=0.5,
            color="grey",
        )

        styles = iter(
            cycler(marker=["o", "x", "v", "^"]) *
            cycler(color=plt.colormaps["tab10"].colors)
        )

        for marker, g in (
            sorted(markers.groupby("marker"), key=lambda g: -g[1].shape[0])
        ):
            style = next(styles)
            ax.scatter(
                g.x, g.y,
                s=10,
                marker=MarkerStyle(style["marker"], fillstyle="none"),
                linewidth=1,
                label=marker,
                color=style["color"],
            )

        ax.axis("off")
        ax.legend(
            frameon=False,
            bbox_to_anchor=(1, 1),
            loc="upper left",
        )

        return fig

    def _repr_png_(self):
        figure = self._figure()
        png_bytes = io.BytesIO()
        with png_bytes:
            figure.savefig(png_bytes, format="png")
            ret_value = png_bytes.getvalue()
        plt.close(figure)
        return ret_value

    def to_png(self, output_file: pathlib.Path):
        output_file = pathlib.Path(output_file)
        with output_file.open("wb") as f:
            f.write(self._repr_png_())

    def summary(self):
        return MarkerSummary(self)


class MarkerSummary:
    def __init__(self, markers: Markers):
        df = (
            markers
            .to_dataframe()
            .assign(nr_proteins=1).loc[:, ["marker", "nr_proteins"]]
        )
        self._counts = df.groupby("marker").count().sort_values(
            "nr_proteins", ascending=False)

    def _repr_markdown_(self):
        return self._counts.to_markdown()

    def __str__(self):
        return "\n".join([
            f"{marker}:\t{nr_proteins}"
            for marker, nr_proteins in self._counts.nr_proteins.items()
        ])
