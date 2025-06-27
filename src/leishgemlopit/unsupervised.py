from collections import Counter
import io
import pathlib
from typing import Mapping
import warnings

from cycler import cycler
from matplotlib import pyplot as plt
from matplotlib.markers import MarkerStyle
import pandas as pd
from hdbscan import HDBSCAN
from sklearn.preprocessing import StandardScaler

from leishgemlopit.lopit import LOPITExperimentCollection
from leishgemlopit.markers import Markers
from leishgemlopit.constants import ANNOTATIONS_BACKGROUND_LIKE
from leishgemlopit.tsne import TSNEAnalysis


def most_common_marker(g: pd.Series):
    c = Counter[str](g)

    markers = {
        m for m in c if c[m] > 1
    }
    markers_no_bg = markers - ANNOTATIONS_BACKGROUND_LIKE
    if markers_no_bg:
        markers = markers_no_bg

    for marker, _ in c.most_common():
        if marker in markers:
            return marker
    return "unknown"


class UnsupervisedHDBSCAN(Mapping[str, str]):
    def __init__(
        self,
        lopit_experiments: LOPITExperimentCollection,
        tsne_analysis: TSNEAnalysis | None = None,
        markers: Markers | None = None,
        vote_threshold: float = 0.25,
        min_samples: int = 5,
        min_cluster_size: int = 5,
    ):
        self.vote_threshold = vote_threshold
        self.min_samples = min_samples
        self.min_cluster_size = min_cluster_size

        data = lopit_experiments.to_dataframe()
        self.tsne_analysis = tsne_analysis
        self.markers = markers

        self._hdbscan(data)

    def _hdbscan(self, data):
        df: pd.DataFrame = data.unstack(["experiment", "tmt_label"])
        df = pd.DataFrame(
            StandardScaler().fit_transform(df),
            index=df.index,
        )

        with warnings.catch_warnings():
            # Ignore HDBSCAN triggering sklearn option rename warning.
            warnings.simplefilter("ignore", FutureWarning)
            hdbscan = HDBSCAN(
                min_samples=self.min_samples,
                min_cluster_size=self.min_cluster_size,
                cluster_selection_method='leaf'
            )
            clusters = pd.Series(
                hdbscan.fit_predict(df),
                index=df.index,
                name="cluster",
            )

        self._data: dict[str, int] = {
            geneid: cluster for geneid, cluster in clusters.items()
        }

        if self.markers is not None:
            markers = pd.Series({
                geneid: marker
                for geneid in clusters.index
                for marker in self.markers[geneid]
            }).to_frame("marker")

            markers = markers.join(
                clusters[clusters >= 0],
                how="inner").fillna("unknown")
            markers = (
                markers.groupby("cluster").marker.apply(most_common_marker)
            )
            self._markers = {
                cluster: marker for cluster, marker in markers.items()
            }
        else:
            self._markers = {}

    def __getitem__(self, gene_id: str):
        return self._data[gene_id]

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def to_dataframe(self):
        clusters = pd.Series(self._data, name="cluster").to_frame()
        if self.tsne_analysis is not None:
            return clusters.join(
                self.tsne_analysis.to_dataframe(), how="outer")
        return clusters

    def _figure(self):
        if self.tsne_analysis is None:
            raise ValueError("Need a TSNEAnalysis to produce figure.")
        if self._markers:
            width = 10
            right = 0.6
        else:
            width = 6
            right = 1
        fig, ax = plt.subplots(1, 1, figsize=(width, 6), dpi=150)
        fig.subplots_adjust(left=0, bottom=0, right=right, top=1)

        df = self.to_dataframe()
        nocluster = df[df.cluster < 0]
        clusters = df[df.cluster >= 0]
        ax.scatter(
            nocluster.x, nocluster.y,
            s=1,
            marker=MarkerStyle("o", fillstyle="none"),
            linewidth=0.5,
            color="black",
        )

        styles = iter(
            cycler(marker=["o", "x", "v", "^"]) *
            cycler(color=plt.colormaps["tab10"].colors)
        )

        for cluster, g in (
            sorted(clusters.groupby("cluster"), key=lambda g: -g[1].shape[0])
        ):
            style = next(styles)
            try:
                label = self._markers[cluster]
            except KeyError:
                label = str(cluster)
            ax.scatter(
                g.x, g.y,
                s=10,
                marker=MarkerStyle(style["marker"], fillstyle="none"),
                linewidth=1,
                label=label,
                color=style["color"],
            )

        ax.axis("off")

        if self._markers:
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
        genes_assigned = [g for g, c in self.items() if c >= 0]
        genes_not_assigned = [g for g, c in self.items() if c < 0]
        return {
            "nr_clusters": len(set(c for c in self.values() if c >= 0)),
            "nr_genes_assigned_cluster": len(genes_assigned),
            "nr_genes_not_assigned": len(genes_not_assigned),
        }
