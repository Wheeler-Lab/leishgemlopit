from collections import Counter, defaultdict
import warnings

import pandas as pd
from hdbscan import HDBSCAN
from sklearn.preprocessing import StandardScaler

from leishgemlopit.lopit import LOPITExperimentCollection
from leishgemlopit.markers import Markers
from leishgemlopit.constants import ANNOTATIONS_BACKGROUND_LIKE
from leishgemlopit.tsne import TSNEAnalysis
from leishgemlopit.utils import TSNEPlotMixin


def most_common_marker(c: Counter[str]):
    markers = {
        m for m in c if c[m] > 1 and m != "unknown"
    }
    markers_no_bg = markers - ANNOTATIONS_BACKGROUND_LIKE
    if markers_no_bg:
        markers = markers_no_bg

    for marker, _ in c.most_common():
        if marker in markers:
            return marker
    return "unknown"


class UnsupervisedHDBSCAN(TSNEPlotMixin):
    def __init__(
        self,
        lopit_experiments: LOPITExperimentCollection,
        markers: Markers,
        tsne_analysis: TSNEAnalysis | None = None,
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

        self._clusters: dict[str, int] = {
            geneid: cluster for geneid, cluster in clusters.items()
        }

        label_candidates = defaultdict[int, Counter[str]](Counter[str])
        for geneid, cluster in self._clusters.items():
            marker = self.markers[geneid]
            if marker is None:
                marker = "unknown"
            label_candidates[cluster][marker] += 1
        self._labels = {
            cluster: most_common_marker(counter)
            for cluster, counter in label_candidates.items()
        }

    def __getitem__(self, gene_id: str):
        return self._clusters[gene_id]

    def get_presentation_label(self, label: int):
        return self._labels[label]

    def was_assigned(self, gene_id: str):
        return self._clusters[gene_id] >= 0

    def __iter__(self):
        return iter(self._clusters)

    def __len__(self):
        return len(self._clusters)

    def to_dataframe(self, include_tsne=True):
        clusters = pd.DataFrame(
            [
                {
                    "geneid": geneid,
                    "cluster": cluster,
                    "label": self._labels[cluster] if cluster in self._labels else "",
                }
                for geneid, cluster in self.items()
            ]
        ).set_index("geneid")

        if include_tsne and self.tsne_analysis is not None:
            return clusters.join(
                self.tsne_analysis.to_dataframe(), how="outer")
        return clusters

    def summary(self):
        genes_assigned = [g for g, c in self.items() if c >= 0]
        genes_not_assigned = [g for g, c in self.items() if c < 0]
        return {
            "nr_clusters": len(set(c for c in self.values() if c >= 0)),
            "nr_genes_assigned_cluster": len(genes_assigned),
            "nr_genes_not_assigned": len(genes_not_assigned),
        }
