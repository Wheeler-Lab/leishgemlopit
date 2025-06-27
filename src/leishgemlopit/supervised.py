from collections import defaultdict
import io
from typing import Mapping, NamedTuple

from cycler import cycler
from matplotlib import pyplot as plt
from matplotlib.markers import MarkerStyle
import pandas as pd
from sklearn.preprocessing import QuantileTransformer

from leishgemlopit.lopit import LOPITExperiment, LOPITExperimentCollection
from leishgemlopit.markers import Markers
from leishgemlopit.tsne import TSNEAnalysis

from tagm import TAGMMAP

from leishgemlopit.utils import PNGMixin


class TAGMResult(NamedTuple):
    label: str
    probability: float


class SupervisedTAGM(Mapping[str, TAGMResult], PNGMixin):
    def __init__(
        self,
        lopit_experiment: LOPITExperiment,
        markers: Markers,
        tsne_analysis: TSNEAnalysis | None = None,
    ):
        data = (
            lopit_experiment.to_dataframe()
            .unstack("tmt_label", fill_value=0.0)
        )
        data = pd.DataFrame(
            QuantileTransformer().fit_transform(data),
            index=data.index.get_level_values("gene_id"),
        )

        self.tsne_analysis = tsne_analysis
        self.markers = markers

        markers = (
            pd.DataFrame([
                {
                    "geneid": geneid,
                    "marker": marker
                }
                for geneid, markers in self.markers.items()
                for marker in markers
            ])
            .groupby("geneid").first()
        )
        markers = (
            data.loc[:, []]
            .join(markers, how="left")
            .fillna("unknown")
            .astype(str)
        )

        tagm = TAGMMAP(
            max_iter=1000,

            trust_training_data_implicitly=True,
        )
        probability = pd.DataFrame(
            tagm.fit_predict_proba(
                data.values,
                markers.values,
                unknown_label="unknown",
            ),
            index=data.index,
            columns=tagm.classes_,
        )
        self._probability = probability
        self._labels: dict[str, TAGMResult] = {
            geneid: TAGMResult(label, probability.loc[geneid, label])
            for geneid, label in probability.idxmax(axis=1).items()
        }

    def __getitem__(self, gene_id: str):
        return self._labels[gene_id]

    def __iter__(self):
        return iter(self._labels)

    def __len__(self):
        return len(self._labels)

    def to_dataframe(self):
        labels = pd.DataFrame([
            {
                "geneid": geneid,
                "label": res.label,
                "probability": res.probability,
            }
            for geneid, res in self.items()
        ]).set_index("geneid")

        if self.tsne_analysis is not None:
            return labels.join(
                self.tsne_analysis.to_dataframe(), how="outer")
        return labels

    def _figure(self):
        if self.tsne_analysis is None:
            raise ValueError("Need a TSNEAnalysis to produce figure.")

        fig, ax = plt.subplots(1, 1, figsize=(10, 6), dpi=150)
        fig.subplots_adjust(left=0, bottom=0, right=0.6, top=1)

        df = self.to_dataframe()
        v = (df.label == "unknown") | (df.probability < 0.5)
        nolabel = df[v]
        labels = df[~v]
        ax.scatter(
            nolabel.x, nolabel.y,
            s=1,
            marker=MarkerStyle("o", fillstyle="none"),
            linewidth=0.5,
            color="black",
        )

        styles = iter(
            cycler(marker=["o", "x", "v", "^"]) *
            cycler(color=plt.colormaps["tab10"].colors)
        )

        for label, g in (
            sorted(labels.groupby("label"), key=lambda g: -g[1].shape[0])
        ):
            style = next(styles)
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


class TAGMResultCollection(Mapping[str, TAGMResult]):
    def __init__(self, results: dict[LOPITExperiment, TAGMResult]):
        self._results = {
            experiment.name: result
            for experiment, result in results.items()
        }

    def __getitem__(self, experiment: str):
        return self._results[experiment]

    def __iter__(self):
        return iter(self._results)

    def __len__(self):
        return len(self._results)

    def assign(self):
        probsum = defaultdict[str, float](lambda: 0.0)
        for result in self.values():
            probsum[result.label] += result.probability

        total = sum(probsum.values())
        confidences = {
            term: value / total
            for term, value in probsum.items()
        }
        term, confidence = max(
            confidences.items(), key=lambda e: e[0])

        if confidence < 0.95:
            return "unknown"
        return term


class SupervisedTAGMCollection(Mapping[str, TAGMResultCollection], PNGMixin):
    def __init__(
        self,
        lopit_experiments: LOPITExperimentCollection,
        markers: Markers,
        tsne_analysis: TSNEAnalysis | None = None,
    ):
        self._tagms = {
            experiment: SupervisedTAGM(
                experiment,
                markers,
                tsne_analysis=tsne_analysis,
            )
            for experiment in lopit_experiments.values()
        }

        self.tsne_analysis = tsne_analysis

        results = defaultdict[str, dict](dict)
        for experiment, tagm in self._tagms.items():
            for geneid, result in tagm.items():
                results[geneid][experiment] = result

        self._results = {
            geneid: TAGMResultCollection(r)
            for geneid, r in results.items()
        }

    def __getitem__(self, gene_id: str):
        return self._results[gene_id]

    def __iter__(self):
        return iter(self._results)

    def __len__(self):
        return len(self._results)

    def to_dataframe(self):
        labels = pd.DataFrame([
            {
                "geneid": geneid,
                "label": res.assign(),
            }
            for geneid, res in self.items()
        ]).set_index("geneid")

        if self.tsne_analysis is not None:
            return labels.join(
                self.tsne_analysis.to_dataframe(), how="outer")
        return labels

    def _figure(self):
        if self.tsne_analysis is None:
            raise ValueError("Need a TSNEAnalysis to produce figure.")

        fig, ax = plt.subplots(1, 1, figsize=(10, 6), dpi=150)
        fig.subplots_adjust(left=0, bottom=0, right=0.6, top=1)

        df = self.to_dataframe()
        v = df.label == "unknown"
        nolabel = df[v]
        labels = df[~v]
        ax.scatter(
            nolabel.x, nolabel.y,
            s=1,
            marker=MarkerStyle("o", fillstyle="none"),
            linewidth=0.5,
            color="black",
        )

        styles = iter(
            cycler(marker=["o", "x", "v", "^"]) *
            cycler(color=plt.colormaps["tab10"].colors)
        )

        for label, g in (
            sorted(labels.groupby("label"), key=lambda g: -g[1].shape[0])
        ):
            style = next(styles)
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
