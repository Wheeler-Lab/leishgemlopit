from __future__ import annotations

from collections import defaultdict
from typing import Mapping, NamedTuple

import pandas as pd
from sklearn.preprocessing import QuantileTransformer

from leishgemlopit.lopit import LOPITExperiment, LOPITExperimentCollection
from leishgemlopit.markers import Markers
from leishgemlopit.tsne import TSNEAnalysis

from tagm import TAGMMAP

from leishgemlopit.utils import TSNEPlotMixin


class TAGMResult(NamedTuple):
    label: str
    probability: float


class SupervisedTAGM(TSNEPlotMixin):
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
                for geneid, marker in self.markers.items()
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


class TAGMResultCollection(Mapping[str, TAGMResult]):
    def __init__(
        self,
        geneid: str,
        results: dict[LOPITExperiment, TAGMResult]
    ):
        self.geneid = geneid
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

    def assign(self) -> tuple[str, float]:
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

        if confidence < 0.5:
            return "unknown", -1
        return term, confidence

    def _representation(self):
        return (self.geneid, tuple(self.values()))

    def __hash__(self):
        return hash(self._representation())

    def __lt__(self, other: TAGMResultCollection):
        return self._representation() < other._representation()

    def __eq__(self, other: TAGMResultCollection):
        return self._representation() == other._representation()


class SupervisedTAGMCollection(TSNEPlotMixin):
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
            geneid: TAGMResultCollection(geneid, r)
            for geneid, r in results.items()
        }

        labels = {
            geneid: result.assign()[0]
            for geneid, result in self._results.items()
        }
        self._labels = {
            geneid: label for geneid, label in labels.items()
        }

    def __getitem__(self, gene_id: str):
        return self._labels[gene_id]

    def was_assigned(self, gene_id: str):
        return self._labels[gene_id] != "unknown"

    def __iter__(self):
        return iter(self._labels)

    def __len__(self):
        return len(self._labels)

    def to_dataframe(self, include_tsne=True):
        data = []
        for geneid, result in self.items():
            label, confidence = result.assign()
            is_unknown = label == ""
            entry = {
                "geneid": geneid,
                "label": "" if is_unknown else label,
                "confidence": "" if is_unknown else confidence,
            }
            entry.update({
                f"{experiment}_{key}": value
                for experiment, exp_result in result.items()
                for key, value in (("label", exp_result.label), ("probability", exp_result.probability))
            })
            data.append(entry)
        labels = pd.DataFrame(data).set_index("geneid")

        if include_tsne and self.tsne_analysis is not None:
            return labels.join(
                self.tsne_analysis.to_dataframe(), how="outer")
        return labels
