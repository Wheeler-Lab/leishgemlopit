from __future__ import annotations
from enum import Enum
from collections import Counter
import pandas as pd
from orthomcl import OrthoMCL

from leishgemlopit.constants import ANNOTATIONS_BACKGROUND_LIKE
from leishgemlopit.tsne import TSNEAnalysis
from leishgemlopit.utils import TSNEPlotMixin


class MarkerConfidence(Enum):
    BACKGROUND_LIKE = 1
    REASONABLE = 2
    EXCELLENT = 3


class Marker:
    def __init__(
        self,
        term: str,
        source: str,
        confidence: MarkerConfidence | None = None,
    ):
        self.term = term
        self.source = source

        if confidence is None:
            if self.term in ANNOTATIONS_BACKGROUND_LIKE:
                self.confidence = MarkerConfidence.BACKGROUND_LIKE
            else:
                self.confidence = MarkerConfidence.REASONABLE
        else:
            self.confidence = confidence


class MarkerCollection:
    def __init__(self, geneid: str, markers: list[Marker] | None = None):
        self.geneid = geneid
        self.markers: list[Marker] = markers if markers else []

    def add_marker(self, marker: Marker):
        self.markers.append(marker)

    def finalise(self, use_multiples=True):
        if not self.markers:
            return None
        candidates = Counter[str]()
        for marker in self.markers:
            candidates[marker.term] += marker.confidence.value
        if (not use_multiples) and (len(candidates) > 1):
            terms = set(candidates)
            if not ANNOTATIONS_BACKGROUND_LIKE.issuperset(terms):
                terms = terms.difference(ANNOTATIONS_BACKGROUND_LIKE)
                if len(terms) > 1:
                    return None
                return list(terms)[0]
            return None

        return candidates.most_common(1)[0][0]

    def merge(self, other: MarkerCollection):
        if self.geneid != other.geneid:
            raise ValueError(
                "Cannot merge MarkerCollections belonging to different genes!")
        self.markers.extend(other.markers)


class MarkerGenerator:
    def all_orthologs_of(self, gene_id: str):
        return {entry.gene_id for entry in OrthoMCL[gene_id].entries}

    def __call__(self, gene_id: str) -> MarkerCollection:
        raise NotImplementedError


class MarkerFactory:
    def __init__(self, generators: list[MarkerGenerator]):
        self.generators = generators

    def __call__(self, gene_id):
        return self.marker_evidence(gene_id).finalise()

    def marker_evidence(self, gene_id: str) -> MarkerCollection:
        markers = None
        for generator in self.generators:
            mc = generator(gene_id)
            if markers is None:
                markers = mc
            else:
                markers.merge(mc)
        return markers


class Markers(TSNEPlotMixin):
    def __init__(
        self,
        marker_factory: MarkerFactory,
        gene_ids: list[str],
        tsne_analysis: TSNEAnalysis | None = None,
    ):
        self._evidence = {
            gene_id: marker_factory.marker_evidence(gene_id)
            for gene_id in gene_ids
        }
        self._data = {
            gene_id: evidence.finalise()
            for gene_id, evidence in
            self._evidence.items()
        }

        self.tsne_analysis = tsne_analysis

    def __getitem__(self, gene_id: str):
        return self._data[gene_id]

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def to_dataframe(self):
        entries: list[dict] = []
        for gene_id, markers in self._evidence.items():
            try:
                description = OrthoMCL.get_entry(gene_id).description
            except KeyError:
                description = ""
            if len(markers.markers) > 0:
                for marker in markers.markers:
                    confidence = (
                        f"{marker.confidence.value} - "
                        f"{marker.confidence.name}"
                    )
                    entries.append({
                        "geneid": gene_id,
                        "description": description,
                        "category": "evidence",
                        "term": marker.term,
                        "source": marker.source,
                        "confidence": confidence,
                    })
                entries.append({
                    "geneid": gene_id,
                    "description": description,
                    "category": "term chosen",
                    "term": markers.finalise(),
                    "source": "",
                    "confidence": "",
                })
        df = (
            pd.DataFrame(entries)
            .set_index(["geneid", "description", "category"])
            .sort_index()
        )

        return df

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
