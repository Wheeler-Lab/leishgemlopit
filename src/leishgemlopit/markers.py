import pandas as pd

from leishgemlopit.constants import ANNOTATIONS_BACKGROUND_LIKE
from leishgemlopit.tsne import TSNEAnalysis
from leishgemlopit.utils import TSNEPlotMixin


class MarkerGenerator:
    def __call__(self, gene_id: str) -> set[str]:
        raise NotImplementedError


class MarkerFactory:
    def __init__(self, generators: list[MarkerGenerator]):
        self.generators = generators

    def __call__(self, gene_id) -> set[str]:
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


class Markers(TSNEPlotMixin):
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
        markers = self._data[gene_id]
        if len(markers) > 1:
            raise ValueError("A gene should only have a single marker annotation.")
        elif len(markers) == 0:
            return None
        return next(iter(markers))

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def to_dataframe(self, include_tsne=True):
        entries: dict[str, str] = []
        for gene_id, markers in self.items():
            for marker in markers:
                entries.append({
                    "geneid": gene_id,
                    "marker": marker,
                })
        df = pd.DataFrame(entries).set_index("geneid")

        if include_tsne and self.tsne_analysis is not None:
            return df.join(self.tsne_analysis.to_dataframe(), how="outer")

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
